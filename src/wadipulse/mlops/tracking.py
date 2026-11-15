from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

from wadipulse.config import Settings


class ExperimentTracker:
    """Dual logger writing run parameters/metrics to MLflow (SQLite) and local JSON manifests."""

    def __init__(self, settings: Settings | None = None) -> None:
        import mlflow

        self._mlflow = mlflow
        self.settings = settings or Settings()
        self.runs_dir = self.settings.artifacts_dir / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        db_uri = f"sqlite:///{(self.settings.project_root / 'mlflow.db').resolve()}"
        artifact_root = (self.settings.project_root / "mlartifacts").resolve().as_uri()
        mlflow.set_tracking_uri(db_uri)
        if mlflow.get_experiment_by_name(self.settings.mlflow_experiment) is None:
            mlflow.create_experiment(
                self.settings.mlflow_experiment,
                artifact_location=artifact_root,
            )
        mlflow.set_experiment(self.settings.mlflow_experiment)

    def log_ablation_run(
        self,
        run_name: str,
        params: dict[str, Any],
        metrics: dict[str, float],
        artifact_paths: list[Path] | None = None,
    ) -> dict[str, Any]:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        manifest: dict[str, Any] = {
            "run_name": run_name,
            "timestamp_utc": timestamp,
            "params": params,
            "metrics": metrics,
        }

        with self._mlflow.start_run(run_name=run_name) as active_run:
            self._mlflow.log_params({k: str(v) for k, v in params.items()})
            self._mlflow.log_metrics(metrics)
            if artifact_paths:
                for path in artifact_paths:
                    if path.exists():
                        self._mlflow.log_artifact(str(path))
            manifest["mlflow_run_id"] = active_run.info.run_id

        out_path = self.runs_dir / f"{run_name}_{timestamp}.json"
        out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        logger.info(f"Logged run '{run_name}' -> {out_path.name} | metrics={metrics}")
        return manifest
