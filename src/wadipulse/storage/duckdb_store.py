from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from wadipulse.config import PivotFieldMetric, Settings


class WadiPulseStore:
    """Analytical DuckDB repository for per-pivot biophysical metrics and temporal tracking."""

    def __init__(self, db_path: Path | None = None) -> None:
        settings = Settings()
        self.db_path = db_path or settings.db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self.db_path))

    def _init_schema(self) -> None:
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS pivot_metrics (
                    field_id VARCHAR PRIMARY KEY,
                    region VARCHAR NOT NULL,
                    scene_id VARCHAR NOT NULL,
                    acquisition_date DATE NOT NULL,
                    centroid_lon DOUBLE NOT NULL,
                    centroid_lat DOUBLE NOT NULL,
                    area_ha DOUBLE NOT NULL,
                    mean_ndvi DOUBLE NOT NULL,
                    mean_ndmi DOUBLE NOT NULL,
                    stress_score DOUBLE NOT NULL,
                    stress_status VARCHAR NOT NULL
                )
                """
            )

    def upsert_metrics(self, records: list[PivotFieldMetric]) -> int:
        if not records:
            return 0
        df = pd.DataFrame([r.model_dump() for r in records])
        with self._connect() as con:
            con.register("incoming_df", df)
            con.execute(
                """
                INSERT OR REPLACE INTO pivot_metrics
                SELECT
                    field_id,
                    region,
                    scene_id,
                    CAST(acquisition_date AS DATE),
                    centroid_lon,
                    centroid_lat,
                    area_ha,
                    mean_ndvi,
                    mean_ndmi,
                    stress_score,
                    stress_status
                FROM incoming_df
                """
            )
        return len(records)

    def get_fields(
        self,
        region: str | None = None,
        status: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if region:
            clauses.append("region = ?")
            params.append(region)
        if status:
            clauses.append("stress_status = ?")
            params.append(status)

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"""
            SELECT
                field_id,
                region,
                scene_id,
                CAST(acquisition_date AS VARCHAR) AS acquisition_date,
                centroid_lon,
                centroid_lat,
                area_ha,
                mean_ndvi,
                mean_ndmi,
                stress_score,
                stress_status
            FROM pivot_metrics
            {where_sql}
            ORDER BY stress_score DESC, area_ha DESC
            LIMIT ?
        """
        params.append(limit)
        with self._connect() as con:
            df = con.execute(query, params).fetchdf()
        return df.to_dict(orient="records")

    def get_region_summary(self, region: str | None = None) -> list[dict[str, Any]]:
        where_sql = "WHERE region = ?" if region else ""
        params = [region] if region else []
        query = f"""
            SELECT
                region,
                CAST(acquisition_date AS VARCHAR) AS acquisition_date,
                COUNT(*) AS total_pivots,
                ROUND(SUM(area_ha), 2) AS total_irrigated_ha,
                ROUND(AVG(mean_ndvi), 4) AS avg_ndvi,
                ROUND(AVG(mean_ndmi), 4) AS avg_ndmi,
                ROUND(AVG(stress_score), 3) AS avg_stress_score,
                SUM(CASE WHEN stress_status = 'severe_stress' THEN 1 ELSE 0 END) AS severe_count,
                SUM(CASE WHEN stress_status = 'moderate_stress' THEN 1 ELSE 0 END) AS moderate_count,
                SUM(CASE WHEN stress_status = 'healthy' THEN 1 ELSE 0 END) AS healthy_count
            FROM pivot_metrics
            {where_sql}
            GROUP BY region, acquisition_date
            ORDER BY region, acquisition_date DESC
        """
        with self._connect() as con:
            df = con.execute(query, params).fetchdf()
        return df.to_dict(orient="records")
