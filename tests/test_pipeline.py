from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from wadipulse.agent.geo_agent import WadiPulseGeoAgent
from wadipulse.api.main import app
from wadipulse.config import PivotFieldMetric
from wadipulse.storage.duckdb_store import WadiPulseStore
from wadipulse.vision.biophysical import compute_spectral_indices


def test_compute_spectral_indices() -> None:
    cube = np.zeros((6, 16, 16), dtype=np.float32)
    cube[2] = 0.08
    cube[3] = 0.48
    cube[4] = 0.20

    indices = compute_spectral_indices(cube)
    assert np.isclose(indices["ndvi"].mean(), (0.48 - 0.08) / (0.48 + 0.08), atol=1e-4)
    assert np.isclose(indices["ndmi"].mean(), (0.48 - 0.20) / (0.48 + 0.20), atol=1e-4)


def test_duckdb_store_and_agent_synthesis(tmp_path: Path) -> None:
    db_path = tmp_path / "test_wadipulse.duckdb"
    store = WadiPulseStore(db_path=db_path)

    sample = [
        PivotFieldMetric(
            field_id="wadi_ad_dawasir_2026-06-30_00_001",
            region="wadi_ad_dawasir",
            scene_id="wadi_ad_dawasir_00",
            acquisition_date="2026-06-30",
            centroid_lon=44.65,
            centroid_lat=20.46,
            area_ha=48.5,
            mean_ndvi=0.31,
            mean_ndmi=-0.01,
            stress_score=0.95,
            stress_status="severe_stress",
        )
    ]
    assert store.upsert_metrics(sample) == 1
    summary = store.get_region_summary("wadi_ad_dawasir")
    assert summary[0]["total_pivots"] == 1
    assert summary[0]["severe_count"] == 1

    agent = WadiPulseGeoAgent(store=store)
    res = agent.ask("Which fields have severe stress?")
    assert "wadi_ad_dawasir_2026-06-30_00_001" in res["answer"]


def test_fastapi_endpoints() -> None:
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

    ablation_resp = client.get("/api/v1/ablation")
    assert ablation_resp.status_code == 200
