from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from wadipulse.agent.geo_agent import WadiPulseGeoAgent
from wadipulse.config import STUDY_REGIONS, Settings
from wadipulse.storage.duckdb_store import WadiPulseStore
from wadipulse.vision.inference import WadiPulseInferenceEngine

settings = Settings()
store = WadiPulseStore(settings.db_path)
engine = WadiPulseInferenceEngine(settings)
agent = WadiPulseGeoAgent(store=store, settings=settings)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not store.get_region_summary():
        engine.bootstrap_from_cached_tiles(store=store)
    yield


app = FastAPI(
    title="WadiPulse-GeoAgent API",
    version="0.1.0",
    description="Geospatial Foundation Model (Prithvi-EO-2.0) & Tool-Calling Agent for Arid Agriculture",
    lifespan=lifespan,
)


class AnalyzeRegionRequest(BaseModel):
    region: str = Field(default="wadi_ad_dawasir")
    date_range: str = Field(default="2026-03-01/2026-06-30")


class AgentQueryRequest(BaseModel):
    question: str = Field(min_length=3)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "gfm_model": settings.gfm_model_id,
        "regions": list(STUDY_REGIONS.keys()),
    }


@app.get("/api/v1/ablation")
def get_ablation_results() -> list[dict[str, Any]]:
    path = settings.artifacts_dir / "ablation_results.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/v1/summary")
def get_summary(region: str | None = None) -> list[dict[str, Any]]:
    return store.get_region_summary(region=region)


@app.get("/api/v1/fields")
def get_fields(
    region: str | None = None,
    status: str | None = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    return store.get_fields(region=region, status=status, limit=limit)


@app.post("/api/v1/analyze-region")
def analyze_region(payload: AnalyzeRegionRequest) -> dict[str, Any]:
    try:
        return engine.analyze_region_live(
            region=payload.region,
            date_range=payload.date_range,
            store=store,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/v1/agent/ask")
def ask_agent(payload: AgentQueryRequest) -> dict[str, Any]:
    return agent.ask(payload.question)
