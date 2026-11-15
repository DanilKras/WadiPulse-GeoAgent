from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PRITHVI_BANDS: tuple[str, ...] = ("B02", "B03", "B04", "B8A", "B11", "B12")

STUDY_REGIONS: dict[str, dict[str, object]] = {
    "wadi_ad_dawasir": {
        "name": "Wadi ad-Dawasir Center-Pivot Oasis",
        "bbox": (44.54, 20.36, 44.76, 20.56),
        "center": (20.46, 44.65),
    },
    "al_kharj": {
        "name": "Al-Kharj Agricultural Basin",
        "bbox": (47.32, 24.06, 47.56, 24.26),
        "center": (24.16, 47.44),
    },
}


class PivotFieldMetric(BaseModel):
    field_id: str
    region: str
    scene_id: str
    acquisition_date: str
    centroid_lon: float
    centroid_lat: float
    area_ha: float
    mean_ndvi: float
    mean_ndmi: float
    stress_score: float = Field(ge=0.0, le=1.0)
    stress_status: Literal["healthy", "moderate_stress", "severe_stress"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="WADIPULSE_",
        env_file=".env",
        extra="ignore",
    )

    project_root: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = project_root / "data"
    few_shot_dir: Path = data_dir / "few_shot_tiles"
    artifacts_dir: Path = data_dir / "artifacts"
    db_path: Path = data_dir / "wadipulse.duckdb"

    stac_url: str = "https://planetarycomputer.microsoft.com/api/stac/v1"
    stac_collection: str = "sentinel-2-l2a"
    chip_size: int = 224
    pixel_size_m: float = 10.0
    max_cloud_cover: float = 10.0

    gfm_model_id: str = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M"
    ndmi_stress_threshold: float = 0.08
    ndvi_active_threshold: float = 0.25
    min_pivot_area_ha: float = 4.0

    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"
    llm_model: str = "qwen2.5:7b-instruct"
    mlflow_experiment: str = "WadiPulse-GeoAgent"
