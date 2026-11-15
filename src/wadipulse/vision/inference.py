from __future__ import annotations

import json
from typing import Any

import numpy as np
import torch
from loguru import logger
from rasterio.transform import Affine

from wadipulse.config import STUDY_REGIONS, PivotFieldMetric, Settings
from wadipulse.data.stac_loader import SentinelSTACLoader
from wadipulse.storage.duckdb_store import WadiPulseStore
from wadipulse.vision.biophysical import compute_spectral_indices, extract_pivot_metrics
from wadipulse.vision.gfm_prithvi import PrithviFewShotSegmentor


class WadiPulseInferenceEngine:
    """End-to-end inference runner combining Prithvi-EO-2.0 segmentation and biophysical extraction."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.device = torch.device("cpu")
        self._model: PrithviFewShotSegmentor | None = None

    @property
    def model(self) -> PrithviFewShotSegmentor:
        if self._model is None:
            seg = PrithviFewShotSegmentor(
                model_id=self.settings.gfm_model_id,
                freeze_encoder=True,
            )
            ckpt_path = self.settings.artifacts_dir / "prithvi_decoder_20shot.pt"
            if ckpt_path.exists():
                state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                seg.load_state_dict(state, strict=False)
                logger.info(f"Loaded fine-tuned decoder weights from {ckpt_path.name}")
            else:
                logger.warning("Decoder checkpoint not found; using untrained decoder head")
            seg.to(self.device).eval()
            self._model = seg
        return self._model

    @torch.no_grad()
    def predict_mask(self, cube: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Run 224x224 tiled inference over a (6, H, W) reflectance cube."""
        _, h, w = cube.shape
        chip_sz = self.settings.chip_size
        prob_map = np.zeros((h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)

        row_starts = list(range(0, max(1, h - chip_sz + 1), chip_sz))
        if row_starts[-1] + chip_sz < h:
            row_starts.append(max(0, h - chip_sz))

        col_starts = list(range(0, max(1, w - chip_sz + 1), chip_sz))
        if col_starts[-1] + chip_sz < w:
            col_starts.append(max(0, w - chip_sz))

        for r0 in row_starts:
            for c0 in col_starts:
                patch = np.zeros((6, chip_sz, chip_sz), dtype=np.float32)
                sub = cube[:, r0 : r0 + chip_sz, c0 : c0 + chip_sz]
                _, sh, sw = sub.shape
                patch[:, :sh, :sw] = sub

                tensor_in = torch.from_numpy(patch).unsqueeze(0).to(self.device)
                probs = torch.sigmoid(self.model(tensor_in)).squeeze().cpu().numpy()

                prob_map[r0 : r0 + sh, c0 : c0 + sw] += probs[:sh, :sw]
                count_map[r0 : r0 + sh, c0 : c0 + sw] += 1.0

        avg_probs = prob_map / np.maximum(count_map, 1.0)
        return (avg_probs >= threshold).astype(np.uint8)

    def analyze_region_live(
        self,
        region: str,
        date_range: str = "2026-03-01/2026-06-30",
        store: WadiPulseStore | None = None,
    ) -> dict[str, Any]:
        if region not in STUDY_REGIONS:
            raise ValueError(f"Unknown region '{region}'. Choose from {list(STUDY_REGIONS.keys())}")

        info = STUDY_REGIONS[region]
        bbox = info["bbox"]
        loader = SentinelSTACLoader(self.settings)
        scenes = loader.search_scenes(
            bbox=bbox, date_range=date_range, max_cloud_cover=10.0, limit=3
        )
        if not scenes:
            raise RuntimeError(
                f"No cloud-free Sentinel-2 scenes found for {region} in {date_range}"
            )

        best = scenes[0]
        chip = loader.read_chip_from_item(best["item"], bbox_wgs84=bbox)
        pred_mask = self.predict_mask(chip.cube) * chip.valid_mask.astype(np.uint8)
        indices = compute_spectral_indices(chip.cube)

        metrics, geojson = extract_pivot_metrics(
            mask=pred_mask,
            cube=chip.cube,
            transform=chip.transform,
            crs=chip.crs,
            region=region,
            scene_id=chip.scene_id,
            acquisition_date=chip.acquisition_date,
            pixel_size_m=self.settings.pixel_size_m,
            min_area_ha=self.settings.min_pivot_area_ha,
            ndmi_stress_threshold=self.settings.ndmi_stress_threshold,
        )

        db = store or WadiPulseStore(self.settings.db_path)
        db.upsert_metrics(metrics)

        return {
            "region": region,
            "region_name": info["name"],
            "scene_id": chip.scene_id,
            "acquisition_date": chip.acquisition_date,
            "cloud_cover": chip.cloud_cover,
            "bounds_wgs84": list(chip.bounds_wgs84),
            "field_count": len(metrics),
            "metrics": [m.model_dump() for m in metrics],
            "geojson": geojson,
            "ndvi_mean": round(
                float(np.nanmean(indices["ndvi"][pred_mask > 0])) if pred_mask.any() else 0.0, 4
            ),
            "ndmi_mean": round(
                float(np.nanmean(indices["ndmi"][pred_mask > 0])) if pred_mask.any() else 0.0, 4
            ),
        }

    def bootstrap_from_cached_tiles(self, store: WadiPulseStore | None = None) -> int:
        """Run fast offline Prithvi inference on cached Saudi tiles to populate DuckDB immediately."""
        manifest_path = self.settings.few_shot_dir / "manifest.json"
        if not manifest_path.exists():
            return 0
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        db = store or WadiPulseStore(self.settings.db_path)
        all_metrics: list[PivotFieldMetric] = []

        for entry in manifest:
            npz = np.load(self.settings.few_shot_dir / entry["file"])
            cube = npz["cube"].astype(np.float32)
            t_arr = npz["transform"]
            affine = Affine(*t_arr.tolist())
            pred_mask = self.predict_mask(cube)

            tile_metrics, _ = extract_pivot_metrics(
                mask=pred_mask,
                cube=cube,
                transform=affine,
                crs=str(entry["crs"]),
                region=str(entry["region"]),
                scene_id=str(entry["tile_id"]),
                acquisition_date=str(entry["acquisition_date"]),
                pixel_size_m=self.settings.pixel_size_m,
                min_area_ha=self.settings.min_pivot_area_ha,
                ndmi_stress_threshold=self.settings.ndmi_stress_threshold,
            )
            all_metrics.extend(tile_metrics)

        return db.upsert_metrics(all_metrics)
