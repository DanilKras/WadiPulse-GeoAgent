from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wadipulse.config import STUDY_REGIONS, Settings
from wadipulse.data.stac_loader import SentinelSTACLoader
from wadipulse.vision.biophysical import compute_spectral_indices


def build_pivot_reference_mask(cube: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Generate clean binary masks of irrigated center-pivot fields against arid desert sand."""
    indices = compute_spectral_indices(cube)
    ndvi = indices["ndvi"]
    ndmi = indices["ndmi"]

    raw_veg = (ndvi > 0.24) & (ndmi > -0.08) & valid_mask
    mask_u8 = raw_veg.astype(np.uint8) * 255

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel_close)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_open)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned)
    final_mask = np.zeros_like(mask_u8, dtype=np.uint8)
    for lbl in range(1, num_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= 300:
            final_mask[labels == lbl] = 1
    return final_mask


def main() -> None:
    settings = Settings()
    settings.few_shot_dir.mkdir(parents=True, exist_ok=True)
    loader = SentinelSTACLoader(settings)

    chip_size = settings.chip_size
    manifest: list[dict[str, object]] = []
    saved_count = 0

    for region_key, info in STUDY_REGIONS.items():
        bbox = info["bbox"]
        scenes = loader.search_scenes(
            bbox=bbox,
            date_range="2026-03-01/2026-06-30",
            max_cloud_cover=5.0,
            limit=3,
        )
        if not scenes:
            logger.warning(f"No scenes found for {region_key}")
            continue

        best_scene = scenes[0]
        logger.info(
            f"Downloading AOI cube for {region_key} from {best_scene['scene_id']} "
            f"({best_scene['acquisition_date']}, cloud={best_scene['cloud_cover']}%)"
        )
        aoi_chip = loader.read_chip_from_item(best_scene["item"], bbox_wgs84=bbox)
        _, h, w = aoi_chip.cube.shape
        to_wgs84 = Transformer.from_crs(aoi_chip.crs, "EPSG:4326", always_xy=True)

        region_saved = 0
        for r0 in range(0, h - chip_size + 1, chip_size):
            for c0 in range(0, w - chip_size + 1, chip_size):
                if region_saved >= 15:
                    break
                sub_cube = aoi_chip.cube[:, r0 : r0 + chip_size, c0 : c0 + chip_size]
                sub_valid = aoi_chip.valid_mask[r0 : r0 + chip_size, c0 : c0 + chip_size]
                if sub_valid.mean() < 0.95:
                    continue

                mask = build_pivot_reference_mask(sub_cube, sub_valid)
                fg_ratio = float(mask.mean())
                if not (0.05 <= fg_ratio <= 0.75):
                    continue

                tile_transform = aoi_chip.transform * aoi_chip.transform.translation(c0, r0)
                minx, maxy = tile_transform * (0, 0)
                maxx, miny = tile_transform * (chip_size, chip_size)
                lon_min, lat_min = to_wgs84.transform(minx, miny)
                lon_max, lat_max = to_wgs84.transform(maxx, maxy)

                tile_id = f"{region_key}_{saved_count:02d}"
                tile_path = settings.few_shot_dir / f"{tile_id}.npz"
                np.savez_compressed(
                    tile_path,
                    cube=sub_cube.astype(np.float32),
                    mask=mask.astype(np.uint8),
                    transform=np.array(tile_transform[:6], dtype=np.float64),
                )

                manifest.append(
                    {
                        "tile_id": tile_id,
                        "region": region_key,
                        "scene_id": aoi_chip.scene_id,
                        "acquisition_date": aoi_chip.acquisition_date,
                        "crs": aoi_chip.crs,
                        "bbox_wgs84": [
                            round(lon_min, 6),
                            round(lat_min, 6),
                            round(lon_max, 6),
                            round(lat_max, 6),
                        ],
                        "fg_ratio": round(fg_ratio, 4),
                        "file": tile_path.name,
                    }
                )
                saved_count += 1
                region_saved += 1

    manifest_path = settings.few_shot_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(f"Saved {saved_count} few-shot tiles -> {manifest_path}")


if __name__ == "__main__":
    main()
