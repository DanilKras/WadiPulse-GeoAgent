from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from pyproj import Transformer
from rasterio.transform import Affine, xy
from scipy import ndimage
from shapely.geometry import Polygon, mapping
from shapely.ops import transform as shapely_transform

from wadipulse.config import PivotFieldMetric

IDX_RED = 2
IDX_NIR = 3
IDX_SWIR1 = 4


def compute_spectral_indices(cube: np.ndarray) -> dict[str, np.ndarray]:
    red = cube[IDX_RED]
    nir = cube[IDX_NIR]
    swir1 = cube[IDX_SWIR1]

    ndvi = (nir - red) / np.maximum(nir + red, 1e-6)
    ndmi = (nir - swir1) / np.maximum(nir + swir1, 1e-6)

    return {
        "ndvi": np.clip(ndvi, -1.0, 1.0).astype(np.float32),
        "ndmi": np.clip(ndmi, -1.0, 1.0).astype(np.float32),
    }


def extract_pivot_metrics(
    mask: np.ndarray,
    cube: np.ndarray,
    transform: Affine,
    crs: str,
    region: str,
    scene_id: str,
    acquisition_date: str,
    pixel_size_m: float = 10.0,
    min_area_ha: float = 4.0,
    ndmi_stress_threshold: float = 0.08,
) -> tuple[list[PivotFieldMetric], dict[str, Any]]:
    """Extract per-pivot biophysical metrics and a WGS84 GeoJSON FeatureCollection."""
    indices = compute_spectral_indices(cube)
    ndvi_map = indices["ndvi"]
    ndmi_map = indices["ndmi"]

    labeled, num_features = ndimage.label(mask.astype(np.uint8))
    to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform
    px_area_ha = (pixel_size_m**2) / 10_000.0

    metrics: list[PivotFieldMetric] = []
    features: list[dict[str, Any]] = []

    for comp_id in range(1, num_features + 1):
        comp_mask = labeled == comp_id
        px_count = int(comp_mask.sum())
        area_ha = px_count * px_area_ha
        if area_ha < min_area_ha:
            continue

        mean_ndvi = float(np.nanmean(ndvi_map[comp_mask]))
        mean_ndmi = float(np.nanmean(ndmi_map[comp_mask]))

        raw_stress = (ndmi_stress_threshold + 0.17 - mean_ndmi) / 0.25
        stress_score = float(np.clip(raw_stress, 0.0, 1.0))

        if stress_score >= 0.65:
            status = "severe_stress"
        elif stress_score >= 0.35:
            status = "moderate_stress"
        else:
            status = "healthy"

        rows, cols = np.where(comp_mask)
        cx_utm, cy_utm = xy(transform, float(rows.mean()), float(cols.mean()))
        cx_lon, cy_lat = to_wgs84(cx_utm, cy_utm)

        # Build simplified contour polygon in WGS84 for interactive map overlay
        contours, _ = cv2.findContours(
            comp_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        largest = max(contours, key=cv2.contourArea)
        if len(largest) < 3:
            continue

        utm_pts = [xy(transform, float(pt[0][1]), float(pt[0][0])) for pt in largest]
        poly_utm = Polygon(utm_pts).buffer(0)
        if poly_utm.is_empty:
            continue
        poly_wgs84 = shapely_transform(to_wgs84, poly_utm)

        short_scene = scene_id.split("_")[-1] if "_" in scene_id else scene_id
        field_id = f"{region}_{acquisition_date}_{short_scene}_{comp_id:03d}"
        metric = PivotFieldMetric(
            field_id=field_id,
            region=region,
            scene_id=scene_id,
            acquisition_date=acquisition_date,
            centroid_lon=round(float(cx_lon), 6),
            centroid_lat=round(float(cy_lat), 6),
            area_ha=round(area_ha, 2),
            mean_ndvi=round(mean_ndvi, 4),
            mean_ndmi=round(mean_ndmi, 4),
            stress_score=round(stress_score, 3),
            stress_status=status,
        )
        metrics.append(metric)
        features.append(
            {
                "type": "Feature",
                "geometry": mapping(poly_wgs84),
                "properties": metric.model_dump(),
            }
        )

    geojson = {"type": "FeatureCollection", "features": features}
    return metrics, geojson
