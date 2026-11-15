from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import planetary_computer
import pystac_client
import rasterio
from loguru import logger
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.windows import Window, from_bounds

from wadipulse.config import PRITHVI_BANDS, Settings

VALID_SCL_CLASSES = (2, 4, 5, 6, 7)


@dataclass
class MultispectralChip:
    scene_id: str
    acquisition_date: str
    cloud_cover: float
    crs: str
    transform: Affine
    bounds_wgs84: tuple[float, float, float, float]
    cube: np.ndarray
    valid_mask: np.ndarray


class SentinelSTACLoader:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self._catalog: pystac_client.Client | None = None

    @property
    def catalog(self) -> pystac_client.Client:
        if self._catalog is None:
            self._catalog = pystac_client.Client.open(
                self.settings.stac_url,
                modifier=planetary_computer.sign_inplace,
            )
        return self._catalog

    def search_scenes(
        self,
        bbox: tuple[float, float, float, float],
        date_range: str,
        max_cloud_cover: float | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        cloud_limit = (
            max_cloud_cover if max_cloud_cover is not None else self.settings.max_cloud_cover
        )
        search = self.catalog.search(
            collections=[self.settings.stac_collection],
            bbox=list(bbox),
            datetime=date_range,
            query={"eo:cloud_cover": {"lt": cloud_limit}},
            max_items=limit,
        )
        items = list(search.items())
        items.sort(key=lambda it: it.properties.get("eo:cloud_cover", 100.0))

        summaries: list[dict[str, Any]] = []
        for item in items:
            dt_str = (item.properties.get("datetime") or "")[:10]
            summaries.append(
                {
                    "scene_id": item.id,
                    "acquisition_date": dt_str,
                    "cloud_cover": round(float(item.properties.get("eo:cloud_cover", 0.0)), 2),
                    "bbox": tuple(item.bbox) if item.bbox else bbox,
                    "item": item,
                }
            )
        logger.info(
            f"STAC found {len(summaries)} scenes for date_range={date_range} "
            f"(cloud < {cloud_limit}%)"
        )
        return summaries

    def read_chip_from_item(
        self,
        item: Any,
        bbox_wgs84: tuple[float, float, float, float],
        out_size: tuple[int, int] | None = None,
    ) -> MultispectralChip:
        """Read aligned 6-band reflectance cube + SCL mask for a WGS84 bounding box."""
        signed_item = planetary_computer.sign(item)
        ref_href = signed_item.assets["B02"].href

        with rasterio.open(ref_href) as ref_src:
            transformer = Transformer.from_crs("EPSG:4326", ref_src.crs, always_xy=True)
            minx, miny = transformer.transform(bbox_wgs84[0], bbox_wgs84[1])
            maxx, maxy = transformer.transform(bbox_wgs84[2], bbox_wgs84[3])

            ref_win = from_bounds(minx, miny, maxx, maxy, transform=ref_src.transform)
            ref_win = ref_win.round_offsets().round_lengths()
            ref_win = ref_win.intersection(Window(0, 0, ref_src.width, ref_src.height))

            if out_size is not None:
                target_h, target_w = out_size
            else:
                target_h = max(1, int(ref_win.height))
                target_w = max(1, int(ref_win.width))

            chip_transform = ref_src.window_transform(ref_win)
            if out_size is not None:
                scale_x = ref_win.width / target_w
                scale_y = ref_win.height / target_h
                chip_transform = chip_transform * Affine.scale(scale_x, scale_y)

            crs_str = str(ref_src.crs)

        bands_data: list[np.ndarray] = []
        for band_name in PRITHVI_BANDS:
            href = signed_item.assets[band_name].href
            with rasterio.open(href) as src:
                band_win = from_bounds(minx, miny, maxx, maxy, transform=src.transform)
                band_win = band_win.round_offsets().round_lengths()
                band_win = band_win.intersection(Window(0, 0, src.width, src.height))
                arr = src.read(
                    1,
                    window=band_win,
                    out_shape=(target_h, target_w),
                    resampling=Resampling.bilinear,
                ).astype(np.float32)
                arr = np.clip(arr / 10000.0, 0.0, 1.0)
                bands_data.append(arr)

        cube = np.stack(bands_data, axis=0)

        scl_href = signed_item.assets["SCL"].href
        with rasterio.open(scl_href) as scl_src:
            scl_win = from_bounds(minx, miny, maxx, maxy, transform=scl_src.transform)
            scl_win = scl_win.round_offsets().round_lengths()
            scl_win = scl_win.intersection(Window(0, 0, scl_src.width, scl_src.height))
            scl = scl_src.read(
                1,
                window=scl_win,
                out_shape=(target_h, target_w),
                resampling=Resampling.nearest,
            )
        valid_mask = np.isin(scl, VALID_SCL_CLASSES) & (cube[0] > 0.0)

        dt_str = (signed_item.properties.get("datetime") or "")[:10]
        cloud_pct = float(signed_item.properties.get("eo:cloud_cover", 0.0))

        return MultispectralChip(
            scene_id=signed_item.id,
            acquisition_date=dt_str,
            cloud_cover=cloud_pct,
            crs=crs_str,
            transform=chip_transform,
            bounds_wgs84=bbox_wgs84,
            cube=cube,
            valid_mask=valid_mask,
        )
