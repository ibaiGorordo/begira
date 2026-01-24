from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass

import numpy as np


@dataclass
class PointCloud:
    id: str
    name: str
    positions: np.ndarray  # float32 (n,3)
    colors: np.ndarray | None  # uint8 (n,3)
    point_size: float
    revision: int
    created_at: float
    updated_at: float


class InMemoryRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._pcs: dict[str, PointCloud] = {}
        self._global_revision = 0
        # Deterministic palette colors for clouds that don't provide per-point colors.
        # Colors are uint8 RGB.
        self._default_palette: list[tuple[int, int, int]] = [
            (31, 119, 180),  # blue
            (255, 127, 14),  # orange
            (44, 160, 44),  # green
            (214, 39, 40),  # red
            (148, 103, 189),  # purple
            (140, 86, 75),  # brown
            (227, 119, 194),  # pink
            (127, 127, 127),  # gray
            (188, 189, 34),  # olive
            (23, 190, 207),  # cyan
        ]

    def _palette_color_for_new_cloud(self) -> tuple[int, int, int]:
        # Choose based on current number of clouds (creation order).
        idx = len(self._pcs) % len(self._default_palette)
        return self._default_palette[idx]

    def global_revision(self) -> int:
        with self._lock:
            return self._global_revision

    def list(self) -> list[PointCloud]:
        with self._lock:
            return list(self._pcs.values())

    def get(self, cloud_id: str) -> PointCloud | None:
        with self._lock:
            return self._pcs.get(cloud_id)

    def update_settings(self, cloud_id: str, *, point_size: float | None = None) -> PointCloud:
        if point_size is not None and (not np.isfinite(point_size) or point_size <= 0):
            raise ValueError("point_size must be a finite positive number")

        with self._lock:
            prev = self._pcs.get(cloud_id)
            if prev is None:
                raise KeyError(cloud_id)

            self._global_revision += 1
            now = time.time()

            pc = PointCloud(
                id=prev.id,
                name=prev.name,
                positions=prev.positions,
                colors=prev.colors,
                point_size=float(point_size) if point_size is not None else prev.point_size,
                revision=prev.revision + 1,
                created_at=prev.created_at,
                updated_at=now,
            )
            self._pcs[cloud_id] = pc
            return pc

    def upsert(
        self,
        *,
        name: str,
        positions: np.ndarray,
        colors: np.ndarray | None,
        point_size: float | None = None,
        cloud_id: str | None = None,
    ) -> PointCloud:
        # Validate + normalize early.
        pos = np.asarray(positions)
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError(f"positions must have shape (N,3), got {pos.shape}")
        pos = np.ascontiguousarray(pos, dtype=np.float32)

        col: np.ndarray | None = None
        if colors is not None:
            c = np.asarray(colors)
            if c.shape != pos.shape:
                raise ValueError(f"colors must have shape {pos.shape}, got {c.shape}")
            if np.issubdtype(c.dtype, np.floating):
                # Accept [0,1] floats.
                c = np.clip(c, 0.0, 1.0) * 255.0
            col = np.ascontiguousarray(c, dtype=np.uint8)

        if point_size is not None and (not np.isfinite(point_size) or point_size <= 0):
            raise ValueError("point_size must be a finite positive number")

        with self._lock:
            if cloud_id is None:
                cloud_id = uuid.uuid4().hex

            self._global_revision += 1
            now = time.time()
            prev = self._pcs.get(cloud_id)

            # New cloud and no explicit colors: assign a deterministic per-cloud palette color.
            if prev is None and col is None:
                r, g, b = self._palette_color_for_new_cloud()
                col = np.empty((pos.shape[0], 3), dtype=np.uint8)
                col[:, 0] = r
                col[:, 1] = g
                col[:, 2] = b

            if prev is None:
                pc = PointCloud(
                    id=cloud_id,
                    name=name,
                    positions=pos,
                    colors=col,
                    point_size=float(point_size) if point_size is not None else 0.02,
                    revision=1,
                    created_at=now,
                    updated_at=now,
                )
            else:
                pc = PointCloud(
                    id=prev.id,
                    name=name,
                    positions=pos,
                    colors=col,
                    point_size=float(point_size) if point_size is not None else prev.point_size,
                    revision=prev.revision + 1,
                    created_at=prev.created_at,
                    updated_at=now,
                )

            self._pcs[cloud_id] = pc
            return pc


REGISTRY = InMemoryRegistry()
