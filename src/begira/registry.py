from __future__ import annotations

import threading
import time
import uuid

import numpy as np

from .elements import ElementBase, PointCloudElement


class InMemoryRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._elements: dict[str, ElementBase] = {}
        self._global_revision = 0
        # Deterministic palette colors for point clouds that don't provide per-point colors.
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

    def _palette_color_for_new_pointcloud(self) -> tuple[int, int, int]:
        idx = len([e for e in self._elements.values() if isinstance(e, PointCloudElement)]) % len(self._default_palette)
        return self._default_palette[idx]

    def global_revision(self) -> int:
        with self._lock:
            return self._global_revision

    def list_elements(self) -> list[ElementBase]:
        with self._lock:
            return list(self._elements.values())

    def get_element(self, element_id: str) -> ElementBase | None:
        with self._lock:
            return self._elements.get(element_id)

    def update_pointcloud_settings(self, element_id: str, *, point_size: float | None = None) -> PointCloudElement:
        if point_size is not None and (not np.isfinite(point_size) or point_size <= 0):
            raise ValueError("point_size must be a finite positive number")

        with self._lock:
            prev = self._elements.get(element_id)
            if not isinstance(prev, PointCloudElement):
                raise KeyError(element_id)

            self._global_revision += 1
            now = time.time()

            pc = PointCloudElement(
                id=prev.id,
                type="pointcloud",
                name=prev.name,
                positions=prev.positions,
                colors=prev.colors,
                point_size=float(point_size) if point_size is not None else prev.point_size,
                revision=prev.revision + 1,
                created_at=prev.created_at,
                updated_at=now,
            )
            self._elements[element_id] = pc
            return pc

    def upsert_pointcloud(
        self,
        *,
        name: str,
        positions: np.ndarray,
        colors: np.ndarray | None,
        point_size: float | None = None,
        element_id: str | None = None,
    ) -> PointCloudElement:
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
                c = np.clip(c, 0.0, 1.0) * 255.0
            col = np.ascontiguousarray(c, dtype=np.uint8)

        if point_size is not None and (not np.isfinite(point_size) or point_size <= 0):
            raise ValueError("point_size must be a finite positive number")

        with self._lock:
            if element_id is None:
                element_id = uuid.uuid4().hex

            self._global_revision += 1
            now = time.time()
            prev = self._elements.get(element_id)
            prev_pc = prev if isinstance(prev, PointCloudElement) else None

            # New pointcloud and no explicit colors: assign a deterministic per-cloud palette color.
            if prev_pc is None and col is None:
                r, g, b = self._palette_color_for_new_pointcloud()
                col = np.empty((pos.shape[0], 3), dtype=np.uint8)
                col[:, 0] = r
                col[:, 1] = g
                col[:, 2] = b

            if prev_pc is None:
                pc = PointCloudElement(
                    id=element_id,
                    type="pointcloud",
                    name=name,
                    positions=pos,
                    colors=col,
                    point_size=float(point_size) if point_size is not None else 0.02,
                    revision=1,
                    created_at=now,
                    updated_at=now,
                )
            else:
                pc = PointCloudElement(
                    id=prev_pc.id,
                    type="pointcloud",
                    name=name,
                    positions=pos,
                    colors=col,
                    point_size=float(point_size) if point_size is not None else prev_pc.point_size,
                    revision=prev_pc.revision + 1,
                    created_at=prev_pc.created_at,
                    updated_at=now,
                )

            self._elements[element_id] = pc
            return pc


REGISTRY = InMemoryRegistry()
