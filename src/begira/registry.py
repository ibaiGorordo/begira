from __future__ import annotations

import threading
import time
import uuid
from dataclasses import replace

import numpy as np

from .elements import ElementBase, PointCloudElement, GaussianSplatElement, CameraElement, ImageElement


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

    @staticmethod
    def _normalize_intrinsic_matrix(
        intrinsic_matrix: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
        | list[list[float]]
        | np.ndarray
        | None,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] | None:
        if intrinsic_matrix is None:
            return None
        k = np.asarray(intrinsic_matrix, dtype=np.float64)
        if k.shape != (3, 3):
            raise ValueError(f"intrinsic_matrix must have shape (3, 3), got {k.shape}")
        return (
            (float(k[0, 0]), float(k[0, 1]), float(k[0, 2])),
            (float(k[1, 0]), float(k[1, 1]), float(k[1, 2])),
            (float(k[2, 0]), float(k[2, 1]), float(k[2, 2])),
        )

    @staticmethod
    def _infer_fov_from_intrinsics(
        intrinsic_matrix: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] | None,
        height: int | None,
    ) -> float | None:
        if intrinsic_matrix is None or height is None:
            return None
        fy = float(intrinsic_matrix[1][1])
        if not np.isfinite(fy) or fy <= 0:
            raise ValueError("intrinsic_matrix[1][1] (fy) must be finite and > 0")
        if int(height) <= 0:
            raise ValueError("height must be a positive integer")
        return float(np.degrees(2.0 * np.arctan(float(height) / (2.0 * fy))))

    def global_revision(self) -> int:
        with self._lock:
            return self._global_revision

    def list_elements(self) -> list[ElementBase]:
        with self._lock:
            return [e for e in self._elements.values() if not getattr(e, "deleted", False)]

    def get_element(self, element_id: str) -> ElementBase | None:
        with self._lock:
            return self._elements.get(element_id)

    def reset(self) -> None:
        with self._lock:
            now = time.time()
            for key, prev in list(self._elements.items()):
                updated = replace(
                    prev,
                    deleted=False,
                    visible=True,
                    position=(0.0, 0.0, 0.0),
                    rotation=(0.0, 0.0, 0.0, 1.0),
                    revision=prev.revision + 1,
                    updated_at=now,
                )
                self._elements[key] = updated
            self._global_revision += 1

    def delete_element(self, element_id: str) -> ElementBase:
        with self._lock:
            prev = self._elements.get(element_id)
            if prev is None:
                raise KeyError(element_id)
            if getattr(prev, "deleted", False):
                return prev
            self._global_revision += 1
            now = time.time()
            updated = replace(prev, deleted=True, revision=prev.revision + 1, updated_at=now)
            self._elements[element_id] = updated
            return updated

    def update_element_meta(
        self,
        element_id: str,
        *,
        position: tuple[float, float, float] | None = None,
        rotation: tuple[float, float, float, float] | None = None,
        visible: bool | None = None,
        deleted: bool | None = None,
        point_size: float | None = None,
        fov: float | None = None,
        near: float | None = None,
        far: float | None = None,
        width: int | None = None,
        height: int | None = None,
        intrinsic_matrix: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
        | list[list[float]]
        | np.ndarray
        | None = None,
    ) -> ElementBase:
        with self._lock:
            prev = self._elements.get(element_id)
            if prev is None:
                raise KeyError(element_id)

            updates: dict[str, object] = {}
            if position is not None:
                updates["position"] = position
            if rotation is not None:
                updates["rotation"] = rotation
            if visible is not None:
                updates["visible"] = visible
            if deleted is not None:
                updates["deleted"] = deleted

            if isinstance(prev, PointCloudElement):
                if point_size is not None:
                    if not np.isfinite(point_size) or point_size <= 0:
                        raise ValueError("point_size must be a finite positive number")
                    updates["point_size"] = float(point_size)
            elif isinstance(prev, GaussianSplatElement):
                if point_size is not None:
                    if not np.isfinite(point_size) or point_size <= 0:
                        raise ValueError("point_size must be a finite positive number")
                    updates["point_size"] = float(point_size)
            elif isinstance(prev, CameraElement):
                if fov is not None:
                    updates["fov"] = float(fov)
                if near is not None:
                    updates["near"] = float(near)
                if far is not None:
                    updates["far"] = float(far)
                if width is not None:
                    width_i = int(width)
                    if width_i <= 0:
                        raise ValueError("width must be a positive integer")
                    updates["width"] = width_i
                if height is not None:
                    height_i = int(height)
                    if height_i <= 0:
                        raise ValueError("height must be a positive integer")
                    updates["height"] = height_i
                if intrinsic_matrix is not None:
                    updates["intrinsic_matrix"] = self._normalize_intrinsic_matrix(intrinsic_matrix)
            elif isinstance(prev, ImageElement):
                # No image-specific mutable fields yet (payload upserts replace the element).
                pass

            if not updates:
                return prev

            if isinstance(prev, CameraElement):
                next_near = float(updates.get("near", prev.near))
                next_far = float(updates.get("far", prev.far))
                if not np.isfinite(next_near) or next_near <= 0:
                    raise ValueError("near must be a finite positive number")
                if not np.isfinite(next_far) or next_far <= next_near:
                    raise ValueError("far must be finite and greater than near")

            self._global_revision += 1
            now = time.time()
            updated = replace(prev, **updates, revision=prev.revision + 1, updated_at=now)
            self._elements[element_id] = updated
            return updated

    def update_pointcloud_settings(self, element_id: str, *, point_size: float | None = None) -> PointCloudElement:
        updated = self.update_element_meta(element_id, point_size=point_size)
        if not isinstance(updated, PointCloudElement):
            raise KeyError(element_id)
        return updated

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
                    position=(0.0, 0.0, 0.0),
                    rotation=(0.0, 0.0, 0.0, 1.0),
                    visible=True,
                    deleted=False,
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
                    position=prev_pc.position,
                    rotation=prev_pc.rotation,
                    visible=prev_pc.visible,
                    deleted=prev_pc.deleted,
                )

            self._elements[element_id] = pc
            return pc

    def update_gaussians_settings(self, element_id: str, *, point_size: float | None = None) -> GaussianSplatElement:
        updated = self.update_element_meta(element_id, point_size=point_size)
        if not isinstance(updated, GaussianSplatElement):
            raise KeyError(element_id)
        return updated

    def upsert_gaussians(
        self,
        *,
        name: str,
        positions: np.ndarray,
        sh0: np.ndarray,
        opacity: np.ndarray,
        scales: np.ndarray,
        rotations: np.ndarray,
        point_size: float | None = None,
        element_id: str | None = None,
    ) -> GaussianSplatElement:
        with self._lock:
            if element_id is None:
                element_id = uuid.uuid4().hex

            self._global_revision += 1
            now = time.time()
            prev = self._elements.get(element_id)
            prev_gs = prev if isinstance(prev, GaussianSplatElement) else None

            if prev_gs is None:
                gs = GaussianSplatElement(
                    id=element_id,
                    type="gaussians",
                    name=name,
                    positions=positions,
                    sh0=sh0,
                    opacity=opacity,
                    scales=scales,
                    rotations=rotations,
                    point_size=float(point_size) if point_size is not None else 1.0,
                    revision=1,
                    created_at=now,
                    updated_at=now,
                    position=(0.0, 0.0, 0.0),
                    rotation=(0.0, 0.0, 0.0, 1.0),
                    visible=True,
                    deleted=False,
                )
            else:
                gs = GaussianSplatElement(
                    id=prev_gs.id,
                    type="gaussians",
                    name=name,
                    positions=positions,
                    sh0=sh0,
                    opacity=opacity,
                    scales=scales,
                    rotations=rotations,
                    point_size=float(point_size) if point_size is not None else prev_gs.point_size,
                    revision=prev_gs.revision + 1,
                    created_at=prev_gs.created_at,
                    updated_at=now,
                    position=prev_gs.position,
                    rotation=prev_gs.rotation,
                    visible=prev_gs.visible,
                    deleted=prev_gs.deleted,
                )

            self._elements[element_id] = gs
            return gs

    def upsert_camera(
        self,
        *,
        name: str,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        fov: float | None = None,
        near: float | None = None,
        far: float | None = None,
        width: int | None = None,
        height: int | None = None,
        intrinsic_matrix: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
        | list[list[float]]
        | np.ndarray
        | None = None,
        element_id: str | None = None,
    ) -> CameraElement:
        k = self._normalize_intrinsic_matrix(intrinsic_matrix)

        with self._lock:
            if element_id is None:
                element_id = uuid.uuid4().hex

            self._global_revision += 1
            now = time.time()
            prev = self._elements.get(element_id)
            prev_cam = prev if isinstance(prev, CameraElement) else None

            width_v = int(width) if width is not None else (prev_cam.width if prev_cam is not None else None)
            if width_v is not None and width_v <= 0:
                raise ValueError("width must be a positive integer")

            height_v = int(height) if height is not None else (prev_cam.height if prev_cam is not None else None)
            if height_v is not None and height_v <= 0:
                raise ValueError("height must be a positive integer")

            k_v = k if k is not None else (prev_cam.intrinsic_matrix if prev_cam is not None else None)

            if fov is None:
                inferred_fov = self._infer_fov_from_intrinsics(k_v, height_v)
                fov_v = float(inferred_fov) if inferred_fov is not None else (prev_cam.fov if prev_cam is not None else 60.0)
            else:
                fov_v = float(fov)

            near_v = float(near) if near is not None else (prev_cam.near if prev_cam is not None else 0.01)
            far_v = float(far) if far is not None else (prev_cam.far if prev_cam is not None else 1000.0)
            if not np.isfinite(near_v) or near_v <= 0:
                raise ValueError("near must be a finite positive number")
            if not np.isfinite(far_v) or far_v <= near_v:
                raise ValueError("far must be finite and greater than near")

            if prev_cam is None:
                cam = CameraElement(
                    id=element_id,
                    type="camera",
                    name=name,
                    fov=fov_v,
                    near=near_v,
                    far=far_v,
                    width=width_v,
                    height=height_v,
                    intrinsic_matrix=k_v,
                    revision=1,
                    created_at=now,
                    updated_at=now,
                    position=position,
                    rotation=rotation,
                    visible=True,
                    deleted=False,
                )
            else:
                cam = CameraElement(
                    id=prev_cam.id,
                    type="camera",
                    name=name,
                    fov=fov_v,
                    near=near_v,
                    far=far_v,
                    width=width_v,
                    height=height_v,
                    intrinsic_matrix=k_v,
                    revision=prev_cam.revision + 1,
                    created_at=prev_cam.created_at,
                    updated_at=now,
                    position=position,
                    rotation=rotation,
                    visible=prev_cam.visible,
                    deleted=prev_cam.deleted,
                )

            self._elements[element_id] = cam
            return cam

    def upsert_image(
        self,
        *,
        name: str,
        image_bytes: bytes,
        mime_type: str,
        width: int,
        height: int,
        channels: int,
        element_id: str | None = None,
    ) -> ImageElement:
        if not image_bytes:
            raise ValueError("image payload is empty")
        if not isinstance(mime_type, str) or "/" not in mime_type:
            raise ValueError("mime_type must be a valid media type, e.g. image/png")
        width_i = int(width)
        height_i = int(height)
        channels_i = int(channels)
        if width_i <= 0 or height_i <= 0:
            raise ValueError("image width/height must be positive integers")
        if channels_i <= 0:
            raise ValueError("image channels must be a positive integer")

        with self._lock:
            if element_id is None:
                element_id = uuid.uuid4().hex

            self._global_revision += 1
            now = time.time()
            prev = self._elements.get(element_id)
            prev_img = prev if isinstance(prev, ImageElement) else None

            if prev_img is None:
                img = ImageElement(
                    id=element_id,
                    type="image",
                    name=name,
                    image_bytes=bytes(image_bytes),
                    mime_type=str(mime_type),
                    width=width_i,
                    height=height_i,
                    channels=channels_i,
                    revision=1,
                    created_at=now,
                    updated_at=now,
                    position=(0.0, 0.0, 0.0),
                    rotation=(0.0, 0.0, 0.0, 1.0),
                    visible=True,
                    deleted=False,
                )
            else:
                img = ImageElement(
                    id=prev_img.id,
                    type="image",
                    name=name,
                    image_bytes=bytes(image_bytes),
                    mime_type=str(mime_type),
                    width=width_i,
                    height=height_i,
                    channels=channels_i,
                    revision=prev_img.revision + 1,
                    created_at=prev_img.created_at,
                    updated_at=now,
                    position=prev_img.position,
                    rotation=prev_img.rotation,
                    visible=prev_img.visible,
                    deleted=prev_img.deleted,
                )

            self._elements[element_id] = img
            return img


REGISTRY = InMemoryRegistry()
