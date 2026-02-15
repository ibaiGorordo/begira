from __future__ import annotations

import threading
import time
import uuid
from dataclasses import replace

import numpy as np

from .elements import ElementBase, PointCloudElement, GaussianSplatElement, CameraElement, ImageElement
from .timeline import WriteTarget, ElementTemporalRecord

class InMemoryRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._elements: dict[str, ElementBase] = {}
        self._timeline: dict[str, ElementTemporalRecord[ElementBase]] = {}
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

    @staticmethod
    def _validate_sample_query(frame: int | None, timestamp: float | None) -> tuple[int | None, float | None]:
        if frame is not None and timestamp is not None:
            raise ValueError("Cannot query with both frame and timestamp")
        frame_v = int(frame) if frame is not None else None
        ts_v: float | None = None
        if timestamp is not None:
            ts_v = float(timestamp)
            if not np.isfinite(ts_v):
                raise ValueError("timestamp must be finite")
        return frame_v, ts_v

    def _resolve_write_target_locked(
        self,
        *,
        static: bool = False,
        frame: int | None = None,
        timestamp: float | None = None,
    ) -> WriteTarget:
        frame_v, ts_v = self._validate_sample_query(frame, timestamp)
        if static and (frame_v is not None or ts_v is not None):
            raise ValueError("static=True cannot be combined with frame or timestamp")

        if static:
            return WriteTarget(axis=None, key=None, auto=False)
        if frame_v is not None:
            return WriteTarget(axis="frame", key=frame_v, auto=False)
        if ts_v is not None:
            return WriteTarget(axis="timestamp", key=ts_v, auto=False)

        # Default behavior: timeless write unless time axis is explicitly provided.
        return WriteTarget(axis=None, key=None, auto=False)

    def _get_record_locked(self, element_id: str) -> ElementTemporalRecord[ElementBase]:
        record = self._timeline.get(element_id)
        if record is None:
            record = ElementTemporalRecord[ElementBase]()
            self._timeline[element_id] = record
        return record

    def _store_sample_locked(self, element_id: str, target: WriteTarget, element: ElementBase) -> None:
        record = self._get_record_locked(element_id)
        record.samples.set_sample(target, element)
        self._elements[element_id] = element

    def _sample_element_locked(self, element_id: str, *, frame: int | None = None, timestamp: float | None = None) -> ElementBase | None:
        record = self._timeline.get(element_id)
        if record is None:
            return None
        return record.samples.sample(frame=frame, timestamp=timestamp)

    def _base_element_for_write_locked(self, element_id: str, target: WriteTarget) -> ElementBase | None:
        sampled = self._sample_element_locked(
            element_id,
            frame=int(target.key) if target.axis == "frame" else None,
            timestamp=float(target.key) if target.axis == "timestamp" else None,
        )
        if sampled is not None:
            return sampled
        if target.axis is None or target.auto:
            return self._elements.get(element_id)
        return None

    @staticmethod
    def _next_revisions(latest: ElementBase | None, *, state_changed: bool, data_changed: bool) -> tuple[int, int, int]:
        if latest is None:
            return 1, 1, 1
        revision = int(latest.revision) + 1
        state_revision = int(latest.state_revision) + (1 if state_changed else 0)
        data_revision = int(latest.data_revision) + (1 if data_changed else 0)
        return revision, state_revision, data_revision

    def global_revision(self) -> int:
        with self._lock:
            return self._global_revision

    def timeline_info(self) -> dict[str, object]:
        with self._lock:
            frame_min: int | None = None
            frame_max: int | None = None
            ts_min: float | None = None
            ts_max: float | None = None

            for record in self._timeline.values():
                fb = record.samples.frame_bounds()
                if fb is not None:
                    lo, hi = fb
                    frame_min = lo if frame_min is None else min(frame_min, lo)
                    frame_max = hi if frame_max is None else max(frame_max, hi)

                tb = record.samples.timestamp_bounds()
                if tb is not None:
                    lo, hi = tb
                    ts_min = lo if ts_min is None else min(ts_min, lo)
                    ts_max = hi if ts_max is None else max(ts_max, hi)

            return {
                "defaultAxis": "frame",
                "axes": [
                    {
                        "axis": "frame",
                        "min": frame_min,
                        "max": frame_max,
                        "hasData": frame_min is not None,
                    },
                    {
                        "axis": "timestamp",
                        "min": ts_min,
                        "max": ts_max,
                        "hasData": ts_min is not None,
                    },
                ],
                "latest": {
                    "frame": frame_max,
                    "timestamp": ts_max,
                },
            }

    def list_elements(self, *, frame: int | None = None, timestamp: float | None = None) -> list[ElementBase]:
        with self._lock:
            frame_v, ts_v = self._validate_sample_query(frame, timestamp)
            if frame_v is None and ts_v is None:
                return [e for e in self._elements.values() if not getattr(e, "deleted", False)]

            out: list[ElementBase] = []
            for element_id in self._timeline:
                e = self._sample_element_locked(element_id, frame=frame_v, timestamp=ts_v)
                if e is None:
                    continue
                if getattr(e, "deleted", False):
                    continue
                out.append(e)
            return out

    def get_element(self, element_id: str, *, frame: int | None = None, timestamp: float | None = None) -> ElementBase | None:
        with self._lock:
            frame_v, ts_v = self._validate_sample_query(frame, timestamp)
            if frame_v is None and ts_v is None:
                return self._elements.get(element_id)
            return self._sample_element_locked(element_id, frame=frame_v, timestamp=ts_v)

    def reset(self) -> None:
        with self._lock:
            for key, prev in list(self._elements.items()):
                self.update_element_meta(
                    key,
                    position=(0.0, 0.0, 0.0),
                    rotation=(0.0, 0.0, 0.0, 1.0),
                    visible=True,
                    deleted=False,
                )
                # update_element_meta stores the sample.

    def delete_element(
        self,
        element_id: str,
        *,
        static: bool = False,
        frame: int | None = None,
        timestamp: float | None = None,
    ) -> ElementBase:
        with self._lock:
            prev = self._elements.get(element_id)
            if prev is None:
                raise KeyError(element_id)
            if getattr(prev, "deleted", False) and frame is None and timestamp is None and not static:
                return prev
            return self.update_element_meta(
                element_id,
                deleted=True,
                static=static,
                frame=frame,
                timestamp=timestamp,
            )

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
        static: bool = False,
        frame: int | None = None,
        timestamp: float | None = None,
    ) -> ElementBase:
        with self._lock:
            latest = self._elements.get(element_id)
            if latest is None:
                raise KeyError(element_id)

            target = self._resolve_write_target_locked(static=static, frame=frame, timestamp=timestamp)
            prev = self._base_element_for_write_locked(element_id, target)
            if prev is None:
                raise KeyError(element_id)

            updates: dict[str, object] = {}
            state_changed = False
            data_changed = False

            if position is not None:
                pos = (float(position[0]), float(position[1]), float(position[2]))
                if pos != prev.position:
                    updates["position"] = pos
                    state_changed = True
            if rotation is not None:
                rot = (float(rotation[0]), float(rotation[1]), float(rotation[2]), float(rotation[3]))
                if rot != prev.rotation:
                    updates["rotation"] = rot
                    state_changed = True
            if visible is not None:
                v = bool(visible)
                if v != bool(prev.visible):
                    updates["visible"] = v
                    state_changed = True
            if deleted is not None:
                d = bool(deleted)
                if d != bool(prev.deleted):
                    updates["deleted"] = d
                    state_changed = True

            if isinstance(prev, PointCloudElement):
                if point_size is not None:
                    if not np.isfinite(point_size) or point_size <= 0:
                        raise ValueError("point_size must be a finite positive number")
                    ps = float(point_size)
                    if ps != prev.point_size:
                        updates["point_size"] = ps
                        state_changed = True
            elif isinstance(prev, GaussianSplatElement):
                if point_size is not None:
                    if not np.isfinite(point_size) or point_size <= 0:
                        raise ValueError("point_size must be a finite positive number")
                    ps = float(point_size)
                    if ps != prev.point_size:
                        updates["point_size"] = ps
                        state_changed = True
            elif isinstance(prev, CameraElement):
                if fov is not None:
                    fv = float(fov)
                    if fv != prev.fov:
                        updates["fov"] = fv
                        state_changed = True
                if near is not None:
                    nv = float(near)
                    if nv != prev.near:
                        updates["near"] = nv
                        state_changed = True
                if far is not None:
                    fv = float(far)
                    if fv != prev.far:
                        updates["far"] = fv
                        state_changed = True
                if width is not None:
                    width_i = int(width)
                    if width_i <= 0:
                        raise ValueError("width must be a positive integer")
                    if width_i != prev.width:
                        updates["width"] = width_i
                        data_changed = True
                if height is not None:
                    height_i = int(height)
                    if height_i <= 0:
                        raise ValueError("height must be a positive integer")
                    if height_i != prev.height:
                        updates["height"] = height_i
                        data_changed = True
                if intrinsic_matrix is not None:
                    k = self._normalize_intrinsic_matrix(intrinsic_matrix)
                    if k != prev.intrinsic_matrix:
                        updates["intrinsic_matrix"] = k
                        data_changed = True
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

            revision, state_revision, data_revision = self._next_revisions(
                latest,
                state_changed=state_changed,
                data_changed=data_changed,
            )
            now = time.time()
            updated = replace(
                prev,
                **updates,
                revision=revision,
                state_revision=state_revision,
                data_revision=data_revision,
                updated_at=now,
            )
            self._global_revision += 1
            self._store_sample_locked(element_id, target, updated)
            return updated

    def update_pointcloud_settings(
        self,
        element_id: str,
        *,
        point_size: float | None = None,
        static: bool = False,
        frame: int | None = None,
        timestamp: float | None = None,
    ) -> PointCloudElement:
        updated = self.update_element_meta(
            element_id,
            point_size=point_size,
            static=static,
            frame=frame,
            timestamp=timestamp,
        )
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
        static: bool = False,
        frame: int | None = None,
        timestamp: float | None = None,
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

            target = self._resolve_write_target_locked(static=static, frame=frame, timestamp=timestamp)
            latest = self._elements.get(element_id)
            base = self._base_element_for_write_locked(element_id, target)
            base_pc = base if isinstance(base, PointCloudElement) else None
            latest_pc = latest if isinstance(latest, PointCloudElement) else None
            inherit_latest = target.axis is None or target.auto
            latest_pc_for_defaults = latest_pc if inherit_latest else None

            # New pointcloud and no explicit colors: assign a deterministic per-cloud palette color.
            if base_pc is None and latest_pc is None and col is None:
                r, g, b = self._palette_color_for_new_pointcloud()
                col = np.empty((pos.shape[0], 3), dtype=np.uint8)
                col[:, 0] = r
                col[:, 1] = g
                col[:, 2] = b

            point_size_value = (
                float(point_size)
                if point_size is not None
                else (base_pc.point_size if base_pc is not None else (latest_pc_for_defaults.point_size if latest_pc_for_defaults is not None else 0.02))
            )

            state_changed = False
            if latest_pc is not None and point_size_value != latest_pc.point_size:
                state_changed = True

            revision, state_revision, data_revision = self._next_revisions(
                latest,
                state_changed=state_changed,
                data_changed=True,
            )
            now = time.time()

            pc = PointCloudElement(
                id=element_id,
                type="pointcloud",
                name=name,
                positions=pos,
                colors=col,
                point_size=point_size_value,
                revision=revision,
                state_revision=state_revision,
                data_revision=data_revision,
                created_at=(latest.created_at if latest is not None else now),
                updated_at=now,
                position=(base_pc.position if base_pc is not None else (latest_pc_for_defaults.position if latest_pc_for_defaults is not None else (0.0, 0.0, 0.0))),
                rotation=(base_pc.rotation if base_pc is not None else (latest_pc_for_defaults.rotation if latest_pc_for_defaults is not None else (0.0, 0.0, 0.0, 1.0))),
                visible=(base_pc.visible if base_pc is not None else (latest_pc_for_defaults.visible if latest_pc_for_defaults is not None else True)),
                deleted=(base_pc.deleted if base_pc is not None else (latest_pc_for_defaults.deleted if latest_pc_for_defaults is not None else False)),
            )

            self._global_revision += 1
            self._store_sample_locked(element_id, target, pc)
            return pc

    def update_gaussians_settings(
        self,
        element_id: str,
        *,
        point_size: float | None = None,
        static: bool = False,
        frame: int | None = None,
        timestamp: float | None = None,
    ) -> GaussianSplatElement:
        updated = self.update_element_meta(
            element_id,
            point_size=point_size,
            static=static,
            frame=frame,
            timestamp=timestamp,
        )
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
        static: bool = False,
        frame: int | None = None,
        timestamp: float | None = None,
    ) -> GaussianSplatElement:
        with self._lock:
            if element_id is None:
                element_id = uuid.uuid4().hex

            target = self._resolve_write_target_locked(static=static, frame=frame, timestamp=timestamp)
            latest = self._elements.get(element_id)
            base = self._base_element_for_write_locked(element_id, target)
            base_gs = base if isinstance(base, GaussianSplatElement) else None
            latest_gs = latest if isinstance(latest, GaussianSplatElement) else None
            inherit_latest = target.axis is None or target.auto
            latest_gs_for_defaults = latest_gs if inherit_latest else None

            point_size_value = (
                float(point_size)
                if point_size is not None
                else (base_gs.point_size if base_gs is not None else (latest_gs_for_defaults.point_size if latest_gs_for_defaults is not None else 1.0))
            )
            state_changed = False
            if latest_gs is not None and point_size_value != latest_gs.point_size:
                state_changed = True

            revision, state_revision, data_revision = self._next_revisions(
                latest,
                state_changed=state_changed,
                data_changed=True,
            )
            now = time.time()

            gs = GaussianSplatElement(
                id=element_id,
                type="gaussians",
                name=name,
                positions=positions,
                sh0=sh0,
                opacity=opacity,
                scales=scales,
                rotations=rotations,
                point_size=point_size_value,
                revision=revision,
                state_revision=state_revision,
                data_revision=data_revision,
                created_at=(latest.created_at if latest is not None else now),
                updated_at=now,
                position=(base_gs.position if base_gs is not None else (latest_gs_for_defaults.position if latest_gs_for_defaults is not None else (0.0, 0.0, 0.0))),
                rotation=(base_gs.rotation if base_gs is not None else (latest_gs_for_defaults.rotation if latest_gs_for_defaults is not None else (0.0, 0.0, 0.0, 1.0))),
                visible=(base_gs.visible if base_gs is not None else (latest_gs_for_defaults.visible if latest_gs_for_defaults is not None else True)),
                deleted=(base_gs.deleted if base_gs is not None else (latest_gs_for_defaults.deleted if latest_gs_for_defaults is not None else False)),
            )

            self._global_revision += 1
            self._store_sample_locked(element_id, target, gs)
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
        static: bool = False,
        frame: int | None = None,
        timestamp: float | None = None,
    ) -> CameraElement:
        k = self._normalize_intrinsic_matrix(intrinsic_matrix)

        with self._lock:
            if element_id is None:
                element_id = uuid.uuid4().hex

            target = self._resolve_write_target_locked(static=static, frame=frame, timestamp=timestamp)
            latest = self._elements.get(element_id)
            base = self._base_element_for_write_locked(element_id, target)
            base_cam = base if isinstance(base, CameraElement) else None
            latest_cam = latest if isinstance(latest, CameraElement) else None
            inherit_latest = target.axis is None or target.auto
            latest_cam_for_defaults = latest_cam if inherit_latest else None

            width_v = int(width) if width is not None else (base_cam.width if base_cam is not None else (latest_cam_for_defaults.width if latest_cam_for_defaults is not None else None))
            if width_v is not None and width_v <= 0:
                raise ValueError("width must be a positive integer")

            height_v = int(height) if height is not None else (base_cam.height if base_cam is not None else (latest_cam_for_defaults.height if latest_cam_for_defaults is not None else None))
            if height_v is not None and height_v <= 0:
                raise ValueError("height must be a positive integer")

            k_v = k if k is not None else (base_cam.intrinsic_matrix if base_cam is not None else (latest_cam_for_defaults.intrinsic_matrix if latest_cam_for_defaults is not None else None))

            if fov is None:
                inferred_fov = self._infer_fov_from_intrinsics(k_v, height_v)
                if inferred_fov is not None:
                    fov_v = float(inferred_fov)
                elif base_cam is not None:
                    fov_v = base_cam.fov
                elif latest_cam_for_defaults is not None:
                    fov_v = latest_cam_for_defaults.fov
                else:
                    fov_v = 60.0
            else:
                fov_v = float(fov)

            near_v = float(near) if near is not None else (base_cam.near if base_cam is not None else (latest_cam_for_defaults.near if latest_cam_for_defaults is not None else 0.01))
            far_v = float(far) if far is not None else (base_cam.far if base_cam is not None else (latest_cam_for_defaults.far if latest_cam_for_defaults is not None else 1000.0))
            if not np.isfinite(near_v) or near_v <= 0:
                raise ValueError("near must be a finite positive number")
            if not np.isfinite(far_v) or far_v <= near_v:
                raise ValueError("far must be finite and greater than near")

            state_changed = True
            data_changed = latest_cam is None or width_v != latest_cam.width or height_v != latest_cam.height or k_v != latest_cam.intrinsic_matrix

            revision, state_revision, data_revision = self._next_revisions(
                latest,
                state_changed=state_changed,
                data_changed=data_changed,
            )
            now = time.time()

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
                revision=revision,
                state_revision=state_revision,
                data_revision=data_revision,
                created_at=(latest.created_at if latest is not None else now),
                updated_at=now,
                position=(float(position[0]), float(position[1]), float(position[2])),
                rotation=(float(rotation[0]), float(rotation[1]), float(rotation[2]), float(rotation[3])),
                visible=(base_cam.visible if base_cam is not None else (latest_cam_for_defaults.visible if latest_cam_for_defaults is not None else True)),
                deleted=(base_cam.deleted if base_cam is not None else (latest_cam_for_defaults.deleted if latest_cam_for_defaults is not None else False)),
            )

            self._global_revision += 1
            self._store_sample_locked(element_id, target, cam)
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
        static: bool = False,
        frame: int | None = None,
        timestamp: float | None = None,
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

            target = self._resolve_write_target_locked(static=static, frame=frame, timestamp=timestamp)
            latest = self._elements.get(element_id)
            base = self._base_element_for_write_locked(element_id, target)
            base_img = base if isinstance(base, ImageElement) else None
            latest_img = latest if isinstance(latest, ImageElement) else None
            inherit_latest = target.axis is None or target.auto
            latest_img_for_defaults = latest_img if inherit_latest else None

            revision, state_revision, data_revision = self._next_revisions(
                latest,
                state_changed=False,
                data_changed=True,
            )
            now = time.time()

            img = ImageElement(
                id=element_id,
                type="image",
                name=name,
                image_bytes=bytes(image_bytes),
                mime_type=str(mime_type),
                width=width_i,
                height=height_i,
                channels=channels_i,
                revision=revision,
                state_revision=state_revision,
                data_revision=data_revision,
                created_at=(latest.created_at if latest is not None else now),
                updated_at=now,
                position=(base_img.position if base_img is not None else (latest_img_for_defaults.position if latest_img_for_defaults is not None else (0.0, 0.0, 0.0))),
                rotation=(base_img.rotation if base_img is not None else (latest_img_for_defaults.rotation if latest_img_for_defaults is not None else (0.0, 0.0, 0.0, 1.0))),
                visible=(base_img.visible if base_img is not None else (latest_img_for_defaults.visible if latest_img_for_defaults is not None else True)),
                deleted=(base_img.deleted if base_img is not None else (latest_img_for_defaults.deleted if latest_img_for_defaults is not None else False)),
            )

            self._global_revision += 1
            self._store_sample_locked(element_id, target, img)
            return img


REGISTRY = InMemoryRegistry()
