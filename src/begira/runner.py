from __future__ import annotations

import contextlib
import os
import socket
import threading
import time
import webbrowser
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import uvicorn

from .animation import CameraAnimationTrack, camera_animation_track_to_dict
from .client import BegiraClient, to_unix_seconds
from .conventions import CoordinateConvention
from .element_projection import element_to_meta_item
from .handles import CameraHandle, GaussianHandle, ImageHandle, PointCloudHandle, Box3DHandle, Ellipsoid3DHandle
from .image_logging import encode_image_payload
from .registry import REGISTRY
from .server import create_app
from .viewer_settings import VIEWER_SETTINGS


@dataclass
class BegiraServer:
    host: str
    port: int
    url: str
    _active_timeline: tuple[str, str, float] | None = None

    def _as_client(self) -> BegiraClient:
        # Reuse HTTP endpoints for any "viewer settings" behavior.
        return BegiraClient(self.url.rstrip("/"))

    def set_time(
        self,
        timeline: str,
        *,
        sequence: int | None = None,
        timestamp: float | datetime | None = None,
    ) -> None:
        timeline_name = str(timeline).strip()
        if not timeline_name:
            raise ValueError("timeline cannot be empty")
        has_sequence = sequence is not None
        has_timestamp = timestamp is not None
        if has_sequence == has_timestamp:
            raise ValueError("Provide exactly one of sequence or timestamp")
        if has_sequence:
            self._active_timeline = (timeline_name, "sequence", float(int(sequence)))
            return
        ts_v = to_unix_seconds(timestamp)
        if ts_v is None:
            raise ValueError("timestamp must not be None")
        self._active_timeline = (timeline_name, "timestamp", float(ts_v))

    def set_times(
        self,
        timeline: str,
        *,
        sequence: int | None = None,
        timestamp: float | datetime | None = None,
    ) -> None:
        self.set_time(timeline, sequence=sequence, timestamp=timestamp)

    def clear_time(self) -> None:
        self._active_timeline = None

    def _effective_read_sample(
        self,
        *,
        frame: int | None = None,
        timestamp: float | datetime | None = None,
    ) -> tuple[int | None, float | None, str | None, float | None]:
        frame_v = int(frame) if frame is not None else None
        ts_v = to_unix_seconds(timestamp)
        if frame_v is not None and ts_v is not None:
            raise ValueError("Only one of frame or timestamp can be provided")
        if frame_v is None and ts_v is None and self._active_timeline is not None:
            return None, None, str(self._active_timeline[0]), float(self._active_timeline[2])
        return frame_v, ts_v, None, None

    def _effective_write_sample(
        self,
        *,
        frame: int | None = None,
        timestamp: float | datetime | None = None,
        static: bool = False,
    ) -> tuple[bool, int | None, float | None, str | None, str | None, float | None]:
        frame_v = int(frame) if frame is not None else None
        ts_v = to_unix_seconds(timestamp)
        if frame_v is not None and ts_v is not None:
            raise ValueError("Only one of frame or timestamp can be provided")
        if static and (frame_v is not None or ts_v is not None):
            raise ValueError("static=True cannot be combined with frame or timestamp")
        if not static and frame_v is None and ts_v is None and self._active_timeline is not None:
            return (
                bool(static),
                None,
                None,
                str(self._active_timeline[0]),
                str(self._active_timeline[1]),
                float(self._active_timeline[2]),
            )
        return bool(static), frame_v, ts_v, None, None, None

    def get_coordinate_convention(self, *, timeout_s: float = 10.0) -> CoordinateConvention:
        """Return the active viewer coordinate convention for this server."""
        return self._as_client().get_coordinate_convention(timeout_s=timeout_s)

    def set_coordinate_convention(self, convention: str | CoordinateConvention, *, timeout_s: float = 10.0) -> None:
        """Set the viewer coordinate convention for this server."""
        self._as_client().set_coordinate_convention(convention, timeout_s=timeout_s)

    def get_element_meta(
        self,
        element_id: str,
        *,
        frame: int | None = None,
        timestamp: float | datetime | None = None,
        timeout_s: float = 10.0,
    ) -> dict:
        _ = timeout_s
        frame_v, ts_v, timeline_v, time_v = self._effective_read_sample(frame=frame, timestamp=timestamp)
        e = REGISTRY.get_element(
            element_id,
            frame=frame_v,
            timestamp=ts_v,
            timeline=timeline_v,
            time_value=time_v,
        )
        if e is None:
            raise RuntimeError(f"Unknown element: {element_id}")
        try:
            return element_to_meta_item(e)
        except ValueError as ex:
            raise RuntimeError(str(ex)) from ex

    def delete_element(
        self,
        element_id: str,
        *,
        frame: int | None = None,
        timestamp: float | datetime | None = None,
        static: bool = False,
        timeout_s: float = 10.0,
    ) -> None:
        _ = timeout_s
        static_v, frame_v, ts_v, timeline_v, timeline_kind_v, timeline_value_v = self._effective_write_sample(
            frame=frame,
            timestamp=timestamp,
            static=static,
        )
        try:
            REGISTRY.delete_element(
                element_id,
                frame=frame_v,
                timestamp=ts_v,
                static=static_v,
                timeline=timeline_v,
                timeline_kind=timeline_kind_v,  # type: ignore[arg-type]
                timeline_value=timeline_value_v,
            )
        except KeyError as e:
            raise RuntimeError(f"Unknown element: {element_id}") from e

    def set_element_visibility(
        self,
        element_id: str,
        visible: bool,
        *,
        frame: int | None = None,
        timestamp: float | datetime | None = None,
        static: bool = False,
        timeout_s: float = 10.0,
    ) -> None:
        _ = timeout_s
        static_v, frame_v, ts_v, timeline_v, timeline_kind_v, timeline_value_v = self._effective_write_sample(
            frame=frame,
            timestamp=timestamp,
            static=static,
        )
        try:
            REGISTRY.update_element_meta(
                element_id,
                visible=bool(visible),
                frame=frame_v,
                timestamp=ts_v,
                static=static_v,
                timeline=timeline_v,
                timeline_kind=timeline_kind_v,  # type: ignore[arg-type]
                timeline_value=timeline_value_v,
            )
        except KeyError as e:
            raise RuntimeError(f"Unknown element: {element_id}") from e

    def log_transform(
        self,
        element_id: str,
        *,
        position: tuple[float, float, float] | list[float] | None = None,
        rotation: tuple[float, float, float, float] | list[float] | None = None,
        frame: int | None = None,
        timestamp: float | datetime | None = None,
        static: bool = False,
        timeout_s: float = 10.0,
    ) -> None:
        _ = timeout_s
        pos_v = tuple(float(x) for x in position) if position is not None else None
        rot_v = tuple(float(x) for x in rotation) if rotation is not None else None
        static_v, frame_v, ts_v, timeline_v, timeline_kind_v, timeline_value_v = self._effective_write_sample(
            frame=frame,
            timestamp=timestamp,
            static=static,
        )
        try:
            REGISTRY.update_element_meta(
                element_id,
                position=pos_v,
                rotation=rot_v,
                frame=frame_v,
                timestamp=ts_v,
                static=static_v,
                timeline=timeline_v,
                timeline_kind=timeline_kind_v,  # type: ignore[arg-type]
                timeline_value=timeline_value_v,
            )
        except KeyError as e:
            raise RuntimeError(f"Unknown element: {element_id}") from e

    @staticmethod
    def _default_animation_up() -> tuple[float, float, float]:
        settings = VIEWER_SETTINGS.get()
        if settings.coordinate_convention == "rh-y-up":
            return (0.0, 1.0, 0.0)
        return (0.0, 0.0, 1.0)

    def get_camera_animation(
        self,
        camera_id: str,
        *,
        timeout_s: float = 10.0,
    ) -> dict | None:
        _ = timeout_s
        track = REGISTRY.get_camera_animation(camera_id)
        if track is None:
            return None
        return camera_animation_track_to_dict(track)

    def set_camera_animation(
        self,
        camera_id: str,
        *,
        mode: str,
        target_id: str,
        start_frame: int,
        end_frame: int,
        step: int = 1,
        turns: float | None = None,
        radius: float | None = None,
        phase_deg: float | None = None,
        up: tuple[float, float, float] | list[float] | np.ndarray | None = None,
        timeout_s: float = 10.0,
    ) -> dict:
        _ = timeout_s
        mode_v = str(mode).strip().lower()
        up_v: tuple[float, float, float]
        if up is None:
            up_v = self._default_animation_up()
        else:
            arr = np.asarray(up, dtype=np.float64).reshape(3)
            up_v = (float(arr[0]), float(arr[1]), float(arr[2]))

        params: dict[str, float] = {}
        if mode_v == "orbit":
            params["turns"] = float(1.0 if turns is None else turns)
            if radius is not None:
                params["radius"] = float(radius)
            params["phaseDeg"] = float(0.0 if phase_deg is None else phase_deg)

        track = CameraAnimationTrack(
            camera_id=str(camera_id),
            mode=mode_v,  # type: ignore[arg-type]
            target_id=str(target_id),
            start_frame=int(start_frame),
            end_frame=int(end_frame),
            step=int(step),
            interpolation="catmull_rom",
            up=up_v,
            params=params,
            control_keys=tuple(),
        )
        try:
            out = REGISTRY.set_camera_animation(track)
        except KeyError as e:
            raise RuntimeError(f"Unknown element: {str(e)}") from e
        except ValueError as e:
            raise RuntimeError(str(e)) from e
        return camera_animation_track_to_dict(out)

    def update_camera_animation_key(
        self,
        camera_id: str,
        *,
        frame: int,
        position: tuple[float, float, float] | list[float] | np.ndarray,
        pull_enabled: bool = False,
        pull_radius_frames: int = 0,
        pull_pinned_ends: bool = False,
        timeout_s: float = 10.0,
    ) -> dict:
        _ = timeout_s
        pos = np.asarray(position, dtype=np.float64).reshape(3)
        try:
            out = REGISTRY.update_camera_animation_key(
                camera_id,
                frame=int(frame),
                new_world_position=(float(pos[0]), float(pos[1]), float(pos[2])),
                pull_enabled=bool(pull_enabled),
                pull_radius_frames=int(pull_radius_frames),
                pull_pinned_ends=bool(pull_pinned_ends),
            )
        except KeyError as e:
            raise RuntimeError(f"Unknown element: {str(e)}") from e
        except ValueError as e:
            raise RuntimeError(str(e)) from e
        return camera_animation_track_to_dict(out)

    def insert_camera_animation_key(
        self,
        camera_id: str,
        *,
        frame: int,
        position: tuple[float, float, float] | list[float] | np.ndarray | None = None,
        timeout_s: float = 10.0,
    ) -> dict:
        _ = timeout_s
        pos: tuple[float, float, float] | None = None
        if position is not None:
            arr = np.asarray(position, dtype=np.float64).reshape(3)
            pos = (float(arr[0]), float(arr[1]), float(arr[2]))
        try:
            out = REGISTRY.insert_camera_animation_key(camera_id, frame=int(frame), world_position=pos)
        except KeyError as e:
            raise RuntimeError(f"Unknown element: {str(e)}") from e
        except ValueError as e:
            raise RuntimeError(str(e)) from e
        return camera_animation_track_to_dict(out)

    def delete_camera_animation_key(
        self,
        camera_id: str,
        *,
        frame: int,
        timeout_s: float = 10.0,
    ) -> dict:
        _ = timeout_s
        try:
            out = REGISTRY.delete_camera_animation_key(camera_id, frame=int(frame))
        except KeyError as e:
            raise RuntimeError(f"Unknown element: {str(e)}") from e
        except ValueError as e:
            raise RuntimeError(str(e)) from e
        return camera_animation_track_to_dict(out)

    def duplicate_camera_animation_key(
        self,
        camera_id: str,
        *,
        source_frame: int,
        target_frame: int,
        timeout_s: float = 10.0,
    ) -> dict:
        _ = timeout_s
        try:
            out = REGISTRY.duplicate_camera_animation_key(
                camera_id,
                source_frame=int(source_frame),
                target_frame=int(target_frame),
            )
        except KeyError as e:
            raise RuntimeError(f"Unknown element: {str(e)}") from e
        except ValueError as e:
            raise RuntimeError(str(e)) from e
        return camera_animation_track_to_dict(out)

    def smooth_camera_animation(
        self,
        camera_id: str,
        *,
        start_frame: int | None = None,
        end_frame: int | None = None,
        passes: int = 1,
        pinned_ends: bool = True,
        timeout_s: float = 10.0,
    ) -> dict:
        _ = timeout_s
        try:
            out = REGISTRY.smooth_camera_animation_keys(
                camera_id,
                start_frame=int(start_frame) if start_frame is not None else None,
                end_frame=int(end_frame) if end_frame is not None else None,
                passes=int(passes),
                pinned_ends=bool(pinned_ends),
            )
        except KeyError as e:
            raise RuntimeError(f"Unknown element: {str(e)}") from e
        except ValueError as e:
            raise RuntimeError(str(e)) from e
        return camera_animation_track_to_dict(out)

    def clear_camera_animation(
        self,
        camera_id: str,
        *,
        timeout_s: float = 10.0,
    ) -> None:
        _ = timeout_s
        REGISTRY.clear_camera_animation(camera_id)

    def get_camera_animation_trajectory(
        self,
        camera_id: str,
        *,
        start_frame: int | None = None,
        end_frame: int | None = None,
        stride: int = 1,
        timeout_s: float = 10.0,
    ) -> dict:
        _ = timeout_s
        try:
            frames, positions = REGISTRY.get_camera_frame_samples(
                camera_id,
                start_frame=start_frame,
                end_frame=end_frame,
                stride=stride,
            )
        except KeyError as e:
            raise RuntimeError(f"Unknown element: {str(e)}") from e
        except ValueError as e:
            raise RuntimeError(str(e)) from e

        if not frames:
            raise RuntimeError("No trajectory samples available")

        return {
            "cameraId": str(camera_id),
            "startFrame": int(frames[0]),
            "endFrame": int(frames[-1]),
            "stride": int(max(1, int(stride))),
            "frames": [int(f) for f in frames],
            "positions": [[float(p[0]), float(p[1]), float(p[2])] for p in positions],
        }

    def log_points(
        self,
        name: str,
        positions: np.ndarray | object,
        colors: np.ndarray | None = None,
        *,
        element_id: str | None = None,
        point_size: float | None = 0.05,
        frame: int | None = None,
        timestamp: float | datetime | None = None,
        static: bool = False,
    ) -> PointCloudHandle:
        """Log (add/update) a pointcloud element."""

        if colors is None and not isinstance(positions, np.ndarray):
            pos_attr = getattr(positions, "positions", None)
            if pos_attr is not None:
                col_attr = getattr(positions, "colors", None)
                positions = pos_attr
                colors = col_attr

        static_v, frame_v, ts_v, timeline_v, timeline_kind_v, timeline_value_v = self._effective_write_sample(
            frame=frame,
            timestamp=timestamp,
            static=static,
        )
        pc = REGISTRY.upsert_pointcloud(
            name=name,
            positions=positions,  # type: ignore[arg-type]
            colors=colors,
            point_size=point_size,
            element_id=element_id,
            frame=frame_v,
            timestamp=ts_v,
            static=static_v,
            timeline=timeline_v,
            timeline_kind=timeline_kind_v,  # type: ignore[arg-type]
            timeline_value=timeline_value_v,
        )
        return PointCloudHandle(pc.id, ops=self, element_type="pointcloud")

    def log_gaussians(
        self,
        name: str,
        positions: np.ndarray | object,
        sh0: np.ndarray | None = None,
        opacity: np.ndarray | None = None,
        scales: np.ndarray | None = None,
        rotations: np.ndarray | None = None,
        *,
        element_id: str | None = None,
        frame: int | None = None,
        timestamp: float | datetime | None = None,
        static: bool = False,
    ) -> GaussianHandle:
        """Log (add/update) a 3D Gaussian Splatting element."""

        # Convenience: allow passing a GaussianSplatData object directly.
        if sh0 is None and not isinstance(positions, np.ndarray):
            pos_attr = getattr(positions, "positions", None)
            if pos_attr is not None:
                sh0_attr = getattr(positions, "f_dc", None)
                opacity_attr = getattr(positions, "opacity", None)
                scales_attr = getattr(positions, "scales", None)
                rot_attr = getattr(positions, "rotations", None)
                positions = pos_attr
                sh0 = sh0_attr
                opacity = opacity_attr
                scales = scales_attr
                rotations = rot_attr

        pos = np.ascontiguousarray(positions, dtype=np.float32)
        n = pos.shape[0]

        if sh0 is None:
            sh0 = np.zeros((n, 3), dtype=np.float32)
        else:
            sh0 = np.ascontiguousarray(sh0, dtype=np.float32)

        if opacity is None:
            opacity = np.ones((n, 1), dtype=np.float32)
        else:
            opacity = np.ascontiguousarray(opacity, dtype=np.float32)
            if opacity.ndim == 1:
                opacity = opacity[:, np.newaxis]

        if scales is None:
            scales = np.zeros((n, 3), dtype=np.float32)
        else:
            scales = np.ascontiguousarray(scales, dtype=np.float32)

        if rotations is None:
            rotations = np.zeros((n, 4), dtype=np.float32)
            rotations[:, 0] = 1.0  # Identity quaternion [1, 0, 0, 0]
        else:
            rotations = np.ascontiguousarray(rotations, dtype=np.float32)

        static_v, frame_v, ts_v, timeline_v, timeline_kind_v, timeline_value_v = self._effective_write_sample(
            frame=frame,
            timestamp=timestamp,
            static=static,
        )
        gs = REGISTRY.upsert_gaussians(
            name=name,
            positions=pos,
            sh0=sh0,
            opacity=opacity,
            scales=scales,
            rotations=rotations,
            element_id=element_id,
            frame=frame_v,
            timestamp=ts_v,
            static=static_v,
            timeline=timeline_v,
            timeline_kind=timeline_kind_v,  # type: ignore[arg-type]
            timeline_value=timeline_value_v,
        )
        return GaussianHandle(gs.id, ops=self, element_type="gaussians")

    def log_camera(
        self,
        name: str,
        *,
        position: tuple[float, float, float] | list[float] = (0.0, 0.0, 0.0),
        rotation: tuple[float, float, float, float] | list[float] = (0.0, 0.0, 0.0, 1.0),
        fov: float | None = None,
        near: float | None = 0.01,
        far: float | None = 1000.0,
        width: int | None = None,
        height: int | None = None,
        intrinsic_matrix: np.ndarray | list[list[float]] | tuple[tuple[float, ...], ...] | None = None,
        frame: int | None = None,
        timestamp: float | datetime | None = None,
        static: bool = False,
        element_id: str | None = None,
    ) -> CameraHandle:
        """Log (add/update) a camera element."""

        static_v, frame_v, ts_v, timeline_v, timeline_kind_v, timeline_value_v = self._effective_write_sample(
            frame=frame,
            timestamp=timestamp,
            static=static,
        )
        cam = REGISTRY.upsert_camera(
            name=name,
            position=(float(position[0]), float(position[1]), float(position[2])),
            rotation=(float(rotation[0]), float(rotation[1]), float(rotation[2]), float(rotation[3])),
            fov=float(fov) if fov is not None else None,
            near=float(near) if near is not None else None,
            far=float(far) if far is not None else None,
            width=int(width) if width is not None else None,
            height=int(height) if height is not None else None,
            intrinsic_matrix=intrinsic_matrix,
            element_id=element_id,
            frame=frame_v,
            timestamp=ts_v,
            static=static_v,
            timeline=timeline_v,
            timeline_kind=timeline_kind_v,  # type: ignore[arg-type]
            timeline_value=timeline_value_v,
        )
        return CameraHandle(cam.id, ops=self, element_type="camera")

    def log_box3d(
        self,
        name: str,
        *,
        size: tuple[float, float, float] | list[float] | np.ndarray = (1.0, 1.0, 1.0),
        color: tuple[float, float, float] | list[float] | np.ndarray = (0.62, 0.8, 1.0),
        position: tuple[float, float, float] | list[float] = (0.0, 0.0, 0.0),
        rotation: tuple[float, float, float, float] | list[float] = (0.0, 0.0, 0.0, 1.0),
        frame: int | None = None,
        timestamp: float | datetime | None = None,
        static: bool = False,
        element_id: str | None = None,
    ) -> Box3DHandle:
        static_v, frame_v, ts_v, timeline_v, timeline_kind_v, timeline_value_v = self._effective_write_sample(
            frame=frame,
            timestamp=timestamp,
            static=static,
        )
        box = REGISTRY.upsert_box3d(
            name=name,
            size=size,  # type: ignore[arg-type]
            color=color,  # type: ignore[arg-type]
            position=(float(position[0]), float(position[1]), float(position[2])),
            rotation=(float(rotation[0]), float(rotation[1]), float(rotation[2]), float(rotation[3])),
            element_id=element_id,
            frame=frame_v,
            timestamp=ts_v,
            static=static_v,
            timeline=timeline_v,
            timeline_kind=timeline_kind_v,  # type: ignore[arg-type]
            timeline_value=timeline_value_v,
        )
        return Box3DHandle(box.id, ops=self, element_type="box3d")

    def log_ellipsoid3d(
        self,
        name: str,
        *,
        radii: tuple[float, float, float] | list[float] | np.ndarray = (0.5, 0.5, 0.5),
        color: tuple[float, float, float] | list[float] | np.ndarray = (0.56, 0.8, 0.62),
        position: tuple[float, float, float] | list[float] = (0.0, 0.0, 0.0),
        rotation: tuple[float, float, float, float] | list[float] = (0.0, 0.0, 0.0, 1.0),
        frame: int | None = None,
        timestamp: float | datetime | None = None,
        static: bool = False,
        element_id: str | None = None,
    ) -> Ellipsoid3DHandle:
        static_v, frame_v, ts_v, timeline_v, timeline_kind_v, timeline_value_v = self._effective_write_sample(
            frame=frame,
            timestamp=timestamp,
            static=static,
        )
        ellipsoid = REGISTRY.upsert_ellipsoid3d(
            name=name,
            radii=radii,  # type: ignore[arg-type]
            color=color,  # type: ignore[arg-type]
            position=(float(position[0]), float(position[1]), float(position[2])),
            rotation=(float(rotation[0]), float(rotation[1]), float(rotation[2]), float(rotation[3])),
            element_id=element_id,
            frame=frame_v,
            timestamp=ts_v,
            static=static_v,
            timeline=timeline_v,
            timeline_kind=timeline_kind_v,  # type: ignore[arg-type]
            timeline_value=timeline_value_v,
        )
        return Ellipsoid3DHandle(ellipsoid.id, ops=self, element_type="ellipsoid3d")

    def log_image(
        self,
        name: str,
        image: object,
        *,
        mime_type: str | None = "image/png",
        color_order: str = "bgr",
        width: int | None = None,
        height: int | None = None,
        channels: int | None = None,
        frame: int | None = None,
        timestamp: float | datetime | None = None,
        static: bool = False,
        element_id: str | None = None,
    ) -> ImageHandle:
        """Log (add/update) an image element."""
        data, mime, width, height, channels = encode_image_payload(
            image,
            mime_type=mime_type,
            color_order=color_order,
            width=width,
            height=height,
            channels=channels,
        )
        static_v, frame_v, ts_v, timeline_v, timeline_kind_v, timeline_value_v = self._effective_write_sample(
            frame=frame,
            timestamp=timestamp,
            static=static,
        )
        img = REGISTRY.upsert_image(
            name=name,
            image_bytes=data,
            mime_type=mime,
            width=int(width),
            height=int(height),
            channels=int(channels),
            element_id=element_id,
            frame=frame_v,
            timestamp=ts_v,
            static=static_v,
            timeline=timeline_v,
            timeline_kind=timeline_kind_v,  # type: ignore[arg-type]
            timeline_value=timeline_value_v,
        )
        return ImageHandle(img.id, ops=self, element_type="image")

    def log_ply(
        self,
        name: str,
        path: str,
        *,
        element_id: str | None = None,
        point_size: float | None = 0.05,
    ) -> PointCloudHandle:
        """Load a `.ply` file and log it as a pointcloud element."""

        from .ply import load_ply_pointcloud

        pc = load_ply_pointcloud(path)
        return self.log_points(
            name,
            pc.positions,
            pc.colors,
            element_id=element_id,
            point_size=point_size,
        )

    def log_ply_gaussians(
        self,
        name: str,
        path: str,
        *,
        element_id: str | None = None,
    ) -> GaussianHandle:
        """Load a `.ply` 3DGS file and log it as a gaussians element."""

        from .ply import load_ply_gaussians

        gs = load_ply_gaussians(path)
        return self.log_gaussians(
            name,
            gs,
            element_id=element_id,
        )

    def get_viewer_settings(self, *, timeout_s: float = 10.0) -> dict:
        """Return the active viewer settings for this server."""
        return self._as_client().get_viewer_settings(timeout_s=timeout_s)

    def open_camera_view(self, camera_id: str, *, timeout_s: float = 10.0) -> None:
        """Request all connected viewers to open a linked view for a camera."""
        _ = timeout_s
        cid = str(camera_id).strip()
        if not cid:
            raise ValueError("camera_id cannot be empty")
        elem = REGISTRY.get_element(cid)
        if elem is None:
            raise RuntimeError(f"Unknown element: {cid}")
        if getattr(elem, "type", None) != "camera":
            raise RuntimeError("open_camera_view requires a camera element id")
        VIEWER_SETTINGS.request_open_camera_view(cid)

    def get_timeline_info(self, *, timeout_s: float = 10.0) -> dict[str, object]:
        """Return timeline axis ranges and latest cursor positions."""
        _ = timeout_s
        return dict(REGISTRY.timeline_info())


def _find_free_port(host: str) -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def _normalize_base_url(url: str) -> str:
    url = url.strip()
    if not url:
        return ""
    # Allow passing just host:port.
    if "://" not in url:
        url = "http://" + url
    return url.rstrip("/")


def _is_server_alive(base_url: str, *, timeout_s: float = 0.2) -> bool:
    """Best-effort probe to determine if a begira server is reachable."""

    # Keep deps light: httpx is already a runtime dependency.
    import httpx

    try:
        with httpx.Client(base_url=base_url, timeout=timeout_s) as client:
            r = client.get("/healthz")
            if r.status_code != 200:
                return False
            data = r.json()
            return bool(data.get("ok"))
    except Exception:
        return False


def run(
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    open_browser: bool = True,
    log_level: str = "info",
    access_log: bool = False,
    new_server: bool = False,
    connect_timeout_s: float = 0.2,
) -> BegiraServer | BegiraClient:
    """Start begira (API + built frontend) with a single Python call.

    Behavior:
    - If BEGIRA_URL is set, we *attach* to that existing server (client mode) unless
      `new_server=True`.
    - Otherwise, if `port != 0` and a server is already reachable at http://{host}:{port},
      we attach to it (client mode) unless `new_server=True`.
    - Otherwise we start a new local server (server mode) and return a `BegiraServer`.

    Notes:
    - `port=0` means "pick a free port", so there's nothing to attach to.
    - We disable Uvicorn's per-request access log by default because the frontend polls
      `/api/events` frequently and it becomes very noisy.
    """

    env_url = _normalize_base_url(os.getenv("BEGIRA_URL", ""))

    # 1) Try attaching to an explicitly provided server.
    if env_url and not new_server:
        if _is_server_alive(env_url, timeout_s=connect_timeout_s):
            if open_browser:
                webbrowser.open(env_url + "/")
            return BegiraClient(env_url)

    # 2) Try attaching to host/port if they are explicitly chosen.
    if port != 0 and not new_server:
        default_url = _normalize_base_url(f"http://{host}:{port}")
        if _is_server_alive(default_url, timeout_s=connect_timeout_s):
            if open_browser:
                webbrowser.open(default_url + "/")
            return BegiraClient(default_url)

    # 3) Start a fresh server.
    if port == 0:
        port = _find_free_port(host)

    app = create_app()

    config = uvicorn.Config(app, host=host, port=port, log_level=log_level, access_log=access_log)
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Give it a moment so a subsequent client probe doesn't race with startup.
    # (Best-effort; doesn't need to be perfect.)
    time.sleep(0.05)

    url = f"http://{host}:{port}/"
    if open_browser:
        webbrowser.open(url)

    return BegiraServer(host=host, port=port, url=url)
