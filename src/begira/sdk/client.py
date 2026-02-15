from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

from ..core.conventions import CoordinateConvention
from .handles import CameraHandle, GaussianHandle, ImageHandle, PointCloudHandle, Box3DHandle, Ellipsoid3DHandle
from ..io.image import encode_image_payload
from .time_context import (
    sample_query_params as _sample_query_params_shared,
    set_active_timeline,
    time_body as _time_body_shared,
    time_fields_to_query_params as _time_fields_to_query_params_shared,
    to_unix_seconds,
)


def _normalize_intrinsic_matrix(
    intrinsic_matrix: np.ndarray | list[list[float]] | tuple[tuple[float, ...], ...] | None,
) -> list[list[float]] | None:
    if intrinsic_matrix is None:
        return None
    k = np.asarray(intrinsic_matrix, dtype=np.float64)
    if k.shape != (3, 3):
        raise ValueError(f"intrinsic_matrix must have shape (3, 3), got {k.shape}")
    return [[float(k[r, c]) for c in range(3)] for r in range(3)]


def _fov_from_intrinsics(k: list[list[float]], height: int) -> float:
    fy = float(k[1][1])
    if not np.isfinite(fy) or fy <= 0:
        raise ValueError("intrinsic_matrix[1][1] (fy) must be finite and > 0")
    if not np.isfinite(height) or int(height) <= 0:
        raise ValueError("height must be a positive integer when deriving fov from intrinsics")
    return float(np.degrees(2.0 * np.arctan(float(height) / (2.0 * fy))))


def _time_body(
    *,
    frame: int | None = None,
    timestamp: float | datetime | None = None,
    static: bool = False,
    timeline: str | None = None,
    timeline_kind: str | None = None,
    sequence: int | None = None,
    timeline_timestamp: float | datetime | None = None,
    active_timeline: tuple[str, str, float] | None = None,
) -> dict[str, Any]:
    return _time_body_shared(
        frame=frame,
        timestamp=timestamp,
        static=static,
        timeline=timeline,
        timeline_kind=timeline_kind,
        sequence=sequence,
        timeline_timestamp=timeline_timestamp,
        active_timeline=active_timeline,
    )


def _sample_query_params(
    *,
    frame: int | None = None,
    timestamp: float | datetime | None = None,
    timeline: str | None = None,
    time_value: float | None = None,
    active_timeline: tuple[str, str, float] | None = None,
) -> dict[str, str]:
    return _sample_query_params_shared(
        frame=frame,
        timestamp=timestamp,
        timeline=timeline,
        time_value=time_value,
        active_timeline=active_timeline,
    )


def _time_fields_to_query_params(time_fields: dict[str, Any]) -> dict[str, str]:
    return _time_fields_to_query_params_shared(time_fields)


class BegiraClient:
    """HTTP client for logging elements to a running begira server."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000") -> None:
        self.base_url = base_url.rstrip("/")
        self._active_timeline: tuple[str, str, float] | None = None

    def set_time(
        self,
        timeline: str,
        *,
        sequence: int | None = None,
        timestamp: float | datetime | None = None,
    ) -> None:
        self._active_timeline = set_active_timeline(
            timeline,
            sequence=sequence,
            timestamp=timestamp,
        )

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

    def get_coordinate_convention(self, *, timeout_s: float = 10.0) -> CoordinateConvention:
        import httpx

        with httpx.Client(base_url=self.base_url, timeout=timeout_s) as client:
            res = client.get("/api/viewer/settings")
            if res.status_code >= 400:
                raise RuntimeError(f"Failed to get viewer settings: {res.status_code} {res.text}")
            data = res.json()
            return CoordinateConvention.from_any(str(data.get("coordinateConvention") or ""))

    def set_coordinate_convention(self, convention: str | CoordinateConvention, *, timeout_s: float = 10.0) -> None:
        import httpx

        conv = CoordinateConvention.from_any(convention)

        with httpx.Client(base_url=self.base_url, timeout=timeout_s) as client:
            res = client.patch("/api/viewer/settings", json={"coordinateConvention": conv.value})
            if res.status_code >= 400:
                raise RuntimeError(f"Failed to update viewer settings: {res.status_code} {res.text}")

    def get_viewer_settings(self, *, timeout_s: float = 10.0) -> dict[str, Any]:
        import httpx

        with httpx.Client(base_url=self.base_url, timeout=timeout_s) as client:
            res = client.get("/api/viewer/settings")
            if res.status_code >= 400:
                raise RuntimeError(f"Failed to get viewer settings: {res.status_code} {res.text}")
            return dict(res.json())

    def open_camera_view(self, camera_id: str, *, timeout_s: float = 10.0) -> None:
        import httpx

        cid = str(camera_id).strip()
        if not cid:
            raise ValueError("camera_id cannot be empty")

        with httpx.Client(base_url=self.base_url, timeout=timeout_s) as client:
            res = client.post("/api/viewer/open-camera-view", json={"cameraId": cid})
            if res.status_code >= 400:
                raise RuntimeError(f"Failed to open camera view: {res.status_code} {res.text}")

    def get_timeline_info(self, *, timeout_s: float = 10.0) -> dict[str, Any]:
        import httpx

        with httpx.Client(base_url=self.base_url, timeout=timeout_s) as client:
            res = client.get("/api/timeline")
            if res.status_code >= 400:
                raise RuntimeError(f"Failed to get timeline info: {res.status_code} {res.text}")
            return dict(res.json())

    def get_element_meta(
        self,
        element_id: str,
        *,
        frame: int | None = None,
        timestamp: float | datetime | None = None,
        timeline: str | None = None,
        time_value: float | None = None,
        timeout_s: float = 10.0,
    ) -> dict[str, Any]:
        import httpx

        params = _sample_query_params(
            frame=frame,
            timestamp=timestamp,
            timeline=timeline,
            time_value=time_value,
            active_timeline=self._active_timeline,
        )

        with httpx.Client(base_url=self.base_url, timeout=timeout_s) as client:
            res = client.get(f"/api/elements/{element_id}/meta", params=params)
            if res.status_code >= 400:
                raise RuntimeError(f"Failed to get element meta: {res.status_code} {res.text}")
            return dict(res.json())

    def reset_project(self, *, timeout_s: float = 10.0) -> None:
        import httpx

        with httpx.Client(base_url=self.base_url, timeout=timeout_s) as client:
            res = client.post("/api/reset")
            if res.status_code >= 400:
                raise RuntimeError(f"Failed to reset project: {res.status_code} {res.text}")

    def delete_element(
        self,
        element_id: str,
        *,
        frame: int | None = None,
        timestamp: float | datetime | None = None,
        static: bool = False,
        timeout_s: float = 10.0,
    ) -> None:
        import httpx

        body = _time_body(frame=frame, timestamp=timestamp, static=static, active_timeline=self._active_timeline)
        params = _time_fields_to_query_params(body)

        with httpx.Client(base_url=self.base_url, timeout=timeout_s) as client:
            res = client.delete(f"/api/elements/{element_id}", params=params)
            if res.status_code >= 400:
                raise RuntimeError(f"Failed to delete element: {res.status_code} {res.text}")

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
        import httpx

        payload: dict[str, Any] = {"visible": bool(visible)}
        payload.update(_time_body(frame=frame, timestamp=timestamp, static=static, active_timeline=self._active_timeline))
        with httpx.Client(base_url=self.base_url, timeout=timeout_s) as client:
            res = client.patch(f"/api/elements/{element_id}/meta", json=payload)
            if res.status_code >= 400:
                raise RuntimeError(f"Failed to update visibility: {res.status_code} {res.text}")

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
        import httpx

        payload: dict[str, object] = {}
        has_transform = False
        if position is not None:
            payload["position"] = [float(x) for x in position]
            has_transform = True
        if rotation is not None:
            payload["rotation"] = [float(x) for x in rotation]
            has_transform = True
        if not has_transform:
            return
        payload.update(_time_body(frame=frame, timestamp=timestamp, static=static, active_timeline=self._active_timeline))

        with httpx.Client(base_url=self.base_url, timeout=timeout_s) as client:
            res = client.patch(f"/api/elements/{element_id}/meta", json=payload)
            if res.status_code >= 400:
                raise RuntimeError(f"Failed to update transform: {res.status_code} {res.text}")

    def log_camera(
        self,
        name: str,
        *,
        position: tuple[float, float, float] | list[float] = (0.0, 0.0, 0.0),
        rotation: tuple[float, float, float, float] | list[float] = (0.0, 0.0, 0.0, 1.0),
        fov: float | None = None,
        near: float = 0.01,
        far: float = 1000.0,
        width: int | None = None,
        height: int | None = None,
        intrinsic_matrix: np.ndarray | list[list[float]] | tuple[tuple[float, ...], ...] | None = None,
        frame: int | None = None,
        timestamp: float | datetime | None = None,
        static: bool = False,
        element_id: str | None = None,
        timeout_s: float = 10.0,
    ) -> CameraHandle:
        """Create/update a camera element.

        If `fov` is omitted and `intrinsic_matrix`+`height` are provided, fov is
        derived from `fy` in the intrinsic matrix.
        """
        import httpx

        if width is not None:
            width = int(width)
            if width <= 0:
                raise ValueError("width must be a positive integer")
        if height is not None:
            height = int(height)
            if height <= 0:
                raise ValueError("height must be a positive integer")

        k = _normalize_intrinsic_matrix(intrinsic_matrix)

        fov_value = float(fov) if fov is not None else None
        if fov_value is None and k is not None and height is not None:
            fov_value = _fov_from_intrinsics(k, height)
        if fov_value is None:
            fov_value = 60.0

        payload: dict[str, Any] = {
            "name": name,
            "position": [float(x) for x in position],
            "rotation": [float(x) for x in rotation],
            "fov": float(fov_value),
            "near": float(near),
            "far": float(far),
            "elementId": element_id,
        }
        if width is not None:
            payload["width"] = int(width)
        if height is not None:
            payload["height"] = int(height)
        if k is not None:
            payload["intrinsicMatrix"] = k
        payload.update(_time_body(frame=frame, timestamp=timestamp, static=static, active_timeline=self._active_timeline))

        with httpx.Client(base_url=self.base_url, timeout=timeout_s) as client:
            res = client.post("/api/elements/cameras", json=payload)
            if res.status_code >= 400:
                raise RuntimeError(f"Failed to create camera: {res.status_code} {res.text}")
            data = res.json()
            eid = str(data.get("id"))
            if not eid:
                raise RuntimeError(f"Invalid camera response: {data}")
            return CameraHandle(eid, ops=self, element_type="camera")

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
        timeout_s: float = 10.0,
    ) -> Box3DHandle:
        import httpx

        s = np.asarray(size, dtype=np.float64).reshape(3)
        c = np.asarray(color, dtype=np.float64).reshape(3)
        payload: dict[str, Any] = {
            "name": str(name),
            "size": [float(s[0]), float(s[1]), float(s[2])],
            "color": [float(c[0]), float(c[1]), float(c[2])],
            "position": [float(position[0]), float(position[1]), float(position[2])],
            "rotation": [float(rotation[0]), float(rotation[1]), float(rotation[2]), float(rotation[3])],
            "elementId": element_id,
        }
        payload.update(_time_body(frame=frame, timestamp=timestamp, static=static, active_timeline=self._active_timeline))

        with httpx.Client(base_url=self.base_url, timeout=timeout_s) as client:
            res = client.post("/api/elements/boxes3d", json=payload)
            if res.status_code >= 400:
                raise RuntimeError(f"Failed to create box3d: {res.status_code} {res.text}")
            data = res.json()
            eid = str(data.get("id"))
            if not eid:
                raise RuntimeError(f"Invalid box3d response: {data}")
            return Box3DHandle(eid, ops=self, element_type="box3d")

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
        timeout_s: float = 10.0,
    ) -> Ellipsoid3DHandle:
        import httpx

        r = np.asarray(radii, dtype=np.float64).reshape(3)
        c = np.asarray(color, dtype=np.float64).reshape(3)
        payload: dict[str, Any] = {
            "name": str(name),
            "radii": [float(r[0]), float(r[1]), float(r[2])],
            "color": [float(c[0]), float(c[1]), float(c[2])],
            "position": [float(position[0]), float(position[1]), float(position[2])],
            "rotation": [float(rotation[0]), float(rotation[1]), float(rotation[2]), float(rotation[3])],
            "elementId": element_id,
        }
        payload.update(_time_body(frame=frame, timestamp=timestamp, static=static, active_timeline=self._active_timeline))

        with httpx.Client(base_url=self.base_url, timeout=timeout_s) as client:
            res = client.post("/api/elements/ellipsoids3d", json=payload)
            if res.status_code >= 400:
                raise RuntimeError(f"Failed to create ellipsoid3d: {res.status_code} {res.text}")
            data = res.json()
            eid = str(data.get("id"))
            if not eid:
                raise RuntimeError(f"Invalid ellipsoid3d response: {data}")
            return Ellipsoid3DHandle(eid, ops=self, element_type="ellipsoid3d")

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
        timeout_s: float = 60.0,
    ) -> PointCloudHandle:
        """Upload (add/update) a pointcloud element into a running server."""

        if colors is None and not isinstance(positions, np.ndarray):
            pos_attr = getattr(positions, "positions", None)
            if pos_attr is not None:
                col_attr = getattr(positions, "colors", None)
                positions = pos_attr
                colors = col_attr

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

        if point_size is not None and (not np.isfinite(point_size) or float(point_size) <= 0):
            raise ValueError("point_size must be a finite positive number")

        has_color = col is not None
        if has_color:
            n = pos.shape[0]
            out = np.empty((n, 15), dtype=np.uint8)
            out[:, 0:12] = pos.view(np.uint8).reshape(n, 12)
            out[:, 12:15] = col.reshape(n, 3)
            payload = out.tobytes(order="C")
        else:
            payload = pos.tobytes(order="C")

        import httpx
        time_fields = _time_body(frame=frame, timestamp=timestamp, static=static, active_timeline=self._active_timeline)

        with httpx.Client(base_url=self.base_url, timeout=timeout_s) as client:
            req = {
                "name": name,
                "elementId": element_id,
                "pointCount": int(pos.shape[0]),
                "hasColor": bool(has_color),
                "pointSize": float(point_size) if point_size is not None else None,
            }
            req.update(time_fields)
            res = client.post("/api/elements/pointclouds/upload", json=req)
            if res.status_code >= 400:
                raise RuntimeError(f"Upload request failed: {res.status_code} {res.text}")

            data = res.json()
            eid = str(data.get("id"))
            upload_url = str(data.get("uploadUrl"))
            if not eid or not upload_url:
                raise RuntimeError(f"Upload request returned invalid response: {data}")

            params = {"name": name, "hasColor": "1" if has_color else "0"}
            if point_size is not None:
                params["pointSize"] = str(float(point_size))
            params.update(_time_fields_to_query_params(time_fields))

            put = client.put(
                upload_url,
                params=params,
                content=payload,
                headers={"content-type": "application/octet-stream"},
            )
            if put.status_code >= 400:
                raise RuntimeError(f"Points upload failed: {put.status_code} {put.text}")

            return PointCloudHandle(eid, ops=self, element_type="pointcloud")

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
        timeout_s: float = 60.0,
    ) -> ImageHandle:
        """Upload (add/update) an image element.

        - `image` can be an OpenCV-style `numpy.ndarray` or a PIL image object.
        - `image` can also be pre-encoded bytes (PNG/JPEG/WEBP) when
          `width/height/channels` are provided.
        - For numpy color images, `color_order` controls channel interpretation.
        """
        data, mime, width, height, channels = encode_image_payload(
            image,
            mime_type=mime_type,
            color_order=color_order,
            width=width,
            height=height,
            channels=channels,
        )

        import httpx
        time_fields = _time_body(frame=frame, timestamp=timestamp, static=static, active_timeline=self._active_timeline)

        with httpx.Client(base_url=self.base_url, timeout=timeout_s) as client:
            req = {
                "name": name,
                "elementId": element_id,
                "width": int(width),
                "height": int(height),
                "channels": int(channels),
                "mimeType": str(mime),
            }
            req.update(time_fields)
            res = client.post("/api/elements/images/upload", json=req)
            if res.status_code >= 400:
                raise RuntimeError(f"Upload request failed: {res.status_code} {res.text}")

            resp = res.json()
            eid = str(resp.get("id"))
            upload_url = str(resp.get("uploadUrl"))
            if not eid or not upload_url:
                raise RuntimeError(f"Upload request returned invalid response: {resp}")

            put = client.put(
                upload_url,
                params={
                    "name": name,
                    "mimeType": str(mime),
                    "width": str(int(width)),
                    "height": str(int(height)),
                    "channels": str(int(channels)),
                    **_time_fields_to_query_params(time_fields),
                },
                content=data,
                headers={"content-type": "application/octet-stream"},
            )
            if put.status_code >= 400:
                raise RuntimeError(f"Image upload failed: {put.status_code} {put.text}")

            out = put.json()
            out_id = str(out.get("id") or eid)
            return ImageHandle(out_id, ops=self, element_type="image")

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
        timeout_s: float = 60.0,
    ) -> GaussianHandle:
        """Upload (add/update) a 3D Gaussian Splatting element into a running server."""

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
            rotations[:, 0] = 1.0
        else:
            rotations = np.ascontiguousarray(rotations, dtype=np.float32)

        payload_data = np.empty((n, 14), dtype="<f4")
        payload_data[:, 0:3] = pos
        payload_data[:, 3:6] = sh0
        payload_data[:, 6:7] = opacity
        payload_data[:, 7:10] = scales
        payload_data[:, 10:14] = rotations
        payload = payload_data.tobytes(order="C")

        import httpx
        time_fields = _time_body(frame=frame, timestamp=timestamp, static=static, active_timeline=self._active_timeline)

        with httpx.Client(base_url=self.base_url, timeout=timeout_s) as client:
            req = {"name": name, "elementId": element_id, "count": int(n), **time_fields}
            res = client.post("/api/elements/gaussians/upload", json=req)
            if res.status_code >= 400:
                raise RuntimeError(f"Upload request failed: {res.status_code} {res.text}")

            data = res.json()
            eid = str(data.get("id"))
            upload_url = str(data.get("uploadUrl"))
            if not eid or not upload_url:
                raise RuntimeError(f"Upload request returned invalid response: {data}")

            put = client.put(
                upload_url,
                params={
                    "name": name,
                    **_time_fields_to_query_params(time_fields),
                },
                content=payload,
                headers={"content-type": "application/octet-stream"},
            )
            if put.status_code >= 400:
                raise RuntimeError(f"Gaussians upload failed: {put.status_code} {put.text}")

            out = put.json()
            out_id = str(out.get("id") or eid)
            return GaussianHandle(out_id, ops=self, element_type="gaussians")

    def get_camera_animation(
        self,
        camera_id: str,
        *,
        timeout_s: float = 10.0,
    ) -> dict[str, Any] | None:
        import httpx

        with httpx.Client(base_url=self.base_url, timeout=timeout_s) as client:
            res = client.get(f"/api/elements/{camera_id}/animation")
            if res.status_code == 404:
                return None
            if res.status_code >= 400:
                raise RuntimeError(f"Failed to get camera animation: {res.status_code} {res.text}")
            return dict(res.json())

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
    ) -> dict[str, Any]:
        import httpx

        payload: dict[str, Any] = {
            "mode": str(mode),
            "targetId": str(target_id),
            "startFrame": int(start_frame),
            "endFrame": int(end_frame),
            "step": int(step),
        }
        if turns is not None:
            payload["turns"] = float(turns)
        if radius is not None:
            payload["radius"] = float(radius)
        if phase_deg is not None:
            payload["phaseDeg"] = float(phase_deg)
        if up is not None:
            u = np.asarray(up, dtype=np.float64).reshape(3)
            payload["up"] = [float(u[0]), float(u[1]), float(u[2])]

        with httpx.Client(base_url=self.base_url, timeout=timeout_s) as client:
            res = client.put(f"/api/elements/{camera_id}/animation", json=payload)
            if res.status_code >= 400:
                raise RuntimeError(f"Failed to set camera animation: {res.status_code} {res.text}")
            return dict(res.json())

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
    ) -> dict[str, Any]:
        import httpx

        p = np.asarray(position, dtype=np.float64).reshape(3)
        payload: dict[str, Any] = {
            "frame": int(frame),
            "position": [float(p[0]), float(p[1]), float(p[2])],
            "pullEnabled": bool(pull_enabled),
            "pullRadiusFrames": int(pull_radius_frames),
            "pullPinnedEnds": bool(pull_pinned_ends),
        }
        with httpx.Client(base_url=self.base_url, timeout=timeout_s) as client:
            res = client.patch(f"/api/elements/{camera_id}/animation/key", json=payload)
            if res.status_code >= 400:
                raise RuntimeError(f"Failed to update camera animation key: {res.status_code} {res.text}")
            return dict(res.json())

    def insert_camera_animation_key(
        self,
        camera_id: str,
        *,
        frame: int,
        position: tuple[float, float, float] | list[float] | np.ndarray | None = None,
        timeout_s: float = 10.0,
    ) -> dict[str, Any]:
        import httpx

        payload: dict[str, Any] = {"frame": int(frame)}
        if position is not None:
            p = np.asarray(position, dtype=np.float64).reshape(3)
            payload["position"] = [float(p[0]), float(p[1]), float(p[2])]

        with httpx.Client(base_url=self.base_url, timeout=timeout_s) as client:
            res = client.post(f"/api/elements/{camera_id}/animation/key", json=payload)
            if res.status_code >= 400:
                raise RuntimeError(f"Failed to insert camera animation key: {res.status_code} {res.text}")
            return dict(res.json())

    def delete_camera_animation_key(
        self,
        camera_id: str,
        *,
        frame: int,
        timeout_s: float = 10.0,
    ) -> dict[str, Any]:
        import httpx

        with httpx.Client(base_url=self.base_url, timeout=timeout_s) as client:
            res = client.delete(f"/api/elements/{camera_id}/animation/key", params={"frame": str(int(frame))})
            if res.status_code >= 400:
                raise RuntimeError(f"Failed to delete camera animation key: {res.status_code} {res.text}")
            return dict(res.json())

    def duplicate_camera_animation_key(
        self,
        camera_id: str,
        *,
        source_frame: int,
        target_frame: int,
        timeout_s: float = 10.0,
    ) -> dict[str, Any]:
        import httpx

        payload = {"sourceFrame": int(source_frame), "targetFrame": int(target_frame)}
        with httpx.Client(base_url=self.base_url, timeout=timeout_s) as client:
            res = client.post(f"/api/elements/{camera_id}/animation/key/duplicate", json=payload)
            if res.status_code >= 400:
                raise RuntimeError(f"Failed to duplicate camera animation key: {res.status_code} {res.text}")
            return dict(res.json())

    def smooth_camera_animation(
        self,
        camera_id: str,
        *,
        start_frame: int | None = None,
        end_frame: int | None = None,
        passes: int = 1,
        pinned_ends: bool = True,
        timeout_s: float = 10.0,
    ) -> dict[str, Any]:
        import httpx

        payload: dict[str, Any] = {
            "passes": int(passes),
            "pinnedEnds": bool(pinned_ends),
        }
        if start_frame is not None:
            payload["startFrame"] = int(start_frame)
        if end_frame is not None:
            payload["endFrame"] = int(end_frame)

        with httpx.Client(base_url=self.base_url, timeout=timeout_s) as client:
            res = client.post(f"/api/elements/{camera_id}/animation/smooth", json=payload)
            if res.status_code >= 400:
                raise RuntimeError(f"Failed to smooth camera animation: {res.status_code} {res.text}")
            return dict(res.json())

    def clear_camera_animation(
        self,
        camera_id: str,
        *,
        timeout_s: float = 10.0,
    ) -> None:
        import httpx

        with httpx.Client(base_url=self.base_url, timeout=timeout_s) as client:
            res = client.delete(f"/api/elements/{camera_id}/animation")
            if res.status_code >= 400:
                raise RuntimeError(f"Failed to clear camera animation: {res.status_code} {res.text}")

    def get_camera_animation_trajectory(
        self,
        camera_id: str,
        *,
        start_frame: int | None = None,
        end_frame: int | None = None,
        stride: int = 1,
        timeout_s: float = 10.0,
    ) -> dict[str, Any]:
        import httpx

        params: dict[str, str] = {"stride": str(max(1, int(stride)))}
        if start_frame is not None:
            params["startFrame"] = str(int(start_frame))
        if end_frame is not None:
            params["endFrame"] = str(int(end_frame))

        with httpx.Client(base_url=self.base_url, timeout=timeout_s) as client:
            res = client.get(f"/api/elements/{camera_id}/animation/trajectory", params=params)
            if res.status_code >= 400:
                raise RuntimeError(f"Failed to get camera animation trajectory: {res.status_code} {res.text}")
            return dict(res.json())
