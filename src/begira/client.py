from __future__ import annotations

from typing import Any

import numpy as np

from .conventions import CoordinateConvention
from .handles import CameraHandle, GaussianHandle, PointCloudHandle


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


class BegiraClient:
    """HTTP client for logging elements to a running begira server."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000") -> None:
        self.base_url = base_url.rstrip("/")

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

    def get_element_meta(self, element_id: str, *, timeout_s: float = 10.0) -> dict[str, Any]:
        import httpx

        with httpx.Client(base_url=self.base_url, timeout=timeout_s) as client:
            res = client.get(f"/api/elements/{element_id}/meta")
            if res.status_code >= 400:
                raise RuntimeError(f"Failed to get element meta: {res.status_code} {res.text}")
            return dict(res.json())

    def reset_project(self, *, timeout_s: float = 10.0) -> None:
        import httpx

        with httpx.Client(base_url=self.base_url, timeout=timeout_s) as client:
            res = client.post("/api/reset")
            if res.status_code >= 400:
                raise RuntimeError(f"Failed to reset project: {res.status_code} {res.text}")

    def delete_element(self, element_id: str, *, timeout_s: float = 10.0) -> None:
        import httpx

        with httpx.Client(base_url=self.base_url, timeout=timeout_s) as client:
            res = client.delete(f"/api/elements/{element_id}")
            if res.status_code >= 400:
                raise RuntimeError(f"Failed to delete element: {res.status_code} {res.text}")

    def set_element_visibility(self, element_id: str, visible: bool, *, timeout_s: float = 10.0) -> None:
        import httpx

        with httpx.Client(base_url=self.base_url, timeout=timeout_s) as client:
            res = client.patch(f"/api/elements/{element_id}/meta", json={"visible": bool(visible)})
            if res.status_code >= 400:
                raise RuntimeError(f"Failed to update visibility: {res.status_code} {res.text}")

    def log_transform(
        self,
        element_id: str,
        *,
        position: tuple[float, float, float] | list[float] | None = None,
        rotation: tuple[float, float, float, float] | list[float] | None = None,
        timeout_s: float = 10.0,
    ) -> None:
        import httpx

        payload: dict[str, object] = {}
        if position is not None:
            payload["position"] = [float(x) for x in position]
        if rotation is not None:
            payload["rotation"] = [float(x) for x in rotation]
        if not payload:
            return

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

        with httpx.Client(base_url=self.base_url, timeout=timeout_s) as client:
            res = client.post("/api/elements/cameras", json=payload)
            if res.status_code >= 400:
                raise RuntimeError(f"Failed to create camera: {res.status_code} {res.text}")
            data = res.json()
            eid = str(data.get("id"))
            if not eid:
                raise RuntimeError(f"Invalid camera response: {data}")
            return CameraHandle(eid, ops=self, element_type="camera")

    def log_points(
        self,
        name: str,
        positions: np.ndarray | object,
        colors: np.ndarray | None = None,
        *,
        element_id: str | None = None,
        point_size: float | None = 0.05,
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

        with httpx.Client(base_url=self.base_url, timeout=timeout_s) as client:
            req = {
                "name": name,
                "elementId": element_id,
                "pointCount": int(pos.shape[0]),
                "hasColor": bool(has_color),
                "pointSize": float(point_size) if point_size is not None else None,
            }
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

            put = client.put(
                upload_url,
                params=params,
                content=payload,
                headers={"content-type": "application/octet-stream"},
            )
            if put.status_code >= 400:
                raise RuntimeError(f"Points upload failed: {put.status_code} {put.text}")

            return PointCloudHandle(eid, ops=self, element_type="pointcloud")

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

        with httpx.Client(base_url=self.base_url, timeout=timeout_s) as client:
            req = {"name": name, "elementId": element_id, "count": int(n)}
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
                params={"name": name},
                content=payload,
                headers={"content-type": "application/octet-stream"},
            )
            if put.status_code >= 400:
                raise RuntimeError(f"Gaussians upload failed: {put.status_code} {put.text}")

            out = put.json()
            out_id = str(out.get("id") or eid)
            return GaussianHandle(out_id, ops=self, element_type="gaussians")

