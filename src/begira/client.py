from __future__ import annotations

import numpy as np

from .conventions import CoordinateConvention


class BegiraClient:
    """HTTP client for logging elements to a running begira server.

    Current supported element types:
      - pointcloud

    Contract (current):
      - POST /api/elements/pointclouds/upload   (JSON)
      - PUT  /api/elements/{id}/payloads/points?name=...&hasColor=...&pointSize=...  (octet-stream)

    The uploaded binary payload is interleaved:
      - XYZ: 3*float32 little-endian
      - optional RGB: 3*uint8
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8000") -> None:
        self.base_url = base_url.rstrip("/")

    def get_coordinate_convention(self, *, timeout_s: float = 10.0) -> CoordinateConvention:
        """Return the active viewer coordinate convention.

        Values:
          - 'rh-y-up'
          - 'rh-z-up'
        """
        import httpx

        with httpx.Client(base_url=self.base_url, timeout=timeout_s) as client:
            res = client.get("/api/viewer/settings")
            if res.status_code >= 400:
                raise RuntimeError(f"Failed to get viewer settings: {res.status_code} {res.text}")
            data = res.json()
            return CoordinateConvention.from_any(str(data.get("coordinateConvention") or ""))

    def set_coordinate_convention(self, convention: str | CoordinateConvention, *, timeout_s: float = 10.0) -> None:
        """Set the viewer coordinate convention.

        You can pass either a CoordinateConvention enum value or a string alias.

        Examples:
            client.set_coordinate_convention(CoordinateConvention.Z_UP)
            client.set_coordinate_convention('z-up')
        """
        import httpx

        conv = CoordinateConvention.from_any(convention)

        with httpx.Client(base_url=self.base_url, timeout=timeout_s) as client:
            res = client.patch("/api/viewer/settings", json={"coordinateConvention": conv.value})
            if res.status_code >= 400:
                raise RuntimeError(f"Failed to update viewer settings: {res.status_code} {res.text}")

    def get_viewer_settings(self, *, timeout_s: float = 10.0) -> dict:
        """Return the raw viewer settings payload from the server."""
        import httpx

        with httpx.Client(base_url=self.base_url, timeout=timeout_s) as client:
            res = client.get("/api/viewer/settings")
            if res.status_code >= 400:
                raise RuntimeError(f"Failed to get viewer settings: {res.status_code} {res.text}")
            return dict(res.json())

    def reset_project(self, *, timeout_s: float = 10.0) -> None:
        """Clear all elements from the server."""
        import httpx

        with httpx.Client(base_url=self.base_url, timeout=timeout_s) as client:
            res = client.post("/api/reset")
            if res.status_code >= 400:
                raise RuntimeError(f"Failed to reset project: {res.status_code} {res.text}")

    def delete_element(self, element_id: str, *, timeout_s: float = 10.0) -> None:
        """Soft-delete an element."""
        import httpx

        with httpx.Client(base_url=self.base_url, timeout=timeout_s) as client:
            res = client.delete(f"/api/elements/{element_id}")
            if res.status_code >= 400:
                raise RuntimeError(f"Failed to delete element: {res.status_code} {res.text}")

    def set_element_visibility(self, element_id: str, visible: bool, *, timeout_s: float = 10.0) -> None:
        """Toggle element visibility."""
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
        """Update the pose of an element (position + rotation quaternion)."""
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
        fov: float = 60.0,
        near: float = 0.01,
        far: float = 1000.0,
        element_id: str | None = None,
        timeout_s: float = 10.0,
    ) -> str:
        """Create/update a camera element."""
        import httpx

        payload = {
            "name": name,
            "position": [float(x) for x in position],
            "rotation": [float(x) for x in rotation],
            "fov": float(fov),
            "near": float(near),
            "far": float(far),
            "elementId": element_id,
        }

        with httpx.Client(base_url=self.base_url, timeout=timeout_s) as client:
            res = client.post("/api/elements/cameras", json=payload)
            if res.status_code >= 400:
                raise RuntimeError(f"Failed to create camera: {res.status_code} {res.text}")
            data = res.json()
            return str(data.get("id"))

    def log_points(
        self,
        name: str,
        positions: np.ndarray | object,
        colors: np.ndarray | None = None,
        *,
        element_id: str | None = None,
        point_size: float | None = 0.05,
        timeout_s: float = 60.0,
    ) -> str:
        """Upload (add/update) a pointcloud element into a running server.

        Returns the element id.
        """

        # Convenience: allow passing a pointcloud object directly.
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

        # Build the interleaved payload.
        if has_color:
            n = pos.shape[0]
            stride = 15

            out = np.empty((n, stride), dtype=np.uint8)
            out[:, 0:12] = pos.view(np.uint8).reshape(n, 12)
            out[:, 12:15] = col.reshape(n, 3)
            payload = out.tobytes(order='C')
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

            params = {
                "name": name,
                "hasColor": "1" if has_color else "0",
            }
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
    ) -> str:
        """Upload (add/update) a 3D Gaussian Splatting element into a running server.

        Returns the element id.
        """

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

        # Build binary payload: [pos(3), sh0(3), opacity(1), scale(3), rot(4)] = 14 floats
        payload_data = np.empty((n, 14), dtype="<f4")
        payload_data[:, 0:3] = pos
        payload_data[:, 3:6] = sh0
        payload_data[:, 6:7] = opacity
        payload_data[:, 7:10] = scales
        payload_data[:, 10:14] = rotations

        payload = payload_data.tobytes(order="C")

        import httpx

        with httpx.Client(base_url=self.base_url, timeout=timeout_s) as client:
            req = {
                "name": name,
                "elementId": element_id,
                "count": int(n),
            }
            res = client.post("/api/elements/gaussians/upload", json=req)
            if res.status_code >= 400:
                raise RuntimeError(f"Upload request failed: {res.status_code} {res.text}")

            data = res.json()
            eid = str(data.get("id"))
            upload_url = str(data.get("uploadUrl"))
            if not eid or not upload_url:
                raise RuntimeError(f"Upload request returned invalid response: {data}")

            params = {"name": name}
            put = client.put(
                upload_url,
                params=params,
                content=payload,
                headers={"content-type": "application/octet-stream"},
            )
            if put.status_code >= 400:
                raise RuntimeError(f"Gaussians upload failed: {put.status_code} {put.text}")

            out = put.json()
            return str(out.get("id") or eid)
