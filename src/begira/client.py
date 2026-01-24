from __future__ import annotations

import numpy as np


class BegiraClient:
    """HTTP client for logging point clouds to a running begira server.

    This is the "remote" companion to `begira.run()` / `BegiraServer.log_points()`.

    Contract (current):
    - POST /api/pointclouds/upload   (JSON)
    - PUT  /api/pointclouds/{id}/points?name=...&hasColor=...&pointSize=...  (octet-stream)

    The uploaded binary payload is interleaved:
      - XYZ: 3*float32 little-endian
      - optional RGB: 3*uint8
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8000") -> None:
        self.base_url = base_url.rstrip("/")

    def log_points(
        self,
        name: str,
        positions: np.ndarray | object,
        colors: np.ndarray | None = None,
        *,
        cloud_id: str | None = None,
        point_size: float | None = 0.05,
        timeout_s: float = 60.0,
    ) -> str:
        """Upload (add/update) a point cloud into a running server.

        Accepts either:
        - positions: array-like (N,3) float
        - OR a pointcloud-like object with `.positions` and optional `.colors`.

        Returns the cloud id.
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

            # Vectorized interleave: create (n, 15) uint8 buffer and fill.
            out = np.empty((n, stride), dtype=np.uint8)
            out[:, 0:12] = pos.view(np.uint8).reshape(n, 12)
            out[:, 12:15] = col.reshape(n, 3)
            payload = out.tobytes(order='C')
        else:
            payload = pos.tobytes(order="C")

        # httpx is a lightweight dependency used for requests.
        import httpx

        with httpx.Client(base_url=self.base_url, timeout=timeout_s) as client:
            # 1) Request upload (server chooses id if not provided).
            req = {
                "name": name,
                "cloudId": cloud_id,
                "pointCount": int(pos.shape[0]),
                "hasColor": bool(has_color),
                "pointSize": float(point_size) if point_size is not None else None,
            }
            res = client.post("/api/pointclouds/upload", json=req)
            if res.status_code >= 400:
                raise RuntimeError(f"Upload request failed: {res.status_code} {res.text}")

            data = res.json()
            cid = str(data.get("id"))
            upload_url = str(data.get("uploadUrl"))
            if not cid or not upload_url:
                raise RuntimeError(f"Upload request returned invalid response: {data}")

            # 2) Upload payload.
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

            out = put.json()
            return str(out.get("id") or cid)
