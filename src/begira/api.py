from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response

from .registry import REGISTRY


def _generate_sample_pointcloud(n: int = 200_000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (positions, colors).

    positions: float32 of shape (n, 3)
    colors: uint8 of shape (n, 3)
    """
    rng = np.random.default_rng(seed)

    # A simple noisy sphere-ish distribution.
    u = rng.random(n)
    v = rng.random(n)
    theta = 2 * np.pi * u
    phi = np.arccos(2 * v - 1)
    r = 1.0 + 0.05 * rng.standard_normal(n)

    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    positions = np.stack([x, y, z], axis=1).astype(np.float32, copy=False)

    # Color by height (z) for a quick visual.
    zc = positions[:, 2]
    t = (zc - float(zc.min())) / (float(np.ptp(zc)) + 1e-9)
    colors = (
        np.stack(
            [
                (255 * t),
                (255 * (1.0 - t)),
                (255 * 0.6 * np.ones_like(t)),
            ],
            axis=1,
        )
        .clip(0, 255)
        .astype(np.uint8, copy=False)
    )

    return positions, colors


@dataclass(frozen=True)
class PointCloud:
    id: str
    name: str
    positions: np.ndarray  # float32 (n,3)
    colors: np.ndarray | None  # uint8 (n,3)


# Optionally seed an initial sample cloud for demos.
# Default is off so users don't see the random sphere when logging their own data.
import os
import uuid

if os.getenv("BEGIRA_SAMPLE", "0") in {"1", "true", "True"}:
    _POS, _COL = _generate_sample_pointcloud()
    REGISTRY.upsert(name="Sample", positions=_POS, colors=_COL, cloud_id="sample")


def create_api_app() -> FastAPI:
    app = FastAPI(title="begira", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/healthz")
    def healthz() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/api/events")
    def events() -> dict[str, int]:
        # Minimal polling endpoint.
        return {"globalRevision": REGISTRY.global_revision()}

    @app.get("/api/pointclouds")
    def list_pointclouds() -> list[dict]:
        pcs = REGISTRY.list()
        # Stable sort to avoid dropdown jumping.
        pcs.sort(key=lambda p: (p.name, p.id))
        out: list[dict] = []
        for pc in pcs:
            pos = pc.positions
            bounds_min = pos.min(axis=0)
            bounds_max = pos.max(axis=0)
            out.append(
                {
                    "id": pc.id,
                    "name": pc.name,
                    "pointCount": int(pc.positions.shape[0]),
                    "revision": int(pc.revision),
                    "createdAt": float(pc.created_at),
                    "pointSize": float(pc.point_size),
                    "bounds": {"min": bounds_min.tolist(), "max": bounds_max.tolist()},
                }
            )
        return out

    @app.get("/api/pointclouds/{cloud_id}/meta")
    def get_pointcloud_meta(cloud_id: str) -> dict:
        pc = REGISTRY.get(cloud_id)
        if pc is None:
            raise HTTPException(status_code=404, detail="Unknown point cloud")

        pos = pc.positions
        bounds_min = pos.min(axis=0)
        bounds_max = pos.max(axis=0)

        schema: dict[str, dict] = {"position": {"type": "float32", "components": 3}}
        if pc.colors is not None:
            schema["color"] = {"type": "uint8", "components": 3, "normalized": True}

        return {
            "id": pc.id,
            "name": pc.name,
            "pointCount": int(pos.shape[0]),
            "revision": int(pc.revision),
            "pointSize": float(pc.point_size),
            "bounds": {"min": bounds_min.tolist(), "max": bounds_max.tolist()},
            "endianness": "little",
            "interleaved": pc.colors is not None,
            "bytesPerPoint": int(12 + (3 if pc.colors is not None else 0)),
            "schema": schema,
            "payload": {
                "url": f"/api/pointclouds/{pc.id}/points",
                "contentType": "application/octet-stream",
            },
        }

    @app.get("/api/pointclouds/{cloud_id}/points")
    def get_pointcloud_points(cloud_id: str) -> Response:
        pc = REGISTRY.get(cloud_id)
        if pc is None:
            raise HTTPException(status_code=404, detail="Unknown point cloud")

        pos = np.ascontiguousarray(pc.positions, dtype=np.float32)

        if pc.colors is None:
            return Response(content=pos.tobytes(order="C"), media_type="application/octet-stream")

        col = np.ascontiguousarray(pc.colors, dtype=np.uint8)
        if col.shape != (pos.shape[0], 3):
            raise HTTPException(status_code=500, detail="Invalid color buffer")

        # Interleave XYZ(float32) + RGB(uint8) per point (15 bytes).
        n = pos.shape[0]
        out = np.empty((n, 15), dtype=np.uint8)
        out[:, 0:12] = pos.view(np.uint8).reshape(n, 12)
        out[:, 12:15] = col
        return Response(content=out.tobytes(order='C'), media_type="application/octet-stream")

    @app.patch("/api/pointclouds/{cloud_id}/settings")
    def update_pointcloud_settings(cloud_id: str, body: dict) -> dict:
        try:
            point_size = body.get("pointSize")
            if point_size is not None:
                point_size = float(point_size)
            pc = REGISTRY.update_settings(cloud_id, point_size=point_size)
            return {"ok": True, "revision": int(pc.revision), "pointSize": float(pc.point_size)}
        except KeyError:
            raise HTTPException(status_code=404, detail="Unknown point cloud")
        except (TypeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/pointclouds/upload")
    def request_pointcloud_upload(body: dict) -> dict:
        """Request an upload slot for a point cloud.

        This is used by `BegiraClient` to push point clouds into an already-running server.

        Body (minimal):
          - name: str
          - pointCount: int
          - hasColor: bool (default false)
          - cloudId: str (optional; if provided, upserts that id)
          - pointSize: float (optional)

        Response:
          - id
          - uploadUrl (PUT)
        """

        try:
            name = str(body.get("name"))
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid name")
        if not name:
            raise HTTPException(status_code=400, detail="Missing name")

        try:
            point_count = int(body.get("pointCount"))
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid pointCount")
        if point_count < 0:
            raise HTTPException(status_code=400, detail="pointCount must be >= 0")

        has_color = bool(body.get("hasColor", False))
        cloud_id = body.get("cloudId")
        if cloud_id is not None:
            cloud_id = str(cloud_id).strip() or None
        if cloud_id is None:
            cloud_id = uuid.uuid4().hex

        point_size = body.get("pointSize")
        if point_size is not None:
            try:
                point_size = float(point_size)
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid pointSize")

        bytes_per_point = 12 + (3 if has_color else 0)

        # We don't allocate anything in the registry here (kept minimal); the PUT will upsert.
        return {
            "id": cloud_id,
            "name": name,
            "pointCount": point_count,
            "hasColor": has_color,
            "bytesPerPoint": bytes_per_point,
            "uploadUrl": f"/api/pointclouds/{cloud_id}/points",
        }

    @app.put("/api/pointclouds/{cloud_id}/points")
    async def upload_pointcloud_points(cloud_id: str, request: Request) -> dict:
        """Upload raw point buffer and upsert into the registry.

        Content-Type: application/octet-stream

        Expected payload format (little-endian, interleaved):
          - XYZ: 3*float32 (12 bytes)
          - optional RGB: 3*uint8 (3 bytes)

        Query params:
          - name: str (required)
          - hasColor: 0/1 (optional; inferred from buffer length if omitted)
          - pointSize: float (optional)
        """

        name = request.query_params.get("name")
        if not name:
            raise HTTPException(status_code=400, detail="Missing query param: name")

        point_size = request.query_params.get("pointSize")
        if point_size is not None:
            try:
                point_size_f = float(point_size)
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid pointSize")
        else:
            point_size_f = None

        raw = await request.body()
        nbytes = len(raw)

        # Determine stride.
        stride = None
        has_color_param = request.query_params.get("hasColor")
        if has_color_param is not None:
            has_color = has_color_param not in {"0", "false", "False"}
            stride = 15 if has_color else 12
            if nbytes % stride != 0:
                raise HTTPException(status_code=400, detail=f"Payload size {nbytes} not divisible by stride {stride}")
        else:
            # Infer: prefer colored if divisible by 15, else require divisible by 12.
            if nbytes % 15 == 0:
                stride = 15
                has_color = True
            elif nbytes % 12 == 0:
                stride = 12
                has_color = False
            else:
                raise HTTPException(status_code=400, detail=f"Payload size {nbytes} is not a valid point buffer (not /12 or /15 divisible)")

        n = nbytes // stride

        if n == 0:
            positions = np.empty((0, 3), dtype=np.float32)
            colors = None
        else:
            if has_color:
                # Parse interleaved XYZ(float32) + RGB(uint8).
                buf = np.frombuffer(raw, dtype=np.uint8)
                try:
                    view = buf.reshape((n, 15))
                except ValueError:
                    raise HTTPException(status_code=400, detail="Invalid interleaved buffer shape")

                pos_bytes = np.ascontiguousarray(view[:, :12]).reshape(-1)
                positions = np.frombuffer(pos_bytes.tobytes(), dtype="<f4").reshape((n, 3))
                colors = np.ascontiguousarray(view[:, 12:15])
            else:
                positions = np.frombuffer(raw, dtype="<f4").reshape((n, 3))
                colors = None

        pc = REGISTRY.upsert(
            name=name,
            positions=positions,
            colors=colors,
            point_size=point_size_f,
            cloud_id=cloud_id,
        )

        return {"ok": True, "id": pc.id, "revision": int(pc.revision), "pointCount": int(pc.positions.shape[0])}

    return app
