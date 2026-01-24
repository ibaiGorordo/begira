from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from fastapi import FastAPI, HTTPException
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

        # Interleave XYZ(float32) + RGB(uint8) per point.
        n = pos.shape[0]
        stride = 15

        out = bytearray(n * stride)
        mv = memoryview(out)

        pos_bytes = pos.tobytes(order="C")
        col_bytes = col.tobytes(order="C")

        for i in range(n):
            src_pos_off = i * 12
            src_col_off = i * 3
            dst_off = i * stride
            mv[dst_off : dst_off + 12] = pos_bytes[src_pos_off : src_pos_off + 12]
            mv[dst_off + 12 : dst_off + 15] = col_bytes[src_col_off : src_col_off + 3]

        return Response(content=bytes(out), media_type="application/octet-stream")

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

    return app
