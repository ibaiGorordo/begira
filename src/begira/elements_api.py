from __future__ import annotations

import uuid
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from starlette.responses import Response

from .elements import PointCloudElement
from .registry import REGISTRY


def _bounds_from_positions(pos: np.ndarray) -> dict[str, list[float]]:
    if pos.size == 0:
        return {"min": [0.0, 0.0, 0.0], "max": [0.0, 0.0, 0.0]}
    bounds_min = pos.min(axis=0)
    bounds_max = pos.max(axis=0)
    return {"min": bounds_min.tolist(), "max": bounds_max.tolist()}


def mount_elements_api(app: FastAPI) -> None:
    """Mount generic element endpoints.

    These are additive and keep the existing pointcloud routes working.
    """

    @app.get("/api/elements")
    def list_elements() -> list[dict[str, Any]]:
        els = REGISTRY.list_elements()
        els.sort(key=lambda e: (e.name, e.id))
        out: list[dict[str, Any]] = []
        for e in els:
            if isinstance(e, PointCloudElement):
                out.append(
                    {
                        "id": e.id,
                        "type": e.type,
                        "name": e.name,
                        "revision": int(e.revision),
                        "createdAt": float(e.created_at),
                        "bounds": _bounds_from_positions(e.positions),
                        "summary": {"pointCount": int(e.positions.shape[0])},
                    }
                )
            else:
                out.append(
                    {
                        "id": e.id,
                        "type": e.type,
                        "name": e.name,
                        "revision": int(e.revision),
                        "createdAt": float(e.created_at),
                    }
                )
        return out

    @app.get("/api/elements/{element_id}/meta")
    def get_element_meta(element_id: str) -> dict[str, Any]:
        e = REGISTRY.get_element(element_id)
        if e is None:
            raise HTTPException(status_code=404, detail="Unknown element")

        if isinstance(e, PointCloudElement):
            schema: dict[str, dict[str, Any]] = {"position": {"type": "float32", "components": 3}}
            if e.colors is not None:
                schema["color"] = {"type": "uint8", "components": 3, "normalized": True}

            return {
                "id": e.id,
                "type": e.type,
                "name": e.name,
                "revision": int(e.revision),
                "bounds": _bounds_from_positions(e.positions),
                "endianness": "little",
                "pointSize": float(e.point_size),
                "pointCount": int(e.positions.shape[0]),
                "interleaved": e.colors is not None,
                "bytesPerPoint": int(12 + (3 if e.colors is not None else 0)),
                "schema": schema,
                "payloads": {
                    "points": {
                        "url": f"/api/elements/{e.id}/payloads/points",
                        "contentType": "application/octet-stream",
                    }
                },
            }

        raise HTTPException(status_code=400, detail=f"Unsupported element type: {e.type}")

    @app.get("/api/elements/{element_id}/payloads/{payload_name}")
    def get_element_payload(element_id: str, payload_name: str) -> Response:
        e = REGISTRY.get_element(element_id)
        if e is None:
            raise HTTPException(status_code=404, detail="Unknown element")

        if isinstance(e, PointCloudElement):
            if payload_name != "points":
                raise HTTPException(status_code=404, detail="Unknown payload")

            pos = np.ascontiguousarray(e.positions, dtype=np.float32)

            if e.colors is None:
                return Response(content=pos.tobytes(order="C"), media_type="application/octet-stream")

            col = np.ascontiguousarray(e.colors, dtype=np.uint8)
            if col.shape != (pos.shape[0], 3):
                raise HTTPException(status_code=500, detail="Invalid color buffer")

            # Interleave XYZ(float32) + RGB(uint8) per point.
            n = pos.shape[0]
            out = np.empty((n, 15), dtype=np.uint8)
            out[:, 0:12] = pos.view(np.uint8).reshape(n, 12)
            out[:, 12:15] = col
            return Response(content=out.tobytes(order="C"), media_type="application/octet-stream")

        raise HTTPException(status_code=400, detail=f"Unsupported element type: {e.type}")

    @app.post("/api/elements/pointclouds/upload")
    def request_pointcloud_upload(body: dict) -> dict:
        """Request an upload slot for a pointcloud element."""

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
        element_id = body.get("elementId") or body.get("cloudId")
        if element_id is not None:
            element_id = str(element_id).strip() or None
        if element_id is None:
            element_id = uuid.uuid4().hex

        point_size = body.get("pointSize")
        if point_size is not None:
            try:
                point_size = float(point_size)
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid pointSize")

        bytes_per_point = 12 + (3 if has_color else 0)

        return {
            "id": element_id,
            "type": "pointcloud",
            "name": name,
            "pointCount": point_count,
            "hasColor": has_color,
            "bytesPerPoint": bytes_per_point,
            "uploadUrl": f"/api/elements/{element_id}/payloads/points",
        }

    @app.put("/api/elements/{element_id}/payloads/points")
    async def upload_pointcloud_points(element_id: str, request: Request) -> dict:
        """Upload raw point buffer and upsert as a pointcloud element."""

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
                raise HTTPException(
                    status_code=400,
                    detail=f"Payload size {nbytes} is not a valid point buffer (not /12 or /15 divisible)",
                )

        n = nbytes // int(stride)

        if n == 0:
            positions = np.empty((0, 3), dtype=np.float32)
            colors = None
        else:
            if has_color:
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

        pc = REGISTRY.upsert_pointcloud(
            name=name,
            positions=positions,
            colors=colors,
            point_size=point_size_f,
            element_id=element_id,
        )

        return {"ok": True, "id": pc.id, "type": pc.type, "revision": int(pc.revision), "pointCount": pc.point_count}

    @app.patch("/api/elements/{element_id}/meta")
    def patch_element_meta(element_id: str, body: dict) -> dict[str, Any]:
        e = REGISTRY.get_element(element_id)
        if e is None:
            raise HTTPException(status_code=404, detail="Unknown element")

        if isinstance(e, PointCloudElement):
            try:
                point_size = body.get("pointSize")
                if point_size is not None:
                    point_size = float(point_size)
                pc = REGISTRY.update_pointcloud_settings(element_id, point_size=point_size)
                return {"ok": True, "id": pc.id, "type": pc.type, "revision": int(pc.revision), "pointSize": float(pc.point_size)}
            except (TypeError, ValueError) as ex:
                raise HTTPException(status_code=400, detail=str(ex))
            except KeyError:
                raise HTTPException(status_code=404, detail="Unknown pointcloud")

        raise HTTPException(status_code=400, detail=f"Unsupported element type: {getattr(e, 'type', None)}")
