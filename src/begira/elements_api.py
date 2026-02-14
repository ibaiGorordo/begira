from __future__ import annotations

import uuid
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from starlette.responses import Response

from .elements import PointCloudElement, GaussianSplatElement, CameraElement
from .registry import REGISTRY


def _bounds_from_positions(pos: np.ndarray) -> dict[str, list[float]]:
    if pos.size == 0:
        return {"min": [0.0, 0.0, 0.0], "max": [0.0, 0.0, 0.0]}
    bounds_min = pos.min(axis=0)
    bounds_max = pos.max(axis=0)
    return {"min": bounds_min.tolist(), "max": bounds_max.tolist()}


def _bounds_from_position(pos: tuple[float, float, float], *, radius: float = 0.5) -> dict[str, list[float]]:
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    r = float(radius)
    return {
        "min": [x - r, y - r, z - r],
        "max": [x + r, y + r, z + r],
    }


def mount_elements_api(app: FastAPI) -> None:
    """Mount generic element endpoints.

    These are additive and keep the existing pointcloud routes working.
    """

    @app.get("/api/elements")
    def list_elements() -> list[dict[str, Any]]:
        els = REGISTRY.list_elements()
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
                        "visible": bool(e.visible),
                        "deleted": bool(e.deleted),
                    }
                )
            elif isinstance(e, GaussianSplatElement):
                out.append(
                    {
                        "id": e.id,
                        "type": e.type,
                        "name": e.name,
                        "revision": int(e.revision),
                        "createdAt": float(e.created_at),
                        "bounds": _bounds_from_positions(e.positions),
                        "summary": {"count": int(e.positions.shape[0])},
                        "visible": bool(e.visible),
                        "deleted": bool(e.deleted),
                    }
                )
            elif isinstance(e, CameraElement):
                out.append(
                    {
                        "id": e.id,
                        "type": e.type,
                        "name": e.name,
                        "revision": int(e.revision),
                        "createdAt": float(e.created_at),
                        "bounds": _bounds_from_position(e.position),
                        "fov": float(e.fov),
                        "near": float(e.near),
                        "far": float(e.far),
                        "width": int(e.width) if e.width is not None else None,
                        "height": int(e.height) if e.height is not None else None,
                        "intrinsicMatrix": [list(row) for row in e.intrinsic_matrix] if e.intrinsic_matrix is not None else None,
                        "position": list(e.position),
                        "rotation": list(e.rotation),
                        "visible": bool(e.visible),
                        "deleted": bool(e.deleted),
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
                "position": list(e.position),
                "rotation": list(e.rotation),
                "visible": bool(e.visible),
                "deleted": bool(e.deleted),
                "schema": schema,
                "payloads": {
                    "points": {
                        "url": f"/api/elements/{e.id}/payloads/points",
                        "contentType": "application/octet-stream",
                    }
                },
            }

        if isinstance(e, GaussianSplatElement):
            schema = {
                "position": {"type": "float32", "components": 3},
                "sh0": {"type": "float32", "components": 3},
                "opacity": {"type": "float32", "components": 1},
                "scale": {"type": "float32", "components": 3},
                "rotation": {"type": "float32", "components": 4},
            }
            return {
                "id": e.id,
                "type": e.type,
                "name": e.name,
                "revision": int(e.revision),
                "bounds": _bounds_from_positions(e.positions),
                "endianness": "little",
                "pointSize": float(e.point_size),
                "count": int(e.positions.shape[0]),
                "position": list(e.position),
                "rotation": list(e.rotation),
                "visible": bool(e.visible),
                "deleted": bool(e.deleted),
                "bytesPerGaussian": 14 * 4,
                "schema": schema,
                "payloads": {
                    "gaussians": {
                        "url": f"/api/elements/{e.id}/payloads/gaussians",
                        "contentType": "application/octet-stream",
                    }
                },
            }

        if isinstance(e, CameraElement):
            return {
                "id": e.id,
                "type": e.type,
                "name": e.name,
                "revision": int(e.revision),
                "fov": float(e.fov),
                "near": float(e.near),
                "far": float(e.far),
                "width": int(e.width) if e.width is not None else None,
                "height": int(e.height) if e.height is not None else None,
                "intrinsicMatrix": [list(row) for row in e.intrinsic_matrix] if e.intrinsic_matrix is not None else None,
                "position": list(e.position),
                "rotation": list(e.rotation),
                "visible": bool(e.visible),
                "deleted": bool(e.deleted),
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

        if isinstance(e, GaussianSplatElement):
            if payload_name != "gaussians":
                raise HTTPException(status_code=404, detail="Unknown payload")

            n = e.positions.shape[0]
            # [pos(3), sh0(3), opacity(1), scale(3), rot(4)] = 14 floats
            out = np.empty((n, 14), dtype="<f4")
            out[:, 0:3] = e.positions
            out[:, 3:6] = e.sh0
            out[:, 6:7] = e.opacity
            out[:, 7:10] = e.scales
            out[:, 10:14] = e.rotations

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

    @app.post("/api/elements/gaussians/upload")
    def request_gaussians_upload(body: dict) -> dict:
        """Request an upload slot for a gaussians element."""

        try:
            name = str(body.get("name"))
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid name")
        if not name:
            raise HTTPException(status_code=400, detail="Missing name")

        try:
            count = int(body.get("count"))
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid count")
        if count < 0:
            raise HTTPException(status_code=400, detail="count must be >= 0")

        element_id = body.get("elementId")
        if element_id is not None:
            element_id = str(element_id).strip() or None
        if element_id is None:
            element_id = uuid.uuid4().hex

        # [pos(3), sh0(3), opacity(1), scale(3), rot(4)] = 14 floats = 56 bytes
        bytes_per_gaussian = 14 * 4

        return {
            "id": element_id,
            "type": "gaussians",
            "name": name,
            "count": count,
            "bytesPerGaussian": bytes_per_gaussian,
            "uploadUrl": f"/api/elements/{element_id}/payloads/gaussians",
        }

    @app.put("/api/elements/{element_id}/payloads/gaussians")
    async def upload_gaussians_payload(element_id: str, request: Request) -> dict:
        """Upload raw gaussians buffer and upsert as a gaussians element."""

        name = request.query_params.get("name")
        if not name:
            raise HTTPException(status_code=400, detail="Missing query param: name")

        raw = await request.body()
        nbytes = len(raw)
        stride_bytes = 14 * 4

        if nbytes % stride_bytes != 0:
            raise HTTPException(
                status_code=400, detail=f"Payload size {nbytes} not divisible by stride {stride_bytes}"
            )

        n = nbytes // stride_bytes

        if n == 0:
            positions = np.empty((0, 3), dtype=np.float32)
            sh0 = np.empty((0, 3), dtype=np.float32)
            opacity = np.empty((0, 1), dtype=np.float32)
            scales = np.empty((0, 3), dtype=np.float32)
            rotations = np.empty((0, 4), dtype=np.float32)
        else:
            # All little-endian float32
            data = np.frombuffer(raw, dtype="<f4").reshape((n, 14))
            positions = np.ascontiguousarray(data[:, 0:3])
            sh0 = np.ascontiguousarray(data[:, 3:6])
            opacity = np.ascontiguousarray(data[:, 6:7])
            scales = np.ascontiguousarray(data[:, 7:10])
            rotations = np.ascontiguousarray(data[:, 10:14])

        gs = REGISTRY.upsert_gaussians(
            name=name,
            positions=positions,
            sh0=sh0,
            opacity=opacity,
            scales=scales,
            rotations=rotations,
            element_id=element_id,
        )

        return {"ok": True, "id": gs.id, "type": gs.type, "revision": int(gs.revision), "count": gs.count}

    @app.post("/api/elements/cameras")
    def create_camera(body: dict) -> dict[str, Any]:
        try:
            name = str(body.get("name") or "Camera")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid name")

        def _as_vec(key: str, length: int, default: list[float]) -> tuple[float, ...]:
            raw = body.get(key, default)
            if raw is None:
                return tuple(default)
            if not isinstance(raw, (list, tuple)) or len(raw) != length:
                raise HTTPException(status_code=400, detail=f"Invalid {key}")
            return tuple(float(x) for x in raw)

        position = _as_vec("position", 3, [0.0, 0.0, 0.0])
        rotation = _as_vec("rotation", 4, [0.0, 0.0, 0.0, 1.0])

        try:
            fov_raw = body.get("fov", None)
            fov = float(fov_raw) if fov_raw is not None else None
            near_raw = body.get("near", 0.01)
            far_raw = body.get("far", 1000.0)
            near = float(near_raw) if near_raw is not None else None
            far = float(far_raw) if far_raw is not None else None
            width_raw = body.get("width", None)
            height_raw = body.get("height", None)
            width = int(width_raw) if width_raw is not None else None
            height = int(height_raw) if height_raw is not None else None
            intrinsic_matrix = body.get("intrinsicMatrix", body.get("intrinsic_matrix", None))
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="Invalid camera intrinsics/projection fields")

        element_id = body.get("elementId")
        if element_id is not None:
            element_id = str(element_id).strip() or None

        try:
            cam = REGISTRY.upsert_camera(
                name=name,
                position=position,
                rotation=rotation,
                fov=fov,
                near=near,
                far=far,
                width=width,
                height=height,
                intrinsic_matrix=intrinsic_matrix,
                element_id=element_id,
            )
            return {"ok": True, "id": cam.id, "type": cam.type, "revision": int(cam.revision)}
        except (TypeError, ValueError) as ex:
            raise HTTPException(status_code=400, detail=str(ex))

    @app.patch("/api/elements/{element_id}/meta")
    def patch_element_meta(element_id: str, body: dict) -> dict[str, Any]:
        e = REGISTRY.get_element(element_id)
        if e is None:
            raise HTTPException(status_code=404, detail="Unknown element")

        def _as_vec(key: str, length: int) -> tuple[float, ...] | None:
            if key not in body:
                return None
            raw = body.get(key)
            if raw is None:
                return None
            if not isinstance(raw, (list, tuple)) or len(raw) != length:
                raise HTTPException(status_code=400, detail=f"Invalid {key}")
            return tuple(float(x) for x in raw)

        try:
            position = _as_vec("position", 3)
            rotation = _as_vec("rotation", 4)
            visible = body.get("visible") if "visible" in body else None
            deleted = body.get("deleted") if "deleted" in body else None

            point_size = body.get("pointSize") if "pointSize" in body else None
            if point_size is not None:
                point_size = float(point_size)

            fov = body.get("fov") if "fov" in body else None
            near = body.get("near") if "near" in body else None
            far = body.get("far") if "far" in body else None
            width = body.get("width") if "width" in body else None
            height = body.get("height") if "height" in body else None
            intrinsic_matrix = None
            if "intrinsicMatrix" in body:
                intrinsic_matrix = body.get("intrinsicMatrix")
            elif "intrinsic_matrix" in body:
                intrinsic_matrix = body.get("intrinsic_matrix")

            updated = REGISTRY.update_element_meta(
                element_id,
                position=position,
                rotation=rotation,
                visible=bool(visible) if visible is not None else None,
                deleted=bool(deleted) if deleted is not None else None,
                point_size=point_size,
                fov=float(fov) if fov is not None else None,
                near=float(near) if near is not None else None,
                far=float(far) if far is not None else None,
                width=int(width) if width is not None else None,
                height=int(height) if height is not None else None,
                intrinsic_matrix=intrinsic_matrix,
            )
            return {"ok": True, "id": updated.id, "type": updated.type, "revision": int(updated.revision)}
        except (TypeError, ValueError) as ex:
            raise HTTPException(status_code=400, detail=str(ex))
        except KeyError:
            raise HTTPException(status_code=404, detail="Unknown element")

    @app.delete("/api/elements/{element_id}")
    def delete_element(element_id: str) -> dict[str, Any]:
        try:
            e = REGISTRY.delete_element(element_id)
            return {"ok": True, "id": e.id, "type": e.type, "revision": int(e.revision)}
        except KeyError:
            raise HTTPException(status_code=404, detail="Unknown element")
