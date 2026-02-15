from __future__ import annotations

import uuid
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from starlette.responses import Response

from .api_time import parse_sample_body, parse_sample_query, parse_sample_query_with_static
from .elements import PointCloudElement, GaussianSplatElement, CameraElement, ImageElement
from .element_projection import element_to_list_item, element_to_meta_item
from .registry import REGISTRY


def mount_elements_api(app: FastAPI) -> None:
    """Mount generic element endpoints.

    These are additive and keep the existing pointcloud routes working.
    """

    @app.get("/api/timeline")
    def get_timeline() -> dict[str, Any]:
        return dict(REGISTRY.timeline_info())

    @app.get("/api/elements")
    def list_elements(frame: int | None = None, timestamp: float | None = None) -> list[dict[str, Any]]:
        try:
            frame_v, ts_v = parse_sample_query(frame, timestamp)
            els = REGISTRY.list_elements(frame=frame_v, timestamp=ts_v)
        except ValueError as ex:
            raise HTTPException(status_code=400, detail=str(ex))
        return [element_to_list_item(e) for e in els]

    @app.get("/api/elements/{element_id}/meta")
    def get_element_meta(element_id: str, frame: int | None = None, timestamp: float | None = None) -> dict[str, Any]:
        try:
            frame_v, ts_v = parse_sample_query(frame, timestamp)
        except ValueError as ex:
            raise HTTPException(status_code=400, detail=str(ex))

        e = REGISTRY.get_element(element_id, frame=frame_v, timestamp=ts_v)
        if e is None:
            raise HTTPException(status_code=404, detail="Unknown element or no sample at requested time")
        try:
            return element_to_meta_item(e)
        except ValueError as ex:
            raise HTTPException(status_code=400, detail=str(ex))

    @app.get("/api/elements/{element_id}/payloads/{payload_name}")
    def get_element_payload(
        element_id: str,
        payload_name: str,
        frame: int | None = None,
        timestamp: float | None = None,
    ) -> Response:
        try:
            frame_v, ts_v = parse_sample_query(frame, timestamp)
        except ValueError as ex:
            raise HTTPException(status_code=400, detail=str(ex))

        e = REGISTRY.get_element(element_id, frame=frame_v, timestamp=ts_v)
        if e is None:
            raise HTTPException(status_code=404, detail="Unknown element or no sample at requested time")

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

        if isinstance(e, ImageElement):
            if payload_name != "image":
                raise HTTPException(status_code=404, detail="Unknown payload")
            return Response(content=e.image_bytes, media_type=str(e.mime_type))

        raise HTTPException(status_code=400, detail=f"Unsupported element type: {e.type}")

    @app.post("/api/elements/pointclouds/upload")
    def request_pointcloud_upload(body: dict[str, Any]) -> dict[str, Any]:
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

        try:
            static_v, frame_v, ts_v = parse_sample_body(body)
        except ValueError as ex:
            raise HTTPException(status_code=400, detail=str(ex))

        bytes_per_point = 12 + (3 if has_color else 0)

        return {
            "id": element_id,
            "type": "pointcloud",
            "name": name,
            "pointCount": point_count,
            "hasColor": has_color,
            "bytesPerPoint": bytes_per_point,
            "uploadUrl": f"/api/elements/{element_id}/payloads/points",
            "static": static_v,
            "frame": frame_v,
            "timestamp": ts_v,
        }

    @app.put("/api/elements/{element_id}/payloads/points")
    async def upload_pointcloud_points(element_id: str, request: Request) -> dict[str, Any]:
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

        try:
            static_v, frame_v, ts_v = parse_sample_query_with_static(
                request.query_params.get("frame"),
                request.query_params.get("timestamp"),
                request.query_params.get("static"),
            )
        except ValueError as ex:
            raise HTTPException(status_code=400, detail=str(ex))

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
            static=static_v,
            frame=frame_v,
            timestamp=ts_v,
        )

        return {
            "ok": True,
            "id": pc.id,
            "type": pc.type,
            "revision": int(pc.revision),
            "stateRevision": int(pc.state_revision),
            "dataRevision": int(pc.data_revision),
            "pointCount": pc.point_count,
        }

    @app.post("/api/elements/gaussians/upload")
    def request_gaussians_upload(body: dict[str, Any]) -> dict[str, Any]:
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

        try:
            static_v, frame_v, ts_v = parse_sample_body(body)
        except ValueError as ex:
            raise HTTPException(status_code=400, detail=str(ex))

        # [pos(3), sh0(3), opacity(1), scale(3), rot(4)] = 14 floats = 56 bytes
        bytes_per_gaussian = 14 * 4

        return {
            "id": element_id,
            "type": "gaussians",
            "name": name,
            "count": count,
            "bytesPerGaussian": bytes_per_gaussian,
            "uploadUrl": f"/api/elements/{element_id}/payloads/gaussians",
            "static": static_v,
            "frame": frame_v,
            "timestamp": ts_v,
        }

    @app.put("/api/elements/{element_id}/payloads/gaussians")
    async def upload_gaussians_payload(element_id: str, request: Request) -> dict[str, Any]:
        """Upload raw gaussians buffer and upsert as a gaussians element."""

        name = request.query_params.get("name")
        if not name:
            raise HTTPException(status_code=400, detail="Missing query param: name")

        try:
            static_v, frame_v, ts_v = parse_sample_query_with_static(
                request.query_params.get("frame"),
                request.query_params.get("timestamp"),
                request.query_params.get("static"),
            )
        except ValueError as ex:
            raise HTTPException(status_code=400, detail=str(ex))

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
            static=static_v,
            frame=frame_v,
            timestamp=ts_v,
        )

        return {
            "ok": True,
            "id": gs.id,
            "type": gs.type,
            "revision": int(gs.revision),
            "stateRevision": int(gs.state_revision),
            "dataRevision": int(gs.data_revision),
            "count": gs.count,
        }

    @app.post("/api/elements/images/upload")
    def request_image_upload(body: dict[str, Any]) -> dict[str, Any]:
        """Request an upload slot for an image element."""
        try:
            name = str(body.get("name"))
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid name")
        if not name:
            raise HTTPException(status_code=400, detail="Missing name")

        try:
            width = int(body.get("width"))
            height = int(body.get("height"))
            channels = int(body.get("channels"))
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image dimensions")

        if width <= 0 or height <= 0:
            raise HTTPException(status_code=400, detail="width/height must be positive integers")
        if channels <= 0:
            raise HTTPException(status_code=400, detail="channels must be a positive integer")

        mime_type_raw = body.get("mimeType", body.get("mime_type", "image/png"))
        try:
            mime_type = str(mime_type_raw)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid mimeType")
        if "/" not in mime_type:
            raise HTTPException(status_code=400, detail="Invalid mimeType")

        element_id = body.get("elementId")
        if element_id is not None:
            element_id = str(element_id).strip() or None
        if element_id is None:
            element_id = uuid.uuid4().hex

        try:
            static_v, frame_v, ts_v = parse_sample_body(body)
        except ValueError as ex:
            raise HTTPException(status_code=400, detail=str(ex))

        return {
            "id": element_id,
            "type": "image",
            "name": name,
            "width": width,
            "height": height,
            "channels": channels,
            "mimeType": mime_type,
            "uploadUrl": f"/api/elements/{element_id}/payloads/image",
            "static": static_v,
            "frame": frame_v,
            "timestamp": ts_v,
        }

    @app.put("/api/elements/{element_id}/payloads/image")
    async def upload_image_payload(element_id: str, request: Request) -> dict[str, Any]:
        name = request.query_params.get("name")
        if not name:
            raise HTTPException(status_code=400, detail="Missing query param: name")

        mime_type = request.query_params.get("mimeType", request.query_params.get("mime_type", "image/png"))
        if "/" not in mime_type:
            raise HTTPException(status_code=400, detail="Invalid query param: mimeType")

        try:
            width = int(request.query_params.get("width", "0"))
            height = int(request.query_params.get("height", "0"))
            channels = int(request.query_params.get("channels", "0"))
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image dimensions")

        if width <= 0 or height <= 0:
            raise HTTPException(status_code=400, detail="width/height must be positive integers")
        if channels <= 0:
            raise HTTPException(status_code=400, detail="channels must be a positive integer")

        try:
            static_v, frame_v, ts_v = parse_sample_query_with_static(
                request.query_params.get("frame"),
                request.query_params.get("timestamp"),
                request.query_params.get("static"),
            )
        except ValueError as ex:
            raise HTTPException(status_code=400, detail=str(ex))

        raw = await request.body()
        if len(raw) == 0:
            raise HTTPException(status_code=400, detail="Image payload is empty")

        try:
            img = REGISTRY.upsert_image(
                name=name,
                image_bytes=raw,
                mime_type=mime_type,
                width=width,
                height=height,
                channels=channels,
                element_id=element_id,
                static=static_v,
                frame=frame_v,
                timestamp=ts_v,
            )
        except (TypeError, ValueError) as ex:
            raise HTTPException(status_code=400, detail=str(ex))

        return {
            "ok": True,
            "id": img.id,
            "type": img.type,
            "revision": int(img.revision),
            "stateRevision": int(img.state_revision),
            "dataRevision": int(img.data_revision),
            "width": int(img.width),
            "height": int(img.height),
            "channels": int(img.channels),
        }

    @app.post("/api/elements/cameras")
    def create_camera(body: dict[str, Any]) -> dict[str, Any]:
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
            static_v, frame_v, ts_v = parse_sample_body(body)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="Invalid camera intrinsics/projection/time fields")

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
                static=static_v,
                frame=frame_v,
                timestamp=ts_v,
            )
            return {
                "ok": True,
                "id": cam.id,
                "type": cam.type,
                "revision": int(cam.revision),
                "stateRevision": int(cam.state_revision),
                "dataRevision": int(cam.data_revision),
            }
        except (TypeError, ValueError) as ex:
            raise HTTPException(status_code=400, detail=str(ex))

    @app.patch("/api/elements/{element_id}/meta")
    def patch_element_meta(element_id: str, body: dict[str, Any]) -> dict[str, Any]:
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

            static_v, frame_v, ts_v = parse_sample_body(body)

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
                static=static_v,
                frame=frame_v,
                timestamp=ts_v,
            )
            return {
                "ok": True,
                "id": updated.id,
                "type": updated.type,
                "revision": int(updated.revision),
                "stateRevision": int(updated.state_revision),
                "dataRevision": int(updated.data_revision),
            }
        except (TypeError, ValueError) as ex:
            raise HTTPException(status_code=400, detail=str(ex))
        except KeyError:
            raise HTTPException(status_code=404, detail="Unknown element")

    @app.delete("/api/elements/{element_id}")
    def delete_element(
        element_id: str,
        frame: int | None = None,
        timestamp: float | None = None,
        static: str | None = None,
    ) -> dict[str, Any]:
        try:
            static_v, frame_v, ts_v = parse_sample_query_with_static(frame, timestamp, static)
            e = REGISTRY.delete_element(element_id, static=static_v, frame=frame_v, timestamp=ts_v)
            return {
                "ok": True,
                "id": e.id,
                "type": e.type,
                "revision": int(e.revision),
                "stateRevision": int(e.state_revision),
                "dataRevision": int(e.data_revision),
            }
        except ValueError as ex:
            raise HTTPException(status_code=400, detail=str(ex))
        except KeyError:
            raise HTTPException(status_code=404, detail="Unknown element")
