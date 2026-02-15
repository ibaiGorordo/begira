from __future__ import annotations

import uuid
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from starlette.responses import Response

from .api_time import parse_sample_body, parse_sample_query, parse_sample_query_with_static
from .animation import CameraAnimationTrack, CameraControlKey, camera_animation_track_to_dict
from .elements import PointCloudElement, GaussianSplatElement, CameraElement, ImageElement
from .element_projection import element_to_list_item, element_to_meta_item
from .registry import REGISTRY
from .viewer_settings import VIEWER_SETTINGS


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

    @app.post("/api/elements/boxes3d")
    def create_box3d(body: dict[str, Any]) -> dict[str, Any]:
        try:
            name = str(body.get("name") or "Box")
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
        size = _as_vec("size", 3, [1.0, 1.0, 1.0])
        color = _as_vec("color", 3, [0.62, 0.8, 1.0])
        element_id = body.get("elementId")
        if element_id is not None:
            element_id = str(element_id).strip() or None

        try:
            static_v, frame_v, ts_v = parse_sample_body(body)
            box = REGISTRY.upsert_box3d(
                name=name,
                position=position,  # type: ignore[arg-type]
                rotation=rotation,  # type: ignore[arg-type]
                size=size,  # type: ignore[arg-type]
                color=color,  # type: ignore[arg-type]
                element_id=element_id,
                static=static_v,
                frame=frame_v,
                timestamp=ts_v,
            )
        except (TypeError, ValueError) as ex:
            raise HTTPException(status_code=400, detail=str(ex))

        return {
            "ok": True,
            "id": box.id,
            "type": box.type,
            "revision": int(box.revision),
            "stateRevision": int(box.state_revision),
            "dataRevision": int(box.data_revision),
        }

    @app.post("/api/elements/ellipsoids3d")
    def create_ellipsoid3d(body: dict[str, Any]) -> dict[str, Any]:
        try:
            name = str(body.get("name") or "Ellipsoid")
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
        radii = _as_vec("radii", 3, [0.5, 0.5, 0.5])
        color = _as_vec("color", 3, [0.56, 0.8, 0.62])
        element_id = body.get("elementId")
        if element_id is not None:
            element_id = str(element_id).strip() or None

        try:
            static_v, frame_v, ts_v = parse_sample_body(body)
            ellipsoid = REGISTRY.upsert_ellipsoid3d(
                name=name,
                position=position,  # type: ignore[arg-type]
                rotation=rotation,  # type: ignore[arg-type]
                radii=radii,  # type: ignore[arg-type]
                color=color,  # type: ignore[arg-type]
                element_id=element_id,
                static=static_v,
                frame=frame_v,
                timestamp=ts_v,
            )
        except (TypeError, ValueError) as ex:
            raise HTTPException(status_code=400, detail=str(ex))

        return {
            "ok": True,
            "id": ellipsoid.id,
            "type": ellipsoid.type,
            "revision": int(ellipsoid.revision),
            "stateRevision": int(ellipsoid.state_revision),
            "dataRevision": int(ellipsoid.data_revision),
        }

    def _default_camera_up() -> tuple[float, float, float]:
        settings = VIEWER_SETTINGS.get()
        if settings.coordinate_convention == "rh-y-up":
            return (0.0, 1.0, 0.0)
        return (0.0, 0.0, 1.0)

    def _assert_camera_element(element_id: str) -> CameraElement:
        elem = REGISTRY.get_element(element_id)
        if elem is None:
            raise HTTPException(status_code=404, detail="Unknown element")
        if not isinstance(elem, CameraElement):
            raise HTTPException(status_code=400, detail="Animation endpoints are only valid for camera elements")
        return elem

    def _parse_frame_range(body: dict[str, Any]) -> tuple[int, int]:
        if "startFrame" not in body or "endFrame" not in body:
            raise HTTPException(status_code=400, detail="startFrame and endFrame are required")
        try:
            start = int(body.get("startFrame"))
            end = int(body.get("endFrame"))
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid frame range")
        if start < 0:
            raise HTTPException(status_code=400, detail="startFrame must be >= 0")
        if end < start:
            raise HTTPException(status_code=400, detail="endFrame must be >= startFrame")
        return start, end

    def _parse_bool_field(body: dict[str, Any], key: str, *, default: bool) -> bool:
        if key not in body:
            return bool(default)
        value = body.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            s = value.strip().lower()
            if s in {"1", "true", "yes", "on"}:
                return True
            if s in {"0", "false", "no", "off"}:
                return False
        raise HTTPException(status_code=400, detail=f"{key} must be a boolean")

    @app.get("/api/elements/{element_id}/animation")
    def get_camera_animation(element_id: str) -> dict[str, Any]:
        _assert_camera_element(element_id)
        track = REGISTRY.get_camera_animation(element_id)
        if track is None:
            raise HTTPException(status_code=404, detail="Camera animation is not configured")
        return camera_animation_track_to_dict(track)

    @app.put("/api/elements/{element_id}/animation")
    def put_camera_animation(element_id: str, body: dict[str, Any]) -> dict[str, Any]:
        _assert_camera_element(element_id)

        if "timestamp" in body:
            raise HTTPException(status_code=400, detail="Camera animations are frame-axis only in v1")

        mode_raw = body.get("mode")
        mode = str(mode_raw).strip().lower()
        if mode not in {"follow", "orbit"}:
            raise HTTPException(status_code=400, detail="mode must be one of: follow, orbit")

        target_id_raw = body.get("targetId")
        if not isinstance(target_id_raw, str) or not target_id_raw.strip():
            raise HTTPException(status_code=400, detail="targetId is required")
        target_id = target_id_raw.strip()

        start_frame, end_frame = _parse_frame_range(body)
        try:
            step = int(body.get("step", 1))
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid step")
        if step <= 0:
            raise HTTPException(status_code=400, detail="step must be a positive integer")

        up = _default_camera_up()
        if "up" in body and body.get("up") is not None:
            raw_up = body.get("up")
            if not isinstance(raw_up, (list, tuple)) or len(raw_up) != 3:
                raise HTTPException(status_code=400, detail="up must be a 3D vector")
            try:
                up = (float(raw_up[0]), float(raw_up[1]), float(raw_up[2]))
            except Exception:
                raise HTTPException(status_code=400, detail="up must be numeric")

        params: dict[str, float] = {}
        control_keys: tuple[Any, ...] = tuple()
        if mode == "orbit":
            try:
                turns = float(body.get("turns", 1.0))
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid orbit turns")
            params["turns"] = turns

            if "radius" in body and body.get("radius") is not None:
                try:
                    params["radius"] = float(body.get("radius"))
                except Exception:
                    raise HTTPException(status_code=400, detail="Invalid orbit radius")
            if "phaseDeg" in body and body.get("phaseDeg") is not None:
                try:
                    params["phaseDeg"] = float(body.get("phaseDeg"))
                except Exception:
                    raise HTTPException(status_code=400, detail="Invalid orbit phaseDeg")
            else:
                params["phaseDeg"] = 0.0

            raw_keys = body.get("controlKeys")
            if raw_keys is not None:
                if not isinstance(raw_keys, list):
                    raise HTTPException(status_code=400, detail="controlKeys must be a list")
                parsed_keys = []
                for item in raw_keys:
                    if not isinstance(item, dict):
                        raise HTTPException(status_code=400, detail="controlKeys items must be objects")
                    if "frame" not in item or "positionLocal" not in item:
                        raise HTTPException(status_code=400, detail="Each control key requires frame and positionLocal")
                    try:
                        key_frame = int(item.get("frame"))
                    except Exception:
                        raise HTTPException(status_code=400, detail="control key frame must be an integer")
                    key_pos = item.get("positionLocal")
                    if not isinstance(key_pos, (list, tuple)) or len(key_pos) != 3:
                        raise HTTPException(status_code=400, detail="control key positionLocal must be a 3D vector")
                    try:
                        pos_local = (float(key_pos[0]), float(key_pos[1]), float(key_pos[2]))
                    except Exception:
                        raise HTTPException(status_code=400, detail="control key positionLocal must be numeric")
                    parsed_keys.append((key_frame, pos_local))
                control_keys = tuple(
                    CameraControlKey(frame=int(frame_v), position_local=pos_local)  # type: ignore[name-defined]
                    for frame_v, pos_local in sorted(parsed_keys, key=lambda x: int(x[0]))
                )

        track = CameraAnimationTrack(
            camera_id=element_id,
            mode=mode,  # type: ignore[arg-type]
            target_id=target_id,
            start_frame=start_frame,
            end_frame=end_frame,
            step=step,
            interpolation="catmull_rom",
            up=up,
            params=params,
            control_keys=control_keys,
        )

        try:
            out = REGISTRY.set_camera_animation(track)
        except KeyError as ex:
            raise HTTPException(status_code=404, detail=f"Unknown element: {str(ex)}")
        except (TypeError, ValueError) as ex:
            raise HTTPException(status_code=400, detail=str(ex))
        return camera_animation_track_to_dict(out)

    @app.patch("/api/elements/{element_id}/animation/key")
    def patch_camera_animation_key(element_id: str, body: dict[str, Any]) -> dict[str, Any]:
        _assert_camera_element(element_id)
        if "timestamp" in body:
            raise HTTPException(status_code=400, detail="Camera animations are frame-axis only in v1")

        if "frame" not in body:
            raise HTTPException(status_code=400, detail="frame is required")
        if "position" not in body:
            raise HTTPException(status_code=400, detail="position is required")

        try:
            frame = int(body.get("frame"))
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid frame")
        if frame < 0:
            raise HTTPException(status_code=400, detail="frame must be >= 0")
        raw_pos = body.get("position")
        if not isinstance(raw_pos, (list, tuple)) or len(raw_pos) != 3:
            raise HTTPException(status_code=400, detail="position must be a 3D vector")
        try:
            pos = (float(raw_pos[0]), float(raw_pos[1]), float(raw_pos[2]))
        except Exception:
            raise HTTPException(status_code=400, detail="position must be numeric")

        pull_enabled = _parse_bool_field(body, "pullEnabled", default=False)
        try:
            pull_radius_frames = int(body.get("pullRadiusFrames", 0))
        except Exception:
            raise HTTPException(status_code=400, detail="pullRadiusFrames must be an integer")
        if pull_radius_frames < 0:
            raise HTTPException(status_code=400, detail="pullRadiusFrames must be >= 0")
        pull_pinned_ends = _parse_bool_field(body, "pullPinnedEnds", default=False)

        try:
            out = REGISTRY.update_camera_animation_key(
                element_id,
                frame=frame,
                new_world_position=pos,
                pull_enabled=pull_enabled,
                pull_radius_frames=pull_radius_frames,
                pull_pinned_ends=pull_pinned_ends,
            )
        except KeyError as ex:
            raise HTTPException(status_code=404, detail=f"Unknown element: {str(ex)}")
        except (TypeError, ValueError) as ex:
            raise HTTPException(status_code=400, detail=str(ex))
        return camera_animation_track_to_dict(out)

    @app.post("/api/elements/{element_id}/animation/key")
    def post_camera_animation_key(element_id: str, body: dict[str, Any]) -> dict[str, Any]:
        _assert_camera_element(element_id)
        if "timestamp" in body:
            raise HTTPException(status_code=400, detail="Camera animations are frame-axis only in v1")
        if "frame" not in body:
            raise HTTPException(status_code=400, detail="frame is required")
        try:
            frame = int(body.get("frame"))
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid frame")
        if frame < 0:
            raise HTTPException(status_code=400, detail="frame must be >= 0")

        pos: tuple[float, float, float] | None = None
        if "position" in body and body.get("position") is not None:
            raw_pos = body.get("position")
            if not isinstance(raw_pos, (list, tuple)) or len(raw_pos) != 3:
                raise HTTPException(status_code=400, detail="position must be a 3D vector")
            try:
                pos = (float(raw_pos[0]), float(raw_pos[1]), float(raw_pos[2]))
            except Exception:
                raise HTTPException(status_code=400, detail="position must be numeric")

        try:
            out = REGISTRY.insert_camera_animation_key(element_id, frame=frame, world_position=pos)
        except KeyError as ex:
            raise HTTPException(status_code=404, detail=f"Unknown element: {str(ex)}")
        except (TypeError, ValueError) as ex:
            raise HTTPException(status_code=400, detail=str(ex))
        return camera_animation_track_to_dict(out)

    @app.delete("/api/elements/{element_id}/animation/key")
    def delete_camera_animation_key(element_id: str, frame: int | None = None, timestamp: float | None = None) -> dict[str, Any]:
        _assert_camera_element(element_id)
        if timestamp is not None:
            raise HTTPException(status_code=400, detail="Camera animations are frame-axis only in v1")
        if frame is None:
            raise HTTPException(status_code=400, detail="frame is required")
        frame_i = int(frame)
        if frame_i < 0:
            raise HTTPException(status_code=400, detail="frame must be >= 0")

        try:
            out = REGISTRY.delete_camera_animation_key(element_id, frame=frame_i)
        except KeyError as ex:
            raise HTTPException(status_code=404, detail=f"Unknown element: {str(ex)}")
        except (TypeError, ValueError) as ex:
            raise HTTPException(status_code=400, detail=str(ex))
        return camera_animation_track_to_dict(out)

    @app.post("/api/elements/{element_id}/animation/key/duplicate")
    def duplicate_camera_animation_key(element_id: str, body: dict[str, Any]) -> dict[str, Any]:
        _assert_camera_element(element_id)
        if "timestamp" in body:
            raise HTTPException(status_code=400, detail="Camera animations are frame-axis only in v1")
        if "sourceFrame" not in body or "targetFrame" not in body:
            raise HTTPException(status_code=400, detail="sourceFrame and targetFrame are required")
        try:
            source_frame = int(body.get("sourceFrame"))
            target_frame = int(body.get("targetFrame"))
        except Exception:
            raise HTTPException(status_code=400, detail="sourceFrame and targetFrame must be integers")
        if source_frame < 0 or target_frame < 0:
            raise HTTPException(status_code=400, detail="frame values must be >= 0")

        try:
            out = REGISTRY.duplicate_camera_animation_key(
                element_id,
                source_frame=source_frame,
                target_frame=target_frame,
            )
        except KeyError as ex:
            raise HTTPException(status_code=404, detail=f"Unknown element: {str(ex)}")
        except (TypeError, ValueError) as ex:
            raise HTTPException(status_code=400, detail=str(ex))
        return camera_animation_track_to_dict(out)

    @app.post("/api/elements/{element_id}/animation/smooth")
    def smooth_camera_animation(element_id: str, body: dict[str, Any]) -> dict[str, Any]:
        _assert_camera_element(element_id)
        if "timestamp" in body:
            raise HTTPException(status_code=400, detail="Camera animations are frame-axis only in v1")
        start_frame: int | None = None
        end_frame: int | None = None
        if "startFrame" in body and body.get("startFrame") is not None:
            try:
                start_frame = int(body.get("startFrame"))
            except Exception:
                raise HTTPException(status_code=400, detail="startFrame must be an integer")
            if start_frame < 0:
                raise HTTPException(status_code=400, detail="startFrame must be >= 0")
        if "endFrame" in body and body.get("endFrame") is not None:
            try:
                end_frame = int(body.get("endFrame"))
            except Exception:
                raise HTTPException(status_code=400, detail="endFrame must be an integer")
            if end_frame < 0:
                raise HTTPException(status_code=400, detail="endFrame must be >= 0")

        try:
            passes = int(body.get("passes", 1))
        except Exception:
            raise HTTPException(status_code=400, detail="passes must be an integer")
        if passes <= 0:
            raise HTTPException(status_code=400, detail="passes must be >= 1")
        pinned_ends = _parse_bool_field(body, "pinnedEnds", default=True)

        try:
            out = REGISTRY.smooth_camera_animation_keys(
                element_id,
                start_frame=start_frame,
                end_frame=end_frame,
                passes=passes,
                pinned_ends=pinned_ends,
            )
        except KeyError as ex:
            raise HTTPException(status_code=404, detail=f"Unknown element: {str(ex)}")
        except (TypeError, ValueError) as ex:
            raise HTTPException(status_code=400, detail=str(ex))
        return camera_animation_track_to_dict(out)

    @app.delete("/api/elements/{element_id}/animation")
    def delete_camera_animation(element_id: str) -> dict[str, Any]:
        _assert_camera_element(element_id)
        removed = REGISTRY.clear_camera_animation(element_id)
        return {"ok": True, "cameraId": element_id, "removed": bool(removed)}

    @app.get("/api/elements/{element_id}/animation/trajectory")
    def get_camera_animation_trajectory(
        element_id: str,
        startFrame: int | None = None,
        endFrame: int | None = None,
        stride: int = 1,
    ) -> dict[str, Any]:
        _assert_camera_element(element_id)
        try:
            frames, positions = REGISTRY.get_camera_frame_samples(
                element_id,
                start_frame=startFrame,
                end_frame=endFrame,
                stride=stride,
            )
        except KeyError as ex:
            raise HTTPException(status_code=404, detail=f"Unknown element: {str(ex)}")
        except (TypeError, ValueError) as ex:
            raise HTTPException(status_code=400, detail=str(ex))

        if not frames:
            raise HTTPException(status_code=404, detail="No trajectory samples available")
        return {
            "cameraId": element_id,
            "startFrame": int(frames[0]),
            "endFrame": int(frames[-1]),
            "stride": int(max(1, int(stride))),
            "frames": [int(f) for f in frames],
            "positions": [[float(p[0]), float(p[1]), float(p[2])] for p in positions],
        }

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
            size = _as_vec("size", 3)
            radii = _as_vec("radii", 3)
            color = _as_vec("color", 3)

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
                size=size,
                radii=radii,
                color=color,
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
