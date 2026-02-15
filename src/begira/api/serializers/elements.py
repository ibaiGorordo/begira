from __future__ import annotations

from typing import Any

import numpy as np

from ...core.elements import (
    ElementBase,
    PointCloudElement,
    GaussianSplatElement,
    CameraElement,
    ImageElement,
    Box3DElement,
    Ellipsoid3DElement,
)


def bounds_from_positions(pos: np.ndarray) -> dict[str, list[float]]:
    if pos.size == 0:
        return {"min": [0.0, 0.0, 0.0], "max": [0.0, 0.0, 0.0]}
    bounds_min = pos.min(axis=0)
    bounds_max = pos.max(axis=0)
    return {"min": bounds_min.tolist(), "max": bounds_max.tolist()}


def bounds_from_position(pos: tuple[float, float, float], *, radius: float = 0.5) -> dict[str, list[float]]:
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    r = float(radius)
    return {
        "min": [x - r, y - r, z - r],
        "max": [x + r, y + r, z + r],
    }


def bounds_from_position_extents(
    pos: tuple[float, float, float],
    extents: tuple[float, float, float],
) -> dict[str, list[float]]:
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    ex, ey, ez = float(extents[0]), float(extents[1]), float(extents[2])
    return {
        "min": [x - ex, y - ey, z - ez],
        "max": [x + ex, y + ey, z + ez],
    }


def element_to_list_item(e: ElementBase) -> dict[str, Any]:
    base = {
        "id": e.id,
        "type": e.type,
        "name": e.name,
        "revision": int(e.revision),
        "stateRevision": int(e.state_revision),
        "dataRevision": int(e.data_revision),
        "createdAt": float(e.created_at),
    }

    if isinstance(e, PointCloudElement):
        return {
            **base,
            "bounds": bounds_from_positions(e.positions),
            "summary": {"pointCount": int(e.positions.shape[0])},
            "visible": bool(e.visible),
            "deleted": bool(e.deleted),
        }

    if isinstance(e, GaussianSplatElement):
        return {
            **base,
            "bounds": bounds_from_positions(e.positions),
            "summary": {"count": int(e.positions.shape[0])},
            "visible": bool(e.visible),
            "deleted": bool(e.deleted),
        }

    if isinstance(e, CameraElement):
        return {
            **base,
            "bounds": bounds_from_position(e.position),
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

    if isinstance(e, ImageElement):
        return {
            **base,
            "summary": {"width": int(e.width), "height": int(e.height), "channels": int(e.channels)},
            "mimeType": str(e.mime_type),
            "visible": bool(e.visible),
            "deleted": bool(e.deleted),
        }

    if isinstance(e, Box3DElement):
        half = (float(e.size[0]) * 0.5, float(e.size[1]) * 0.5, float(e.size[2]) * 0.5)
        return {
            **base,
            "bounds": bounds_from_position_extents(e.position, half),
            "summary": {"size": [float(e.size[0]), float(e.size[1]), float(e.size[2])]},
            "position": list(e.position),
            "rotation": list(e.rotation),
            "size": [float(e.size[0]), float(e.size[1]), float(e.size[2])],
            "color": [float(e.color[0]), float(e.color[1]), float(e.color[2])],
            "visible": bool(e.visible),
            "deleted": bool(e.deleted),
        }

    if isinstance(e, Ellipsoid3DElement):
        return {
            **base,
            "bounds": bounds_from_position_extents(e.position, e.radii),
            "summary": {"radii": [float(e.radii[0]), float(e.radii[1]), float(e.radii[2])]},
            "position": list(e.position),
            "rotation": list(e.rotation),
            "radii": [float(e.radii[0]), float(e.radii[1]), float(e.radii[2])],
            "color": [float(e.color[0]), float(e.color[1]), float(e.color[2])],
            "visible": bool(e.visible),
            "deleted": bool(e.deleted),
        }

    return base


def element_to_meta_item(e: ElementBase) -> dict[str, Any]:
    base = {
        "id": e.id,
        "type": e.type,
        "name": e.name,
        "revision": int(e.revision),
        "stateRevision": int(e.state_revision),
        "dataRevision": int(e.data_revision),
    }

    if isinstance(e, PointCloudElement):
        schema: dict[str, dict[str, Any]] = {"position": {"type": "float32", "components": 3}}
        if e.colors is not None:
            schema["color"] = {"type": "uint8", "components": 3, "normalized": True}

        return {
            **base,
            "bounds": bounds_from_positions(e.positions),
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
            **base,
            "bounds": bounds_from_positions(e.positions),
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
            **base,
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

    if isinstance(e, ImageElement):
        return {
            **base,
            "width": int(e.width),
            "height": int(e.height),
            "channels": int(e.channels),
            "mimeType": str(e.mime_type),
            "position": list(e.position),
            "rotation": list(e.rotation),
            "visible": bool(e.visible),
            "deleted": bool(e.deleted),
            "payloads": {
                "image": {
                    "url": f"/api/elements/{e.id}/payloads/image",
                    "contentType": str(e.mime_type),
                }
            },
        }

    if isinstance(e, Box3DElement):
        half = (float(e.size[0]) * 0.5, float(e.size[1]) * 0.5, float(e.size[2]) * 0.5)
        return {
            **base,
            "bounds": bounds_from_position_extents(e.position, half),
            "position": list(e.position),
            "rotation": list(e.rotation),
            "size": [float(e.size[0]), float(e.size[1]), float(e.size[2])],
            "color": [float(e.color[0]), float(e.color[1]), float(e.color[2])],
            "visible": bool(e.visible),
            "deleted": bool(e.deleted),
        }

    if isinstance(e, Ellipsoid3DElement):
        return {
            **base,
            "bounds": bounds_from_position_extents(e.position, e.radii),
            "position": list(e.position),
            "rotation": list(e.rotation),
            "radii": [float(e.radii[0]), float(e.radii[1]), float(e.radii[2])],
            "color": [float(e.color[0]), float(e.color[1]), float(e.color[2])],
            "visible": bool(e.visible),
            "deleted": bool(e.deleted),
        }

    raise ValueError(f"Unsupported element type: {e.type}")
