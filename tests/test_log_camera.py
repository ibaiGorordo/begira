from __future__ import annotations

from begira.core.registry import REGISTRY
from begira.runtime.server import BegiraServer


def test_log_camera_from_server_object() -> None:
    server = BegiraServer(host="127.0.0.1", port=0, url="http://127.0.0.1:0/")

    element_id = server.log_camera(
        "cam0",
        position=(1.0, 2.0, 3.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
        fov=70.0,
        near=0.05,
        far=500.0,
    )

    stored = REGISTRY.get_element(element_id)
    assert stored is not None
    assert stored.type == "camera"
    assert stored.position == (1.0, 2.0, 3.0)  # type: ignore[attr-defined]
    assert stored.rotation == (0.0, 0.0, 0.0, 1.0)  # type: ignore[attr-defined]
    assert stored.fov == 70.0  # type: ignore[attr-defined]
    assert stored.near == 0.05  # type: ignore[attr-defined]
    assert stored.far == 500.0  # type: ignore[attr-defined]


def test_log_camera_supports_intrinsics_and_size() -> None:
    server = BegiraServer(host="127.0.0.1", port=0, url="http://127.0.0.1:0/")

    cam = server.log_camera(
        "cam1",
        width=1280,
        height=720,
        intrinsic_matrix=((600.0, 0.0, 640.0), (0.0, 610.0, 360.0), (0.0, 0.0, 1.0)),
        fov=None,
    )

    stored = REGISTRY.get_element(cam)
    assert stored is not None
    assert stored.type == "camera"
    assert stored.width == 1280  # type: ignore[attr-defined]
    assert stored.height == 720  # type: ignore[attr-defined]
    assert stored.intrinsic_matrix == ((600.0, 0.0, 640.0), (0.0, 610.0, 360.0), (0.0, 0.0, 1.0))  # type: ignore[attr-defined]
