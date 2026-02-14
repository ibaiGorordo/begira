from __future__ import annotations

import numpy as np
import pytest

from begira.registry import REGISTRY
from begira.runner import BegiraServer


def test_logged_handles_support_transform_visibility_and_delete() -> None:
    server = BegiraServer(host="127.0.0.1", port=0, url="http://127.0.0.1:0/")

    gs = server.log_gaussians(
        "gs",
        positions=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        sh0=np.zeros((1, 3), dtype=np.float32),
        opacity=np.ones((1, 1), dtype=np.float32),
        scales=np.zeros((1, 3), dtype=np.float32),
        rotations=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    )

    t = np.eye(4, dtype=np.float64)
    t[:3, 3] = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    gs.set_transform(t)

    stored = REGISTRY.get_element(gs)
    assert stored is not None
    assert stored.position == (1.0, 2.0, 3.0)  # type: ignore[attr-defined]

    gs.disable()
    stored = REGISTRY.get_element(gs)
    assert stored is not None
    assert stored.visible is False

    gs.enable()
    stored = REGISTRY.get_element(gs)
    assert stored is not None
    assert stored.visible is True

    gs.delete()
    stored = REGISTRY.get_element(gs)
    assert stored is not None
    assert stored.deleted is True


def test_camera_handle_look_at_another_handle() -> None:
    server = BegiraServer(host="127.0.0.1", port=0, url="http://127.0.0.1:0/")

    points = server.log_points(
        "pts",
        np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        np.array([[255, 0, 0]], dtype=np.uint8),
    )
    points.set_transform(np.array([[1.0, 0.0, 0.0, 3.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64))

    cam = server.log_camera("cam", position=(0.0, 0.0, 0.0))
    cam.look_at(points)

    stored = REGISTRY.get_element(cam)
    assert stored is not None
    q = stored.rotation  # type: ignore[attr-defined]
    assert not np.allclose(np.array(q, dtype=np.float64), np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64))


def test_camera_handle_look_at_with_distance_repositions_camera() -> None:
    server = BegiraServer(host="127.0.0.1", port=0, url="http://127.0.0.1:0/")

    points = server.log_points(
        "pts2",
        np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        np.array([[255, 0, 0]], dtype=np.uint8),
    )
    points.set_transform(np.array([[1.0, 0.0, 0.0, 3.0], [0.0, 1.0, 0.0, -1.0], [0.0, 0.0, 1.0, 0.5], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64))

    cam = server.log_camera("cam2", position=(0.0, 0.0, 0.0))
    cam.look_at(points, 1.0)

    stored = REGISTRY.get_element(cam)
    assert stored is not None
    cam_pos = np.asarray(stored.position, dtype=np.float64)  # type: ignore[attr-defined]
    target_pos = np.array([3.0, -1.0, 0.5], dtype=np.float64)
    assert np.isclose(np.linalg.norm(target_pos - cam_pos), 1.0, atol=1e-6)


def test_camera_handle_look_at_with_invalid_distance_raises() -> None:
    server = BegiraServer(host="127.0.0.1", port=0, url="http://127.0.0.1:0/")
    cam = server.log_camera("cam3", position=(0.0, 0.0, 0.0))

    with pytest.raises(ValueError, match="distance must be a positive finite number"):
        cam.look_at((1.0, 0.0, 0.0), 0.0)


def test_handle_properties_expose_pose_name_and_count() -> None:
    server = BegiraServer(host="127.0.0.1", port=0, url="http://127.0.0.1:0/")

    points = server.log_points(
        "pts_meta",
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8),
    )
    qz90 = np.array([0.0, 0.0, np.sin(np.pi / 4.0), np.cos(np.pi / 4.0)], dtype=np.float64)
    points.set_pose(position=(2.0, 3.0, 4.0), rotation=qz90)

    assert points.name == "pts_meta"
    assert points.kind == "pointcloud"
    assert points.revision >= 1
    assert points.position == (2.0, 3.0, 4.0)
    assert np.allclose(np.asarray(points.orientation), qz90, atol=1e-6)
    assert points.count == 2
    assert points.visible is True
    assert points.deleted is False

    r = points.rotation_matrix
    assert r.shape == (3, 3)
    x_axis = r @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
    assert np.allclose(x_axis, np.array([0.0, 1.0, 0.0], dtype=np.float64), atol=1e-6)

    points.disable()
    assert points.visible is False
    points.delete()
    assert points.deleted is True

    gs = server.log_gaussians(
        "gs_meta",
        positions=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        sh0=np.zeros((1, 3), dtype=np.float32),
        opacity=np.ones((1, 1), dtype=np.float32),
        scales=np.zeros((1, 3), dtype=np.float32),
        rotations=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    )
    assert gs.count == 1

    cam = server.log_camera(
        "cam_meta",
        fov=70.0,
        near=0.05,
        far=500.0,
        width=1280,
        height=720,
        intrinsic_matrix=((600.0, 0.0, 640.0), (0.0, 610.0, 360.0), (0.0, 0.0, 1.0)),
    )
    assert cam.kind == "camera"
    assert cam.fov == 70.0
    assert cam.near == 0.05
    assert cam.far == 500.0
    assert cam.width == 1280
    assert cam.height == 720
    assert cam.intrinsic_matrix is not None
    assert np.asarray(cam.intrinsic_matrix).shape == (3, 3)
