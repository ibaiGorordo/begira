from __future__ import annotations

import numpy as np

from begira.animation import pose_matrix
from begira.runner import BegiraServer


def _relative_transform(
    cam_pos: list[float] | tuple[float, float, float],
    cam_rot: list[float] | tuple[float, float, float, float],
    target_pos: list[float] | tuple[float, float, float],
    target_rot: list[float] | tuple[float, float, float, float],
) -> np.ndarray:
    t_cam = pose_matrix(cam_pos, cam_rot)
    t_target = pose_matrix(target_pos, target_rot)
    return np.linalg.inv(t_target) @ t_cam


def test_camera_animator_follow_and_orbit_and_clear() -> None:
    server = BegiraServer(host="127.0.0.1", port=0, url="http://127.0.0.1:0/")

    target = server.log_points(
        "target",
        np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        np.asarray([[255, 0, 0]], dtype=np.uint8),
        static=True,
    )
    cam = server.log_camera(
        "cam",
        position=(0.0, -2.0, 0.5),
        rotation=(0.0, 0.0, 0.0, 1.0),
        fov=60.0,
        near=0.01,
        far=100.0,
        static=True,
    )

    for frame in range(8):
        target.set_pose(position=(float(frame) * 0.3, 0.0, 0.0), frame=frame)

    cam.animate.follow(target, start_frame=0, end_frame=7)
    track_follow = cam.animate.get_track()
    assert track_follow is not None
    assert track_follow["mode"] == "follow"

    rel0 = _relative_transform(
        server.get_element_meta(cam.id, frame=0)["position"],
        server.get_element_meta(cam.id, frame=0)["rotation"],
        server.get_element_meta(target.id, frame=0)["position"],
        server.get_element_meta(target.id, frame=0)["rotation"],
    )
    rel7 = _relative_transform(
        server.get_element_meta(cam.id, frame=7)["position"],
        server.get_element_meta(cam.id, frame=7)["rotation"],
        server.get_element_meta(target.id, frame=7)["position"],
        server.get_element_meta(target.id, frame=7)["rotation"],
    )
    assert np.allclose(rel0, rel7, atol=1e-4)

    cam.animate.orbit(target, start_frame=0, end_frame=7, turns=1.0, radius=2.5, phase_deg=0.0, step=2)
    track_orbit = cam.animate.get_track()
    assert track_orbit is not None
    assert track_orbit["mode"] == "orbit"
    assert len(track_orbit["controlKeys"]) >= 3

    trajectory = cam.animate.get_trajectory(start_frame=0, end_frame=7, stride=1)
    assert len(trajectory["frames"]) == 8
    assert trajectory["frames"][0] == 0
    assert trajectory["frames"][-1] == 7

    key_frame = int(track_orbit["controlKeys"][0]["frame"])
    key_pos_before = np.asarray(server.get_element_meta(cam.id, frame=key_frame)["position"], dtype=np.float64)
    cam.animate.update_key(
        key_frame,
        key_pos_before + np.asarray([0.0, -1.0, 0.25], dtype=np.float64),
        pull_enabled=True,
        pull_radius_frames=6,
        pull_pinned_ends=True,
    )
    key_pos_after = np.asarray(server.get_element_meta(cam.id, frame=key_frame)["position"], dtype=np.float64)
    assert not np.allclose(key_pos_before, key_pos_after, atol=1e-4)

    cam.animate.insert_key(3)
    track_after_insert = cam.animate.get_track()
    assert track_after_insert is not None
    assert any(int(k["frame"]) == 3 for k in track_after_insert["controlKeys"])

    cam.animate.duplicate_key(3, 4)
    track_after_duplicate = cam.animate.get_track()
    assert track_after_duplicate is not None
    assert any(int(k["frame"]) == 4 for k in track_after_duplicate["controlKeys"])

    cam.animate.smooth(passes=1, pinned_ends=True)
    track_after_smooth = cam.animate.get_track()
    assert track_after_smooth is not None

    cam.animate.delete_key(4)
    track_after_delete = cam.animate.get_track()
    assert track_after_delete is not None
    assert all(int(k["frame"]) != 4 for k in track_after_delete["controlKeys"])

    cam.animate.clear()
    assert cam.animate.get_track() is None
