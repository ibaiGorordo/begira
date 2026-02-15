from __future__ import annotations

import numpy as np

from begira.core.animation import CameraAnimationTrack, pose_matrix
from begira.core.registry import InMemoryRegistry


def _single_point(offset: float = 0.0) -> np.ndarray:
    return np.asarray([[offset, 0.0, 0.0]], dtype=np.float32)


def _quat_close(a: np.ndarray, b: np.ndarray, atol: float = 1e-5) -> bool:
    # q and -q represent the same orientation.
    return bool(np.allclose(a, b, atol=atol) or np.allclose(a, -b, atol=atol))


def test_follow_preserves_full_relative_transform() -> None:
    reg = InMemoryRegistry()
    target = reg.upsert_pointcloud(name="target", positions=_single_point(), colors=None, static=True)
    cam = reg.upsert_camera(
        name="cam",
        position=(0.0, -2.0, 1.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
        fov=60.0,
        near=0.01,
        far=100.0,
        static=True,
    )

    for frame in range(6):
        reg.update_element_meta(
            target.id,
            position=(float(frame) * 0.5, float(frame) * 0.2, 0.0),
            rotation=(0.0, 0.0, np.sin(0.25 * frame), np.cos(0.25 * frame)),
            frame=frame,
        )

    track = CameraAnimationTrack(
        camera_id=cam.id,
        mode="follow",
        target_id=target.id,
        start_frame=0,
        end_frame=5,
        step=1,
    )
    reg.set_camera_animation(track)

    cam0 = reg.get_element(cam.id, frame=0)
    tgt0 = reg.get_element(target.id, frame=0)
    assert cam0 is not None
    assert tgt0 is not None

    t_rel0 = np.linalg.inv(pose_matrix(tgt0.position, tgt0.rotation)) @ pose_matrix(cam0.position, cam0.rotation)

    for frame in range(6):
        cam_f = reg.get_element(cam.id, frame=frame)
        tgt_f = reg.get_element(target.id, frame=frame)
        assert cam_f is not None
        assert tgt_f is not None
        t_rel_f = np.linalg.inv(pose_matrix(tgt_f.position, tgt_f.rotation)) @ pose_matrix(cam_f.position, cam_f.rotation)
        assert np.allclose(t_rel_f, t_rel0, atol=1e-4)


def test_orbit_tracks_moving_target_and_key_edit_rebakes() -> None:
    reg = InMemoryRegistry()
    target = reg.upsert_pointcloud(name="target", positions=_single_point(), colors=None, static=True)
    cam = reg.upsert_camera(
        name="cam",
        position=(0.0, -3.0, 0.5),
        rotation=(0.0, 0.0, 0.0, 1.0),
        fov=60.0,
        near=0.01,
        far=100.0,
        static=True,
    )

    for frame in range(21):
        reg.update_element_meta(
            target.id,
            position=(float(frame) * 0.1, 0.0, 0.0),
            rotation=(0.0, 0.0, 0.0, 1.0),
            frame=frame,
        )

    track = CameraAnimationTrack(
        camera_id=cam.id,
        mode="orbit",
        target_id=target.id,
        start_frame=0,
        end_frame=20,
        step=5,
        params={"turns": 1.0, "radius": 3.0, "phaseDeg": 0.0},
    )
    baked = reg.set_camera_animation(track)
    assert baked.mode == "orbit"
    assert len(baked.control_keys) >= 3

    for frame in [int(k.frame) for k in baked.control_keys]:
        cam_f = reg.get_element(cam.id, frame=frame)
        tgt_f = reg.get_element(target.id, frame=frame)
        assert cam_f is not None
        assert tgt_f is not None
        cam_pos = np.asarray(cam_f.position, dtype=np.float64)
        tgt_pos = np.asarray(tgt_f.position, dtype=np.float64)
        dist = float(np.linalg.norm(cam_pos - tgt_pos))
        assert np.isclose(dist, 3.0, atol=1e-2)

    before = reg.get_element(cam.id, frame=10)
    assert before is not None
    target_f10 = reg.get_element(target.id, frame=10)
    assert target_f10 is not None
    new_world = np.asarray(target_f10.position, dtype=np.float64) + np.asarray([0.0, -5.0, 1.0], dtype=np.float64)

    updated = reg.update_camera_animation_key(cam.id, frame=10, new_world_position=new_world)
    assert updated.revision > baked.revision

    after = reg.get_element(cam.id, frame=10)
    assert after is not None
    assert not np.allclose(np.asarray(before.position, dtype=np.float64), np.asarray(after.position, dtype=np.float64), atol=1e-4)
    assert np.allclose(np.asarray(after.position, dtype=np.float64), new_world, atol=1e-3)

    q_after = np.asarray(after.rotation, dtype=np.float64)
    # Ensure orientation changed in response to key edit (non-identity sanity).
    assert not _quat_close(q_after, np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64), atol=1e-3)


def test_orbit_pull_insert_delete_duplicate_and_smooth() -> None:
    reg = InMemoryRegistry()
    target = reg.upsert_pointcloud(name="target", positions=_single_point(), colors=None, static=True)
    cam = reg.upsert_camera(
        name="cam",
        position=(0.0, -3.0, 1.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
        fov=60.0,
        near=0.01,
        far=100.0,
        static=True,
    )

    for frame in range(21):
        reg.update_element_meta(
            target.id,
            position=(float(frame) * 0.15, 0.1 * np.sin(0.2 * frame), 0.0),
            rotation=(0.0, 0.0, 0.0, 1.0),
            frame=frame,
        )

    baked = reg.set_camera_animation(
        CameraAnimationTrack(
            camera_id=cam.id,
            mode="orbit",
            target_id=target.id,
            start_frame=0,
            end_frame=20,
            step=5,
            params={"turns": 1.0, "radius": 3.0, "phaseDeg": 0.0},
        )
    )
    assert baked.mode == "orbit"
    assert len(baked.control_keys) >= 4

    by_frame_before = {int(k.frame): np.asarray(k.position_local, dtype=np.float64) for k in baked.control_keys}
    key_mid = int(baked.control_keys[len(baked.control_keys) // 2].frame)
    start_frame = int(baked.start_frame)
    end_frame = int(baked.end_frame)

    cam_mid = reg.get_element(cam.id, frame=key_mid)
    assert cam_mid is not None
    moved_world = np.asarray(cam_mid.position, dtype=np.float64) + np.asarray([0.0, 0.8, 0.2], dtype=np.float64)

    pulled = reg.update_camera_animation_key(
        cam.id,
        frame=key_mid,
        new_world_position=moved_world,
        pull_enabled=True,
        pull_radius_frames=8,
        pull_pinned_ends=True,
    )
    by_frame_pulled = {int(k.frame): np.asarray(k.position_local, dtype=np.float64) for k in pulled.control_keys}
    assert not np.allclose(by_frame_pulled[key_mid], by_frame_before[key_mid], atol=1e-6)
    assert np.allclose(by_frame_pulled[start_frame], by_frame_before[start_frame], atol=1e-6)
    assert np.allclose(by_frame_pulled[end_frame], by_frame_before[end_frame], atol=1e-6)

    pulled_unpinned = reg.update_camera_animation_key(
        cam.id,
        frame=key_mid,
        new_world_position=moved_world + np.asarray([0.0, 0.5, 0.0], dtype=np.float64),
        pull_enabled=True,
        pull_radius_frames=100,
        pull_pinned_ends=False,
    )
    by_frame_unpinned = {int(k.frame): np.asarray(k.position_local, dtype=np.float64) for k in pulled_unpinned.control_keys}
    assert not np.allclose(by_frame_unpinned[start_frame], by_frame_pulled[start_frame], atol=1e-6)

    # Orbit pull is cyclic: moving start-frame key should also influence end-frame key.
    start_pose = reg.get_element(cam.id, frame=start_frame)
    assert start_pose is not None
    start_world = np.asarray(start_pose.position, dtype=np.float64)
    cycled = reg.update_camera_animation_key(
        cam.id,
        frame=start_frame,
        new_world_position=start_world + np.asarray([0.2, 0.0, 0.0], dtype=np.float64),
        pull_enabled=True,
        pull_radius_frames=2,
        pull_pinned_ends=False,
    )
    by_frame_cycled = {int(k.frame): np.asarray(k.position_local, dtype=np.float64) for k in cycled.control_keys}
    assert not np.allclose(by_frame_cycled[end_frame], by_frame_unpinned[end_frame], atol=1e-6)

    inserted = reg.insert_camera_animation_key(cam.id, frame=7)
    frames_inserted = {int(k.frame) for k in inserted.control_keys}
    assert 7 in frames_inserted

    duplicated = reg.duplicate_camera_animation_key(cam.id, source_frame=7, target_frame=9)
    frames_duplicated = {int(k.frame) for k in duplicated.control_keys}
    assert 9 in frames_duplicated

    before_smooth = {int(k.frame): np.asarray(k.position_local, dtype=np.float64) for k in duplicated.control_keys}
    smoothed = reg.smooth_camera_animation_keys(cam.id, passes=1, pinned_ends=True)
    after_smooth = {int(k.frame): np.asarray(k.position_local, dtype=np.float64) for k in smoothed.control_keys}
    interior_frames = [f for f in sorted(after_smooth.keys()) if f not in {start_frame, end_frame}]
    assert any(not np.allclose(after_smooth[f], before_smooth[f], atol=1e-8) for f in interior_frames)

    deleted = reg.delete_camera_animation_key(cam.id, frame=9)
    frames_deleted = {int(k.frame) for k in deleted.control_keys}
    assert 9 not in frames_deleted
