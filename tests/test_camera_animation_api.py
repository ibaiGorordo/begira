from __future__ import annotations

import uuid

import numpy as np

from begira.core.registry import REGISTRY
from begira.runtime.app import create_app


def _skip(msg: str) -> None:  # pragma: no cover
    try:
        import pytest  # type: ignore

        pytest.skip(msg)
    except Exception:
        raise RuntimeError(msg)


def _seed_camera_and_target(prefix: str) -> tuple[str, str]:
    target_id = f"{prefix}_target_{uuid.uuid4().hex}"
    camera_id = f"{prefix}_camera_{uuid.uuid4().hex}"
    REGISTRY.upsert_pointcloud(
        name="target",
        positions=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        colors=np.asarray([[255, 0, 0]], dtype=np.uint8),
        element_id=target_id,
        static=True,
    )
    REGISTRY.upsert_camera(
        name="camera",
        position=(0.0, -2.0, 0.5),
        rotation=(0.0, 0.0, 0.0, 1.0),
        fov=60.0,
        near=0.01,
        far=100.0,
        element_id=camera_id,
        static=True,
    )
    for frame in range(10):
        REGISTRY.update_element_meta(
            target_id,
            position=(float(frame) * 0.25, 0.0, 0.0),
            rotation=(0.0, 0.0, 0.0, 1.0),
            frame=frame,
        )
    return camera_id, target_id


def test_camera_animation_api_crud_and_trajectory() -> None:
    try:
        from fastapi.testclient import TestClient
    except Exception as e:  # pragma: no cover
        _skip(f"TestClient not available ({e!r}); install test extras to run this test")
        return

    camera_id, target_id = _seed_camera_and_target("anim_api")
    client = TestClient(create_app())

    create = client.put(
        f"/api/elements/{camera_id}/animation",
        json={
            "mode": "orbit",
            "targetId": target_id,
            "startFrame": 0,
            "endFrame": 9,
            "step": 3,
            "turns": 1.0,
            "radius": 2.5,
            "phaseDeg": 0.0,
        },
    )
    assert create.status_code == 200
    track = create.json()
    assert track["cameraId"] == camera_id
    assert track["mode"] == "orbit"
    assert len(track["controlKeys"]) >= 3

    fetch = client.get(f"/api/elements/{camera_id}/animation")
    assert fetch.status_code == 200
    fetched = fetch.json()
    assert fetched["revision"] >= 1

    traj = client.get(
        f"/api/elements/{camera_id}/animation/trajectory",
        params={"startFrame": 0, "endFrame": 9, "stride": 1},
    )
    assert traj.status_code == 200
    trajectory = traj.json()
    assert trajectory["cameraId"] == camera_id
    assert trajectory["frames"] == sorted(trajectory["frames"])
    assert len(trajectory["frames"]) == len(trajectory["positions"])
    assert trajectory["startFrame"] == 0
    assert trajectory["endFrame"] == 9

    key_frame = int(track["controlKeys"][0]["frame"])
    key_pos = trajectory["positions"][min(key_frame, len(trajectory["positions"]) - 1)]
    patch = client.patch(
        f"/api/elements/{camera_id}/animation/key",
        json={
            "frame": key_frame,
            "position": [key_pos[0], key_pos[1] - 1.0, key_pos[2] + 0.25],
            "pullEnabled": True,
            "pullRadiusFrames": 6,
            "pullPinnedEnds": True,
        },
    )
    assert patch.status_code == 200
    assert patch.json()["revision"] > fetched["revision"]

    add_key = client.post(
        f"/api/elements/{camera_id}/animation/key",
        json={"frame": 4},
    )
    assert add_key.status_code == 200
    assert any(int(k["frame"]) == 4 for k in add_key.json()["controlKeys"])

    duplicate = client.post(
        f"/api/elements/{camera_id}/animation/key/duplicate",
        json={"sourceFrame": 4, "targetFrame": 6},
    )
    assert duplicate.status_code == 200
    assert any(int(k["frame"]) == 6 for k in duplicate.json()["controlKeys"])

    smooth = client.post(
        f"/api/elements/{camera_id}/animation/smooth",
        json={"passes": 1, "pinnedEnds": True},
    )
    assert smooth.status_code == 200

    del_key = client.delete(
        f"/api/elements/{camera_id}/animation/key",
        params={"frame": 6},
    )
    assert del_key.status_code == 200
    assert all(int(k["frame"]) != 6 for k in del_key.json()["controlKeys"])

    delete = client.delete(f"/api/elements/{camera_id}/animation")
    assert delete.status_code == 200
    assert bool(delete.json()["removed"]) is True

    missing = client.get(f"/api/elements/{camera_id}/animation")
    assert missing.status_code == 404


def test_camera_animation_api_validation() -> None:
    try:
        from fastapi.testclient import TestClient
    except Exception as e:  # pragma: no cover
        _skip(f"TestClient not available ({e!r}); install test extras to run this test")
        return

    camera_id, target_id = _seed_camera_and_target("anim_api_validation")
    client = TestClient(create_app())

    bad_timestamp = client.put(
        f"/api/elements/{camera_id}/animation",
        json={
            "mode": "follow",
            "targetId": target_id,
            "startFrame": 0,
            "endFrame": 9,
            "timestamp": 1.0,
        },
    )
    assert bad_timestamp.status_code == 400

    bad_target = client.put(
        f"/api/elements/{camera_id}/animation",
        json={
            "mode": "follow",
            "targetId": "missing_target",
            "startFrame": 0,
            "endFrame": 9,
        },
    )
    assert bad_target.status_code == 404

    non_camera = client.put(
        f"/api/elements/{target_id}/animation",
        json={
            "mode": "follow",
            "targetId": camera_id,
            "startFrame": 0,
            "endFrame": 9,
        },
    )
    assert non_camera.status_code == 400

    no_frame_delete = client.delete(f"/api/elements/{camera_id}/animation/key")
    assert no_frame_delete.status_code == 400
