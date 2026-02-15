from __future__ import annotations

import numpy as np

from begira.runner import BegiraServer


def test_static_payload_temporal_transform_via_handle() -> None:
    server = BegiraServer(host='127.0.0.1', port=0, url='http://127.0.0.1:0/')

    pc = server.log_points(
        'map',
        np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        np.asarray([[255, 0, 0], [0, 255, 0]], dtype=np.uint8),
        static=True,
    )

    t0 = np.eye(4, dtype=np.float64)
    t1 = np.eye(4, dtype=np.float64)
    t2 = np.eye(4, dtype=np.float64)
    t1[:3, 3] = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    t2[:3, 3] = np.asarray([2.0, 0.0, 0.0], dtype=np.float64)

    pc.set_transform(t0, frame=0)
    pc.set_transform(t1, frame=1)
    pc.set_transform(t2, frame=2)

    meta0 = server.get_element_meta(pc.id, frame=0)
    meta1 = server.get_element_meta(pc.id, frame=1)
    meta2 = server.get_element_meta(pc.id, frame=2)

    assert meta0['position'] == [0.0, 0.0, 0.0]
    assert meta1['position'] == [1.0, 0.0, 0.0]
    assert meta2['position'] == [2.0, 0.0, 0.0]

    # Payload is static; only state revision should change across transform-only frames.
    assert meta0['dataRevision'] == meta1['dataRevision'] == meta2['dataRevision']
    assert meta2['stateRevision'] >= meta1['stateRevision'] >= meta0['stateRevision']


def test_camera_look_at_supports_time_args() -> None:
    server = BegiraServer(host='127.0.0.1', port=0, url='http://127.0.0.1:0/')

    pts = server.log_points(
        'pts',
        np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        np.asarray([[255, 0, 0]], dtype=np.uint8),
        static=True,
    )

    cam = server.log_camera('cam', position=(0.0, -2.0, 0.0), static=True)
    cam.look_at(pts, distance=1.0, frame=5)

    cm = server.get_element_meta(cam.id, frame=5)
    cp = np.asarray(cm['position'], dtype=np.float64)
    tp = np.asarray(server.get_element_meta(pts.id, frame=5)['position'], dtype=np.float64)
    assert np.isclose(np.linalg.norm(tp - cp), 1.0, atol=1e-6)
