from __future__ import annotations

import uuid

import numpy as np


def _skip(msg: str) -> None:  # pragma: no cover
    try:
        import pytest  # type: ignore

        pytest.skip(msg)
    except Exception:
        raise RuntimeError(msg)


def test_timeline_api_queries_and_birth() -> None:
    from begira.runtime.app import create_app
    from begira.core.registry import REGISTRY

    try:
        from fastapi.testclient import TestClient
    except Exception as e:  # pragma: no cover
        _skip(f'TestClient not available ({e!r}); install test extras to run this test')
        return

    eid = f'timeline_api_{uuid.uuid4().hex}'
    REGISTRY.upsert_pointcloud(
        name='pc_api',
        positions=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        colors=np.asarray([[255, 0, 0]], dtype=np.uint8),
        element_id=eid,
        frame=10,
    )

    client = TestClient(create_app())

    bad = client.get('/api/elements', params={'frame': 1, 'timestamp': 1.0})
    assert bad.status_code == 400

    at_5 = client.get('/api/elements', params={'frame': 5})
    assert at_5.status_code == 200
    assert not any(e['id'] == eid for e in at_5.json())

    born_meta = client.get(f'/api/elements/{eid}/meta', params={'frame': 10})
    assert born_meta.status_code == 200
    assert born_meta.json()['id'] == eid

    not_born_meta = client.get(f'/api/elements/{eid}/meta', params={'frame': 5})
    assert not_born_meta.status_code == 404

    not_born_payload = client.get(f'/api/elements/{eid}/payloads/points', params={'frame': 5})
    assert not_born_payload.status_code == 404

    timeline = client.get('/api/timeline')
    assert timeline.status_code == 200
    info = timeline.json()
    assert info['defaultAxis'] == 'frame'
    assert isinstance(info['axes'], list)
    assert isinstance(info['latest'], dict)


def test_api_exposes_state_and_data_revisions() -> None:
    from begira.runtime.app import create_app
    from begira.core.registry import REGISTRY

    try:
        from fastapi.testclient import TestClient
    except Exception as e:  # pragma: no cover
        _skip(f'TestClient not available ({e!r}); install test extras to run this test')
        return

    eid = f'timeline_api_rev_{uuid.uuid4().hex}'
    REGISTRY.upsert_pointcloud(
        name='pc_rev',
        positions=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        colors=np.asarray([[255, 0, 0]], dtype=np.uint8),
        element_id=eid,
        static=True,
    )
    REGISTRY.update_element_meta(eid, position=(1.0, 0.0, 0.0), frame=0)

    client = TestClient(create_app())

    elements = client.get('/api/elements').json()
    hit = next(e for e in elements if e['id'] == eid)
    assert 'stateRevision' in hit
    assert 'dataRevision' in hit

    meta = client.get(f'/api/elements/{eid}/meta').json()
    assert 'stateRevision' in meta
    assert 'dataRevision' in meta
