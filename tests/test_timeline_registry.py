from __future__ import annotations

import numpy as np

from begira.core.registry import InMemoryRegistry


def _tiny_positions(offset: float = 0.0) -> np.ndarray:
    return np.asarray([[offset, 0.0, 0.0], [offset + 1.0, 0.0, 0.0]], dtype=np.float32)


def test_default_timeless_and_static_sampling() -> None:
    reg = InMemoryRegistry()

    p0 = reg.upsert_pointcloud(name='pc_auto', positions=_tiny_positions(0.0), colors=None)
    p1 = reg.upsert_pointcloud(name='pc_auto', positions=_tiny_positions(2.0), colors=None, element_id=p0.id)

    assert p1.id == p0.id
    assert reg.timeline_info()['latest']['frame'] is None

    at_frame0 = reg.get_element(p0.id, frame=0)
    at_frame1 = reg.get_element(p0.id, frame=1)
    assert at_frame0 is not None
    assert at_frame1 is not None
    assert isinstance(at_frame0.positions, np.ndarray)  # type: ignore[attr-defined]
    assert isinstance(at_frame1.positions, np.ndarray)  # type: ignore[attr-defined]
    assert float(at_frame0.positions[0, 0]) == 2.0  # type: ignore[attr-defined]
    assert float(at_frame1.positions[0, 0]) == 2.0  # type: ignore[attr-defined]

    static_pc = reg.upsert_pointcloud(name='pc_static', positions=_tiny_positions(10.0), colors=None, static=True)
    assert reg.get_element(static_pc.id, frame=123) is not None
    assert reg.get_element(static_pc.id, timestamp=123.0) is not None


def test_hold_last_sampling_and_duplicate_key_overwrite() -> None:
    reg = InMemoryRegistry()
    pc = reg.upsert_pointcloud(name='pc', positions=_tiny_positions(), colors=None, static=True)

    reg.update_element_meta(pc.id, position=(1.0, 0.0, 0.0), frame=0)
    reg.update_element_meta(pc.id, position=(5.0, 0.0, 0.0), frame=2)

    at_1 = reg.get_element(pc.id, frame=1)
    assert at_1 is not None
    assert at_1.position == (1.0, 0.0, 0.0)

    reg.update_element_meta(pc.id, position=(8.0, 0.0, 0.0), timestamp=1.0)
    reg.update_element_meta(pc.id, position=(9.0, 0.0, 0.0), timestamp=3.0)

    at_t2 = reg.get_element(pc.id, timestamp=2.0)
    assert at_t2 is not None
    assert at_t2.position == (8.0, 0.0, 0.0)

    reg.update_element_meta(pc.id, position=(11.0, 0.0, 0.0), frame=5)
    reg.update_element_meta(pc.id, position=(12.0, 0.0, 0.0), frame=5)
    at_5 = reg.get_element(pc.id, frame=5)
    assert at_5 is not None
    assert at_5.position == (12.0, 0.0, 0.0)


def test_state_and_data_revisions_split() -> None:
    reg = InMemoryRegistry()
    pc = reg.upsert_pointcloud(name='pc', positions=_tiny_positions(0.0), colors=None, static=True)

    assert pc.state_revision == 1
    assert pc.data_revision == 1

    pose = reg.update_element_meta(pc.id, position=(1.0, 2.0, 3.0), frame=0)
    assert pose.state_revision == 2
    assert pose.data_revision == 1

    data = reg.upsert_pointcloud(name='pc', positions=_tiny_positions(5.0), colors=None, element_id=pc.id, frame=1)
    assert data.state_revision == pose.state_revision
    assert data.data_revision == pose.data_revision + 1
