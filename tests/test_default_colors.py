import numpy as np

from begira.registry import InMemoryRegistry


def test_default_colors_are_deterministic_by_creation_order() -> None:
    reg = InMemoryRegistry()

    pos = np.zeros((5, 3), dtype=np.float32)

    a = reg.upsert_pointcloud(name="a", positions=pos, colors=None)
    b = reg.upsert_pointcloud(name="b", positions=pos, colors=None)

    assert a.colors is not None
    assert b.colors is not None

    # All points in a cloud get the same default color.
    assert (a.colors == a.colors[0]).all()
    assert (b.colors == b.colors[0]).all()

    # First and second clouds get different palette colors.
    assert not np.array_equal(a.colors[0], b.colors[0])


def test_explicit_colors_override_defaults() -> None:
    reg = InMemoryRegistry()

    pos = np.zeros((3, 3), dtype=np.float32)
    col = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)

    pc = reg.upsert_pointcloud(name="c", positions=pos, colors=col)
    assert pc.colors is not None
    assert np.array_equal(pc.colors, col)
