from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from begira.ply import load_ply, load_ply_pointcloud


@pytest.mark.parametrize(
    "fname,expect_colors",
    [
        ("triangle_ascii_color.ply", True),
        ("triangle_ascii_nocolor.ply", False),
    ],
)
def test_load_ply_shapes_and_dtypes(fname: str, expect_colors: bool) -> None:
    path = Path(__file__).parent / "assets" / fname
    positions, colors = load_ply(path)

    assert positions.shape == (3, 3)
    assert positions.dtype == np.float32

    if expect_colors:
        assert colors is not None
        assert colors.shape == (3, 3)
        assert colors.dtype == np.uint8
    else:
        assert colors is None


@pytest.mark.parametrize(
    "fname,expect_colors",
    [
        ("triangle_ascii_color.ply", True),
        ("triangle_ascii_nocolor.ply", False),
    ],
)
def test_load_ply_pointcloud_dataclass(fname: str, expect_colors: bool) -> None:
    path = Path(__file__).parent / "assets" / fname
    pc = load_ply_pointcloud(path)

    assert isinstance(pc.positions, np.ndarray)
    assert pc.positions.shape == (3, 3)
    assert pc.positions.dtype == np.float32

    if expect_colors:
        assert pc.colors is not None
        assert pc.colors.shape == (3, 3)
        assert pc.colors.dtype == np.uint8
    else:
        assert pc.colors is None
