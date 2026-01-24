from __future__ import annotations

from pathlib import Path

import numpy as np

import begira
from begira.ply import load_ply_pointcloud
from begira.registry import REGISTRY


def test_log_points_accepts_pointcloud_object_and_uses_colors() -> None:
    server = begira.run(open_browser=False)

    pc = load_ply_pointcloud(Path(__file__).parent / "assets" / "triangle_ascii_color.ply")
    cloud_id = server.log_points("tri", pc)

    stored = REGISTRY.get(cloud_id)
    assert stored is not None

    assert stored.positions.shape == (3, 3)
    assert stored.positions.dtype == np.float32

    assert stored.colors is not None
    assert stored.colors.shape == (3, 3)
    assert stored.colors.dtype == np.uint8

    # Ensure colors came from the PLY (not a default palette fill).
    assert stored.colors[0].tolist() == [255, 0, 0]
    assert stored.colors[1].tolist() == [0, 255, 0]
    assert stored.colors[2].tolist() == [0, 0, 255]
