from __future__ import annotations

from pathlib import Path

import numpy as np

import begira
from begira.io.ply import load_ply_pointcloud
from begira.core.registry import REGISTRY


def test_log_points_accepts_pointcloud_object_and_uses_colors() -> None:
    server = begira.run(open_browser=False)

    pc = load_ply_pointcloud(Path(__file__).parent / "assets" / "triangle_ascii_color.ply")
    element_id = server.log_points("tri", pc)

    stored = REGISTRY.get_element(element_id)
    assert stored is not None

    # The test fixture is a pointcloud element; assert its buffers.
    assert stored.type == "pointcloud"

    positions = stored.positions  # type: ignore[attr-defined]
    colors = stored.colors  # type: ignore[attr-defined]

    assert positions.shape == (3, 3)
    assert positions.dtype == np.float32

    assert colors is not None
    assert colors.shape == (3, 3)
    assert colors.dtype == np.uint8

    # Ensure colors came from the PLY (not a default palette fill).
    assert colors[0].tolist() == [255, 0, 0]
    assert colors[1].tolist() == [0, 255, 0]
    assert colors[2].tolist() == [0, 0, 255]
