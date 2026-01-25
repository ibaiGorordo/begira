from __future__ import annotations

from pathlib import Path

import numpy as np

from begira.ply import load_ply_gaussians


def test_load_ply_gaussians_smoke() -> None:
    # Uses the small-ish example asset shipped with the repo.
    path = Path(__file__).parent.parent / "examples" / "assets" / "gaussians.ply"
    gs = load_ply_gaussians(path)

    assert gs.positions.ndim == 2
    assert gs.positions.shape[1] == 3
    assert gs.positions.dtype == np.float32

    # These are expected for 3DGS exports.
    assert gs.f_dc is not None
    assert gs.f_dc.shape == gs.positions.shape
    assert gs.f_dc.dtype == np.float32

    assert gs.colors_rgb8 is not None
    assert gs.colors_rgb8.shape == gs.positions.shape
    assert gs.colors_rgb8.dtype == np.uint8

    # If present, check shapes.
    if gs.opacity is not None:
        assert gs.opacity.shape == (gs.positions.shape[0],)
    if gs.scales is not None:
        assert gs.scales.shape == gs.positions.shape
    if gs.rotations is not None:
        assert gs.rotations.shape == (gs.positions.shape[0], 4)
