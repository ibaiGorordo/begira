from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PointCloudData:
    """In-memory point cloud data (no Begira registry id attached)."""

    positions: np.ndarray  # float32 (N, 3)
    colors: np.ndarray | None  # uint8 (N, 3) or None


def load_ply_pointcloud(path: str | Path) -> PointCloudData:
    """Load a point cloud from a PLY file.

    Supported:
    - ASCII PLY
    - Binary little-endian PLY

    The loader expects a `vertex` element with at least `x`, `y`, `z` properties.
    If color properties exist (`red/green/blue` or `r/g/b`), they will be returned
    as uint8 (N, 3).
    """

    # Import lazily so begira can stay lightweight if users never load PLY.
    try:
        from plyfile import PlyData  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "PLY loading requires the optional dependency 'plyfile'. "
            "Install it with: pip install 'begira[ply]' (or just pip install plyfile)."
        ) from e

    p = Path(path)
    ply = PlyData.read(str(p))

    if "vertex" not in ply:
        raise ValueError("PLY file has no 'vertex' element")

    v = ply["vertex"].data

    for k in ("x", "y", "z"):
        if k not in v.dtype.names:
            raise ValueError(f"PLY vertex element missing required property '{k}'")

    positions = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32, copy=False)
    positions = np.ascontiguousarray(positions, dtype=np.float32)

    colors: np.ndarray | None = None
    color_keys: tuple[str, str, str] | None = None
    if {"red", "green", "blue"}.issubset(v.dtype.names):
        color_keys = ("red", "green", "blue")
    elif {"r", "g", "b"}.issubset(v.dtype.names):
        color_keys = ("r", "g", "b")

    if color_keys is not None:
        c = np.stack([v[color_keys[0]], v[color_keys[1]], v[color_keys[2]]], axis=1)

        # Normalize a few common encodings.
        if np.issubdtype(c.dtype, np.floating):
            c = np.clip(c, 0.0, 1.0) * 255.0
        else:
            # If integer colors appear to be 0..1, scale up.
            cmax = float(np.max(c)) if c.size else 0.0
            if 0.0 < cmax <= 1.0:
                c = c.astype(np.float32) * 255.0

        colors = np.ascontiguousarray(c, dtype=np.uint8)

    return PointCloudData(positions=positions, colors=colors)


def load_ply(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Backward-compatible wrapper that returns (positions, colors)."""

    pc = load_ply_pointcloud(path)
    return pc.positions, pc.colors


@dataclass(frozen=True)
class GaussianSplatData:
    """In-memory 3D Gaussian Splat data loaded from a 3DGS-style PLY.

    This matches the common output of 3D Gaussian Splatting pipelines:
    - positions: x/y/z
    - f_dc_0/1/2: DC (0th-order SH) color coefficients
    - opacity: usually stored in logit space
    - scale_0/1/2: usually stored in log space
    - rot_0/1/2/3: quaternion (often normalized)

    Notes:
    - Begira currently visualizes these as a point cloud (one point per Gaussian).
      The returned `colors_rgb8` is a best-effort conversion of f_dc_* to displayable RGB.
    """

    positions: np.ndarray  # float32 (N, 3)
    colors_rgb8: np.ndarray | None  # uint8 (N, 3) - derived from f_dc_* if present

    f_dc: np.ndarray | None  # float32 (N, 3)
    opacity: np.ndarray | None  # float32 (N,)
    scales: np.ndarray | None  # float32 (N, 3)
    rotations: np.ndarray | None  # float32 (N, 4)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def load_ply_gaussians(path: str | Path) -> GaussianSplatData:
    """Load a 3D Gaussian Splat PLY.

    This parses the common 3DGS PLY schema (binary_little_endian) that contains
    f_dc_0/1/2, opacity, scale_0/1/2, rot_0/1/2/3.

    Returns a `GaussianSplatData`.

    Visualization tip:
    - You can pass `data.positions` + `data.colors_rgb8` to `BegiraServer.log_points()`
      to view the splats as a colored point cloud.
    """

    try:
        from plyfile import PlyData  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "PLY loading requires the optional dependency 'plyfile'. "
            "Install it with: pip install 'begira[ply]' (or just pip install plyfile)."
        ) from e

    p = Path(path)
    ply = PlyData.read(str(p))

    if "vertex" not in ply:
        raise ValueError("PLY file has no 'vertex' element")

    v = ply["vertex"].data
    names = set(v.dtype.names or ())

    for k in ("x", "y", "z"):
        if k not in names:
            raise ValueError(f"PLY vertex element missing required property '{k}'")

    positions = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32, copy=False)
    positions = np.ascontiguousarray(positions, dtype=np.float32)

    f_dc: np.ndarray | None = None
    if {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(names):
        f_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(np.float32, copy=False)
        f_dc = np.ascontiguousarray(f_dc, dtype=np.float32)

    opacity: np.ndarray | None = None
    if "opacity" in names:
        opacity = np.asarray(v["opacity"], dtype=np.float32)
        opacity = np.ascontiguousarray(opacity, dtype=np.float32)

    scales: np.ndarray | None = None
    if {"scale_0", "scale_1", "scale_2"}.issubset(names):
        scales = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1).astype(np.float32, copy=False)
        scales = np.ascontiguousarray(scales, dtype=np.float32)

    rotations: np.ndarray | None = None
    if {"rot_0", "rot_1", "rot_2", "rot_3"}.issubset(names):
        rotations = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1).astype(
            np.float32, copy=False
        )
        rotations = np.ascontiguousarray(rotations, dtype=np.float32)

    # Derive a display color. 3DGS stores SH DC coefficients; a common convention is:
    # rgb = sigmoid(f_dc) or rgb = clamp(0.5 + f_dc, 0..1). We try sigmoid first.
    colors_rgb8: np.ndarray | None = None
    if f_dc is not None:
        rgb01 = _sigmoid(f_dc)
        rgb01 = np.clip(rgb01, 0.0, 1.0)
        colors_rgb8 = np.ascontiguousarray(rgb01 * 255.0, dtype=np.uint8)

    return GaussianSplatData(
        positions=positions,
        colors_rgb8=colors_rgb8,
        f_dc=f_dc,
        opacity=opacity,
        scales=scales,
        rotations=rotations,
    )
