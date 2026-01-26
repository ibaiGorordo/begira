from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


ElementType = Literal[
    "pointcloud",
    "gaussians",
    "camera",
    # Future:
    # "lines3d",
    # "boxes3d",
]


@dataclass(frozen=True, kw_only=True)
class ElementBase:
    """Common metadata for anything renderable in the viewer.

    Notes:
    - This is intentionally small. Type-specific payloads live in dedicated dataclasses.
    - `revision` is bumped on any change to metadata or payloads.
    """

    id: str
    type: ElementType
    name: str
    revision: int
    created_at: float
    updated_at: float
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    visible: bool = True
    deleted: bool = False


@dataclass(frozen=True)
class PointCloudElement(ElementBase):
    type: Literal["pointcloud"]
    positions: np.ndarray  # float32 (n,3)
    colors: np.ndarray | None  # uint8 (n,3)
    point_size: float

    @property
    def point_count(self) -> int:
        return int(self.positions.shape[0])


@dataclass(frozen=True)
class GaussianSplatElement(ElementBase):
    type: Literal["gaussians"]
    positions: np.ndarray  # float32 (n,3)
    # SH-0 coefficients (f_dc_0, f_dc_1, f_dc_2)
    sh0: np.ndarray  # float32 (n,3)
    opacity: np.ndarray  # float32 (n,1)
    scales: np.ndarray  # float32 (n,3)
    rotations: np.ndarray  # float32 (n,4)
    point_size: float  # Multiplier for 3D scales

    @property
    def count(self) -> int:
        return int(self.positions.shape[0])


@dataclass(frozen=True)
class CameraElement(ElementBase):
    type: Literal["camera"]
    fov: float
    near: float
    far: float
