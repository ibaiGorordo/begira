from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


ElementType = Literal[
    "pointcloud",
    # Future:
    # "gaussians",
    # "lines3d",
    # "boxes3d",
]


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class PointCloudElement(ElementBase):
    type: Literal["pointcloud"]
    positions: np.ndarray  # float32 (n,3)
    colors: np.ndarray | None  # uint8 (n,3)
    point_size: float

    @property
    def point_count(self) -> int:
        return int(self.positions.shape[0])
