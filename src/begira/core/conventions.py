from __future__ import annotations

from enum import Enum
from typing import Any


class CoordinateConvention(str, Enum):
    """Viewer coordinate convention.

    These values describe *how the viewer interprets axes*.

    Notes:
    - This does not modify your point data.
    - It only affects camera up direction + how the frontend displays the scene.

    Currently only right-handed conventions are supported, so the enum uses short names.
    """

    Y_UP = "rh-y-up"
    Z_UP = "rh-z-up"

    @classmethod
    def from_any(cls, value: Any) -> "CoordinateConvention":
        if isinstance(value, cls):
            return value

        v = str(value).strip().lower().replace("_", "-")
        aliases: dict[str, CoordinateConvention] = {
            # canonical
            "rh-y-up": cls.Y_UP,
            "rh-z-up": cls.Z_UP,
            # short aliases
            "y-up": cls.Y_UP,
            "z-up": cls.Z_UP,
            "y": cls.Y_UP,
            "z": cls.Z_UP,
            # accepted enum-like strings
            "yup": cls.Y_UP,
            "zup": cls.Z_UP,
            # historical variants
            "right-hand-y-up": cls.Y_UP,
            "right-hand-z-up": cls.Z_UP,
            "right-handed-y-up": cls.Y_UP,
            "right-handed-z-up": cls.Z_UP,
        }
        if v in aliases:
            return aliases[v]

        raise ValueError(
            "Unsupported coordinate convention. Use CoordinateConvention.Z_UP / Y_UP (or 'z-up' / 'y-up')."
        )
