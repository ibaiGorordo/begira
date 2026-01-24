from __future__ import annotations

import threading
from dataclasses import dataclass

from .conventions import CoordinateConvention


@dataclass
class ViewerSettings:
    """Server-side viewer preferences.

    This is intentionally small and safe to mutate at runtime.

    Notes:
    - The point data is always stored in whatever coordinates the user logs.
    - The viewer convention controls how the frontend interprets and displays that data.
    """

    coordinate_convention: str = CoordinateConvention.Z_UP.value


class ViewerSettingsStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._settings = ViewerSettings()

    def get(self) -> ViewerSettings:
        with self._lock:
            return ViewerSettings(
                coordinate_convention=self._settings.coordinate_convention,
            )

    def set_coordinate_convention(self, convention: str | CoordinateConvention) -> ViewerSettings:
        conv = CoordinateConvention.from_any(convention)

        with self._lock:
            self._settings.coordinate_convention = conv.value
            return self.get()


VIEWER_SETTINGS = ViewerSettingsStore()
