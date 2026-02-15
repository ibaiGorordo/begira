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
        self._open_camera_view_seq = 0
        self._last_open_camera_view: tuple[int, str] | None = None

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

    def request_open_camera_view(self, camera_id: str) -> dict[str, object]:
        cid = str(camera_id).strip()
        if not cid:
            raise ValueError("camera_id cannot be empty")
        with self._lock:
            self._open_camera_view_seq += 1
            self._last_open_camera_view = (self._open_camera_view_seq, cid)
            return {"seq": int(self._open_camera_view_seq), "cameraId": cid}

    def get_open_camera_view_request(self) -> dict[str, object] | None:
        with self._lock:
            if self._last_open_camera_view is None:
                return None
            seq, cid = self._last_open_camera_view
            return {"seq": int(seq), "cameraId": str(cid)}


VIEWER_SETTINGS = ViewerSettingsStore()
