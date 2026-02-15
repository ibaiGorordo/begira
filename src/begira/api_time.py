from __future__ import annotations

from typing import Any

import numpy as np


def parse_bool(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        raise ValueError(f"Missing {field}")
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "on"}:
        return True
    if s in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid {field}")


def parse_sample_query(frame: Any, timestamp: Any) -> tuple[int | None, float | None]:
    frame_v: int | None = None
    ts_v: float | None = None

    if frame is not None:
        try:
            frame_v = int(frame)
        except Exception as ex:
            raise ValueError("Invalid frame") from ex

    if timestamp is not None:
        try:
            ts_v = float(timestamp)
        except Exception as ex:
            raise ValueError("Invalid timestamp") from ex
        if not np.isfinite(ts_v):
            raise ValueError("Invalid timestamp")

    if frame_v is not None and ts_v is not None:
        raise ValueError("Provide only one of frame or timestamp")

    return frame_v, ts_v


def parse_sample_body(body: dict[str, Any]) -> tuple[bool, int | None, float | None]:
    raw_static = body.get("static", False)
    if isinstance(raw_static, str):
        static = parse_bool(raw_static, field="static")
    else:
        static = bool(raw_static)

    frame = body.get("frame") if "frame" in body else None
    timestamp = body.get("timestamp") if "timestamp" in body else None

    frame_v, ts_v = parse_sample_query(frame, timestamp)
    if static and (frame_v is not None or ts_v is not None):
        raise ValueError("static=True cannot be combined with frame or timestamp")
    return static, frame_v, ts_v


def parse_sample_query_with_static(frame: Any, timestamp: Any, static: Any) -> tuple[bool, int | None, float | None]:
    frame_v, ts_v = parse_sample_query(frame, timestamp)

    static_v = False
    if static is not None:
        static_v = parse_bool(static, field="static")
    if static_v and (frame_v is not None or ts_v is not None):
        raise ValueError("static=true cannot be combined with frame or timestamp")

    return static_v, frame_v, ts_v
