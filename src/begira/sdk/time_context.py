from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

ActiveTimeline = tuple[str, str, float] | None


def to_unix_seconds(timestamp: float | datetime | None) -> float | None:
    if timestamp is None:
        return None
    if isinstance(timestamp, datetime):
        return float(timestamp.timestamp())
    value = float(timestamp)
    if not np.isfinite(value):
        raise ValueError("timestamp must be finite")
    return value


def set_active_timeline(
    timeline: str,
    *,
    sequence: int | None = None,
    timestamp: float | datetime | None = None,
) -> tuple[str, str, float]:
    timeline_name = str(timeline).strip()
    if not timeline_name:
        raise ValueError("timeline cannot be empty")
    has_sequence = sequence is not None
    has_timestamp = timestamp is not None
    if has_sequence == has_timestamp:
        raise ValueError("Provide exactly one of sequence or timestamp")
    if has_sequence:
        return (timeline_name, "sequence", float(int(sequence)))
    ts_v = to_unix_seconds(timestamp)
    if ts_v is None:
        raise ValueError("timestamp must not be None")
    return (timeline_name, "timestamp", float(ts_v))


def normalize_time_args(
    *,
    frame: int | None = None,
    timestamp: float | datetime | None = None,
    static: bool = False,
) -> tuple[bool, int | None, float | None]:
    frame_v = int(frame) if frame is not None else None
    ts_v = to_unix_seconds(timestamp)
    if frame_v is not None and ts_v is not None:
        raise ValueError("Only one of frame or timestamp can be provided")
    if static and (frame_v is not None or ts_v is not None):
        raise ValueError("static=True cannot be combined with frame or timestamp")
    return bool(static), frame_v, ts_v


def time_body(
    *,
    frame: int | None = None,
    timestamp: float | datetime | None = None,
    static: bool = False,
    timeline: str | None = None,
    timeline_kind: str | None = None,
    sequence: int | None = None,
    timeline_timestamp: float | datetime | None = None,
    active_timeline: ActiveTimeline = None,
) -> dict[str, Any]:
    static_v, frame_v, ts_v = normalize_time_args(frame=frame, timestamp=timestamp, static=static)

    timeline_name = str(timeline).strip() if timeline is not None else None
    if timeline_name == "":
        raise ValueError("timeline cannot be empty")

    has_explicit_timeline = timeline_name is not None or sequence is not None or timeline_timestamp is not None
    if frame_v is not None or ts_v is not None:
        if has_explicit_timeline:
            raise ValueError("frame/timestamp cannot be combined with timeline fields")
    if static_v and has_explicit_timeline:
        raise ValueError("static=True cannot be combined with timeline fields")

    def _emit_timeline(name: str, kind: str, value: float) -> dict[str, Any]:
        out_t: dict[str, Any] = {
            "timeline": name,
            "timelineKind": kind,
        }
        if kind == "sequence":
            out_t["sequence"] = int(value)
        elif kind == "timestamp":
            out_t["timelineTimestamp"] = float(value)
        else:
            raise ValueError("timeline kind must be 'sequence' or 'timestamp'")
        return out_t

    out: dict[str, Any] = {}
    if static_v:
        out["static"] = True
        return out

    if frame_v is not None:
        out["frame"] = frame_v
        return out
    if ts_v is not None:
        out["timestamp"] = ts_v
        return out

    if has_explicit_timeline:
        if timeline_name is None:
            raise ValueError("timeline is required when providing sequence/timeline_timestamp")
        has_sequence = sequence is not None
        has_timeline_ts = timeline_timestamp is not None
        if has_sequence == has_timeline_ts:
            raise ValueError("Provide exactly one of sequence or timeline_timestamp")

        if has_sequence:
            kind = timeline_kind or "sequence"
            out.update(_emit_timeline(timeline_name, str(kind), float(int(sequence))))
            return out

        ts2_v = to_unix_seconds(timeline_timestamp)
        if ts2_v is None:
            raise ValueError("timeline_timestamp must not be None")
        kind = timeline_kind or "timestamp"
        out.update(_emit_timeline(timeline_name, str(kind), float(ts2_v)))
        return out

    if active_timeline is not None:
        active_name, active_kind, active_value = active_timeline
        out.update(_emit_timeline(str(active_name), str(active_kind), float(active_value)))

    return out


def sample_query_params(
    *,
    frame: int | None = None,
    timestamp: float | datetime | None = None,
    timeline: str | None = None,
    time_value: float | None = None,
    active_timeline: ActiveTimeline = None,
) -> dict[str, str]:
    params: dict[str, str] = {}
    if frame is not None and timestamp is not None:
        raise ValueError("Only one of frame or timestamp can be provided")
    if (timeline is None) != (time_value is None):
        raise ValueError("timeline and time_value must be provided together")
    if frame is not None:
        params["frame"] = str(int(frame))
        return params
    if timestamp is not None:
        ts_v = to_unix_seconds(timestamp)
        if ts_v is None:
            raise ValueError("timestamp must not be None")
        params["timestamp"] = str(ts_v)
        return params
    if timeline is not None and time_value is not None:
        name = str(timeline).strip()
        if not name:
            raise ValueError("timeline cannot be empty")
        params["timeline"] = name
        params["time"] = str(float(time_value))
        return params
    if active_timeline is not None:
        params["timeline"] = str(active_timeline[0])
        params["time"] = str(float(active_timeline[2]))
    return params


def effective_read_sample(
    *,
    frame: int | None = None,
    timestamp: float | datetime | None = None,
    active_timeline: ActiveTimeline = None,
) -> tuple[int | None, float | None, str | None, float | None]:
    frame_v = int(frame) if frame is not None else None
    ts_v = to_unix_seconds(timestamp)
    if frame_v is not None and ts_v is not None:
        raise ValueError("Only one of frame or timestamp can be provided")
    if frame_v is None and ts_v is None and active_timeline is not None:
        return None, None, str(active_timeline[0]), float(active_timeline[2])
    return frame_v, ts_v, None, None


def effective_write_sample(
    *,
    frame: int | None = None,
    timestamp: float | datetime | None = None,
    static: bool = False,
    active_timeline: ActiveTimeline = None,
) -> tuple[bool, int | None, float | None, str | None, str | None, float | None]:
    frame_v = int(frame) if frame is not None else None
    ts_v = to_unix_seconds(timestamp)
    if frame_v is not None and ts_v is not None:
        raise ValueError("Only one of frame or timestamp can be provided")
    if static and (frame_v is not None or ts_v is not None):
        raise ValueError("static=True cannot be combined with frame or timestamp")
    if not static and frame_v is None and ts_v is None and active_timeline is not None:
        return (
            bool(static),
            None,
            None,
            str(active_timeline[0]),
            str(active_timeline[1]),
            float(active_timeline[2]),
        )
    return bool(static), frame_v, ts_v, None, None, None


def time_fields_to_query_params(time_fields: dict[str, Any]) -> dict[str, str]:
    params: dict[str, str] = {}
    if "frame" in time_fields:
        params["frame"] = str(int(time_fields["frame"]))
    if "timestamp" in time_fields:
        params["timestamp"] = str(float(time_fields["timestamp"]))
    if "timeline" in time_fields:
        params["timeline"] = str(time_fields["timeline"])
    if "timelineKind" in time_fields:
        params["timelineKind"] = str(time_fields["timelineKind"])
    if "sequence" in time_fields:
        params["sequence"] = str(int(time_fields["sequence"]))
    if "timelineTimestamp" in time_fields:
        params["timelineTimestamp"] = str(float(time_fields["timelineTimestamp"]))
    if "static" in time_fields:
        params["static"] = "true"
    return params

