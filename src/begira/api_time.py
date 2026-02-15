from __future__ import annotations

from typing import Any
from typing import Literal

import numpy as np

TimelineKind = Literal["sequence", "timestamp"]


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


def parse_sample_query(
    frame: Any,
    timestamp: Any,
    timeline: Any = None,
    time_value: Any = None,
) -> tuple[int | None, float | None, str | None, float | None]:
    frame_v: int | None = None
    ts_v: float | None = None
    timeline_v: str | None = None
    time_v: float | None = None

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

    if timeline is not None:
        timeline_v = str(timeline).strip()
        if not timeline_v:
            raise ValueError("Invalid timeline")
    if time_value is not None:
        try:
            time_v = float(time_value)
        except Exception as ex:
            raise ValueError("Invalid time") from ex
        if not np.isfinite(time_v):
            raise ValueError("Invalid time")

    if timeline_v is not None or time_v is not None:
        if frame_v is not None or ts_v is not None:
            raise ValueError("Provide either frame/timestamp or timeline/time, not both")
        if timeline_v is None or time_v is None:
            raise ValueError("timeline and time must be provided together")

    return frame_v, ts_v, timeline_v, time_v


def parse_sample_body(
    body: dict[str, Any],
) -> tuple[bool, int | None, float | None, str | None, TimelineKind | None, float | None]:
    raw_static = body.get("static", False)
    if isinstance(raw_static, str):
        static = parse_bool(raw_static, field="static")
    else:
        static = bool(raw_static)

    frame = body.get("frame") if "frame" in body else None
    timestamp = body.get("timestamp") if "timestamp" in body else None
    timeline = body.get("timeline") if "timeline" in body else None
    timeline_kind_raw = body.get("timelineKind") if "timelineKind" in body else None
    sequence = body.get("sequence") if "sequence" in body else None
    timeline_timestamp = (
        body.get("timelineTimestamp")
        if "timelineTimestamp" in body
        else (body.get("time") if "time" in body else None)
    )

    frame_v, ts_v, timeline_v, _ = parse_sample_query(frame, timestamp, None, None)

    timeline_kind: TimelineKind | None = None
    timeline_value: float | None = None
    if timeline_v is None and timeline is not None:
        timeline_v = str(timeline).strip()
        if not timeline_v:
            raise ValueError("Invalid timeline")

    if timeline_v is not None:
        if frame_v is not None or ts_v is not None:
            raise ValueError("timeline writes cannot be combined with frame/timestamp")

        if timeline_kind_raw is not None:
            timeline_kind_txt = str(timeline_kind_raw).strip().lower()
            if timeline_kind_txt not in {"sequence", "timestamp"}:
                raise ValueError("timelineKind must be 'sequence' or 'timestamp'")
            timeline_kind = timeline_kind_txt  # type: ignore[assignment]

        has_sequence = sequence is not None
        has_timeline_ts = timeline_timestamp is not None
        if has_sequence and has_timeline_ts:
            raise ValueError("Provide only one of sequence or timelineTimestamp for timeline writes")
        if not has_sequence and not has_timeline_ts:
            raise ValueError("Timeline writes require sequence or timelineTimestamp")

        if has_sequence:
            try:
                seq_v = int(sequence)
            except Exception as ex:
                raise ValueError("Invalid sequence") from ex
            frame_v = seq_v
            timeline_kind = timeline_kind or "sequence"
            timeline_value = float(seq_v)
        else:
            try:
                ts2_v = float(timeline_timestamp)
            except Exception as ex:
                raise ValueError("Invalid timelineTimestamp") from ex
            if not np.isfinite(ts2_v):
                raise ValueError("Invalid timelineTimestamp")
            ts_v = ts2_v
            timeline_kind = timeline_kind or "timestamp"
            timeline_value = ts2_v

    if static and (frame_v is not None or ts_v is not None or timeline_v is not None):
        raise ValueError("static=True cannot be combined with temporal sample fields")
    return static, frame_v, ts_v, timeline_v, timeline_kind, timeline_value


def parse_sample_query_with_static(
    frame: Any,
    timestamp: Any,
    static: Any,
    timeline: Any = None,
    time_value: Any = None,
) -> tuple[bool, int | None, float | None, str | None, float | None]:
    frame_v, ts_v, timeline_v, time_v = parse_sample_query(frame, timestamp, timeline, time_value)

    static_v = False
    if static is not None:
        static_v = parse_bool(static, field="static")
    if static_v and (frame_v is not None or ts_v is not None or timeline_v is not None):
        raise ValueError("static=true cannot be combined with temporal sample fields")

    return static_v, frame_v, ts_v, timeline_v, time_v
