from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from typing import Generic, Literal, TypeVar


T = TypeVar("T")
TimeAxis = Literal["frame", "timestamp"]


@dataclass(frozen=True)
class WriteTarget:
    axis: TimeAxis | None
    key: int | float | None
    auto: bool = False


@dataclass
class TemporalChannel(Generic[T]):
    timeless: T | None = None
    frame_samples: dict[int, T] = field(default_factory=dict)
    timestamp_samples: list[tuple[float, T]] = field(default_factory=list)

    def set_sample(self, target: WriteTarget, value: T) -> None:
        if target.axis is None:
            self.timeless = value
            return
        if target.axis == "frame":
            self.frame_samples[int(target.key)] = value
            return

        ts = float(target.key)
        keys = [k for k, _ in self.timestamp_samples]
        idx = bisect_left(keys, ts)
        if idx < len(self.timestamp_samples) and self.timestamp_samples[idx][0] == ts:
            self.timestamp_samples[idx] = (ts, value)
        else:
            self.timestamp_samples.insert(idx, (ts, value))

    def sample(self, *, frame: int | None = None, timestamp: float | None = None) -> T | None:
        if frame is not None and timestamp is not None:
            raise ValueError("Cannot sample with both frame and timestamp")

        if frame is not None:
            best_key: int | None = None
            for k in self.frame_samples:
                if k <= frame and (best_key is None or k > best_key):
                    best_key = k
            if best_key is not None:
                return self.frame_samples[best_key]
            return self.timeless

        if timestamp is not None:
            keys = [k for k, _ in self.timestamp_samples]
            idx = bisect_right(keys, timestamp) - 1
            if idx >= 0:
                return self.timestamp_samples[idx][1]
            return self.timeless

        if self.timestamp_samples:
            return self.timestamp_samples[-1][1]
        if self.frame_samples:
            latest_frame = max(self.frame_samples)
            return self.frame_samples[latest_frame]
        return self.timeless

    def frame_bounds(self) -> tuple[int, int] | None:
        if not self.frame_samples:
            return None
        keys = list(self.frame_samples.keys())
        return min(keys), max(keys)

    def timestamp_bounds(self) -> tuple[float, float] | None:
        if not self.timestamp_samples:
            return None
        return self.timestamp_samples[0][0], self.timestamp_samples[-1][0]


@dataclass
class ElementTemporalRecord(Generic[T]):
    samples: TemporalChannel[T] = field(default_factory=TemporalChannel)
