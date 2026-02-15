from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np


CameraAnimationMode = Literal["follow", "orbit"]
CameraInterpolation = Literal["catmull_rom"]


@dataclass(frozen=True)
class CameraControlKey:
    frame: int
    position_local: tuple[float, float, float]


@dataclass(frozen=True)
class CameraAnimationTrack:
    camera_id: str
    mode: CameraAnimationMode
    target_id: str
    start_frame: int
    end_frame: int
    step: int = 1
    interpolation: CameraInterpolation = "catmull_rom"
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    params: dict[str, float] = field(default_factory=dict)
    control_keys: tuple[CameraControlKey, ...] = field(default_factory=tuple)
    revision: int = 1
    updated_at: float = field(default_factory=time.time)


def normalized_vec3(v: np.ndarray | tuple[float, float, float] | list[float], fallback: tuple[float, float, float]) -> np.ndarray:
    out = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(out))
    if n < 1e-12:
        out = np.asarray(fallback, dtype=np.float64).reshape(3)
        n = float(np.linalg.norm(out))
        if n < 1e-12:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return out / n


def quat_xyzw_to_matrix(
    q_xyzw: tuple[float, float, float, float] | list[float] | np.ndarray,
) -> np.ndarray:
    q = np.asarray(q_xyzw, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        raise ValueError("Quaternion norm is too close to zero")
    x, y, z, w = (q / n).tolist()
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def quat_xyzw_from_matrix(m: np.ndarray) -> tuple[float, float, float, float]:
    tr = float(m[0, 0] + m[1, 1] + m[2, 2])
    if tr > 0.0:
        s = float(np.sqrt(tr + 1.0) * 2.0)
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = float(np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0)
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = float(np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0)
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = float(np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0)
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    q /= np.linalg.norm(q) + 1e-12
    return float(q[0]), float(q[1]), float(q[2]), float(q[3])


def pose_matrix(
    position: tuple[float, float, float] | list[float] | np.ndarray,
    rotation_xyzw: tuple[float, float, float, float] | list[float] | np.ndarray,
) -> np.ndarray:
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = quat_xyzw_to_matrix(rotation_xyzw)
    p = np.asarray(position, dtype=np.float64).reshape(3)
    t[:3, 3] = p
    return t


def pose_from_matrix(t: np.ndarray) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    m = np.asarray(t, dtype=np.float64)
    if m.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 matrix, got {m.shape}")
    pos = (float(m[0, 3]), float(m[1, 3]), float(m[2, 3]))
    quat = quat_xyzw_from_matrix(m[:3, :3])
    return pos, quat


def look_at_quaternion(
    position: np.ndarray | tuple[float, float, float] | list[float],
    target: np.ndarray | tuple[float, float, float] | list[float],
    up: np.ndarray | tuple[float, float, float] | list[float],
) -> tuple[float, float, float, float]:
    pos = np.asarray(position, dtype=np.float64).reshape(3)
    tar = np.asarray(target, dtype=np.float64).reshape(3)
    forward = tar - pos
    norm_f = float(np.linalg.norm(forward))
    if norm_f < 1e-12:
        raise ValueError("look_at target is too close to the camera position")
    forward /= norm_f

    up_v = normalized_vec3(up, (0.0, 0.0, 1.0))
    right = np.cross(forward, up_v)
    if float(np.linalg.norm(right)) < 1e-8:
        fallback_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(forward, fallback_up))) > 0.95:
            fallback_up = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        right = np.cross(forward, fallback_up)
    right = normalized_vec3(right, (1.0, 0.0, 0.0))
    cam_up = normalized_vec3(np.cross(right, forward), (0.0, 1.0, 0.0))

    rot = np.stack([right, cam_up, -forward], axis=1).astype(np.float64)
    return quat_xyzw_from_matrix(rot)


def sample_catmull_rom(
    control_keys: tuple[CameraControlKey, ...],
    frame: int,
) -> np.ndarray:
    if len(control_keys) == 0:
        raise ValueError("control_keys cannot be empty")
    if len(control_keys) == 1:
        return np.asarray(control_keys[0].position_local, dtype=np.float64)

    keys = sorted(control_keys, key=lambda k: int(k.frame))
    f = int(frame)
    if f <= int(keys[0].frame):
        return np.asarray(keys[0].position_local, dtype=np.float64)
    if f >= int(keys[-1].frame):
        return np.asarray(keys[-1].position_local, dtype=np.float64)

    i1 = 0
    for i in range(len(keys) - 1):
        if int(keys[i].frame) <= f <= int(keys[i + 1].frame):
            i1 = i
            break

    i0 = max(0, i1 - 1)
    i2 = i1 + 1
    i3 = min(len(keys) - 1, i2 + 1)

    k0, k1, k2, k3 = keys[i0], keys[i1], keys[i2], keys[i3]
    f1 = float(k1.frame)
    f2 = float(k2.frame)
    if f2 - f1 < 1e-9:
        return np.asarray(k1.position_local, dtype=np.float64)

    t = (float(f) - f1) / (f2 - f1)

    p0 = np.asarray(k0.position_local, dtype=np.float64)
    p1 = np.asarray(k1.position_local, dtype=np.float64)
    p2 = np.asarray(k2.position_local, dtype=np.float64)
    p3 = np.asarray(k3.position_local, dtype=np.float64)

    t2 = t * t
    t3 = t2 * t
    # Uniform Catmull-Rom spline.
    out = 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    )
    return out.astype(np.float64)


def camera_control_key_to_dict(key: CameraControlKey) -> dict[str, Any]:
    return {
        "frame": int(key.frame),
        "positionLocal": [float(key.position_local[0]), float(key.position_local[1]), float(key.position_local[2])],
    }


def camera_animation_track_to_dict(track: CameraAnimationTrack) -> dict[str, Any]:
    return {
        "cameraId": track.camera_id,
        "mode": track.mode,
        "targetId": track.target_id,
        "startFrame": int(track.start_frame),
        "endFrame": int(track.end_frame),
        "step": int(track.step),
        "interpolation": track.interpolation,
        "up": [float(track.up[0]), float(track.up[1]), float(track.up[2])],
        "params": {str(k): float(v) for k, v in track.params.items()},
        "controlKeys": [camera_control_key_to_dict(k) for k in track.control_keys],
        "revision": int(track.revision),
        "updatedAt": float(track.updated_at),
    }


def bump_track_revision(track: CameraAnimationTrack) -> CameraAnimationTrack:
    return replace(
        track,
        revision=int(track.revision) + 1,
        updated_at=time.time(),
    )
