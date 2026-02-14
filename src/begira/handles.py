from __future__ import annotations

from typing import Any, Protocol, Self

import numpy as np


class ElementOps(Protocol):
    def delete_element(self, element_id: str, *, timeout_s: float = 10.0) -> None: ...
    def set_element_visibility(self, element_id: str, visible: bool, *, timeout_s: float = 10.0) -> None: ...
    def log_transform(
        self,
        element_id: str,
        *,
        position: tuple[float, float, float] | list[float] | None = None,
        rotation: tuple[float, float, float, float] | list[float] | None = None,
        timeout_s: float = 10.0,
    ) -> None: ...
    def get_element_meta(self, element_id: str, *, timeout_s: float = 10.0) -> dict[str, Any]: ...


def _quat_from_matrix(m: np.ndarray) -> tuple[float, float, float, float]:
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


def _coerce_target_position(ops: ElementOps, target: object, *, timeout_s: float = 10.0) -> np.ndarray:
    if isinstance(target, ElementHandle):
        meta = ops.get_element_meta(target.id, timeout_s=timeout_s)
        pos = meta.get("position")
        if isinstance(pos, list) and len(pos) == 3:
            return np.asarray(pos, dtype=np.float64)
        bounds = meta.get("bounds")
        if isinstance(bounds, dict) and isinstance(bounds.get("min"), list) and isinstance(bounds.get("max"), list):
            bmin = np.asarray(bounds["min"], dtype=np.float64)
            bmax = np.asarray(bounds["max"], dtype=np.float64)
            return 0.5 * (bmin + bmax)
        raise ValueError(f"Target element '{target.id}' does not expose a targetable position")

    arr = np.asarray(target, dtype=np.float64)
    if arr.shape == (3,):
        return arr
    if arr.shape == (1, 3):
        return arr.reshape(3)
    raise ValueError("target must be a 3D position or another logged object handle")


def _camera_look_at_quaternion(position: np.ndarray, target: np.ndarray, up: np.ndarray) -> tuple[float, float, float, float]:
    forward = target - position
    norm_f = float(np.linalg.norm(forward))
    if norm_f < 1e-12:
        raise ValueError("camera.look_at target is too close to the camera position")
    forward /= norm_f

    up = np.asarray(up, dtype=np.float64).reshape(3)
    up_norm = float(np.linalg.norm(up))
    if up_norm < 1e-12:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        up /= up_norm

    right = np.cross(forward, up)
    if float(np.linalg.norm(right)) < 1e-8:
        # Pick a fallback up that is not collinear with forward.
        fallback_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(forward, fallback_up))) > 0.95:
            fallback_up = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        right = np.cross(forward, fallback_up)
    right /= np.linalg.norm(right) + 1e-12

    cam_up = np.cross(right, forward)
    cam_up /= np.linalg.norm(cam_up) + 1e-12

    # Camera local -Z looks forward in world space.
    rot = np.stack([right, cam_up, -forward], axis=1).astype(np.float64)
    return _quat_from_matrix(rot)


def _camera_position_with_distance(current_position: np.ndarray, target_position: np.ndarray, distance: float) -> np.ndarray:
    if not np.isfinite(distance) or distance <= 0.0:
        raise ValueError("distance must be a positive finite number")
    offset = current_position - target_position
    norm = float(np.linalg.norm(offset))
    if norm < 1e-12:
        # Degenerate case: current camera exactly on target.
        offset = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        norm = 1.0
    return target_position + (offset / norm) * float(distance)


def _rotation_matrix_from_quaternion_xyzw(
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


class ElementHandle(str):
    def __new__(
        cls,
        element_id: str,
        *,
        ops: ElementOps,
        element_type: str | None = None,
    ) -> Self:
        obj = str.__new__(cls, element_id)
        obj._ops = ops
        obj._element_type = element_type
        return obj

    @property
    def id(self) -> str:
        return str(self)

    @property
    def element_type(self) -> str | None:
        return self._element_type

    def get_meta(self, *, timeout_s: float = 10.0) -> dict[str, Any]:
        return self._ops.get_element_meta(self.id, timeout_s=timeout_s)

    @property
    def meta(self) -> dict[str, Any]:
        return self.get_meta()

    @property
    def kind(self) -> str:
        value = self.get_meta().get("type")
        if isinstance(value, str) and value:
            return value
        if self._element_type is not None:
            return self._element_type
        raise ValueError(f"Element '{self.id}' metadata does not contain a valid type")

    @property
    def revision(self) -> int:
        value = self.get_meta().get("revision")
        if value is None:
            raise ValueError(f"Element '{self.id}' metadata does not contain revision")
        return int(value)

    @property
    def name(self) -> str:
        value = self.get_meta().get("name")
        if not isinstance(value, str):
            raise ValueError(f"Element '{self.id}' metadata does not contain a valid name")
        return value

    @property
    def position(self) -> tuple[float, float, float]:
        value = self.get_meta().get("position")
        if not isinstance(value, list) or len(value) != 3:
            raise ValueError(f"Element '{self.id}' metadata does not contain a valid position")
        return (float(value[0]), float(value[1]), float(value[2]))

    @property
    def orientation(self) -> tuple[float, float, float, float]:
        value = self.get_meta().get("rotation")
        if not isinstance(value, list) or len(value) != 4:
            raise ValueError(f"Element '{self.id}' metadata does not contain a valid rotation quaternion")
        q = np.asarray(value, dtype=np.float64).reshape(4)
        n = float(np.linalg.norm(q))
        if n < 1e-12:
            raise ValueError(f"Element '{self.id}' metadata contains a near-zero rotation quaternion")
        q /= n
        return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))

    @property
    def quaternion(self) -> tuple[float, float, float, float]:
        return self.orientation

    @property
    def orientation_pyquaternion(self) -> Any:
        try:
            from pyquaternion import Quaternion  # type: ignore
        except Exception as exc:
            raise ImportError(
                "pyquaternion is not installed. Install with `python -m pip install pyquaternion` to use orientation_pyquaternion."
            ) from exc
        x, y, z, w = self.orientation
        return Quaternion(w=w, x=x, y=y, z=z)

    @property
    def rotation_matrix(self) -> np.ndarray:
        return _rotation_matrix_from_quaternion_xyzw(self.orientation)

    @property
    def visible(self) -> bool:
        value = self.get_meta().get("visible")
        return bool(value)

    @property
    def deleted(self) -> bool:
        value = self.get_meta().get("deleted")
        if value is None:
            return False
        return bool(value)

    @property
    def bounds(self) -> dict[str, list[float]] | None:
        value = self.get_meta().get("bounds")
        if isinstance(value, dict):
            return value
        return None

    def set_visibility(self, visible: bool, *, timeout_s: float = 10.0) -> Self:
        self._ops.set_element_visibility(self.id, bool(visible), timeout_s=timeout_s)
        return self

    def disable(self, *, timeout_s: float = 10.0) -> Self:
        return self.set_visibility(False, timeout_s=timeout_s)

    def enable(self, *, timeout_s: float = 10.0) -> Self:
        return self.set_visibility(True, timeout_s=timeout_s)

    def delete(self, *, timeout_s: float = 10.0) -> None:
        self._ops.delete_element(self.id, timeout_s=timeout_s)

    def set_pose(
        self,
        *,
        position: tuple[float, float, float] | list[float] | np.ndarray | None = None,
        rotation: tuple[float, float, float, float] | list[float] | np.ndarray | None = None,
        timeout_s: float = 10.0,
    ) -> Self:
        pos_payload: tuple[float, float, float] | None = None
        rot_payload: tuple[float, float, float, float] | None = None
        if position is not None:
            p = np.asarray(position, dtype=np.float64).reshape(3)
            pos_payload = (float(p[0]), float(p[1]), float(p[2]))
        if rotation is not None:
            q = np.asarray(rotation, dtype=np.float64).reshape(4)
            q /= np.linalg.norm(q) + 1e-12
            rot_payload = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
        self._ops.log_transform(self.id, position=pos_payload, rotation=rot_payload, timeout_s=timeout_s)
        return self

    def set_transform(self, transform: np.ndarray | list[list[float]] | tuple[tuple[float, ...], ...], *, timeout_s: float = 10.0) -> Self:
        t = np.asarray(transform, dtype=np.float64)
        if t.shape != (4, 4):
            raise ValueError("transform must be a 4x4 homogeneous matrix")
        rot = t[:3, :3]
        pos = t[:3, 3]
        quat = _quat_from_matrix(rot)
        return self.set_pose(position=pos, rotation=quat, timeout_s=timeout_s)

    def look_at(
        self,
        target: object,
        distance: float | None = None,
        *,
        up: tuple[float, float, float] | list[float] | np.ndarray = (0.0, 0.0, 1.0),
        timeout_s: float = 10.0,
    ) -> Self:
        et = self._element_type
        if et is None:
            meta = self._ops.get_element_meta(self.id, timeout_s=timeout_s)
            et = str(meta.get("type") or "")
        if et != "camera":
            raise TypeError(f"look_at is only supported for camera handles (got type={et!r})")

        cam_meta = self._ops.get_element_meta(self.id, timeout_s=timeout_s)
        cam_pos_raw = cam_meta.get("position")
        if not isinstance(cam_pos_raw, list) or len(cam_pos_raw) != 3:
            raise ValueError("camera metadata does not contain a valid position")

        cam_pos = np.asarray(cam_pos_raw, dtype=np.float64)
        target_pos = _coerce_target_position(self._ops, target, timeout_s=timeout_s)
        if distance is not None:
            cam_pos = _camera_position_with_distance(cam_pos, target_pos, float(distance))
        up_v = np.asarray(up, dtype=np.float64).reshape(3)
        quat = _camera_look_at_quaternion(cam_pos, target_pos, up_v)
        if distance is not None:
            return self.set_pose(position=cam_pos, rotation=quat, timeout_s=timeout_s)
        return self.set_pose(rotation=quat, timeout_s=timeout_s)


class PointCloudHandle(ElementHandle):
    @property
    def count(self) -> int:
        meta = self.get_meta()
        if isinstance(meta.get("pointCount"), int):
            return int(meta["pointCount"])
        summary = meta.get("summary")
        if isinstance(summary, dict) and isinstance(summary.get("pointCount"), int):
            return int(summary["pointCount"])
        if isinstance(meta.get("count"), int):
            return int(meta["count"])
        raise ValueError(f"Pointcloud element '{self.id}' metadata does not expose point count")


class GaussianHandle(ElementHandle):
    @property
    def count(self) -> int:
        meta = self.get_meta()
        if isinstance(meta.get("count"), int):
            return int(meta["count"])
        summary = meta.get("summary")
        if isinstance(summary, dict) and isinstance(summary.get("count"), int):
            return int(summary["count"])
        if isinstance(meta.get("pointCount"), int):
            return int(meta["pointCount"])
        raise ValueError(f"Gaussian element '{self.id}' metadata does not expose count")


class CameraHandle(ElementHandle):
    @property
    def fov(self) -> float:
        value = self.get_meta().get("fov")
        if value is None:
            raise ValueError(f"Camera '{self.id}' metadata does not expose fov")
        return float(value)

    @property
    def near(self) -> float:
        value = self.get_meta().get("near")
        if value is None:
            raise ValueError(f"Camera '{self.id}' metadata does not expose near")
        return float(value)

    @property
    def far(self) -> float:
        value = self.get_meta().get("far")
        if value is None:
            raise ValueError(f"Camera '{self.id}' metadata does not expose far")
        return float(value)

    @property
    def width(self) -> int | None:
        value = self.get_meta().get("width")
        if value is None:
            return None
        return int(value)

    @property
    def height(self) -> int | None:
        value = self.get_meta().get("height")
        if value is None:
            return None
        return int(value)

    @property
    def intrinsic_matrix(self) -> np.ndarray | None:
        meta = self.get_meta()
        value = meta.get("intrinsicMatrix")
        if value is None:
            value = meta.get("intrinsic_matrix")
        if value is None:
            return None
        k = np.asarray(value, dtype=np.float64)
        if k.shape != (3, 3):
            raise ValueError(f"Camera '{self.id}' intrinsic matrix has invalid shape {k.shape}")
        return k


class ImageHandle(ElementHandle):
    @property
    def width(self) -> int:
        value = self.get_meta().get("width")
        if value is None:
            raise ValueError(f"Image '{self.id}' metadata does not expose width")
        return int(value)

    @property
    def height(self) -> int:
        value = self.get_meta().get("height")
        if value is None:
            raise ValueError(f"Image '{self.id}' metadata does not expose height")
        return int(value)

    @property
    def channels(self) -> int:
        value = self.get_meta().get("channels")
        if value is None:
            raise ValueError(f"Image '{self.id}' metadata does not expose channels")
        return int(value)

    @property
    def mime_type(self) -> str:
        meta = self.get_meta()
        value = meta.get("mimeType")
        if value is None:
            value = meta.get("mime_type")
        if not isinstance(value, str):
            raise ValueError(f"Image '{self.id}' metadata does not expose mimeType")
        return value
