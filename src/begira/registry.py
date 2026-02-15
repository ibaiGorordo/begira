from __future__ import annotations

import threading
import time
import uuid
from dataclasses import replace
import numpy as np

from .animation import (
    CameraAnimationTrack,
    CameraControlKey,
    bump_track_revision,
    look_at_quaternion,
    normalized_vec3,
    pose_from_matrix,
    pose_matrix,
    sample_catmull_rom,
)
from .elements import (
    ElementBase,
    PointCloudElement,
    GaussianSplatElement,
    CameraElement,
    ImageElement,
    Box3DElement,
    Ellipsoid3DElement,
)
from .timeline import WriteTarget, ElementTemporalRecord

class InMemoryRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._elements: dict[str, ElementBase] = {}
        self._timeline: dict[str, ElementTemporalRecord[ElementBase]] = {}
        self._camera_animations: dict[str, CameraAnimationTrack] = {}
        self._global_revision = 0
        # Deterministic palette colors for point clouds that don't provide per-point colors.
        # Colors are uint8 RGB.
        self._default_palette: list[tuple[int, int, int]] = [
            (31, 119, 180),  # blue
            (255, 127, 14),  # orange
            (44, 160, 44),  # green
            (214, 39, 40),  # red
            (148, 103, 189),  # purple
            (140, 86, 75),  # brown
            (227, 119, 194),  # pink
            (127, 127, 127),  # gray
            (188, 189, 34),  # olive
            (23, 190, 207),  # cyan
        ]

    def _palette_color_for_new_pointcloud(self) -> tuple[int, int, int]:
        idx = len([e for e in self._elements.values() if isinstance(e, PointCloudElement)]) % len(self._default_palette)
        return self._default_palette[idx]

    @staticmethod
    def _normalize_intrinsic_matrix(
        intrinsic_matrix: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
        | list[list[float]]
        | np.ndarray
        | None,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] | None:
        if intrinsic_matrix is None:
            return None
        k = np.asarray(intrinsic_matrix, dtype=np.float64)
        if k.shape != (3, 3):
            raise ValueError(f"intrinsic_matrix must have shape (3, 3), got {k.shape}")
        return (
            (float(k[0, 0]), float(k[0, 1]), float(k[0, 2])),
            (float(k[1, 0]), float(k[1, 1]), float(k[1, 2])),
            (float(k[2, 0]), float(k[2, 1]), float(k[2, 2])),
        )

    @staticmethod
    def _infer_fov_from_intrinsics(
        intrinsic_matrix: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] | None,
        height: int | None,
    ) -> float | None:
        if intrinsic_matrix is None or height is None:
            return None
        fy = float(intrinsic_matrix[1][1])
        if not np.isfinite(fy) or fy <= 0:
            raise ValueError("intrinsic_matrix[1][1] (fy) must be finite and > 0")
        if int(height) <= 0:
            raise ValueError("height must be a positive integer")
        return float(np.degrees(2.0 * np.arctan(float(height) / (2.0 * fy))))

    @staticmethod
    def _validate_sample_query(frame: int | None, timestamp: float | None) -> tuple[int | None, float | None]:
        if frame is not None and timestamp is not None:
            raise ValueError("Cannot query with both frame and timestamp")
        frame_v = int(frame) if frame is not None else None
        ts_v: float | None = None
        if timestamp is not None:
            ts_v = float(timestamp)
            if not np.isfinite(ts_v):
                raise ValueError("timestamp must be finite")
        return frame_v, ts_v

    @staticmethod
    def _normalize_positive_vec3(
        value: tuple[float, float, float] | list[float] | np.ndarray,
        *,
        name: str,
    ) -> tuple[float, float, float]:
        arr = np.asarray(value, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must contain finite numeric values")
        if np.any(arr <= 0.0):
            raise ValueError(f"{name} components must be > 0")
        return (float(arr[0]), float(arr[1]), float(arr[2]))

    @staticmethod
    def _normalize_color3(
        value: tuple[float, float, float] | list[float] | np.ndarray,
        *,
        name: str = "color",
    ) -> tuple[float, float, float]:
        arr = np.asarray(value, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must contain finite numeric values")
        arr = np.clip(arr, 0.0, 1.0)
        return (float(arr[0]), float(arr[1]), float(arr[2]))

    def _resolve_write_target_locked(
        self,
        *,
        static: bool = False,
        frame: int | None = None,
        timestamp: float | None = None,
    ) -> WriteTarget:
        frame_v, ts_v = self._validate_sample_query(frame, timestamp)
        if static and (frame_v is not None or ts_v is not None):
            raise ValueError("static=True cannot be combined with frame or timestamp")

        if static:
            return WriteTarget(axis=None, key=None, auto=False)
        if frame_v is not None:
            return WriteTarget(axis="frame", key=frame_v, auto=False)
        if ts_v is not None:
            return WriteTarget(axis="timestamp", key=ts_v, auto=False)

        # Default behavior: timeless write unless time axis is explicitly provided.
        return WriteTarget(axis=None, key=None, auto=False)

    def _get_record_locked(self, element_id: str) -> ElementTemporalRecord[ElementBase]:
        record = self._timeline.get(element_id)
        if record is None:
            record = ElementTemporalRecord[ElementBase]()
            self._timeline[element_id] = record
        return record

    def _store_sample_locked(self, element_id: str, target: WriteTarget, element: ElementBase) -> None:
        record = self._get_record_locked(element_id)
        record.samples.set_sample(target, element)
        self._elements[element_id] = element

    def _sample_element_locked(self, element_id: str, *, frame: int | None = None, timestamp: float | None = None) -> ElementBase | None:
        record = self._timeline.get(element_id)
        if record is None:
            return None
        return record.samples.sample(frame=frame, timestamp=timestamp)

    def _base_element_for_write_locked(self, element_id: str, target: WriteTarget) -> ElementBase | None:
        sampled = self._sample_element_locked(
            element_id,
            frame=int(target.key) if target.axis == "frame" else None,
            timestamp=float(target.key) if target.axis == "timestamp" else None,
        )
        if sampled is not None:
            return sampled
        if target.axis is None or target.auto:
            return self._elements.get(element_id)
        return None

    @staticmethod
    def _next_revisions(latest: ElementBase | None, *, state_changed: bool, data_changed: bool) -> tuple[int, int, int]:
        if latest is None:
            return 1, 1, 1
        revision = int(latest.revision) + 1
        state_revision = int(latest.state_revision) + (1 if state_changed else 0)
        data_revision = int(latest.data_revision) + (1 if data_changed else 0)
        return revision, state_revision, data_revision

    def global_revision(self) -> int:
        with self._lock:
            return self._global_revision

    def _require_element_for_frame_locked(self, element_id: str, frame: int) -> ElementBase:
        elem = self._sample_element_locked(element_id, frame=int(frame), timestamp=None)
        if elem is None:
            raise ValueError(f"Element '{element_id}' has no sample at frame {int(frame)}")
        return elem

    def _require_pose_for_frame_locked(self, element_id: str, frame: int) -> tuple[np.ndarray, np.ndarray]:
        elem = self._require_element_for_frame_locked(element_id, frame)
        pos = np.asarray(elem.position, dtype=np.float64).reshape(3)
        rot = np.asarray(elem.rotation, dtype=np.float64).reshape(4)
        return pos, rot

    @staticmethod
    def _to_pose_tuple(position: np.ndarray, rotation: np.ndarray) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        pos = (float(position[0]), float(position[1]), float(position[2]))
        q = np.asarray(rotation, dtype=np.float64).reshape(4)
        n = float(np.linalg.norm(q))
        if n < 1e-12:
            raise ValueError("Rotation quaternion norm is too close to zero")
        q = q / n
        quat = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
        return pos, quat

    def _validate_camera_animation_track_locked(self, track: CameraAnimationTrack) -> None:
        camera = self._elements.get(track.camera_id)
        if camera is None or not isinstance(camera, CameraElement):
            raise KeyError(track.camera_id)

        target = self._elements.get(track.target_id)
        if target is None:
            raise KeyError(track.target_id)
        if track.camera_id == track.target_id:
            raise ValueError("camera_id and target_id must be different")
        if isinstance(target, ImageElement):
            raise ValueError("Image elements cannot be used as animation targets")

        start = int(track.start_frame)
        end = int(track.end_frame)
        if start < 0:
            raise ValueError("start_frame must be >= 0")
        if end < start:
            raise ValueError("end_frame must be >= start_frame")
        if int(track.step) <= 0:
            raise ValueError("step must be a positive integer")

        self._require_element_for_frame_locked(track.target_id, start)
        self._require_element_for_frame_locked(track.camera_id, start)

        up_v = np.asarray(track.up, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(up_v)):
            raise ValueError("up must contain finite numeric values")
        if float(np.linalg.norm(up_v)) < 1e-12:
            raise ValueError("up vector cannot be near zero")

        if track.mode == "orbit":
            if track.interpolation != "catmull_rom":
                raise ValueError("Only catmull_rom interpolation is supported")
            turns = float(track.params.get("turns", 1.0))
            if not np.isfinite(turns) or abs(turns) < 1e-9:
                raise ValueError("orbit turns must be a finite non-zero number")
            radius = track.params.get("radius")
            if radius is not None:
                radius_v = float(radius)
                if not np.isfinite(radius_v) or radius_v <= 0.0:
                    raise ValueError("orbit radius must be a positive finite number")
            phase = float(track.params.get("phaseDeg", 0.0))
            if not np.isfinite(phase):
                raise ValueError("orbit phaseDeg must be finite")
            for key in track.control_keys:
                if int(key.frame) < start or int(key.frame) > end:
                    raise ValueError("control key frame is outside animation range")
        elif track.mode != "follow":
            raise ValueError(f"Unsupported animation mode: {track.mode!r}")

    def _build_default_orbit_keys_locked(self, track: CameraAnimationTrack) -> tuple[CameraControlKey, ...]:
        start = int(track.start_frame)
        end = int(track.end_frame)
        step = max(1, int(track.step))

        cam_pos, cam_rot = self._require_pose_for_frame_locked(track.camera_id, start)
        target_pos, target_rot = self._require_pose_for_frame_locked(track.target_id, start)

        t_cam_start = pose_matrix(cam_pos, cam_rot)
        t_target_start = pose_matrix(target_pos, target_rot)
        t_local = np.linalg.inv(t_target_start) @ t_cam_start
        local_start_offset = np.asarray(t_local[:3, 3], dtype=np.float64)

        radius_param = track.params.get("radius")
        if radius_param is None:
            radius = float(np.linalg.norm(local_start_offset))
            if not np.isfinite(radius) or radius <= 1e-6:
                radius = 1.0
        else:
            radius = float(radius_param)

        up_world = normalized_vec3(track.up, (0.0, 0.0, 1.0))
        r_target_start = pose_matrix((0.0, 0.0, 0.0), target_rot)[:3, :3]
        up_local = normalized_vec3(r_target_start.T @ up_world, (0.0, 0.0, 1.0))

        proj = local_start_offset - up_local * float(np.dot(local_start_offset, up_local))
        proj_n = float(np.linalg.norm(proj))
        if proj_n < 1e-6:
            fallback = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            if abs(float(np.dot(fallback, up_local))) > 0.95:
                fallback = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            proj = np.cross(up_local, fallback)
            proj_n = float(np.linalg.norm(proj))
            if proj_n < 1e-6:
                proj = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                proj_n = 1.0
        axis_a = (proj / proj_n) * float(radius)
        axis_b = normalized_vec3(np.cross(up_local, axis_a), (0.0, 1.0, 0.0)) * float(radius)

        frames = list(range(start, end + 1, step))
        if not frames:
            frames = [start]
        if frames[-1] != end:
            frames.append(end)
        if start == end:
            frames = [start]

        turns = float(track.params.get("turns", 1.0))
        phase = np.deg2rad(float(track.params.get("phaseDeg", 0.0)))
        span = max(1, end - start)

        out: list[CameraControlKey] = []
        for f in frames:
            t = float(f - start) / float(span)
            angle = phase + (2.0 * np.pi * turns * t)
            local = axis_a * float(np.cos(angle)) + axis_b * float(np.sin(angle))
            out.append(
                CameraControlKey(
                    frame=int(f),
                    position_local=(float(local[0]), float(local[1]), float(local[2])),
                )
            )
        return tuple(out)

    def _bake_follow_track_locked(self, track: CameraAnimationTrack) -> None:
        start = int(track.start_frame)
        end = int(track.end_frame)

        cam_pos_0, cam_rot_0 = self._require_pose_for_frame_locked(track.camera_id, start)
        target_pos_0, target_rot_0 = self._require_pose_for_frame_locked(track.target_id, start)

        t_cam_0 = pose_matrix(cam_pos_0, cam_rot_0)
        t_target_0 = pose_matrix(target_pos_0, target_rot_0)
        t_rel = np.linalg.inv(t_target_0) @ t_cam_0

        for frame in range(start, end + 1):
            target_pos_f, target_rot_f = self._require_pose_for_frame_locked(track.target_id, frame)
            t_target_f = pose_matrix(target_pos_f, target_rot_f)
            t_cam_f = t_target_f @ t_rel
            pos_t, quat_t = pose_from_matrix(t_cam_f)
            self.update_element_meta(
                track.camera_id,
                position=pos_t,
                rotation=quat_t,
                frame=int(frame),
            )

    def _bake_orbit_track_locked(self, track: CameraAnimationTrack) -> None:
        start = int(track.start_frame)
        end = int(track.end_frame)
        up_world = normalized_vec3(track.up, (0.0, 0.0, 1.0))
        control_keys = tuple(sorted(track.control_keys, key=lambda k: int(k.frame)))
        if not control_keys:
            raise ValueError("orbit track requires control keys")

        for frame in range(start, end + 1):
            local_offset = sample_catmull_rom(control_keys, frame)
            target_pos_f, target_rot_f = self._require_pose_for_frame_locked(track.target_id, frame)
            t_target_f = pose_matrix(target_pos_f, target_rot_f)
            world_h = t_target_f @ np.array([float(local_offset[0]), float(local_offset[1]), float(local_offset[2]), 1.0], dtype=np.float64)
            cam_pos = world_h[:3]
            cam_quat = look_at_quaternion(cam_pos, target_pos_f, up_world)
            pos_t, quat_t = self._to_pose_tuple(cam_pos, np.asarray(cam_quat, dtype=np.float64))
            self.update_element_meta(
                track.camera_id,
                position=pos_t,
                rotation=quat_t,
                frame=int(frame),
            )

    def _bake_camera_track_locked(self, track: CameraAnimationTrack) -> CameraAnimationTrack:
        built = track
        if built.mode == "orbit" and len(built.control_keys) == 0:
            built = replace(built, control_keys=self._build_default_orbit_keys_locked(built))

        if built.mode == "follow":
            self._bake_follow_track_locked(built)
        elif built.mode == "orbit":
            self._bake_orbit_track_locked(built)
        else:
            raise ValueError(f"Unsupported animation mode: {built.mode!r}")
        return built

    def get_camera_animation(self, camera_id: str) -> CameraAnimationTrack | None:
        with self._lock:
            return self._camera_animations.get(camera_id)

    def set_camera_animation(self, track: CameraAnimationTrack) -> CameraAnimationTrack:
        with self._lock:
            self._validate_camera_animation_track_locked(track)
            prev = self._camera_animations.get(track.camera_id)
            if prev is None:
                working = replace(track, revision=1, updated_at=time.time())
            else:
                working = bump_track_revision(replace(track, revision=prev.revision, updated_at=prev.updated_at))

            baked = self._bake_camera_track_locked(working)
            self._camera_animations[baked.camera_id] = baked
            self._global_revision += 1
            return baked

    def clear_camera_animation(self, camera_id: str) -> bool:
        with self._lock:
            existed = camera_id in self._camera_animations
            if existed:
                del self._camera_animations[camera_id]
                self._global_revision += 1
            return existed

    def update_camera_animation_key(
        self,
        camera_id: str,
        frame: int,
        new_world_position: tuple[float, float, float] | list[float] | np.ndarray,
        *,
        pull_enabled: bool = False,
        pull_radius_frames: int = 0,
        pull_pinned_ends: bool = True,
    ) -> CameraAnimationTrack:
        with self._lock:
            track = self._camera_animations.get(camera_id)
            if track is None:
                raise KeyError(camera_id)
            if track.mode != "orbit":
                raise ValueError("Only orbit animation keys are editable")

            frame_i = int(frame)
            if frame_i < int(track.start_frame) or frame_i > int(track.end_frame):
                raise ValueError("frame is outside animation range")

            world = np.asarray(new_world_position, dtype=np.float64).reshape(3)
            target_pos, target_rot = self._require_pose_for_frame_locked(track.target_id, frame_i)
            t_target = pose_matrix(target_pos, target_rot)
            local_h = np.linalg.inv(t_target) @ np.array([float(world[0]), float(world[1]), float(world[2]), 1.0], dtype=np.float64)
            new_local = np.asarray([float(local_h[0]), float(local_h[1]), float(local_h[2])], dtype=np.float64)

            by_frame: dict[int, CameraControlKey] = {int(k.frame): k for k in track.control_keys}
            if frame_i in by_frame:
                old_local = np.asarray(by_frame[frame_i].position_local, dtype=np.float64).reshape(3)
            else:
                old_local = sample_catmull_rom(track.control_keys, frame_i)
            delta_local = new_local - old_local

            radius_i = max(0, int(pull_radius_frames))
            sigma = max(float(radius_i) / 2.5, 1e-6)
            start_frame_i = int(track.start_frame)
            end_frame_i = int(track.end_frame)
            frame_span = max(1, end_frame_i - start_frame_i)
            use_pull = bool(pull_enabled) and radius_i > 0

            updated_by_frame: dict[int, CameraControlKey] = {}
            for key in track.control_keys:
                key_frame = int(key.frame)
                base_local = np.asarray(key.position_local, dtype=np.float64).reshape(3)
                next_local = base_local

                if key_frame == frame_i:
                    next_local = new_local
                elif use_pull:
                    if bool(pull_pinned_ends) and key_frame in (start_frame_i, end_frame_i):
                        next_local = base_local
                    else:
                        d_linear = abs(key_frame - frame_i)
                        # Orbit keys are a closed loop over [start_frame, end_frame].
                        d = min(d_linear, frame_span - d_linear)
                        if d <= radius_i:
                            w = float(np.exp(-0.5 * ((float(d) / sigma) ** 2)))
                            next_local = base_local + delta_local * w

                updated_by_frame[key_frame] = CameraControlKey(
                    frame=key_frame,
                    position_local=(float(next_local[0]), float(next_local[1]), float(next_local[2])),
                )

            if frame_i not in updated_by_frame:
                updated_by_frame[frame_i] = CameraControlKey(
                    frame=frame_i,
                    position_local=(float(new_local[0]), float(new_local[1]), float(new_local[2])),
                )

            keys = tuple(sorted(updated_by_frame.values(), key=lambda k: int(k.frame)))
            if len(keys) < 2:
                raise ValueError("orbit animation requires at least 2 control keys")

            updated = bump_track_revision(replace(track, control_keys=keys))
            baked = self._bake_camera_track_locked(updated)
            self._camera_animations[camera_id] = baked
            self._global_revision += 1
            return baked

    def insert_camera_animation_key(
        self,
        camera_id: str,
        frame: int,
        world_position: tuple[float, float, float] | list[float] | np.ndarray | None = None,
    ) -> CameraAnimationTrack:
        with self._lock:
            track = self._camera_animations.get(camera_id)
            if track is None:
                raise KeyError(camera_id)
            if track.mode != "orbit":
                raise ValueError("Only orbit animation keys are editable")

            frame_i = int(frame)
            if frame_i < int(track.start_frame) or frame_i > int(track.end_frame):
                raise ValueError("frame is outside animation range")

            if world_position is None:
                if len(track.control_keys) > 0:
                    local = sample_catmull_rom(track.control_keys, frame_i)
                else:
                    cam_elem = self._require_element_for_frame_locked(camera_id, frame_i)
                    cam_world = np.asarray(cam_elem.position, dtype=np.float64).reshape(3)
                    target_pos, target_rot = self._require_pose_for_frame_locked(track.target_id, frame_i)
                    t_target = pose_matrix(target_pos, target_rot)
                    local_h = np.linalg.inv(t_target) @ np.array([float(cam_world[0]), float(cam_world[1]), float(cam_world[2]), 1.0], dtype=np.float64)
                    local = np.asarray(local_h[:3], dtype=np.float64)
            else:
                target_pos, target_rot = self._require_pose_for_frame_locked(track.target_id, frame_i)
                t_target = pose_matrix(target_pos, target_rot)
                world = np.asarray(world_position, dtype=np.float64).reshape(3)
                local_h = np.linalg.inv(t_target) @ np.array([float(world[0]), float(world[1]), float(world[2]), 1.0], dtype=np.float64)
                local = np.asarray(local_h[:3], dtype=np.float64)

            by_frame: dict[int, CameraControlKey] = {int(k.frame): k for k in track.control_keys}
            by_frame[frame_i] = CameraControlKey(
                frame=frame_i,
                position_local=(float(local[0]), float(local[1]), float(local[2])),
            )
            keys = tuple(sorted(by_frame.values(), key=lambda k: int(k.frame)))
            if len(keys) < 2:
                raise ValueError("orbit animation requires at least 2 control keys")

            updated = bump_track_revision(replace(track, control_keys=keys))
            baked = self._bake_camera_track_locked(updated)
            self._camera_animations[camera_id] = baked
            self._global_revision += 1
            return baked

    def delete_camera_animation_key(
        self,
        camera_id: str,
        frame: int,
    ) -> CameraAnimationTrack:
        with self._lock:
            track = self._camera_animations.get(camera_id)
            if track is None:
                raise KeyError(camera_id)
            if track.mode != "orbit":
                raise ValueError("Only orbit animation keys are editable")

            frame_i = int(frame)
            by_frame: dict[int, CameraControlKey] = {int(k.frame): k for k in track.control_keys}
            if frame_i not in by_frame:
                raise ValueError("control key does not exist at requested frame")

            del by_frame[frame_i]
            keys = tuple(sorted(by_frame.values(), key=lambda k: int(k.frame)))
            if len(keys) < 2:
                raise ValueError("orbit animation requires at least 2 control keys")

            updated = bump_track_revision(replace(track, control_keys=keys))
            baked = self._bake_camera_track_locked(updated)
            self._camera_animations[camera_id] = baked
            self._global_revision += 1
            return baked

    def duplicate_camera_animation_key(
        self,
        camera_id: str,
        *,
        source_frame: int,
        target_frame: int,
    ) -> CameraAnimationTrack:
        with self._lock:
            track = self._camera_animations.get(camera_id)
            if track is None:
                raise KeyError(camera_id)
            if track.mode != "orbit":
                raise ValueError("Only orbit animation keys are editable")

            src_i = int(source_frame)
            dst_i = int(target_frame)
            if dst_i < int(track.start_frame) or dst_i > int(track.end_frame):
                raise ValueError("target_frame is outside animation range")

            by_frame: dict[int, CameraControlKey] = {int(k.frame): k for k in track.control_keys}
            src = by_frame.get(src_i)
            if src is None:
                raise ValueError("source_frame does not contain a control key")

            by_frame[dst_i] = CameraControlKey(frame=dst_i, position_local=src.position_local)
            keys = tuple(sorted(by_frame.values(), key=lambda k: int(k.frame)))
            if len(keys) < 2:
                raise ValueError("orbit animation requires at least 2 control keys")

            updated = bump_track_revision(replace(track, control_keys=keys))
            baked = self._bake_camera_track_locked(updated)
            self._camera_animations[camera_id] = baked
            self._global_revision += 1
            return baked

    def smooth_camera_animation_keys(
        self,
        camera_id: str,
        *,
        start_frame: int | None = None,
        end_frame: int | None = None,
        passes: int = 1,
        pinned_ends: bool = True,
    ) -> CameraAnimationTrack:
        with self._lock:
            track = self._camera_animations.get(camera_id)
            if track is None:
                raise KeyError(camera_id)
            if track.mode != "orbit":
                raise ValueError("Only orbit animation keys are editable")

            keys = list(sorted(track.control_keys, key=lambda k: int(k.frame)))
            if len(keys) < 3:
                raise ValueError("Need at least 3 control keys to smooth")

            start_i = int(track.start_frame) if start_frame is None else int(start_frame)
            end_i = int(track.end_frame) if end_frame is None else int(end_frame)
            if end_i < start_i:
                raise ValueError("end_frame must be >= start_frame")
            if start_i < int(track.start_frame) or end_i > int(track.end_frame):
                raise ValueError("smoothing range must lie within animation range")

            passes_i = int(passes)
            if passes_i <= 0:
                raise ValueError("passes must be a positive integer")

            work = [np.asarray(k.position_local, dtype=np.float64).reshape(3) for k in keys]
            frames = [int(k.frame) for k in keys]

            for _ in range(passes_i):
                next_work = [p.copy() for p in work]
                for idx in range(1, len(work) - 1):
                    frame_i = frames[idx]
                    if frame_i < start_i or frame_i > end_i:
                        continue
                    if bool(pinned_ends) and frame_i in (int(track.start_frame), int(track.end_frame)):
                        continue
                    prev_p = work[idx - 1]
                    cur_p = work[idx]
                    next_p = work[idx + 1]
                    next_work[idx] = 0.5 * cur_p + 0.25 * prev_p + 0.25 * next_p
                work = next_work

            smoothed_keys = tuple(
                CameraControlKey(
                    frame=frames[idx],
                    position_local=(float(work[idx][0]), float(work[idx][1]), float(work[idx][2])),
                )
                for idx in range(len(work))
            )

            updated = bump_track_revision(replace(track, control_keys=smoothed_keys))
            baked = self._bake_camera_track_locked(updated)
            self._camera_animations[camera_id] = baked
            self._global_revision += 1
            return baked

    def get_camera_frame_samples(
        self,
        camera_id: str,
        *,
        start_frame: int | None = None,
        end_frame: int | None = None,
        stride: int = 1,
    ) -> tuple[list[int], list[tuple[float, float, float]]]:
        with self._lock:
            track = self._camera_animations.get(camera_id)
            if track is None:
                raise KeyError(camera_id)

            start = int(start_frame) if start_frame is not None else int(track.start_frame)
            end = int(end_frame) if end_frame is not None else int(track.end_frame)
            if end < start:
                raise ValueError("end_frame must be >= start_frame")
            stride_i = max(1, int(stride))

            frames: list[int] = []
            positions: list[tuple[float, float, float]] = []
            for frame in range(start, end + 1, stride_i):
                elem = self._sample_element_locked(camera_id, frame=frame, timestamp=None)
                if not isinstance(elem, CameraElement):
                    continue
                frames.append(int(frame))
                positions.append((float(elem.position[0]), float(elem.position[1]), float(elem.position[2])))
            return frames, positions

    def timeline_info(self) -> dict[str, object]:
        with self._lock:
            frame_min: int | None = None
            frame_max: int | None = None
            ts_min: float | None = None
            ts_max: float | None = None

            for record in self._timeline.values():
                fb = record.samples.frame_bounds()
                if fb is not None:
                    lo, hi = fb
                    frame_min = lo if frame_min is None else min(frame_min, lo)
                    frame_max = hi if frame_max is None else max(frame_max, hi)

                tb = record.samples.timestamp_bounds()
                if tb is not None:
                    lo, hi = tb
                    ts_min = lo if ts_min is None else min(ts_min, lo)
                    ts_max = hi if ts_max is None else max(ts_max, hi)

            return {
                "defaultAxis": "frame",
                "axes": [
                    {
                        "axis": "frame",
                        "min": frame_min,
                        "max": frame_max,
                        "hasData": frame_min is not None,
                    },
                    {
                        "axis": "timestamp",
                        "min": ts_min,
                        "max": ts_max,
                        "hasData": ts_min is not None,
                    },
                ],
                "latest": {
                    "frame": frame_max,
                    "timestamp": ts_max,
                },
            }

    def list_elements(self, *, frame: int | None = None, timestamp: float | None = None) -> list[ElementBase]:
        with self._lock:
            frame_v, ts_v = self._validate_sample_query(frame, timestamp)
            if frame_v is None and ts_v is None:
                return [e for e in self._elements.values() if not getattr(e, "deleted", False)]

            out: list[ElementBase] = []
            for element_id in self._timeline:
                e = self._sample_element_locked(element_id, frame=frame_v, timestamp=ts_v)
                if e is None:
                    continue
                if getattr(e, "deleted", False):
                    continue
                out.append(e)
            return out

    def get_element(self, element_id: str, *, frame: int | None = None, timestamp: float | None = None) -> ElementBase | None:
        with self._lock:
            frame_v, ts_v = self._validate_sample_query(frame, timestamp)
            if frame_v is None and ts_v is None:
                return self._elements.get(element_id)
            return self._sample_element_locked(element_id, frame=frame_v, timestamp=ts_v)

    def reset(self) -> None:
        with self._lock:
            for key, prev in list(self._elements.items()):
                self.update_element_meta(
                    key,
                    position=(0.0, 0.0, 0.0),
                    rotation=(0.0, 0.0, 0.0, 1.0),
                    visible=True,
                    deleted=False,
                )
                # update_element_meta stores the sample.
            self._camera_animations.clear()
            self._global_revision += 1

    def delete_element(
        self,
        element_id: str,
        *,
        static: bool = False,
        frame: int | None = None,
        timestamp: float | None = None,
    ) -> ElementBase:
        with self._lock:
            prev = self._elements.get(element_id)
            if prev is None:
                raise KeyError(element_id)
            if getattr(prev, "deleted", False) and frame is None and timestamp is None and not static:
                return prev
            updated = self.update_element_meta(
                element_id,
                deleted=True,
                static=static,
                frame=frame,
                timestamp=timestamp,
            )
            if isinstance(prev, CameraElement):
                self._camera_animations.pop(element_id, None)
            return updated

    def update_element_meta(
        self,
        element_id: str,
        *,
        position: tuple[float, float, float] | None = None,
        rotation: tuple[float, float, float, float] | None = None,
        visible: bool | None = None,
        deleted: bool | None = None,
        point_size: float | None = None,
        size: tuple[float, float, float] | list[float] | np.ndarray | None = None,
        radii: tuple[float, float, float] | list[float] | np.ndarray | None = None,
        color: tuple[float, float, float] | list[float] | np.ndarray | None = None,
        fov: float | None = None,
        near: float | None = None,
        far: float | None = None,
        width: int | None = None,
        height: int | None = None,
        intrinsic_matrix: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
        | list[list[float]]
        | np.ndarray
        | None = None,
        static: bool = False,
        frame: int | None = None,
        timestamp: float | None = None,
    ) -> ElementBase:
        with self._lock:
            latest = self._elements.get(element_id)
            if latest is None:
                raise KeyError(element_id)

            target = self._resolve_write_target_locked(static=static, frame=frame, timestamp=timestamp)
            prev = self._base_element_for_write_locked(element_id, target)
            if prev is None:
                raise KeyError(element_id)

            updates: dict[str, object] = {}
            state_changed = False
            data_changed = False

            if position is not None:
                pos = (float(position[0]), float(position[1]), float(position[2]))
                if pos != prev.position:
                    updates["position"] = pos
                    state_changed = True
            if rotation is not None:
                rot = (float(rotation[0]), float(rotation[1]), float(rotation[2]), float(rotation[3]))
                if rot != prev.rotation:
                    updates["rotation"] = rot
                    state_changed = True
            if visible is not None:
                v = bool(visible)
                if v != bool(prev.visible):
                    updates["visible"] = v
                    state_changed = True
            if deleted is not None:
                d = bool(deleted)
                if d != bool(prev.deleted):
                    updates["deleted"] = d
                    state_changed = True

            if isinstance(prev, PointCloudElement):
                if point_size is not None:
                    if not np.isfinite(point_size) or point_size <= 0:
                        raise ValueError("point_size must be a finite positive number")
                    ps = float(point_size)
                    if ps != prev.point_size:
                        updates["point_size"] = ps
                        state_changed = True
            elif isinstance(prev, GaussianSplatElement):
                if point_size is not None:
                    if not np.isfinite(point_size) or point_size <= 0:
                        raise ValueError("point_size must be a finite positive number")
                    ps = float(point_size)
                    if ps != prev.point_size:
                        updates["point_size"] = ps
                        state_changed = True
            elif isinstance(prev, CameraElement):
                if fov is not None:
                    fv = float(fov)
                    if fv != prev.fov:
                        updates["fov"] = fv
                        state_changed = True
                if near is not None:
                    nv = float(near)
                    if nv != prev.near:
                        updates["near"] = nv
                        state_changed = True
                if far is not None:
                    fv = float(far)
                    if fv != prev.far:
                        updates["far"] = fv
                        state_changed = True
                if width is not None:
                    width_i = int(width)
                    if width_i <= 0:
                        raise ValueError("width must be a positive integer")
                    if width_i != prev.width:
                        updates["width"] = width_i
                        data_changed = True
                if height is not None:
                    height_i = int(height)
                    if height_i <= 0:
                        raise ValueError("height must be a positive integer")
                    if height_i != prev.height:
                        updates["height"] = height_i
                        data_changed = True
                if intrinsic_matrix is not None:
                    k = self._normalize_intrinsic_matrix(intrinsic_matrix)
                    if k != prev.intrinsic_matrix:
                        updates["intrinsic_matrix"] = k
                        data_changed = True
            elif isinstance(prev, Box3DElement):
                if size is not None:
                    size_v = self._normalize_positive_vec3(size, name="size")
                    if size_v != prev.size:
                        updates["size"] = size_v
                        state_changed = True
                if color is not None:
                    color_v = self._normalize_color3(color, name="color")
                    if color_v != prev.color:
                        updates["color"] = color_v
                        state_changed = True
            elif isinstance(prev, Ellipsoid3DElement):
                if radii is not None:
                    radii_v = self._normalize_positive_vec3(radii, name="radii")
                    if radii_v != prev.radii:
                        updates["radii"] = radii_v
                        state_changed = True
                if color is not None:
                    color_v = self._normalize_color3(color, name="color")
                    if color_v != prev.color:
                        updates["color"] = color_v
                        state_changed = True
            elif isinstance(prev, ImageElement):
                # No image-specific mutable fields yet (payload upserts replace the element).
                pass

            if not updates:
                return prev

            if isinstance(prev, CameraElement):
                next_near = float(updates.get("near", prev.near))
                next_far = float(updates.get("far", prev.far))
                if not np.isfinite(next_near) or next_near <= 0:
                    raise ValueError("near must be a finite positive number")
                if not np.isfinite(next_far) or next_far <= next_near:
                    raise ValueError("far must be finite and greater than near")

            revision, state_revision, data_revision = self._next_revisions(
                latest,
                state_changed=state_changed,
                data_changed=data_changed,
            )
            now = time.time()
            updated = replace(
                prev,
                **updates,
                revision=revision,
                state_revision=state_revision,
                data_revision=data_revision,
                updated_at=now,
            )
            self._global_revision += 1
            self._store_sample_locked(element_id, target, updated)
            return updated

    def update_pointcloud_settings(
        self,
        element_id: str,
        *,
        point_size: float | None = None,
        static: bool = False,
        frame: int | None = None,
        timestamp: float | None = None,
    ) -> PointCloudElement:
        updated = self.update_element_meta(
            element_id,
            point_size=point_size,
            static=static,
            frame=frame,
            timestamp=timestamp,
        )
        if not isinstance(updated, PointCloudElement):
            raise KeyError(element_id)
        return updated

    def upsert_pointcloud(
        self,
        *,
        name: str,
        positions: np.ndarray,
        colors: np.ndarray | None,
        point_size: float | None = None,
        element_id: str | None = None,
        static: bool = False,
        frame: int | None = None,
        timestamp: float | None = None,
    ) -> PointCloudElement:
        # Validate + normalize early.
        pos = np.asarray(positions)
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError(f"positions must have shape (N,3), got {pos.shape}")
        pos = np.ascontiguousarray(pos, dtype=np.float32)

        col: np.ndarray | None = None
        if colors is not None:
            c = np.asarray(colors)
            if c.shape != pos.shape:
                raise ValueError(f"colors must have shape {pos.shape}, got {c.shape}")
            if np.issubdtype(c.dtype, np.floating):
                c = np.clip(c, 0.0, 1.0) * 255.0
            col = np.ascontiguousarray(c, dtype=np.uint8)

        if point_size is not None and (not np.isfinite(point_size) or point_size <= 0):
            raise ValueError("point_size must be a finite positive number")

        with self._lock:
            if element_id is None:
                element_id = uuid.uuid4().hex

            target = self._resolve_write_target_locked(static=static, frame=frame, timestamp=timestamp)
            latest = self._elements.get(element_id)
            base = self._base_element_for_write_locked(element_id, target)
            base_pc = base if isinstance(base, PointCloudElement) else None
            latest_pc = latest if isinstance(latest, PointCloudElement) else None
            inherit_latest = target.axis is None or target.auto
            latest_pc_for_defaults = latest_pc if inherit_latest else None

            # New pointcloud and no explicit colors: assign a deterministic per-cloud palette color.
            if base_pc is None and latest_pc is None and col is None:
                r, g, b = self._palette_color_for_new_pointcloud()
                col = np.empty((pos.shape[0], 3), dtype=np.uint8)
                col[:, 0] = r
                col[:, 1] = g
                col[:, 2] = b

            point_size_value = (
                float(point_size)
                if point_size is not None
                else (base_pc.point_size if base_pc is not None else (latest_pc_for_defaults.point_size if latest_pc_for_defaults is not None else 0.02))
            )

            state_changed = False
            if latest_pc is not None and point_size_value != latest_pc.point_size:
                state_changed = True

            revision, state_revision, data_revision = self._next_revisions(
                latest,
                state_changed=state_changed,
                data_changed=True,
            )
            now = time.time()

            pc = PointCloudElement(
                id=element_id,
                type="pointcloud",
                name=name,
                positions=pos,
                colors=col,
                point_size=point_size_value,
                revision=revision,
                state_revision=state_revision,
                data_revision=data_revision,
                created_at=(latest.created_at if latest is not None else now),
                updated_at=now,
                position=(base_pc.position if base_pc is not None else (latest_pc_for_defaults.position if latest_pc_for_defaults is not None else (0.0, 0.0, 0.0))),
                rotation=(base_pc.rotation if base_pc is not None else (latest_pc_for_defaults.rotation if latest_pc_for_defaults is not None else (0.0, 0.0, 0.0, 1.0))),
                visible=(base_pc.visible if base_pc is not None else (latest_pc_for_defaults.visible if latest_pc_for_defaults is not None else True)),
                deleted=(base_pc.deleted if base_pc is not None else (latest_pc_for_defaults.deleted if latest_pc_for_defaults is not None else False)),
            )

            self._global_revision += 1
            self._store_sample_locked(element_id, target, pc)
            return pc

    def update_gaussians_settings(
        self,
        element_id: str,
        *,
        point_size: float | None = None,
        static: bool = False,
        frame: int | None = None,
        timestamp: float | None = None,
    ) -> GaussianSplatElement:
        updated = self.update_element_meta(
            element_id,
            point_size=point_size,
            static=static,
            frame=frame,
            timestamp=timestamp,
        )
        if not isinstance(updated, GaussianSplatElement):
            raise KeyError(element_id)
        return updated

    def upsert_gaussians(
        self,
        *,
        name: str,
        positions: np.ndarray,
        sh0: np.ndarray,
        opacity: np.ndarray,
        scales: np.ndarray,
        rotations: np.ndarray,
        point_size: float | None = None,
        element_id: str | None = None,
        static: bool = False,
        frame: int | None = None,
        timestamp: float | None = None,
    ) -> GaussianSplatElement:
        with self._lock:
            if element_id is None:
                element_id = uuid.uuid4().hex

            target = self._resolve_write_target_locked(static=static, frame=frame, timestamp=timestamp)
            latest = self._elements.get(element_id)
            base = self._base_element_for_write_locked(element_id, target)
            base_gs = base if isinstance(base, GaussianSplatElement) else None
            latest_gs = latest if isinstance(latest, GaussianSplatElement) else None
            inherit_latest = target.axis is None or target.auto
            latest_gs_for_defaults = latest_gs if inherit_latest else None

            point_size_value = (
                float(point_size)
                if point_size is not None
                else (base_gs.point_size if base_gs is not None else (latest_gs_for_defaults.point_size if latest_gs_for_defaults is not None else 1.0))
            )
            state_changed = False
            if latest_gs is not None and point_size_value != latest_gs.point_size:
                state_changed = True

            revision, state_revision, data_revision = self._next_revisions(
                latest,
                state_changed=state_changed,
                data_changed=True,
            )
            now = time.time()

            gs = GaussianSplatElement(
                id=element_id,
                type="gaussians",
                name=name,
                positions=positions,
                sh0=sh0,
                opacity=opacity,
                scales=scales,
                rotations=rotations,
                point_size=point_size_value,
                revision=revision,
                state_revision=state_revision,
                data_revision=data_revision,
                created_at=(latest.created_at if latest is not None else now),
                updated_at=now,
                position=(base_gs.position if base_gs is not None else (latest_gs_for_defaults.position if latest_gs_for_defaults is not None else (0.0, 0.0, 0.0))),
                rotation=(base_gs.rotation if base_gs is not None else (latest_gs_for_defaults.rotation if latest_gs_for_defaults is not None else (0.0, 0.0, 0.0, 1.0))),
                visible=(base_gs.visible if base_gs is not None else (latest_gs_for_defaults.visible if latest_gs_for_defaults is not None else True)),
                deleted=(base_gs.deleted if base_gs is not None else (latest_gs_for_defaults.deleted if latest_gs_for_defaults is not None else False)),
            )

            self._global_revision += 1
            self._store_sample_locked(element_id, target, gs)
            return gs

    def upsert_camera(
        self,
        *,
        name: str,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        fov: float | None = None,
        near: float | None = None,
        far: float | None = None,
        width: int | None = None,
        height: int | None = None,
        intrinsic_matrix: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
        | list[list[float]]
        | np.ndarray
        | None = None,
        element_id: str | None = None,
        static: bool = False,
        frame: int | None = None,
        timestamp: float | None = None,
    ) -> CameraElement:
        k = self._normalize_intrinsic_matrix(intrinsic_matrix)

        with self._lock:
            if element_id is None:
                element_id = uuid.uuid4().hex

            target = self._resolve_write_target_locked(static=static, frame=frame, timestamp=timestamp)
            latest = self._elements.get(element_id)
            base = self._base_element_for_write_locked(element_id, target)
            base_cam = base if isinstance(base, CameraElement) else None
            latest_cam = latest if isinstance(latest, CameraElement) else None
            inherit_latest = target.axis is None or target.auto
            latest_cam_for_defaults = latest_cam if inherit_latest else None

            width_v = int(width) if width is not None else (base_cam.width if base_cam is not None else (latest_cam_for_defaults.width if latest_cam_for_defaults is not None else None))
            if width_v is not None and width_v <= 0:
                raise ValueError("width must be a positive integer")

            height_v = int(height) if height is not None else (base_cam.height if base_cam is not None else (latest_cam_for_defaults.height if latest_cam_for_defaults is not None else None))
            if height_v is not None and height_v <= 0:
                raise ValueError("height must be a positive integer")

            k_v = k if k is not None else (base_cam.intrinsic_matrix if base_cam is not None else (latest_cam_for_defaults.intrinsic_matrix if latest_cam_for_defaults is not None else None))

            if fov is None:
                inferred_fov = self._infer_fov_from_intrinsics(k_v, height_v)
                if inferred_fov is not None:
                    fov_v = float(inferred_fov)
                elif base_cam is not None:
                    fov_v = base_cam.fov
                elif latest_cam_for_defaults is not None:
                    fov_v = latest_cam_for_defaults.fov
                else:
                    fov_v = 60.0
            else:
                fov_v = float(fov)

            near_v = float(near) if near is not None else (base_cam.near if base_cam is not None else (latest_cam_for_defaults.near if latest_cam_for_defaults is not None else 0.01))
            far_v = float(far) if far is not None else (base_cam.far if base_cam is not None else (latest_cam_for_defaults.far if latest_cam_for_defaults is not None else 1000.0))
            if not np.isfinite(near_v) or near_v <= 0:
                raise ValueError("near must be a finite positive number")
            if not np.isfinite(far_v) or far_v <= near_v:
                raise ValueError("far must be finite and greater than near")

            state_changed = True
            data_changed = latest_cam is None or width_v != latest_cam.width or height_v != latest_cam.height or k_v != latest_cam.intrinsic_matrix

            revision, state_revision, data_revision = self._next_revisions(
                latest,
                state_changed=state_changed,
                data_changed=data_changed,
            )
            now = time.time()

            cam = CameraElement(
                id=element_id,
                type="camera",
                name=name,
                fov=fov_v,
                near=near_v,
                far=far_v,
                width=width_v,
                height=height_v,
                intrinsic_matrix=k_v,
                revision=revision,
                state_revision=state_revision,
                data_revision=data_revision,
                created_at=(latest.created_at if latest is not None else now),
                updated_at=now,
                position=(float(position[0]), float(position[1]), float(position[2])),
                rotation=(float(rotation[0]), float(rotation[1]), float(rotation[2]), float(rotation[3])),
                visible=(base_cam.visible if base_cam is not None else (latest_cam_for_defaults.visible if latest_cam_for_defaults is not None else True)),
                deleted=(base_cam.deleted if base_cam is not None else (latest_cam_for_defaults.deleted if latest_cam_for_defaults is not None else False)),
            )

            self._global_revision += 1
            self._store_sample_locked(element_id, target, cam)
            return cam

    def upsert_box3d(
        self,
        *,
        name: str,
        size: tuple[float, float, float] | list[float] | np.ndarray = (1.0, 1.0, 1.0),
        color: tuple[float, float, float] | list[float] | np.ndarray = (0.62, 0.8, 1.0),
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        element_id: str | None = None,
        static: bool = False,
        frame: int | None = None,
        timestamp: float | None = None,
    ) -> Box3DElement:
        size_v = self._normalize_positive_vec3(size, name="size")
        color_v = self._normalize_color3(color, name="color")

        with self._lock:
            if element_id is None:
                element_id = uuid.uuid4().hex

            target = self._resolve_write_target_locked(static=static, frame=frame, timestamp=timestamp)
            latest = self._elements.get(element_id)
            base = self._base_element_for_write_locked(element_id, target)
            base_box = base if isinstance(base, Box3DElement) else None
            latest_box = latest if isinstance(latest, Box3DElement) else None
            inherit_latest = target.axis is None or target.auto
            latest_box_for_defaults = latest_box if inherit_latest else None

            state_changed = (
                latest_box is None
                or size_v != latest_box.size
                or color_v != latest_box.color
                or tuple(float(x) for x in position) != latest_box.position
                or tuple(float(x) for x in rotation) != latest_box.rotation
            )
            revision, state_revision, data_revision = self._next_revisions(
                latest,
                state_changed=state_changed,
                data_changed=False,
            )
            now = time.time()

            box = Box3DElement(
                id=element_id,
                type="box3d",
                name=name,
                size=size_v,
                color=color_v,
                revision=revision,
                state_revision=state_revision,
                data_revision=data_revision,
                created_at=(latest.created_at if latest is not None else now),
                updated_at=now,
                position=(float(position[0]), float(position[1]), float(position[2])),
                rotation=(float(rotation[0]), float(rotation[1]), float(rotation[2]), float(rotation[3])),
                visible=(base_box.visible if base_box is not None else (latest_box_for_defaults.visible if latest_box_for_defaults is not None else True)),
                deleted=(base_box.deleted if base_box is not None else (latest_box_for_defaults.deleted if latest_box_for_defaults is not None else False)),
            )

            self._global_revision += 1
            self._store_sample_locked(element_id, target, box)
            return box

    def upsert_ellipsoid3d(
        self,
        *,
        name: str,
        radii: tuple[float, float, float] | list[float] | np.ndarray = (0.5, 0.5, 0.5),
        color: tuple[float, float, float] | list[float] | np.ndarray = (0.56, 0.8, 0.62),
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        element_id: str | None = None,
        static: bool = False,
        frame: int | None = None,
        timestamp: float | None = None,
    ) -> Ellipsoid3DElement:
        radii_v = self._normalize_positive_vec3(radii, name="radii")
        color_v = self._normalize_color3(color, name="color")

        with self._lock:
            if element_id is None:
                element_id = uuid.uuid4().hex

            target = self._resolve_write_target_locked(static=static, frame=frame, timestamp=timestamp)
            latest = self._elements.get(element_id)
            base = self._base_element_for_write_locked(element_id, target)
            base_ellipsoid = base if isinstance(base, Ellipsoid3DElement) else None
            latest_ellipsoid = latest if isinstance(latest, Ellipsoid3DElement) else None
            inherit_latest = target.axis is None or target.auto
            latest_ellipsoid_for_defaults = latest_ellipsoid if inherit_latest else None

            state_changed = (
                latest_ellipsoid is None
                or radii_v != latest_ellipsoid.radii
                or color_v != latest_ellipsoid.color
                or tuple(float(x) for x in position) != latest_ellipsoid.position
                or tuple(float(x) for x in rotation) != latest_ellipsoid.rotation
            )
            revision, state_revision, data_revision = self._next_revisions(
                latest,
                state_changed=state_changed,
                data_changed=False,
            )
            now = time.time()

            ellipsoid = Ellipsoid3DElement(
                id=element_id,
                type="ellipsoid3d",
                name=name,
                radii=radii_v,
                color=color_v,
                revision=revision,
                state_revision=state_revision,
                data_revision=data_revision,
                created_at=(latest.created_at if latest is not None else now),
                updated_at=now,
                position=(float(position[0]), float(position[1]), float(position[2])),
                rotation=(float(rotation[0]), float(rotation[1]), float(rotation[2]), float(rotation[3])),
                visible=(
                    base_ellipsoid.visible
                    if base_ellipsoid is not None
                    else (latest_ellipsoid_for_defaults.visible if latest_ellipsoid_for_defaults is not None else True)
                ),
                deleted=(
                    base_ellipsoid.deleted
                    if base_ellipsoid is not None
                    else (latest_ellipsoid_for_defaults.deleted if latest_ellipsoid_for_defaults is not None else False)
                ),
            )

            self._global_revision += 1
            self._store_sample_locked(element_id, target, ellipsoid)
            return ellipsoid

    def upsert_image(
        self,
        *,
        name: str,
        image_bytes: bytes,
        mime_type: str,
        width: int,
        height: int,
        channels: int,
        element_id: str | None = None,
        static: bool = False,
        frame: int | None = None,
        timestamp: float | None = None,
    ) -> ImageElement:
        if not image_bytes:
            raise ValueError("image payload is empty")
        if not isinstance(mime_type, str) or "/" not in mime_type:
            raise ValueError("mime_type must be a valid media type, e.g. image/png")
        width_i = int(width)
        height_i = int(height)
        channels_i = int(channels)
        if width_i <= 0 or height_i <= 0:
            raise ValueError("image width/height must be positive integers")
        if channels_i <= 0:
            raise ValueError("image channels must be a positive integer")

        with self._lock:
            if element_id is None:
                element_id = uuid.uuid4().hex

            target = self._resolve_write_target_locked(static=static, frame=frame, timestamp=timestamp)
            latest = self._elements.get(element_id)
            base = self._base_element_for_write_locked(element_id, target)
            base_img = base if isinstance(base, ImageElement) else None
            latest_img = latest if isinstance(latest, ImageElement) else None
            inherit_latest = target.axis is None or target.auto
            latest_img_for_defaults = latest_img if inherit_latest else None

            revision, state_revision, data_revision = self._next_revisions(
                latest,
                state_changed=False,
                data_changed=True,
            )
            now = time.time()

            img = ImageElement(
                id=element_id,
                type="image",
                name=name,
                image_bytes=bytes(image_bytes),
                mime_type=str(mime_type),
                width=width_i,
                height=height_i,
                channels=channels_i,
                revision=revision,
                state_revision=state_revision,
                data_revision=data_revision,
                created_at=(latest.created_at if latest is not None else now),
                updated_at=now,
                position=(base_img.position if base_img is not None else (latest_img_for_defaults.position if latest_img_for_defaults is not None else (0.0, 0.0, 0.0))),
                rotation=(base_img.rotation if base_img is not None else (latest_img_for_defaults.rotation if latest_img_for_defaults is not None else (0.0, 0.0, 0.0, 1.0))),
                visible=(base_img.visible if base_img is not None else (latest_img_for_defaults.visible if latest_img_for_defaults is not None else True)),
                deleted=(base_img.deleted if base_img is not None else (latest_img_for_defaults.deleted if latest_img_for_defaults is not None else False)),
            )

            self._global_revision += 1
            self._store_sample_locked(element_id, target, img)
            return img


REGISTRY = InMemoryRegistry()
