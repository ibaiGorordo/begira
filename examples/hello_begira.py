import time
from pathlib import Path

import numpy as np

import begira
from begira.ply import load_ply_gaussians


def _quat_from_matrix(m: np.ndarray) -> tuple[float, float, float, float]:
    # Convert 3x3 rotation matrix to quaternion (x, y, z, w).
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
    q = np.array([x, y, z, w], dtype=np.float32)
    q /= np.linalg.norm(q) + 1e-12
    return float(q[0]), float(q[1]), float(q[2]), float(q[3])


def _look_at_quat(position: np.ndarray, target: np.ndarray, up: np.ndarray) -> tuple[float, float, float, float]:
    # Build camera world rotation where local -Z points from position -> target.
    forward = target - position
    forward /= np.linalg.norm(forward) + 1e-12
    right = np.cross(forward, up)
    if float(np.linalg.norm(right)) < 1e-8:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    right /= np.linalg.norm(right) + 1e-12
    cam_up = np.cross(right, forward)
    cam_up /= np.linalg.norm(cam_up) + 1e-12
    rot = np.stack([right, cam_up, -forward], axis=1).astype(np.float32)
    return _quat_from_matrix(rot)


def main() -> None:
    client = begira.run(port=57793)

    client.set_coordinate_convention(begira.CoordinateConvention.Z_UP)

    assets_dir = Path(__file__).resolve().parent / "assets"
    gs = load_ply_gaussians(str(assets_dir / "gaussians.ply"))
    client.log_gaussians("gaussians", gs)
    client.log_points("points", gs.positions * np.array([1, -1, 1]) + np.array([0, -5, 0]), gs.colors_rgb8, point_size=0.025)

    center = gs.positions.mean(axis=0)
    # Editor-style camera pose that clearly sees the subject.
    cam_pos_np = center + np.array([2.2, -2.2, 1.2], dtype=np.float32)
    cam_rot = _look_at_quat(cam_pos_np, center, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    cam_pos = tuple(cam_pos_np.tolist())
    client.log_camera(
        "main_camera",
        position=cam_pos,
        rotation=cam_rot,
        fov=60.0,
        near=0.01,
        far=1000.0,
    )

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
