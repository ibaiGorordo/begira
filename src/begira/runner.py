from __future__ import annotations

import contextlib
import os
import socket
import threading
import time
import webbrowser
from dataclasses import dataclass

import numpy as np
import uvicorn

from .client import BegiraClient
from .conventions import CoordinateConvention
from .registry import REGISTRY
from .server import create_app


@dataclass(frozen=True)
class BegiraServer:
    host: str
    port: int
    url: str

    def _as_client(self) -> BegiraClient:
        # Reuse HTTP endpoints for any "viewer settings" behavior.
        return BegiraClient(self.url.rstrip("/"))

    def get_coordinate_convention(self, *, timeout_s: float = 10.0) -> CoordinateConvention:
        """Return the active viewer coordinate convention for this server."""
        return self._as_client().get_coordinate_convention(timeout_s=timeout_s)

    def set_coordinate_convention(self, convention: str | CoordinateConvention, *, timeout_s: float = 10.0) -> None:
        """Set the viewer coordinate convention for this server."""
        self._as_client().set_coordinate_convention(convention, timeout_s=timeout_s)

    def log_points(
        self,
        name: str,
        positions: np.ndarray | object,
        colors: np.ndarray | None = None,
        *,
        element_id: str | None = None,
        point_size: float | None = 0.05,
    ) -> str:
        """Log (add/update) a pointcloud element."""

        if colors is None and not isinstance(positions, np.ndarray):
            pos_attr = getattr(positions, "positions", None)
            if pos_attr is not None:
                col_attr = getattr(positions, "colors", None)
                positions = pos_attr
                colors = col_attr

        pc = REGISTRY.upsert_pointcloud(
            name=name,
            positions=positions,  # type: ignore[arg-type]
            colors=colors,
            point_size=point_size,
            element_id=element_id,
        )
        return pc.id

    def log_gaussians(
        self,
        name: str,
        positions: np.ndarray | object,
        sh0: np.ndarray | None = None,
        opacity: np.ndarray | None = None,
        scales: np.ndarray | None = None,
        rotations: np.ndarray | None = None,
        *,
        element_id: str | None = None,
    ) -> str:
        """Log (add/update) a 3D Gaussian Splatting element."""

        # Convenience: allow passing a GaussianSplatData object directly.
        if sh0 is None and not isinstance(positions, np.ndarray):
            pos_attr = getattr(positions, "positions", None)
            if pos_attr is not None:
                sh0_attr = getattr(positions, "f_dc", None)
                opacity_attr = getattr(positions, "opacity", None)
                scales_attr = getattr(positions, "scales", None)
                rot_attr = getattr(positions, "rotations", None)
                positions = pos_attr
                sh0 = sh0_attr
                opacity = opacity_attr
                scales = scales_attr
                rotations = rot_attr

        pos = np.ascontiguousarray(positions, dtype=np.float32)
        n = pos.shape[0]

        if sh0 is None:
            sh0 = np.zeros((n, 3), dtype=np.float32)
        else:
            sh0 = np.ascontiguousarray(sh0, dtype=np.float32)

        if opacity is None:
            opacity = np.ones((n, 1), dtype=np.float32)
        else:
            opacity = np.ascontiguousarray(opacity, dtype=np.float32)
            if opacity.ndim == 1:
                opacity = opacity[:, np.newaxis]

        if scales is None:
            scales = np.zeros((n, 3), dtype=np.float32)
        else:
            scales = np.ascontiguousarray(scales, dtype=np.float32)

        if rotations is None:
            rotations = np.zeros((n, 4), dtype=np.float32)
            rotations[:, 0] = 1.0  # Identity quaternion [1, 0, 0, 0]
        else:
            rotations = np.ascontiguousarray(rotations, dtype=np.float32)

        gs = REGISTRY.upsert_gaussians(
            name=name,
            positions=pos,
            sh0=sh0,
            opacity=opacity,
            scales=scales,
            rotations=rotations,
            element_id=element_id,
        )
        return gs.id

    def log_ply(
        self,
        name: str,
        path: str,
        *,
        element_id: str | None = None,
        point_size: float | None = 0.05,
    ) -> str:
        """Load a `.ply` file and log it as a pointcloud element."""

        from .ply import load_ply_pointcloud

        pc = load_ply_pointcloud(path)
        return self.log_points(
            name,
            pc.positions,
            pc.colors,
            element_id=element_id,
            point_size=point_size,
        )

    def log_ply_gaussians(
        self,
        name: str,
        path: str,
        *,
        element_id: str | None = None,
    ) -> str:
        """Load a `.ply` 3DGS file and log it as a gaussians element."""

        from .ply import load_ply_gaussians

        gs = load_ply_gaussians(path)
        return self.log_gaussians(
            name,
            gs,
            element_id=element_id,
        )

    def get_viewer_settings(self, *, timeout_s: float = 10.0) -> dict:
        """Return the active viewer settings for this server."""
        return self._as_client().get_viewer_settings(timeout_s=timeout_s)


def _find_free_port(host: str) -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def _normalize_base_url(url: str) -> str:
    url = url.strip()
    if not url:
        return ""
    # Allow passing just host:port.
    if "://" not in url:
        url = "http://" + url
    return url.rstrip("/")


def _is_server_alive(base_url: str, *, timeout_s: float = 0.2) -> bool:
    """Best-effort probe to determine if a begira server is reachable."""

    # Keep deps light: httpx is already a runtime dependency.
    import httpx

    try:
        with httpx.Client(base_url=base_url, timeout=timeout_s) as client:
            r = client.get("/healthz")
            if r.status_code != 200:
                return False
            data = r.json()
            return bool(data.get("ok"))
    except Exception:
        return False


def run(
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    open_browser: bool = True,
    log_level: str = "info",
    access_log: bool = False,
    new_server: bool = False,
    connect_timeout_s: float = 0.2,
) -> BegiraServer | BegiraClient:
    """Start begira (API + built frontend) with a single Python call.

    Behavior:
    - If BEGIRA_URL is set, we *attach* to that existing server (client mode) unless
      `new_server=True`.
    - Otherwise, if `port != 0` and a server is already reachable at http://{host}:{port},
      we attach to it (client mode) unless `new_server=True`.
    - Otherwise we start a new local server (server mode) and return a `BegiraServer`.

    Notes:
    - `port=0` means "pick a free port", so there's nothing to attach to.
    - We disable Uvicorn's per-request access log by default because the frontend polls
      `/api/events` frequently and it becomes very noisy.
    """

    env_url = _normalize_base_url(os.getenv("BEGIRA_URL", ""))

    # 1) Try attaching to an explicitly provided server.
    if env_url and not new_server:
        if _is_server_alive(env_url, timeout_s=connect_timeout_s):
            if open_browser:
                webbrowser.open(env_url + "/")
            return BegiraClient(env_url)

    # 2) Try attaching to host/port if they are explicitly chosen.
    if port != 0 and not new_server:
        default_url = _normalize_base_url(f"http://{host}:{port}")
        if _is_server_alive(default_url, timeout_s=connect_timeout_s):
            if open_browser:
                webbrowser.open(default_url + "/")
            return BegiraClient(default_url)

    # 3) Start a fresh server.
    if port == 0:
        port = _find_free_port(host)

    app = create_app()

    config = uvicorn.Config(app, host=host, port=port, log_level=log_level, access_log=access_log)
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Give it a moment so a subsequent client probe doesn't race with startup.
    # (Best-effort; doesn't need to be perfect.)
    time.sleep(0.05)

    url = f"http://{host}:{port}/"
    if open_browser:
        webbrowser.open(url)

    return BegiraServer(host=host, port=port, url=url)
