from __future__ import annotations

import contextlib
import socket
import threading
import webbrowser
from dataclasses import dataclass

import numpy as np
import uvicorn

from .registry import REGISTRY
from .server import create_app


@dataclass(frozen=True)
class BegiraServer:
    host: str
    port: int
    url: str

    def log_points(
        self,
        name: str,
        positions: np.ndarray,
        colors: np.ndarray | None = None,
        *,
        cloud_id: str | None = None,
        point_size: float | None = 0.05,
    ) -> str:
        """Log (add/update) a point cloud.

        - positions: array-like (N,3)
        - colors (optional): array-like (N,3), uint8 [0..255] or float [0..1]
        - point_size (optional): rendered point radius/size (in world units; default ~0.02)

        Returns the point cloud id.
        """

        pc = REGISTRY.upsert(
            name=name,
            positions=positions,
            colors=colors,
            point_size=point_size,
            cloud_id=cloud_id,
        )
        return pc.id


def _find_free_port(host: str) -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def run(
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    open_browser: bool = True,
    log_level: str = "info",
    access_log: bool = False,
) -> BegiraServer:
    """Start begira (API + built frontend) with a single Python call.

    By default we disable Uvicorn's per-request access log because the frontend polls
    `/api/events` frequently and it becomes very noisy.
    """

    if port == 0:
        port = _find_free_port(host)

    app = create_app()

    config = uvicorn.Config(app, host=host, port=port, log_level=log_level, access_log=access_log)
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    url = f"http://{host}:{port}/"
    if open_browser:
        webbrowser.open(url)

    return BegiraServer(host=host, port=port, url=url)
