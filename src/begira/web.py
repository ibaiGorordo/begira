from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


def mount_frontend(app: FastAPI, dist_dir: Path) -> None:
    """Serve the built frontend (Vite dist/) from the same FastAPI app."""

    dist_dir = dist_dir.resolve()
    index_html = dist_dir / "index.html"
    assets_dir = dist_dir / "assets"

    if not index_html.exists():
        raise FileNotFoundError(
            f"Frontend not built. Expected {index_html}. "
            "Run: (cd frontend && npm install && npm run build)"
        )

    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/", include_in_schema=False)
    @app.get("/{path:path}", include_in_schema=False)
    def _spa_fallback(path: str):  # noqa: ARG001
        return FileResponse(str(index_html))

