from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from .api import create_api_app
from .web import mount_frontend


def create_app() -> FastAPI:
    """Create the full app: API + (optional) built frontend."""

    app = create_api_app()

    # If the frontend has been built, serve it. Otherwise API-only still works.
    dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
    try:
        mount_frontend(app, dist)
    except FileNotFoundError:
        pass

    return app


# Convenience for uvicorn: `uvicorn begira.server:app`
app = create_app()

