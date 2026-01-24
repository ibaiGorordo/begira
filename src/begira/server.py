from __future__ import annotations

from fastapi import FastAPI

from .api import create_api_app
from .web import mount_frontend


def create_app() -> FastAPI:
    """Create the full app: API + packaged frontend."""

    app = create_api_app()
    mount_frontend(app)
    return app


# Convenience for uvicorn: `uvicorn begira.server:app`
app = create_app()
