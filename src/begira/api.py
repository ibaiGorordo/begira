from __future__ import annotations

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .registry import REGISTRY
from .viewer_settings import VIEWER_SETTINGS
from .elements_api import mount_elements_api


def create_api_app() -> FastAPI:
    app = FastAPI(title="begira", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    mount_elements_api(app)

    @app.get("/healthz")
    def healthz() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/api/events")
    def events() -> dict[str, int]:
        # Minimal polling endpoint.
        return {"globalRevision": REGISTRY.global_revision()}

    @app.get("/api/viewer/settings")
    def get_viewer_settings() -> dict:
        s = VIEWER_SETTINGS.get()
        return {"coordinateConvention": s.coordinate_convention}

    @app.patch("/api/viewer/settings")
    def update_viewer_settings(body: dict) -> dict:
        # Supported:
        # - coordinateConvention: 'rh-y-up' | 'rh-z-up'
        updated = VIEWER_SETTINGS.get()

        if "coordinateConvention" in body:
            try:
                updated = VIEWER_SETTINGS.set_coordinate_convention(body.get("coordinateConvention"))
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))

        if "coordinateConvention" not in body:
            raise HTTPException(status_code=400, detail="Missing field: coordinateConvention")

        return {
            "ok": True,
            "coordinateConvention": updated.coordinate_convention,
        }

    return app
