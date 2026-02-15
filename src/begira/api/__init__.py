from __future__ import annotations

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from ..core.elements import CameraElement
from ..core.registry.service import REGISTRY
from ..core.viewer_settings import VIEWER_SETTINGS
from .routes.elements import mount_elements_api


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
    def events() -> dict:
        # Minimal polling endpoint + lightweight viewer commands.
        return {
            "globalRevision": REGISTRY.global_revision(),
            "viewerCommands": {
                "openCameraView": VIEWER_SETTINGS.get_open_camera_view_request(),
            },
        }

    @app.post("/api/reset")
    def reset_project() -> dict[str, bool]:
        REGISTRY.reset()
        return {"ok": True}

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

    @app.post("/api/viewer/open-camera-view")
    def open_camera_view(body: dict) -> dict:
        camera_id = str(body.get("cameraId", "")).strip()
        if not camera_id:
            raise HTTPException(status_code=400, detail="cameraId is required")
        elem = REGISTRY.get_element(camera_id)
        if elem is None:
            raise HTTPException(status_code=404, detail=f"Unknown element: {camera_id}")
        if not isinstance(elem, CameraElement):
            raise HTTPException(status_code=400, detail="cameraId must reference a camera element")
        try:
            cmd = VIEWER_SETTINGS.request_open_camera_view(camera_id)
        except ValueError as ex:
            raise HTTPException(status_code=400, detail=str(ex))
        return {"ok": True, **cmd}

    return app
