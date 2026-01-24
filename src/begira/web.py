from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


def _mount_dist(app: FastAPI, *, dist_root: Path) -> None:
    index_path = dist_root / "index.html"
    assets_path = dist_root / "assets"

    if not index_path.exists():
        raise FileNotFoundError(str(index_path))

    if assets_path.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_path)), name="assets")

    @app.get("/", include_in_schema=False)
    @app.get("/{path:path}", include_in_schema=False)
    def _spa_index(path: str):  # noqa: ARG001
        return FileResponse(str(index_path))


def mount_frontend(app: FastAPI) -> None:
    """Serve the built frontend.

    We support two locations:

    1) Installed wheel: packaged assets under `begira/_frontend/dist/`.
    2) Editable/source checkout: `frontend/dist/` in the repository.

    In both cases this serves the *full* Vite build output (no fallback UI).
    """

    # 1) Preferred: packaged wheel assets.
    try:
        from importlib import resources as importlib_resources

        dist_root = importlib_resources.files("begira._frontend").joinpath("dist")
        with importlib_resources.as_file(dist_root) as dist_root_path:
            dist_root_path = Path(dist_root_path)
            # Only treat as present if index.html exists.
            if (dist_root_path / "index.html").exists():
                _mount_dist(app, dist_root=dist_root_path)
                return
    except ModuleNotFoundError:
        pass

    # 2) Editable/source: repo frontend/dist.
    repo_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
    if (repo_dist / "index.html").exists():
        _mount_dist(app, dist_root=repo_dist)
        return

    raise FileNotFoundError(
        "begira frontend assets are missing. Expected either packaged assets at "
        "begira/_frontend/dist (wheel install) OR a built repo frontend at frontend/dist. "
        "Run: (cd frontend && npm install && npm run build)"
    )
