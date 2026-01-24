from __future__ import annotations

import shutil
from pathlib import Path

from setuptools import setup  # type: ignore
from setuptools.command.build_py import build_py as _build_py  # type: ignore
from setuptools.command.sdist import sdist as _sdist  # type: ignore


ROOT = Path(__file__).resolve().parent
FRONTEND_DIST = ROOT / "frontend" / "dist"
PKG_DIST = ROOT / "src" / "begira" / "_frontend" / "dist"


def _sync_frontend_into_package() -> None:
    """If a maintainer built the frontend in-repo, sync it into the packaged location.

    We intentionally do NOT run npm here.
    Regular users installing from an sdist should not need Node/npm.

    Maintainers should run:
      python scripts/sync_frontend_dist.py
    before publishing.
    """

    if not (FRONTEND_DIST / "index.html").exists():
        # Nothing to sync (likely building from sdist, or frontend not built in this checkout).
        return

    if PKG_DIST.exists():
        shutil.rmtree(PKG_DIST)
    PKG_DIST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(FRONTEND_DIST, PKG_DIST)


class build_py(_build_py):
    def run(self):
        _sync_frontend_into_package()
        super().run()


class sdist(_sdist):
    def run(self):
        _sync_frontend_into_package()
        super().run()


setup(
    cmdclass={
        "build_py": build_py,
        "sdist": sdist,
    }
)
