from __future__ import annotations

"""Build the Vite frontend and copy the dist output into the Python package.

This is meant for maintainers before publishing to PyPI.
It makes sure the repository contains `src/begira/_frontend/dist`, so:
- sdists can be installed without Node/npm
- wheels always include the built UI

Usage:
  python scripts/sync_frontend_dist.py
"""

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT / "frontend"
FRONTEND_DIST = FRONTEND_DIR / "dist"
PKG_DIST = ROOT / "src" / "begira" / "_frontend" / "dist"


def run(cmd: list[str], *, cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    if not FRONTEND_DIR.exists():
        raise SystemExit("frontend/ directory not found")

    lock = FRONTEND_DIR / "package-lock.json"
    if lock.exists():
        run(["npm", "ci"], cwd=FRONTEND_DIR)
    else:
        run(["npm", "install"], cwd=FRONTEND_DIR)

    run(["npm", "run", "build"], cwd=FRONTEND_DIR)

    if not (FRONTEND_DIST / "index.html").exists():
        raise SystemExit("frontend build did not produce dist/index.html")

    if PKG_DIST.exists():
        shutil.rmtree(PKG_DIST)
    PKG_DIST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(FRONTEND_DIST, PKG_DIST)

    print(f"Synced {FRONTEND_DIST} -> {PKG_DIST}")


if __name__ == "__main__":
    main()
