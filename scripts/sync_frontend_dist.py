from __future__ import annotations

"""Build the Vite frontend.

Historically this script built into `frontend/dist` and then copied into
`src/begira/_frontend/dist`.

The frontend build is now configured to write directly into the Python package
(`src/begira/_frontend/dist`), so there is nothing to sync/copy anymore.
For Git installs (`pip install begira@git+...`) this built directory should be
committed so users do not need npm installed locally.

Usage:
  python scripts/sync_frontend_dist.py
"""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT / "frontend"
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

    if not (PKG_DIST / "index.html").exists():
        raise SystemExit("frontend build did not produce src/begira/_frontend/dist/index.html")

    print(f"Built frontend into {PKG_DIST}")


if __name__ == "__main__":
    main()
