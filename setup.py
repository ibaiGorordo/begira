from __future__ import annotations

from setuptools import setup  # type: ignore
from setuptools.command.build_py import build_py as _build_py  # type: ignore
from setuptools.command.sdist import sdist as _sdist  # type: ignore


class build_py(_build_py):
    def run(self):
        # We intentionally do NOT run npm here.
        # Maintainers should build the frontend beforehand (see scripts/sync_frontend_dist.py),
        # which now builds directly into src/begira/_frontend/dist.
        super().run()


class sdist(_sdist):
    def run(self):
        # Same rationale as build_py.
        super().run()


setup(
    cmdclass={
        "build_py": build_py,
        "sdist": sdist,
    }
)
