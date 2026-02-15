from __future__ import annotations

"""Registry state compatibility module.

This module exists to make the registry architecture explicit during the
incremental refactor. Core state is currently owned by `service.py`.
"""

from .service import InMemoryRegistry, REGISTRY

__all__ = ["InMemoryRegistry", "REGISTRY"]
