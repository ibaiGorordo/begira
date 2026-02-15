from __future__ import annotations

"""Element-oriented registry operations.

The concrete implementation remains in `service.py` while call sites migrate.
"""

from .service import InMemoryRegistry, REGISTRY

__all__ = ["InMemoryRegistry", "REGISTRY"]
