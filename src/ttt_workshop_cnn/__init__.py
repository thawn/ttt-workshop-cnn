"""Top-level package for TTT Workshop CNN utilities.

This package provides small, self-contained helper functions used across
the workshop notebooks (seeding, small image ops, and paths).

Notes
-----
- Keep functions minimal and dependency-light.
- Prefer pure functions with clear inputs/outputs.
"""

from ._version import __version__
from .utils import ensure_seed, normalize_minmax, project_root

__all__ = [
    "__version__",
    "ensure_seed",
    "normalize_minmax",
    "project_root",
]
