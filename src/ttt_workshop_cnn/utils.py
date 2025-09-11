"""General utilities for workshop notebooks.

Functions here are intentionally lightweight, with no heavy deps.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Iterable, Optional, Tuple


def ensure_seed(seed: int = 42) -> int:
    """Seed common RNGs for reproducibility.

    This seeds Python's ``random`` and, if available, NumPy and PyTorch.

    Parameters
    ----------
    seed : int, optional
        The integer seed to use, by default ``42``.

    Returns
    -------
    int
        The seed that was set (echoed back for convenience).

    Notes
    -----
    - If NumPy is installed, ``numpy.random.default_rng(seed)`` and the legacy
      ``numpy.random.seed(seed)`` will be set.
    - If PyTorch is installed, ``torch.manual_seed(seed)`` and CUDA seeds will
      be set when CUDA is available.
    """

    random.seed(seed)

    # Optional: numpy
    try:  # pragma: no cover - optional dependency
        import numpy as np  # type: ignore

        np.random.seed(seed)
        # Creating a RNG instance encourages modern APIs
        _ = np.random.default_rng(seed)
    except Exception:
        pass

    # Optional: torch
    try:  # pragma: no cover - optional dependency
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
    except Exception:
        pass

    return seed


def normalize_minmax(x, min_val: Optional[float] = None, max_val: Optional[float] = None):
    """Normalize a numeric array-like to the [0, 1] range.

    Parameters
    ----------
    x : array-like
        Input numeric data. Supports NumPy arrays, PyTorch tensors, or similar.
    min_val : float, optional
        Minimum value for normalization. If ``None``, use ``x.min()``.
    max_val : float, optional
        Maximum value for normalization. If ``None``, use ``x.max()``.

    Returns
    -------
    array-like
        Normalized data with the same type as the input.

    Raises
    ------
    ValueError
        If ``max_val`` equals ``min_val`` (would cause divide-by-zero).
    """

    # Handle numpy & torch without importing them at module level.
    np = None
    torch = None
    try:  # pragma: no cover - optional dep detection
        import numpy as _np  # type: ignore

        np = _np
    except Exception:
        pass
    try:  # pragma: no cover - optional dep detection
        import torch as _torch  # type: ignore

        torch = _torch
    except Exception:
        pass

    # Determine min/max
    if min_val is None:
        if torch is not None and hasattr(x, "min") and isinstance(x, torch.Tensor):
            min_val = float(x.min().item())
        elif np is not None and hasattr(x, "min"):
            min_val = float(x.min())
        else:
            # Generic fallback for sequences
            min_val = float(min(x))

    if max_val is None:
        if torch is not None and hasattr(x, "max") and isinstance(x, torch.Tensor):
            max_val = float(x.max().item())
        elif np is not None and hasattr(x, "max"):
            max_val = float(x.max())
        else:
            max_val = float(max(x))

    if max_val == min_val:
        raise ValueError("min_val and max_val must differ for normalization.")

    # Compute normalized
    if torch is not None and isinstance(x, torch.Tensor):
        return (x - min_val) / (max_val - min_val)
    if np is not None and hasattr(x, "astype"):
        return (x - min_val) / (max_val - min_val)

    # Fallback: list/tuple -> list of floats
    return [(xi - min_val) / (max_val - min_val) for xi in x]


def project_root(start: Optional[os.PathLike[str] | str] = None) -> Path:
    """Return the project root path (directory containing pyproject.toml).

    Parameters
    ----------
    start : path-like, optional
        Starting directory for the search. Defaults to the current working
        directory.

    Returns
    -------
    pathlib.Path
        The absolute path to the repository/project root.
    """

    cur = Path(start or os.getcwd()).resolve()
    for p in [cur, *cur.parents]:
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p
    return cur
