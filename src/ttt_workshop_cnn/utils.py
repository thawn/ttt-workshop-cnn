"""General utilities for workshop notebooks.

Functions here are intentionally lightweight, with no heavy deps.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt


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


def draw_smiley() -> np.ndarray:
    """Return a 16x16 NumPy array representing a smiley face.

    Returns
    -------
    np.ndarray
        A 10x10 array with values 0 (background) and 1 (smiley).
    """
    smiley = np.array(
        [
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
            [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    return smiley


def draw_gaussian_kernel(size: int = 3, sigma: float = 1.0) -> np.ndarray:
    """Generate a 2D Gaussian kernel using scipy.

    Parameters
    ----------
    size : int, optional
        The size of the kernel (size x size), by default 3.
    sigma : float, optional
        The standard deviation of the Gaussian, by default 1.0.

    Returns
    -------
    np.ndarray
        A 2D array representing the normalized Gaussian kernel.
    """

    # Create a temporary array to generate the kernel
    temp = np.zeros((size, size))
    temp[size // 2, size // 2] = 1

    # Apply Gaussian filter to create the kernel
    kernel = ndimage.gaussian_filter(temp, sigma=sigma)

    # Normalize the kernel
    return kernel / np.sum(kernel)


def animate_convolution(image, kernel, interval=50):
    """
    Create an animation that applies convolution step by step.

    Parameters
    ----------
    image : numpy.ndarray
        Input image to convolve.
    kernel : numpy.ndarray
        Convolution kernel to apply.
    interval : int, optional
        Animation frame interval in milliseconds, by default 50.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        Animation object showing the convolution process.
    """
    from matplotlib import animation

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(image, cmap="inferno")
    axs[0].set_title("Input Image")

    # Create a black background for kernel animation
    kernel_bg = np.zeros_like(image)
    kernel_im = axs[1].imshow(kernel_bg, cmap="inferno", vmin=0, vmax=1)
    axs[1].set_title("Kernel Position")

    out = np.zeros((image.shape[0] - kernel.shape[0] + 1, image.shape[1] - kernel.shape[1] + 1), dtype=np.float32)
    im = axs[2].imshow(out, cmap="inferno", vmin=0, vmax=1)
    axs[2].set_title("Convolved Output")

    for ax in axs[:2]:
        ax.set_xticks(np.arange(-0.5, image.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, image.shape[0], 1), minor=True)

    # shift axs[2] by one pixel to the right and down so that the output is centered
    axs[2].set_xticks(np.arange(-1.5, image.shape[1] - 1, 1), minor=True)
    axs[2].set_yticks(np.arange(-1.5, image.shape[0] - 1, 1), minor=True)

    for ax in axs:
        ax.grid(True, which="minor", alpha=0.3, linewidth=0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(which="both", size=0)

    def update(frame):
        """
        Update function for animation frames.

        Parameters
        ----------
        frame : tuple
            Tuple containing (i, j) coordinates for current kernel position.

        Returns
        -------
        list
            List of artists to be redrawn.
        """
        i, j = frame

        # Clear the output at the beginning of each animation cycle
        if i == 0 and j == 0:
            out.fill(0)

        # Update kernel position on black background
        kernel_bg_current = np.zeros_like(image)
        kernel_bg_current[i : i + kernel.shape[0], j : j + kernel.shape[1]] = kernel
        kernel_im.set_array(kernel_bg_current)

        # Update convolution output
        region = image[i : i + kernel.shape[0], j : j + kernel.shape[1]]
        out[i, j] = float(np.sum(region * kernel))
        im.set_array(out)

        return [kernel_im, im]

    frames = [(i, j) for i in range(out.shape[0]) for j in range(out.shape[1])]
    anim = animation.FuncAnimation(fig, update, frames=frames, blit=False, repeat=True, interval=interval)

    # Prevent automatic display in Jupyter notebooks
    plt.close(fig)

    return anim


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
