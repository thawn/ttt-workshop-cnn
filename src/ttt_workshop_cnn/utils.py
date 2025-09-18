"""General utilities for workshop notebooks.

Functions here are intentionally lightweight, with no heavy deps.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
from scipy import signal, ndimage
import matplotlib.pyplot as plt

plt.rcParams["svg.fonttype"] = "none"  # Keep text as text in SVGs

output_path = Path("output")
output_path.mkdir(exist_ok=True, parents=True)


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


def draw_sobel_kernel() -> np.ndarray:
    """Return a 3x3 Sobel kernel for horizontal edge detection.

    Returns
    -------
    np.ndarray
        A 3x3 array representing the Sobel kernel.
    """
    return np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)


def draw_laplacian_kernel() -> np.ndarray:
    """Return a 3x3 Laplacian kernel for edge detection.

    Returns
    -------
    np.ndarray
        A 3x3 array representing the Laplacian kernel.
    """
    return np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)


def add_axis_grid(ax: plt.Axes, pad=0.0, offset: float = 0.0):
    """Add a grid to a matplotlib axis.

    Parameters
    ----------
    ax : plt.Axes
        The axis to add the grid to.

    """
    if offset >= 0:
        start_offset = -offset
        end_offset = 1
    else:
        start_offset = 0
        end_offset = 1 - offset
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_xticks(np.arange(xlim[0] - pad + start_offset, xlim[1] + pad + end_offset, 1), minor=True)
    ax.set_yticks(np.arange(ylim[1] - pad + start_offset, ylim[0] + pad + end_offset, 1), minor=True)
    ax.grid(True, which="minor", alpha=0.3, linewidth=0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(which="both", size=0)


def animate_convolution(image, kernel, stride=1, interval=50):
    """
    Create an animation that applies convolution step by step.

    Parameters
    ----------
    image : numpy.ndarray
        Input image to convolve.
    kernel : numpy.ndarray
        Convolution kernel to apply.
    stride : int, optional
        Stride for convolution operation, by default 1.
    interval : int, optional
        Animation frame interval in milliseconds, by default 50.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        Animation object showing the convolution process.
    """
    from matplotlib import animation

    fig, axs = plt.subplots(1, 3, figsize=(7, 2.3))
    axs[0].imshow(image, cmap="inferno")
    axs[0].set_title("Input Image")

    # Create a black background for kernel animation
    kernel_bg = np.zeros_like(image)
    kernel_im = axs[1].imshow(kernel_bg, cmap="inferno", vmin=0, vmax=kernel.max())
    axs[1].set_title("Kernel Position")

    # Calculate output size with stride
    out_h = (image.shape[0] - kernel.shape[0]) // stride + 1
    out_w = (image.shape[1] - kernel.shape[1]) // stride + 1
    out = np.zeros((out_h, out_w), dtype=np.float32)
    im = axs[2].imshow(out, cmap="inferno", vmin=image.min(), vmax=image.max())
    axs[2].set_title("Convolved Map")

    for ax in axs[:2]:
        add_axis_grid(ax)

    # shift axs[2] by one pixel to the right and down so that the output is centered
    add_axis_grid(axs[2], pad=1 + (image.shape[0] - image.shape[0] // stride) // 2)

    fig.tight_layout()

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
        # Clear any existing patches
        for patch in axs[0].patches:
            patch.remove()

        # Add dark transparent rectangle to show convolution region
        rect = plt.Rectangle(
            (j - 0.5, i - 0.5),
            kernel.shape[1],
            kernel.shape[0],
            linewidth=2,
            edgecolor="cyan",
            facecolor="black",
            alpha=0.3,
        )
        axs[0].add_patch(rect)

        # Update kernel position on black background
        kernel_bg_current = np.zeros_like(image)
        kernel_bg_current[i : i + kernel.shape[0], j : j + kernel.shape[1]] = kernel
        kernel_im.set_array(kernel_bg_current)

        # Update convolution output
        region = image[i : i + kernel.shape[0], j : j + kernel.shape[1]]
        out_i, out_j = i // stride, j // stride
        out[out_i, out_j] = float(np.sum(region * kernel))
        im.set_array(out)

        return [kernel_im, im, rect]

    frames = [
        (i, j)
        for i in range(0, image.shape[0] - kernel.shape[0] + 1, stride)
        for j in range(0, image.shape[1] - kernel.shape[1] + 1, stride)
    ]
    anim = animation.FuncAnimation(fig, update, frames=frames, blit=False, repeat=True, interval=interval)
    # Save animation as MP4 if requested
    filename = f"convolution_animation{stride}_{interval}ms"
    try:
        # Try to save as MP4 using ffmpeg writer
        writer = animation.FFMpegWriter(fps=1000 // interval, bitrate=1800)
        anim.save(output_path / f"{filename}.mp4", writer=writer)
    except Exception:
        # Fallback: try pillow writer for GIF
        try:
            anim.save(output_path / f"{filename}.gif", writer="pillow", fps=1000 // interval)
        except Exception:
            # If both fail, just continue without saving
            pass
    # Prevent automatic display in Jupyter notebooks
    plt.close(fig)

    return anim


def plot_input_kernel_output(
    image: np.ndarray,
    kernel: np.ndarray,
    axs: Optional[Iterable[plt.Axes]] = None,
    titles: Iterable[str] = ("Input Image", "Kernel", "Convolved Map"),
) -> Tuple[plt.Figure, Iterable[plt.Axes]]:
    """Plot the input image, kernel, and convolved output side by side.

    Parameters
    ----------
    image : np.ndarray
        Input image to convolve.
    kernel : np.ndarray
        Convolution kernel to apply.
    axs : Iterable[plt.Axes], optional
        Pre-existing axes to use for plotting. If ``None``, new axes will be created.
    titles : Iterable[str], optional
        Titles for each panel. Default: ("Input Image", "Kernel", "Convolved Map").

    Returns
    -------
    Tuple[plt.Figure, Iterable[plt.Axes]]
        The figure and axes containing the plots.
    """
    if axs is None:
        fig, axs = plt.subplots(1, 3, figsize=(7, 2.3))
    else:
        fig = axs[0].figure

    axs[0].imshow(image, cmap="inferno")
    axs[0].set_title(titles[0])
    add_axis_grid(axs[0])

    axs[1].imshow(kernel, cmap="inferno")
    axs[1].set_title(titles[1])
    add_axis_grid(axs[1], pad=6, offset=1.0)

    convolved = signal.correlate(image, kernel, mode="valid")

    axs[2].imshow(convolved, cmap="inferno")
    axs[2].set_title(titles[2])
    add_axis_grid(axs[2], pad=1)
    fig.tight_layout()
    # Save figure as SVG
    fig.savefig(output_path / f"input_kernel_output_{'_'.join(titles)}.svg", format="svg", bbox_inches="tight")
    return fig, axs


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


def plot_three_kernels(
    image: np.ndarray,
    kernels: Iterable[np.ndarray],
    titles: Iterable[str] = ("Gaussian", "Laplacian", "Sobel"),
    axs: Optional[Iterable[plt.Axes]] = None,
) -> Tuple[plt.Figure, Iterable[plt.Axes]]:
    """Plot three kernels side by side.

    Parameters
    ----------
    image : np.ndarray
        Input image to convolve.
    kernels : Iterable[np.ndarray]
        Three kernels to convolve with the image.
    titles : Iterable[str], optional
        Titles for each kernel, by default ("Gaussian", "Laplacian", "Sobel").
    axs : Iterable[plt.Axes], optional
        Pre-existing axes to use for plotting. If ``None``, new axes will be created.

    Returns
    -------
    Tuple[plt.Figure, Iterable[plt.Axes]]
        The figure and axes containing the plots.
    """
    if axs is None:
        layout = [[f"image{i}", f"kernel{i}", f"output{i}"] for i in range(len(kernels))]
        fig, axes_dict = plt.subplot_mosaic(layout, figsize=(5, 5))
        axs = list(axes_dict.values())
    else:
        fig = axs[0].figure
    # fig.suptitle(
    #     f"Convolution of image with shape {(1,) + image.shape} with {len(kernels)} Kernels\nresults in {(len(kernels), image.shape[0] - 2, image.shape[1] - 2)} Maps"
    # )
    for i, (kernel, title) in enumerate(zip(kernels, titles)):
        start_idx = i * 3
        plot_input_kernel_output(
            image,
            kernel,
            titles=(f"Input Image ", f"{title} Kernel", f"{title} Map"),
            axs=axs[start_idx : start_idx + 3],
        )
    # draw arrows instead of redundant images
    for ax in [axs[3], axs[6]]:
        ax.remove()
    plt.tight_layout()
    fig.savefig(output_path / f"input_kernel_output_{'_'.join(titles)}.svg", format="svg", bbox_inches="tight")
    return fig, axs


def plot_receptive_field(
    image: np.ndarray, kernel_layers: int = 2, kernel_sizes: Tuple[int] = (3, 5)
) -> Tuple[plt.Figure, Iterable[plt.Axes]]:

    # draw the smiley image
    num_kernels = len(kernel_sizes)
    images = [f"image"] * num_kernels
    layout = []
    for kernel_size in kernel_sizes:
        kernels_output = []
        for n in range(kernel_layers):
            kernels_output += [f"kernel{kernel_size}_{n}", f"output{kernel_size}_{n}"]
        layout.append(images + kernels_output)
    height = 7 / (1 + 2 * kernel_layers / num_kernels)
    fig, axs = plt.subplot_mosaic(layout, figsize=(7, height))
    axs["image"].imshow(image, cmap="inferno")
    add_axis_grid(axs["image"])

    rects = []

    # draw the kernels
    y_pad = -0.5
    for i, kernel_size in enumerate(kernel_sizes):
        need_padding = image.shape[1] // num_kernels - kernel_size
        pad = (need_padding) // 2
        offset = 1 if need_padding % 2 == 1 else 0
        y_pad += pad + offset
        for n in range(kernel_layers):
            receptive_field = (kernel_size - 1) * (kernel_layers - n) + 1
            kernel = draw_gaussian_kernel(size=kernel_size, sigma=1)
            if n < 1:
                # receptive_field -= kernel_size - 1
                rect_ax = f"image"
                convolved = ndimage.convolve(image, kernel, mode="constant", cval=0.0)
                annotation = "Receptive\nField"
            else:
                convolved = ndimage.convolve(convolved, kernel, mode="constant", cval=0.0)
                annotation = ""
                rect_ax = f"output{kernel_size}_{n-1}"
            x_pos = image.shape[1] // 2 - receptive_field / 2
            y_pos = y_pad - (receptive_field - kernel_size) // 2 + i * kernel_size
            axs[f"kernel{kernel_size}_{n}"].imshow(kernel, cmap="inferno")
            add_axis_grid(axs[f"kernel{kernel_size}_{n}"], pad=pad, offset=offset)
            axs[f"output{kernel_size}_{n}"].imshow(convolved, cmap="inferno")
            add_axis_grid(axs[f"output{kernel_size}_{n}"])
            rect = plt.Rectangle(
                (x_pos, y_pos),
                receptive_field,
                receptive_field,
                linewidth=2,
                edgecolor="cyan",
                facecolor="none",
            )
            axs[rect_ax].add_patch(rect)
            text = axs[rect_ax].annotate(
                annotation,
                xy=(x_pos + receptive_field / 2, y_pos),
                xytext=(x_pos + receptive_field / 2, y_pos - 0.2),
                color="cyan",
                fontsize=8,
                ha="center",
                va="bottom",
            )
            text.set_bbox(dict(facecolor="black", alpha=0.5, edgecolor="none", pad=0.5))
            rects.append(rect)

    fig.tight_layout(pad=0.0)
    fig.savefig(
        output_path / f"receptive_field_layers{kernel_layers}_kernel_sizes{kernel_sizes}.svg",
        format="svg",
        bbox_inches="tight",
    )


def draw_lines(rect, kernel3, fig, axs):
    from matplotlib.patches import ConnectionPatch

    # draw lines connecting the corners of the rectangle to the corners of the kernel
    rect_corners = [
        (rect.get_x(), rect.get_y()),  # bottom-left
        (rect.get_x() + rect.get_width(), rect.get_y()),  # bottom-right
        (rect.get_x(), rect.get_y() + rect.get_height()),  # top-left
        (rect.get_x() + rect.get_width(), rect.get_y() + rect.get_height()),
    ]  # top-right

    kernel_corners = [
        (-0.5, -0.5),
        (kernel3.shape[1] - 0.5, -0.5),
        (-0.5, kernel3.shape[0] - 0.5),
        (kernel3.shape[1] - 0.5, kernel3.shape[0] - 0.5),
    ]

    for rect_corner, kernel_corner in zip(rect_corners, kernel_corners):
        con = ConnectionPatch(
            rect_corner,
            kernel_corner,
            "data",
            "data",
            axesA=axs[0],
            axesB=axs[1],
            color="cyan",
            linestyle="--",
            alpha=0.7,
        )
        fig.add_artist(con)
