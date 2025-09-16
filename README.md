# Jupyter Book for CNN Train-the-Trainer

This repo contains a Jupyter Book with notebooks runnable in Google Colab. Use the Open in Colab badges at the top of each notebook.

## Link to the book

[https://thawn.github.io/ttt-workshop-cnn/](https://thawn.github.io/ttt-workshop-cnn/)

## Purpose

The purpose of this repo is twofold:

* A hands-on workshop to teach how convolutional neural networks (CNNs) work under the hood, and how to train them for image segmentation tasks.
* A comprehensive set of materials for instructors to use in their own teaching, giving examples of classical lecture slides, flipped classroom lessons, and pair programming exercises.

## Prerequisites

* Basic Python experience, including Jupyter notebooks.
* Basic machine learning experience is helpful but not required.

## Setup (local)

Requires Python 3.9+ (recommended: a virtual environment).

```bash
# 1) Create & activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate
python -m pip install uv

# 2) Install the package (includes notebook/runtime deps like numpy, torch, ipykernel)
uv pip install -e .

# 3) Optional: developer tools (linting, testing)
uv pip install -e .[dev]

# 4) Optional: build the Jupyter Book site
uv pip install -e .[book]
jupyter-book build book/
```

Notes
- Torch CPU wheels are installed by default. GPU acceleration may require a different wheel/index per your platform—see PyTorch.org for instructions.
- After installing the dev extra, you can run tests with `pytest`.


## Run notebooks in Google Colab

- Click the badge at the top of a notebook to open it in Colab.
- In Colab, set Runtime -> Change runtime type -> T4 GPU (optional for U-Net training).
- First cell installs dependencies for that notebook.

## Preview the built site locally (optional)

Open `book/_build/html/index.html` after a successful build.

## Publish on GitHub Pages

This repo includes a GitHub Actions workflow `.github/workflows/jupyterbook.yml` that builds and publishes on pushes to `main`. Enable Pages in repo settings -> Pages -> Source: `GitHub Actions`.

## Data

The U-Net exercise uses BBBC039. The notebook downloads a small subset automatically for class-time constraints.

## License and attribution

- Apache 2.0
