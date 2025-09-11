# Jupyter Book for CNN Train-the-Trainer

This repo contains a Jupyter Book with notebooks runnable in Google Colab. Use the Open in Colab badges at the top of each notebook.

## Structure

- `book/_config.yml` and `book/_toc.yml`: Jupyter Book configuration and table of contents
- `book/intro.ipynb`: Course home and navigation
- `book/slides/lecture_slides.ipynb`: 20-min slide deck (RISE-compatible in Colab)
- `book/exercises/*`: Hands-on exercises
- `book/self_teach/*`: Self-teaching notebooks

## Run notebooks in Google Colab

- Click the badge at the top of a notebook to open it in Colab.
- In Colab, set Runtime -> Change runtime type -> T4 GPU (optional for U-Net training).
- First cell installs dependencies for that notebook.

## Build locally (optional)

You can build the book locally if you want to preview:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip jupyter-book
jupyter-book build book/
```

Open `_build/html/index.html`.

## Publish on GitHub Pages

This repo includes a GitHub Actions workflow `.github/workflows/jupyterbook.yml` that builds and publishes on pushes to `main`. Enable Pages in repo settings -> Pages -> Source: `GitHub Actions`.

## Data

The U-Net exercise uses BBBC039. The notebook downloads a small subset automatically for class-time constraints.

## License and attribution

- Slides and notebooks: CC-BY 4.0 (update as needed)
- Code examples: MIT (update as needed)
