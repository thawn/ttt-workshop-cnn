# PDF to Jupyter Slides Conversion - README

## Overview
This project has been updated to include presentation slides converted from PDF files in the `slides/` directory.

## Files Converted
- `02_Deep_Learning.pdf` (8 pages) → Embedded as slides in lecture_slides.ipynb
- `CNN_Train_The_Trainer.pdf` (64 pages) → Embedded as slides in lecture_slides.ipynb

## Presentation Features
The `book/slides/lecture_slides.ipynb` notebook now includes:
- **89 total cells** (1 original header + 88 slides)
- **86 slides with embedded images** from the converted PDFs
- **RISE presentation configuration** for slideshow mode
- **Proper slideshow metadata** for each slide cell

## How to Use the Presentation

### Option 1: Using RISE (Recommended)
1. Open the notebook in Jupyter:
   ```bash
   jupyter notebook book/slides/lecture_slides.ipynb
   ```
2. Click the "slideshow" button in the toolbar (looks like a bar chart)
3. Or press `Alt+R` to start the presentation
4. Use arrow keys to navigate between slides

### Option 2: Using Jupyter Book
Since this appears to be a Jupyter Book project, you can also build the book:
```bash
jupyter-book build book/
```

### Presentation Controls
- **Arrow Keys**: Navigate slides
- **Space**: Next slide
- **Shift+Space**: Previous slide
- **ESC**: Exit presentation mode
- **?**: Show help with all keyboard shortcuts

## Technical Details
- **Images**: Converted to base64-encoded PNG format for embedding
- **Resolution**: 150 DPI for optimal quality/size balance
- **Theme**: White theme with slide transitions
- **Metadata**: Each slide has proper `slideshow.slide_type: "slide"` metadata

## File Size
The notebook is approximately 29MB due to the embedded base64 images. This is normal for presentations with many images.

## Troubleshooting
If the slideshow button doesn't appear:
1. Ensure RISE is installed: `pip install RISE`
2. Restart Jupyter notebook
3. Check that the notebook has slideshow metadata in cells