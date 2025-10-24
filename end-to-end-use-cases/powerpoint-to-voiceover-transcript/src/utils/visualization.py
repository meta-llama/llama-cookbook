"""Visualization utilities for displaying slide images and analysis results."""

import matplotlib.pyplot as plt
from PIL import Image
import math
from typing import List, Union, Tuple
from pathlib import Path


def display_slide_grid(
    image_files: List[Union[str, Path]],
    max_cols: int = 3,
    figsize_per_image: Tuple[int, int] = (4, 3)
) -> None:
    """
    Display slide images in a grid layout for Jupyter notebooks.

    Args:
        image_files: List of image file paths
        max_cols: Maximum number of columns in the grid (default: 3)
        figsize_per_image: Size of each image subplot as (width, height) (default: (4, 3))

    Example:
        # Display first 6 slides in a 3-column grid
        display_slide_grid(image_files[:6], max_cols=3, figsize_per_image=(4, 3))

        # Display all slides in a 2-column grid with larger images
        display_slide_grid(image_files, max_cols=2, figsize_per_image=(6, 4))
    """
    if not image_files:
        print("No images to display")
        return

    num_images = len(image_files)
    cols = min(max_cols, num_images)
    rows = math.ceil(num_images / cols)

    # Calculate figure size
    fig_width = cols * figsize_per_image[0]
    fig_height = rows * figsize_per_image[1]

    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))

    # Handle single row/column cases
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    for i, img_path in enumerate(image_files):
        try:
            # Load and display image
            img = Image.open(img_path)
            axes[i].imshow(img)
            axes[i].set_title(f"Slide {i+1}", fontsize=10, pad=5)
            axes[i].axis('off')
        except Exception as e:
            # Handle error case
            img_name = Path(img_path).name if hasattr(img_path, 'name') else str(img_path)
            axes[i].text(0.5, 0.5, f"Error loading\n{img_name}",
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f"Slide {i+1} (Error)", fontsize=10, pad=5)
            axes[i].axis('off')

    # Hide any unused subplots
    for i in range(num_images, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def display_slide_preview(
    image_files: List[Union[str, Path]],
    num_slides: int = 6,
    max_cols: int = 3,
    figsize_per_image: Tuple[int, int] = (4, 3)
) -> None:
    """
    Display a preview of the first N slide images.

    Args:
        image_files: List of image file paths
        num_slides: Number of slides to display (default: 6)
        max_cols: Maximum number of columns in the grid (default: 3)
        figsize_per_image: Size of each image subplot as (width, height) (default: (4, 3))

    Example:
        # Show first 6 slides
        display_slide_preview(image_files, num_slides=6)

        # Show first 8 slides in 4 columns
        display_slide_preview(image_files, num_slides=8, max_cols=4)
    """
    total_slides = len(image_files)
    slides_to_show = min(num_slides, total_slides)

    print(f"\nDisplaying first {slides_to_show} of {total_slides} slide images:")
    display_slide_grid(
        image_files[:slides_to_show],
        max_cols=max_cols,
        figsize_per_image=figsize_per_image
    )
