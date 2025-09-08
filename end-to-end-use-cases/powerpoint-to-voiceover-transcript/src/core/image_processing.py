"""Image processing utilities for PPTX to Transcript."""

import base64
from typing import Union
from pathlib import Path


def encode_image(image_path: Union[str, Path]) -> str:
    """
    Encode an image file as a base64 string.

    Args:
        image_path: Path to the image file

    Returns:
        Base64-encoded image data as a string

    Raises:
        FileNotFoundError: If the image file doesn't exist
        IOError: If there's an error reading the image file
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except IOError as e:
        raise IOError(f"Error reading image file {image_path}: {e}")
