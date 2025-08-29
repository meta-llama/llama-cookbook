import base64
import json
from pathlib import Path
from typing import Dict, Union

import yaml

from PIL import Image


def is_base64_encoded(s: str) -> bool:
    """Check if a string is already base64 encoded."""
    try:
        # Basic character check - base64 only contains these characters
        if not all(
            c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
            for c in s
        ):
            return False

        # Try to decode - if it fails, it's not valid base64
        decoded = base64.b64decode(s, validate=True)

        # Re-encode and compare - if they match, it was valid base64
        re_encoded = base64.b64encode(decoded).decode("utf-8")
        return s == re_encoded or s == re_encoded.rstrip(
            "="
        )  # Handle padding differences
    except Exception:
        return False


def image_to_base64(image: Union[str, list, Image.Image]):
    if isinstance(image, str):
        # Check if the string is already base64 encoded
        if is_base64_encoded(image):
            return image
        # Otherwise, treat it as a file path
        with open(image, "rb") as img:
            return base64.b64encode(img.read()).decode("utf-8")
    elif isinstance(image, Image.Image):
        return base64.b64encode(image.tobytes()).decode("utf-8")
    elif isinstance(image, list):
        return [image_to_base64(img) for img in image]


def read_config(config_path: str) -> Dict:
    """
    Read the configuration file (supports both JSON and YAML formats).

    Args:
        config_path: Path to the configuration file

    Returns:
        dict: Configuration parameters

    Raises:
        ValueError: If the file format is not supported
        ImportError: If the required package for the file format is not installed
    """
    file_extension = Path(config_path).suffix.lower()

    with open(config_path, "r") as f:
        if file_extension in [".json"]:
            config = json.load(f)
        elif file_extension in [".yaml", ".yml"]:
            config = yaml.safe_load(f)
        else:
            raise ValueError(
                f"Unsupported config file format: {file_extension}. "
                f"Supported formats are: .json, .yaml, .yml"
            )

    return config
