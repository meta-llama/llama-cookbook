import base64
import io
from multiprocessing.pool import ThreadPool
from typing import Any, Union

import yaml
from tqdm import tqdm


def map_with_progress(
    f: callable, xs: list[Any], num_threads: int = 50, show_progress: bool = True
):
    """
    Apply f to each element of xs, using a ThreadPool, and show progress.
    """
    if show_progress:
        with ThreadPool(min(num_threads, len(xs))) as pool:
            return list(tqdm(pool.imap(f, xs), total=len(xs)))
    else:
        with ThreadPool(min(num_threads, len(xs))) as pool:
            return list(pool.imap(f, xs))


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


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


def image_to_base64_url(image: Union[str, list, Image.Image]):
    if isinstance(image, str):
        # Check if the string is already base64 encoded
        if is_base64_encoded(image):
            return image
        # Otherwise, treat it as a file path
        with open(image, "rb") as img:
            img_format = image.split(".")[-1]
            b64_string = base64.b64encode(img.read()).decode("utf-8")
            return f"data:image/{img_format};base64,{b64_string}"
    elif isinstance(image, Image.Image):
        try:
            img_format = image.format.lower()
        except AttributeError:
            img_format = "png"  # Default format
        buffer = io.BytesIO()
        image.save(buffer, format=img_format)
        b64_string = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/{img_format};base64,{b64_string}"
    elif isinstance(image, list):
        return [image_to_base64_url(img) for img in image]
