"""Core functionality for PPTX to Transcript processing."""

from .image_processing import encode_image
from .pptx_processor import extract_pptx_notes, pptx_to_images_and_notes
from .llama_client import LlamaClient
from .file_utils import check_libreoffice

__all__ = [
    "encode_image",
    "extract_pptx_notes",
    "pptx_to_images_and_notes",
    "LlamaClient",
    "check_libreoffice"
]
