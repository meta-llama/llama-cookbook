"""
PPTX to Transcript - A tool for converting PowerPoint presentations to AI-generated transcripts.

This package provides functionality to:
- Extract speaker notes and slide content from PPTX files
- Convert slides to images
- Generate AI-powered transcripts using Llama models
"""

__version__ = "0.1.0"
__author__ = "Yuce Dincer"

from .core.pptx_processor import extract_pptx_notes, pptx_to_images_and_notes
from .processors.unified_transcript_generator import (
    UnifiedTranscriptProcessor,
    process_slides,
    process_slides_with_narrative
)
from .config.settings import load_config

__all__ = [
    "extract_pptx_notes",
    "pptx_to_images_and_notes",
    "UnifiedTranscriptProcessor",
    "process_slides",
    "process_slides_with_narrative",
    "load_config"
]
