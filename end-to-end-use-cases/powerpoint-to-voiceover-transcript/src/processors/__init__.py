"""Processing modules for PPTX to Transcript."""

from .unified_transcript_generator import (
    UnifiedTranscriptProcessor,
    process_slides,
    process_slides_with_narrative
)

__all__ = [
    "UnifiedTranscriptProcessor",
    "process_slides",
    "process_slides_with_narrative"
]
