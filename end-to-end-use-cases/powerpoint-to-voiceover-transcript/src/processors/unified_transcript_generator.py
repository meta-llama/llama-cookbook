"""Unified transcript generation processor with optional narrative continuity."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from tqdm import tqdm

from ..config.settings import get_processing_config, get_system_prompt
from ..core.llama_client import LlamaClient


class SlideContext:
    """Simple container for slide context information."""

    def __init__(self, slide_number: int, title: str, transcript: str):
        self.slide_number = slide_number
        self.title = title
        self.transcript = transcript

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slide_number": int(self.slide_number),  # Convert to native Python int
            "title": str(self.title),  # Ensure it's a string
            "transcript": str(self.transcript),  # Ensure it's a string
        }


class UnifiedTranscriptProcessor:
    """Unified processor for generating transcripts with optional narrative continuity."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        use_narrative: bool = True,
        context_window_size: int = 5,
    ):
        """
        Initialize unified transcript processor.

        Args:
            api_key: Llama API key. If None, will be loaded from config/environment.
            use_narrative: Whether to use narrative continuity (default: True)
            context_window_size: Number of previous slides to include in context when use_narrative=True (default: 5)
        """
        self.client = LlamaClient(api_key=api_key)
        self.processing_config = get_processing_config()
        self.use_narrative = use_narrative
        self.context_window_size = context_window_size
        self.slide_contexts: List[SlideContext] = []

    def _build_context_prompt(
        self, current_slide_number: int, slide_contexts: List[SlideContext]
    ) -> str:
        """
        Build enhanced system prompt with previous slide transcripts as context.
        Only used when use_narrative=True.

        Args:
            current_slide_number: Number of the current slide being processed
            slide_contexts: List of previous slide contexts

        Returns:
            Enhanced system prompt with context
        """
        base_prompt = get_system_prompt()

        if not slide_contexts or not self.use_narrative:
            return base_prompt

        # Build context section
        context_section = "\n\n## PREVIOUS SLIDE CONTEXT\n\n"
        context_section += f"You are currently processing slide {current_slide_number} of this presentation. "
        context_section += "Here are the transcripts from the previous slides to maintain narrative continuity:\n\n"

        # Add previous slides context (use last N slides based on context window)
        recent_contexts = slide_contexts[-self.context_window_size :]

        for context in recent_contexts:
            context_section += (
                f'**Slide {context.slide_number} - "{context.title}":**\n'
            )
            context_section += f"{context.transcript}\n\n"

        # Add continuity instructions
        continuity_instructions = """
## NARRATIVE CONTINUITY REQUIREMENTS

When generating the transcript for this slide, ensure:

1. **Smooth Transitions**: Reference previous concepts when appropriate (e.g., "Building on what we discussed about...", "As we saw in the previous section...")

2. **Consistent Terminology**: Use the same terms and definitions established in previous slides

3. **Logical Flow**: Ensure this slide's content logically follows from previous slides

4. **Avoid Repetition**: Don't repeat information already covered unless it's for emphasis or summary

5. **Forward References**: If this slide sets up future content, use appropriate language (e.g., "We'll explore this further...", "This leads us to...")

6. **Contextual Awareness**: Understand where this slide fits in the overall presentation narrative

"""

        return base_prompt + context_section + continuity_instructions

    def process_single_slide(
        self,
        image_path: Union[str, Path],
        speaker_notes: str = "",
        system_prompt: Optional[str] = None,
        slide_number: Optional[int] = None,
        slide_title: str = "",
    ) -> str:
        """
        Process a single slide to generate transcript.

        Args:
            image_path: Path to the slide image
            speaker_notes: Speaker notes for the slide
            system_prompt: Custom system prompt. If None, uses default from config.
            slide_number: Number of the current slide (used for narrative continuity)
            slide_title: Title of the current slide (used for narrative continuity)

        Returns:
            Generated transcript text
        """
        if self.use_narrative and slide_number is not None:
            # Use narrative-aware processing
            enhanced_prompt = self._build_context_prompt(
                slide_number, self.slide_contexts
            )

            # Generate transcript with context
            transcript = self.client.generate_transcript(
                image_path=str(image_path),
                speaker_notes=speaker_notes,
                system_prompt=enhanced_prompt,
                stream=False,
            )

            # Create and store slide context for future slides
            slide_context = SlideContext(
                slide_number=slide_number,
                title=slide_title,
                transcript=transcript,
            )
            self.slide_contexts.append(slide_context)

            return transcript
        else:
            # Use standard processing
            return self.client.generate_transcript(
                image_path=str(image_path),
                speaker_notes=speaker_notes,
                system_prompt=system_prompt,
                stream=False,
            )

    def process_slides_dataframe(
        self,
        df: pd.DataFrame,
        output_dir: Union[str, Path],
        system_prompt: Optional[str] = None,
        save_context: bool = True,
    ) -> pd.DataFrame:
        """
        Process slides from a DataFrame containing slide information.

        Args:
            df: DataFrame with slide information (from extract_pptx_notes)
            output_dir: Directory containing slide images
            system_prompt: Custom system prompt. If None, uses default from config.
            save_context: Whether to save context information to file (only used when use_narrative=True)

        Returns:
            DataFrame with added 'ai_transcript' column and context columns if using narrative mode
        """
        output_dir = Path(output_dir)
        df_copy = df.copy()

        if self.use_narrative:
            print(f"Processing {len(df_copy)} slides with narrative continuity...")
            print(f"Using context window of {self.context_window_size} previous slides")
        else:
            print(f"Processing {len(df_copy)} slides in standard mode...")

        for i in tqdm(range(len(df_copy)), desc="Processing slides"):
            # Get data for current slide
            slide_row = df_copy.iloc[i]
            slide_number = slide_row["slide_number"]
            slide_title = slide_row.get("slide_title", "")
            slide_filename = slide_row["image_filename"]
            speaker_notes = (
                slide_row["speaker_notes"]
                if pd.notna(slide_row["speaker_notes"])
                else ""
            )

            image_path = output_dir / slide_filename

            # Generate transcript
            if self.use_narrative:
                transcript = self.process_single_slide(
                    image_path=image_path,
                    speaker_notes=speaker_notes,
                    system_prompt=system_prompt,
                    slide_number=slide_number,
                    slide_title=slide_title,
                )
                # Add context information to dataframe
                df_copy.loc[i, "context_slides_used"] = min(
                    len(self.slide_contexts) - 1, self.context_window_size
                )
            else:
                transcript = self.process_single_slide(
                    image_path=image_path,
                    speaker_notes=speaker_notes,
                    system_prompt=system_prompt,
                )

            # Add transcript to dataframe
            df_copy.loc[i, "ai_transcript"] = transcript

        # Save context information if requested and using narrative mode
        if save_context and self.use_narrative:
            self._save_context_information(output_dir)

        return df_copy

    def _save_context_information(self, output_dir: Path):
        """Save context information to files. Only used when use_narrative=True."""
        if not self.use_narrative:
            return

        context_dir = output_dir / "narrative_context"
        context_dir.mkdir(exist_ok=True)

        # Save slide contexts
        contexts_data = [context.to_dict() for context in self.slide_contexts]
        with open(context_dir / "slide_contexts.json", "w") as f:
            json.dump(contexts_data, f, indent=2)

        # Save simple summary
        summary = {
            "total_slides": len(self.slide_contexts),
            "context_window_size": self.context_window_size,
            "slide_progression": [
                {
                    "slide_number": int(
                        ctx.slide_number
                    ),  # Convert to native Python int
                    "title": str(ctx.title),  # Ensure it's a string
                }
                for ctx in self.slide_contexts
            ],
        }

        with open(context_dir / "narrative_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"Context information saved to: {context_dir}")


# Convenience functions for backward compatibility
def process_slides(
    df: pd.DataFrame,
    output_dir: Union[str, Path] = "slide_images",
    api_key: Optional[str] = None,
    system_prompt: Optional[str] = None,
    use_narrative: bool = False,
) -> pd.DataFrame:
    """
    Process slides from a DataFrame to generate transcripts.

    Args:
        df: DataFrame with slide information (from extract_pptx_notes)
        output_dir: Directory containing slide images
        api_key: Llama API key. If None, will be loaded from config/environment.
        system_prompt: Custom system prompt. If None, uses default from config.
        use_narrative: Whether to use narrative continuity (default: False for backward compatibility)

    Returns:
        DataFrame with added 'ai_transcript' column
    """
    processor = UnifiedTranscriptProcessor(api_key=api_key, use_narrative=use_narrative)
    return processor.process_slides_dataframe(df, output_dir, system_prompt)


def process_slides_with_narrative(
    df: pd.DataFrame,
    output_dir: Union[str, Path] = "slide_images",
    api_key: Optional[str] = None,
    context_window_size: int = 5,
    save_context: bool = True,
) -> pd.DataFrame:
    """
    Process slides from a DataFrame to generate transcripts with narrative continuity.

    Args:
        df: DataFrame with slide information (from extract_pptx_notes)
        output_dir: Directory containing slide images
        api_key: Llama API key. If None, will be loaded from config/environment.
        context_window_size: Number of previous slides to include in context (default: 5)
        save_context: Whether to save context information to files

    Returns:
        DataFrame with added transcript and context columns
    """
    processor = UnifiedTranscriptProcessor(
        api_key=api_key, use_narrative=True, context_window_size=context_window_size
    )
    return processor.process_slides_dataframe(df, output_dir, save_context=save_context)
