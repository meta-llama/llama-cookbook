"""Transcript generation processor for PPTX slides."""

from pathlib import Path
from typing import Optional, Union

import pandas as pd
from tqdm import tqdm

from ..config.settings import get_processing_config

from ..core.llama_client import LlamaClient


class TranscriptProcessor:
    """Processor for generating transcripts from slide images and notes."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize transcript processor.

        Args:
            api_key: Llama API key. If None, will be loaded from config/environment.
        """
        self.client = LlamaClient(api_key=api_key)
        self.processing_config = get_processing_config()

    def process_single_slide(
        self,
        image_path: Union[str, Path],
        speaker_notes: str = "",
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Process a single slide to generate transcript.

        Args:
            image_path: Path to the slide image
            speaker_notes: Speaker notes for the slide
            system_prompt: Custom system prompt. If None, uses default from config.

        Returns:
            Generated transcript text
        """
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
    ) -> pd.DataFrame:
        """
        Process slides from a DataFrame containing slide information.

        Args:
            df: DataFrame with slide information (from extract_pptx_notes)
            output_dir: Directory containing slide images
            system_prompt: Custom system prompt. If None, uses default from config.

        Returns:
            DataFrame with added 'ai_transcript' column
        """
        output_dir = Path(output_dir)
        df_copy = df.copy()

        for i in tqdm(range(len(df_copy)), desc="Processing slides"):
            # Get data for current slide
            slide_filename = df_copy.iloc[i]["image_filename"]
            speaker_notes = (
                df_copy.iloc[i]["speaker_notes"]
                if pd.notna(df_copy.iloc[i]["speaker_notes"])
                else ""
            )

            image_path = output_dir / slide_filename

            # Generate transcript
            transcript = self.process_single_slide(
                image_path=image_path,
                speaker_notes=speaker_notes,
                system_prompt=system_prompt,
            )

            # Add to dataframe
            df_copy.loc[i, "ai_transcript"] = transcript

        return df_copy


def process_slides(
    df: pd.DataFrame,
    output_dir: Union[str, Path] = "slide_images",
    api_key: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> pd.DataFrame:
    """
    Legacy function for backward compatibility with notebook code.
    Process slides from a DataFrame to generate transcripts.

    Args:
        df: DataFrame with slide information (from extract_pptx_notes)
        output_dir: Directory containing slide images
        api_key: Llama API key. If None, will be loaded from config/environment.
        system_prompt: Custom system prompt. If None, uses default from config.

    Returns:
        DataFrame with added 'ai_transcript' column
    """
    processor = TranscriptProcessor(api_key=api_key)
    return processor.process_slides_dataframe(df, output_dir, system_prompt)
