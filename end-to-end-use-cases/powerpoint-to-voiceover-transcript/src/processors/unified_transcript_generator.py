"""Unified transcript generation processor with optional narrative continuity."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from tqdm import tqdm

from ..config.settings import get_processing_config, get_system_prompt, is_knowledge_enabled
from ..core.groq_client import GroqClient


logger = logging.getLogger(__name__)


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
        knowledge_base_dir: Optional[str] = None,
        enable_knowledge: Optional[bool] = None,
    ):
        """
        Initialize unified transcript processor.

        Args:
            api_key: Llama API key. If None, will be loaded from config/environment.
            use_narrative: Whether to use narrative continuity (default: True)
            context_window_size: Number of previous slides to include in context when use_narrative=True (default: 5)
            knowledge_base_dir: Path to knowledge base directory (optional)
            enable_knowledge: Override knowledge base enable setting (optional)
        """
        self.client = GroqClient(api_key=api_key)
        self.processing_config = get_processing_config()
        self.use_narrative = use_narrative
        self.context_window_size = context_window_size
        self.slide_contexts: List[SlideContext] = []

        # Knowledge base integration
        self.knowledge_manager = None
        self.context_manager = None
        self.enable_knowledge = enable_knowledge if enable_knowledge is not None else is_knowledge_enabled()

        if self.enable_knowledge:
            self._initialize_knowledge_components(knowledge_base_dir)
    def _initialize_knowledge_components(self, knowledge_base_dir: Optional[str] = None) -> None:
        """Initialize knowledge base components with error handling."""
        try:
            from ..knowledge.faiss_knowledge import FAISSKnowledgeManager
            from ..knowledge.context_manager import ContextManager
            from ..config.settings import load_config

            # Get configuration
            config = load_config()
            knowledge_config = config.get('knowledge', {})

            # Get knowledge base directory from config if not provided
            if knowledge_base_dir is None:
                knowledge_base_dir = knowledge_config.get('knowledge_base_dir', 'knowledge_base')

            # Ensure we have a valid path
            if not knowledge_base_dir:
                raise ValueError("Knowledge base directory not specified")

            # Get FAISS configuration
            vector_config = knowledge_config.get('vector_store', {})
            embedding_config = knowledge_config.get('embedding', {})

            # Initialize FAISS knowledge manager with configuration
            self.knowledge_manager = FAISSKnowledgeManager(
                knowledge_base_dir=knowledge_base_dir,
                index_type=vector_config.get('index_type', 'flat'),
                embedding_model=embedding_config.get('model_name', 'all-MiniLM-L6-v2'),
                use_gpu=vector_config.get('use_gpu', False)
            )
            self.knowledge_manager.initialize()

            # Initialize context manager
            self.context_manager = ContextManager()

            logger.info("FAISS knowledge base components initialized successfully")

        except Exception as e:
            logger.warning(f"Failed to initialize knowledge base: {e}")
            # Graceful degradation - disable knowledge features
            self.enable_knowledge = False
            self.knowledge_manager = None
            self.context_manager = None

    def _retrieve_knowledge_chunks(self, slide_content: str, speaker_notes: str) -> tuple[List[Any], Dict[str, Any]]:
        """
        Retrieve relevant knowledge chunks for the current slide.

        Args:
            slide_content: Content extracted from slide image (if available)
            speaker_notes: Speaker notes for the slide

        Returns:
            Tuple of (knowledge chunks, knowledge metadata)
        """
        if not self.enable_knowledge or not self.knowledge_manager:
            return [], {}

        try:
            # Combine slide content and speaker notes for search query
            search_query = f"{slide_content} {speaker_notes}".strip()

            if not search_query:
                return [], {}

            # Search for relevant chunks
            chunks = self.knowledge_manager.search(search_query)

            # Create metadata about knowledge usage
            knowledge_metadata = {
                'search_query': search_query[:200] + '...' if len(search_query) > 200 else search_query,
                'chunks_found': len(chunks),
                'knowledge_sources': [],
                'knowledge_sections': [],
                'search_scores': []
            }

            # Extract metadata from chunks
            for chunk in chunks:
                if hasattr(chunk, 'file_path'):
                    source_file = Path(chunk.file_path).name
                    if source_file not in knowledge_metadata['knowledge_sources']:
                        knowledge_metadata['knowledge_sources'].append(source_file)

                if hasattr(chunk, 'section') and chunk.section:
                    if chunk.section not in knowledge_metadata['knowledge_sections']:
                        knowledge_metadata['knowledge_sections'].append(chunk.section)

                if hasattr(chunk, 'metadata') and chunk.metadata and 'search_score' in chunk.metadata:
                    knowledge_metadata['search_scores'].append(round(chunk.metadata['search_score'], 3))

            logger.debug(f"Retrieved {len(chunks)} knowledge chunks for query: {search_query[:100]}...")

            return chunks, knowledge_metadata

        except Exception as e:
            logger.warning(f"Failed to retrieve knowledge chunks: {e}")
            return [], {}

    def _build_context_bundle(
        self,
        knowledge_chunks: List[Any],
        slide_number: Optional[int] = None
    ) -> Optional[Any]:
        """
        Build context bundle combining knowledge and narrative contexts.

        Args:
            knowledge_chunks: Retrieved knowledge chunks
            slide_number: Current slide number for narrative context

        Returns:
            ContextBundle or None if no context available
        """
        if not self.enable_knowledge or not self.context_manager:
            return None

        try:
            # Get narrative context if using narrative mode
            narrative_context = None
            previous_slides = None

            if self.use_narrative and slide_number is not None and self.slide_contexts:
                # Build narrative context from previous slides
                recent_contexts = self.slide_contexts[-self.context_window_size:]
                narrative_parts = []

                for context in recent_contexts:
                    narrative_parts.append(
                        f"Slide {context.slide_number} - {context.title}: {context.transcript[:200]}..."
                    )

                narrative_context = "\n".join(narrative_parts)
                previous_slides = [ctx.to_dict() for ctx in recent_contexts]

            # Create context bundle
            context_bundle = self.context_manager.create_context_bundle(
                knowledge_chunks=knowledge_chunks,
                narrative_context=narrative_context,
                previous_slides=previous_slides
            )

            return context_bundle

        except Exception as e:
            logger.warning(f"Failed to build context bundle: {e}")
            return None

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
    ) -> tuple[str, Dict[str, Any]]:
        """
        Process a single slide to generate transcript with optional knowledge integration.

        Args:
            image_path: Path to the slide image
            speaker_notes: Speaker notes for the slide
            system_prompt: Custom system prompt. If None, uses default from config.
            slide_number: Number of the current slide (used for narrative continuity)
            slide_title: Title of the current slide (used for narrative continuity)

        Returns:
            Tuple of (generated transcript text, knowledge metadata)
        """
        # Retrieve knowledge chunks if knowledge base is enabled
        knowledge_chunks = []
        knowledge_metadata = {}
        if self.enable_knowledge:
            # Use slide title and speaker notes as search context
            slide_content = slide_title  # Could be enhanced with OCR in the future
            knowledge_chunks, knowledge_metadata = self._retrieve_knowledge_chunks(slide_content, speaker_notes)

        # Build context bundle
        context_bundle = None
        if knowledge_chunks or (self.use_narrative and slide_number is not None):
            context_bundle = self._build_context_bundle(knowledge_chunks, slide_number)

        if self.use_narrative and slide_number is not None:
            # Use narrative-aware processing with optional knowledge integration
            enhanced_prompt = self._build_context_prompt(
                slide_number, self.slide_contexts
            )

            # Generate transcript with context bundle
            transcript = self.client.generate_transcript(
                image_path=str(image_path),
                speaker_notes=speaker_notes,
                system_prompt=enhanced_prompt,
                context_bundle=context_bundle,
                stream=False,
            )

            # Create and store slide context for future slides
            slide_context = SlideContext(
                slide_number=slide_number,
                title=slide_title,
                transcript=transcript,
            )
            self.slide_contexts.append(slide_context)

            return transcript, knowledge_metadata
        else:
            # Use standard processing with optional knowledge integration
            transcript = self.client.generate_transcript(
                image_path=str(image_path),
                speaker_notes=speaker_notes,
                system_prompt=system_prompt,
                context_bundle=context_bundle,
                stream=False,
            )
            return transcript, knowledge_metadata

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

            # Initialize knowledge metadata for this slide
            knowledge_metadata = {}

            # Generate transcript
            if self.use_narrative:
                transcript, knowledge_metadata = self.process_single_slide(
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
                transcript, knowledge_metadata = self.process_single_slide(
                    image_path=image_path,
                    speaker_notes=speaker_notes,
                    system_prompt=system_prompt,
                )

            # Add transcript to dataframe
            df_copy.loc[i, "ai_transcript"] = transcript

            # Add knowledge metadata if available
            if knowledge_metadata:
                df_copy.loc[i, "knowledge_chunks_used"] = knowledge_metadata.get('chunks_found', 0)
                df_copy.loc[i, "knowledge_sources"] = ', '.join(knowledge_metadata.get('knowledge_sources', []))
                df_copy.loc[i, "knowledge_sections"] = ', '.join(knowledge_metadata.get('knowledge_sections', []))
                df_copy.loc[i, "knowledge_search_query"] = knowledge_metadata.get('search_query', '')
                if knowledge_metadata.get('search_scores'):
                    df_copy.loc[i, "avg_knowledge_score"] = sum(knowledge_metadata['search_scores']) / len(knowledge_metadata['search_scores'])
                else:
                    df_copy.loc[i, "avg_knowledge_score"] = 0.0
            else:
                # Initialize knowledge metadata columns with default values
                df_copy.loc[i, "knowledge_chunks_used"] = 0
                df_copy.loc[i, "knowledge_sources"] = ""
                df_copy.loc[i, "knowledge_sections"] = ""
                df_copy.loc[i, "knowledge_search_query"] = ""
                df_copy.loc[i, "avg_knowledge_score"] = 0.0

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
