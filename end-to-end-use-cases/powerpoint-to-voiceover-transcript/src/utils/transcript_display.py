"""
Utility functions for displaying transcripts with knowledge enhancement details.
"""

import pandas as pd
from typing import Optional, List, Dict, Any
from pathlib import Path


def display_enhanced_transcripts(
    processed_df: pd.DataFrame,
    knowledge_manager=None,
    num_slides: int = 5,
    show_knowledge_details: bool = True,
    show_search_scores: bool = True
) -> None:
    """
    Display transcripts with knowledge enhancement details.

    Args:
        processed_df: DataFrame with processed transcripts
        knowledge_manager: FAISS knowledge manager instance
        num_slides: Number of slides to display
        show_knowledge_details: Whether to show knowledge chunk details
        show_search_scores: Whether to show similarity scores
    """

    print(f'Displaying first {num_slides} slide transcripts with knowledge enhancement\n')
    print('=' * 100)

    for idx, row in processed_df.head(num_slides).iterrows():
        print(f'\nSLIDE {row["slide_number"]} - {row["slide_title"]}')
        print('=' * 80)

        # Display transcript
        print('\nTRANSCRIPT:')
        print(f'"{row["ai_transcript"]}"')

        # Display knowledge enhancement details if available
        if show_knowledge_details and knowledge_manager:
            _display_knowledge_details(row, knowledge_manager, show_search_scores)

        # Display basic knowledge stats from DataFrame if available
        elif 'knowledge_chunks_used' in row and pd.notna(row['knowledge_chunks_used']):
            _display_basic_knowledge_stats(row)

        print('\n' + '-' * 80)


def _display_knowledge_details(
    row: pd.Series,
    knowledge_manager,
    show_search_scores: bool = True
) -> None:
    """Display detailed knowledge chunk information."""

    # Try to reconstruct the search that was performed
    search_query = ""
    if 'knowledge_search_query' in row and pd.notna(row['knowledge_search_query']):
        search_query = row['knowledge_search_query']
    else:
        # Fallback: use slide title and notes
        search_query = f"{row.get('slide_title', '')} {row.get('speaker_notes', '')}".strip()

    if not search_query:
        print('\nKNOWLEDGE ENHANCEMENT: No search query available')
        return

    try:
        # Perform the same search to get chunk details
        chunks = knowledge_manager.search(search_query, top_k=5)

        if not chunks:
            print('\nKNOWLEDGE ENHANCEMENT: No relevant knowledge found')
            return

        print(f'\nKNOWLEDGE ENHANCEMENT:')
        print(f'   Search Query: "{search_query[:100]}{"..." if len(search_query) > 100 else ""}"')
        print(f'   Chunks Found: {len(chunks)}')

        print('\nKNOWLEDGE CHUNKS USED:')

        for i, chunk in enumerate(chunks, 1):
            print(f'\n   Chunk {i}:')
            print(f'      Source: {Path(chunk.file_path).name}')

            if hasattr(chunk, 'section') and chunk.section:
                print(f'      Section: {chunk.section}')

            if show_search_scores and hasattr(chunk, 'metadata') and chunk.metadata:
                score = chunk.metadata.get('search_score', 'N/A')
                if score != 'N/A':
                    print(f'      Similarity Score: {score:.3f}')

            # Show content preview
            content_preview = chunk.content[:200] + '...' if len(chunk.content) > 200 else chunk.content
            print(f'      Content: "{content_preview}"')

    except Exception as e:
        print(f'\nKNOWLEDGE ENHANCEMENT: Error retrieving details - {e}')


def _display_basic_knowledge_stats(row: pd.Series) -> None:
    """Display basic knowledge statistics from DataFrame."""

    print(f'\nKNOWLEDGE ENHANCEMENT:')

    if 'knowledge_chunks_used' in row and pd.notna(row['knowledge_chunks_used']):
        print(f'   Chunks Used: {int(row["knowledge_chunks_used"])}')

    if 'knowledge_sources' in row and pd.notna(row['knowledge_sources']) and row['knowledge_sources']:
        sources = row['knowledge_sources'].split(', ') if isinstance(row['knowledge_sources'], str) else []
        print(f'   Sources: {", ".join(sources)}')

    if 'knowledge_sections' in row and pd.notna(row['knowledge_sections']) and row['knowledge_sections']:
        sections = row['knowledge_sections'].split(', ') if isinstance(row['knowledge_sections'], str) else []
        print(f'   Sections: {", ".join(sections)}')

    if 'avg_knowledge_score' in row and pd.notna(row['avg_knowledge_score']):
        print(f'   Avg Similarity Score: {row["avg_knowledge_score"]:.3f}')


def display_knowledge_base_summary(knowledge_manager) -> None:
    """Display summary of knowledge base contents."""

    if not knowledge_manager:
        print("No knowledge manager available")
        return

    try:
        stats = knowledge_manager.get_stats()

        print('\nKNOWLEDGE BASE SUMMARY')
        print('=' * 50)
        print(f'Total Chunks: {stats.get("total_chunks", 0)}')
        print(f'Index Type: {stats.get("index_type", "unknown").upper()}')
        print(f'Embedding Model: {stats.get("embedding_model", "unknown")}')
        print(f'Content Size: {stats.get("content_size_mb", 0):.1f} MB')
        print(f'Avg Chunk Size: {stats.get("avg_chunk_size", 0):.0f} characters')

        # Show knowledge sources
        if hasattr(knowledge_manager, 'chunks') and knowledge_manager.chunks:
            sources = set()
            sections = set()

            for chunk in knowledge_manager.chunks:
                if hasattr(chunk, 'file_path'):
                    sources.add(Path(chunk.file_path).name)
                if hasattr(chunk, 'section') and chunk.section:
                    sections.add(chunk.section)

            print(f'\nKnowledge Sources ({len(sources)}):')
            for source in sorted(sources):
                print(f'  • {source}')

            if sections:
                print(f'\nKnowledge Sections ({len(sections)}):')
                for section in sorted(list(sections)[:10]):  # Show first 10
                    print(f'  • {section}')
                if len(sections) > 10:
                    print(f'  ... and {len(sections) - 10} more')

        print('=' * 50)

    except Exception as e:
        print(f"Error displaying knowledge base summary: {e}")


# Convenience function for notebook use
def show_transcripts_with_knowledge(
    processed_df: pd.DataFrame,
    knowledge_manager=None,
    num_slides: int = 5
) -> None:
    """
    Convenience function for displaying transcripts with knowledge details in notebooks.

    Usage in notebook:
        from src.utils.transcript_display import show_transcripts_with_knowledge
        show_transcripts_with_knowledge(processed_df, knowledge_manager, num_slides=3)
    """

    # Display knowledge base summary first
    if knowledge_manager:
        display_knowledge_base_summary(knowledge_manager)

    # Display enhanced transcripts
    display_enhanced_transcripts(
        processed_df,
        knowledge_manager,
        num_slides=num_slides,
        show_knowledge_details=True,
        show_search_scores=True
    )
