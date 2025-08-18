from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class ContextBundle:
    knowledge_context: str = ""
    narrative_context: str = ""
    combined_context: str = ""

class ContextManager:
    def __init__(self):
        pass

    def create_context_bundle(self, knowledge_chunks=None, narrative_context="", previous_slides=None) -> ContextBundle:
        """Create a context bundle from knowledge chunks and narrative context"""
        knowledge_context = ""

        # Ensure narrative_context is not None
        narrative_context = narrative_context or ""

        if knowledge_chunks:
            knowledge_context = "\n\n".join([
                f"From {chunk.file_path}: {chunk.content[:200]}..."
                for chunk in knowledge_chunks
            ])

        combined_context = f"{narrative_context}\n\n{knowledge_context}".strip()

        return ContextBundle(
            knowledge_context=knowledge_context,
            narrative_context=narrative_context,
            combined_context=combined_context
        )
    def get_context_stats(self, context_bundle: ContextBundle) -> Dict[str, Any]:
        """Get statistics about the context bundle"""
        return {
            'knowledge_length': len(context_bundle.knowledge_context),
            'narrative_length': len(context_bundle.narrative_context),
            'combined_length': len(context_bundle.combined_context),
            'total_words': len(context_bundle.combined_context.split())
        }
    def get_context_for_integration(self, context_bundle: ContextBundle, integration_method: str = "system_prompt") -> Dict[str, Any]:
        """
        Get context data formatted for integration into prompts.

        Args:
            context_bundle: The context bundle to format
            integration_method: How to integrate context ("system_prompt" or "user_message")

        Returns:
            Dictionary with context_addition and integration_point
        """
        if not context_bundle:
            return {}

        # Safely get context strings, handling None values
        combined_context = getattr(context_bundle, 'combined_context', '') or ''
        knowledge_context = getattr(context_bundle, 'knowledge_context', '') or ''
        narrative_context = getattr(context_bundle, 'narrative_context', '') or ''

        if not combined_context.strip():
            return {}

        # Format the context for integration
        context_parts = []

        if knowledge_context.strip():
            context_parts.append("## RELEVANT KNOWLEDGE\n\n" + knowledge_context)

        if narrative_context.strip():
            context_parts.append("## PREVIOUS SLIDES CONTEXT\n\n" + narrative_context)

        if not context_parts:
            return {}

        context_addition = "\n\n".join(context_parts)

        # Add instructions for using the context
        if integration_method == "system_prompt":
            context_addition += "\n\n## CONTEXT USAGE INSTRUCTIONS\n\n"
            context_addition += "Use the above knowledge and context information to enhance your transcript generation. "
            context_addition += "Incorporate relevant facts and maintain consistency with previous slides when appropriate."

        return {
            'context_addition': context_addition,
            'integration_point': 'before_instructions' if integration_method == "system_prompt" else 'prefix'
        }
