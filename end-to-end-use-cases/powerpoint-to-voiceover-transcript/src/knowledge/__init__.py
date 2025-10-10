"""
Knowledge base package for PowerPoint to Voiceover Transcript Generator.

This package provides knowledge base integration capabilities using FAISS
for efficient vector search and retrieval of domain-specific information.
"""

# Import main classes for easy access
try:
    from .faiss_knowledge import FAISSKnowledgeManager, KnowledgeChunk
    from .context_manager import ContextManager, ContextBundle

    # Backward compatibility
    MarkdownKnowledgeManager = FAISSKnowledgeManager

    __all__ = [
        'FAISSKnowledgeManager',
        'MarkdownKnowledgeManager',  # Backward compatibility alias
        'KnowledgeChunk',
        'ContextManager',
        'ContextBundle'
    ]

except ImportError as e:
    # Graceful degradation if dependencies are missing
    import warnings
    warnings.warn(f"Knowledge base components not fully available: {e}")

    __all__ = []
