"""
FAISS-based Knowledge Manager for enhanced vector search and storage.
Replaces the numpy-based approach with production-ready vector database.
"""

import faiss
import pickle
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import json
from datetime import datetime

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class KnowledgeChunk:
    """Enhanced knowledge chunk with FAISS indexing support"""
    content: str
    file_path: str
    section: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    chunk_id: int = 0
    embedding_hash: Optional[str] = None
    created_at: Optional[str] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.metadata is None:
            self.metadata = {}

class FAISSKnowledgeManager:
    """
    Production-ready knowledge manager using FAISS for vector search.

    Features:
    - Multiple index types (Flat, IVF, HNSW)
    - Persistent caching with automatic invalidation
    - Incremental document updates
    - Memory-efficient storage
    - GPU acceleration support
    """

    def __init__(
        self,
        knowledge_base_dir: str,
        index_type: str = "flat",
        embedding_model: str = "all-MiniLM-L6-v2",
        use_gpu: bool = False
    ):
        self.knowledge_base_dir = Path(knowledge_base_dir)
        self.chunks = []
        self.index = None
        self.model = None
        self.index_type = index_type.lower()
        self.embedding_model = embedding_model
        self.use_gpu = use_gpu

        # Cache and metadata
        self.cache_dir = self.knowledge_base_dir / ".faiss_cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.metadata_file = self.cache_dir / "metadata.json"

        # Performance tracking
        self.stats = {
            'total_searches': 0,
            'cache_hits': 0,
            'index_builds': 0,
            'last_updated': None
        }

    def initialize(self) -> None:
        """Initialize the FAISS knowledge manager"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence_transformers and numpy are required for knowledge base functionality")

        logger.info(f"Initializing FAISS Knowledge Manager with {self.index_type} index")

        # Load embedding model
        self.model = SentenceTransformer(self.embedding_model)
        logger.info(f"Loaded embedding model: {self.embedding_model}")

        # Try to load cached index
        if self._should_rebuild_index():
            logger.info("Building new FAISS index...")
            self._build_index()
            self._save_index()
        else:
            logger.info("Loading cached FAISS index...")
            if not self._load_cached_index():
                logger.warning("Failed to load cached index, rebuilding...")
                self._build_index()
                self._save_index()

    def _should_rebuild_index(self) -> bool:
        """Check if index needs to be rebuilt based on file changes"""
        if not self.metadata_file.exists():
            return True

        try:
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)

            # Check if knowledge base files have changed
            current_hash = self._get_knowledge_base_hash()
            stored_hash = metadata.get('knowledge_base_hash')

            return current_hash != stored_hash

        except Exception as e:
            logger.warning(f"Error reading metadata: {e}")
            return True

    def _get_knowledge_base_hash(self) -> str:
        """Generate hash of all knowledge base files for change detection"""
        md_files = sorted(self.knowledge_base_dir.rglob("*.md"))
        hash_content = ""

        for md_file in md_files:
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                hash_content += f"{md_file.name}:{len(content)}:{hash(content)}"
            except Exception as e:
                logger.warning(f"Error reading {md_file}: {e}")

        return hashlib.md5(hash_content.encode()).hexdigest()

    def _build_index(self) -> None:
        """Build FAISS index from knowledge base"""
        self._load_knowledge_base()

        if not self.chunks:
            logger.warning("No knowledge chunks found")
            return

        logger.info(f"Processing {len(self.chunks)} knowledge chunks...")

        # Generate embeddings with progress tracking
        texts = [chunk.content for chunk in self.chunks]
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True
        )

        # Create FAISS index based on type
        dimension = embeddings.shape[1]

        if self.index_type == "flat":
            # Exact search - best for small to medium datasets
            self.index = faiss.IndexFlatIP(dimension)

        elif self.index_type == "ivf":
            # Inverted File index - good balance of speed and accuracy
            nlist = min(100, max(10, len(self.chunks) // 10))
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)

            # Train the index
            logger.info(f"Training IVF index with {nlist} clusters...")
            self.index.train(embeddings.astype('float32'))

        elif self.index_type == "hnsw":
            # Hierarchical Navigable Small World - very fast approximate search
            self.index = faiss.IndexHNSWFlat(dimension, 32)
            self.index.hnsw.efConstruction = 200
            self.index.hnsw.efSearch = 50

        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")

        # Normalize embeddings for cosine similarity BEFORE adding to index
        embeddings_normalized = embeddings.astype('float32').copy()
        faiss.normalize_L2(embeddings_normalized)

        # Add embeddings to index
        self.index.add(embeddings_normalized)

        # Move to GPU if requested and available (after adding data)
        if self.use_gpu and faiss.get_num_gpus() > 0:
            logger.info("Moving index to GPU...")
            gpu_res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(gpu_res, 0, self.index)

        # Update statistics
        self.stats['index_builds'] += 1
        self.stats['last_updated'] = datetime.now().isoformat()

        logger.info(f"Built {self.index_type.upper()} index with {len(self.chunks)} chunks")

    def _load_knowledge_base(self) -> None:
        """Load and chunk markdown files from knowledge base"""
        md_files = list(self.knowledge_base_dir.rglob("*.md"))
        self.chunks = []
        chunk_id = 0

        logger.info(f"Loading {len(md_files)} markdown files...")

        for md_file in md_files:
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Enhanced chunking by sections with better parsing
                chunks = self._chunk_content(content, str(md_file))

                for chunk_content, section_title in chunks:
                    if chunk_content.strip():
                        chunk = KnowledgeChunk(
                            content=chunk_content.strip(),
                            file_path=str(md_file),
                            section=section_title,
                            chunk_id=chunk_id,
                            metadata={
                                'file_size': len(content),
                                'chunk_size': len(chunk_content),
                                'source_file': md_file.name
                            }
                        )
                        self.chunks.append(chunk)
                        chunk_id += 1

            except Exception as e:
                logger.error(f"Error processing {md_file}: {e}")

        logger.info(f"Created {len(self.chunks)} knowledge chunks from {len(md_files)} files")

    def _chunk_content(self, content: str, file_path: str) -> List[Tuple[str, Optional[str]]]:
        """Enhanced content chunking with better section detection"""
        chunks = []

        # Split by main headers (# and ##)
        sections = content.split('\n## ')

        for i, section in enumerate(sections):
            if not section.strip():
                continue

            if i == 0:
                # First section might not have ## prefix
                if section.startswith('# '):
                    # Extract title and content
                    lines = section.split('\n', 1)
                    title = lines[0].replace('# ', '').strip()
                    content_part = lines[1] if len(lines) > 1 else ""
                    chunks.append((content_part, title))
                else:
                    chunks.append((section, None))
            else:
                # Subsequent sections
                lines = section.split('\n', 1)
                section_title = lines[0].strip()
                section_content = lines[1] if len(lines) > 1 else ""

                # Further split large sections by ### if needed
                if len(section_content) > 2000:  # Large section threshold
                    subsections = section_content.split('\n### ')
                    for j, subsection in enumerate(subsections):
                        if subsection.strip():
                            if j == 0:
                                chunks.append((subsection, section_title))
                            else:
                                sub_lines = subsection.split('\n', 1)
                                sub_title = f"{section_title} - {sub_lines[0].strip()}"
                                sub_content = sub_lines[1] if len(sub_lines) > 1 else ""
                                chunks.append((sub_content, sub_title))
                else:
                    chunks.append((section_content, section_title))

        return chunks

    def search(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.3
    ) -> List[KnowledgeChunk]:
        """Search for relevant knowledge chunks using FAISS"""
        if not self.chunks or self.index is None:
            logger.warning("No index available for search")
            return []

        self.stats['total_searches'] += 1

        try:
            # Encode query
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding.astype('float32'))

            # Search FAISS index
            scores, indices = self.index.search(query_embedding.astype('float32'), top_k)

            # Filter by similarity threshold and return chunks
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if score >= similarity_threshold and 0 <= idx < len(self.chunks):
                    chunk = self.chunks[idx]
                    # Add search score to metadata
                    chunk.metadata['search_score'] = float(score)
                    results.append(chunk)

            logger.debug(f"Search query: '{query}' returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    def add_document(self, file_path: str, content: str) -> bool:
        """Add new document to existing index"""
        try:
            # Process new content into chunks
            new_chunks = []
            chunks = self._chunk_content(content, file_path)

            start_id = len(self.chunks)
            for i, (chunk_content, section_title) in enumerate(chunks):
                if chunk_content.strip():
                    chunk = KnowledgeChunk(
                        content=chunk_content.strip(),
                        file_path=file_path,
                        section=section_title,
                        chunk_id=start_id + i,
                        metadata={
                            'file_size': len(content),
                            'chunk_size': len(chunk_content),
                            'source_file': Path(file_path).name
                        }
                    )
                    new_chunks.append(chunk)

            if not new_chunks:
                logger.warning(f"No chunks created from {file_path}")
                return False

            # Generate embeddings for new chunks
            texts = [chunk.content for chunk in new_chunks]
            embeddings = self.model.encode(texts, convert_to_numpy=True)

            # Normalize embeddings before adding to index
            embeddings_normalized = embeddings.astype('float32').copy()
            faiss.normalize_L2(embeddings_normalized)

            # Add to index
            self.index.add(embeddings_normalized)

            # Add to chunks list
            self.chunks.extend(new_chunks)

            # Save updated index
            self._save_index()

            logger.info(f"Added {len(new_chunks)} chunks from {file_path}")
            return True

        except Exception as e:
            logger.error(f"Error adding document {file_path}: {e}")
            return False

    def _save_index(self) -> None:
        """Save FAISS index and chunks to cache"""
        if self.index is None:
            return

        try:
            index_path = self.cache_dir / "faiss.index"
            chunks_path = self.cache_dir / "chunks.pkl"

            # Save FAISS index (move to CPU first if on GPU)
            index_to_save = self.index
            if self.use_gpu and faiss.get_num_gpus() > 0:
                index_to_save = faiss.index_gpu_to_cpu(self.index)

            faiss.write_index(index_to_save, str(index_path))

            # Save chunks
            with open(chunks_path, 'wb') as f:
                pickle.dump(self.chunks, f)

            # Save metadata
            metadata = {
                'knowledge_base_hash': self._get_knowledge_base_hash(),
                'index_type': self.index_type,
                'embedding_model': self.embedding_model,
                'total_chunks': len(self.chunks),
                'created_at': datetime.now().isoformat(),
                'stats': self.stats
            }

            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Saved FAISS index with {len(self.chunks)} chunks")

        except Exception as e:
            logger.error(f"Error saving index: {e}")

    def _load_cached_index(self) -> bool:
        """Load cached FAISS index and chunks"""
        index_path = self.cache_dir / "faiss.index"
        chunks_path = self.cache_dir / "chunks.pkl"

        if not (index_path.exists() and chunks_path.exists()):
            return False

        try:
            # Load FAISS index
            self.index = faiss.read_index(str(index_path))

            # Move to GPU if requested
            if self.use_gpu and faiss.get_num_gpus() > 0:
                logger.info("Moving cached index to GPU...")
                self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)

            # Load chunks
            with open(chunks_path, 'rb') as f:
                self.chunks = pickle.load(f)

            # Load metadata and stats
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                    self.stats.update(metadata.get('stats', {}))

            logger.info(f"Loaded cached index with {len(self.chunks)} chunks")
            return True

        except Exception as e:
            logger.error(f"Failed to load cached index: {e}")
            return False

    def rebuild_index(self) -> None:
        """Force rebuild of the index"""
        logger.info("Force rebuilding FAISS index...")
        self._build_index()
        self._save_index()

    def clear_cache(self) -> None:
        """Clear all cached data"""
        try:
            import shutil
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(exist_ok=True)
            logger.info("Cleared FAISS cache")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive knowledge base statistics"""
        stats = {
            'total_chunks': len(self.chunks),
            'index_type': self.index_type,
            'embedding_model': self.embedding_model,
            'model_loaded': self.model is not None,
            'index_loaded': self.index is not None,
            'use_gpu': self.use_gpu,
            'cache_dir': str(self.cache_dir),
            'knowledge_base_dir': str(self.knowledge_base_dir),
        }

        # Add FAISS-specific stats
        if self.index:
            stats.update({
                'index_size': self.index.ntotal,
                'dimension': self.index.d,
                'is_trained': getattr(self.index, 'is_trained', True)
            })

        # Add performance stats
        stats.update(self.stats)

        # Memory usage estimation
        if self.chunks:
            total_content_size = sum(len(chunk.content) for chunk in self.chunks)
            stats['content_size_mb'] = total_content_size / (1024 * 1024)
            stats['avg_chunk_size'] = total_content_size / len(self.chunks)

        return stats

    def get_chunk_by_id(self, chunk_id: int) -> Optional[KnowledgeChunk]:
        """Get specific chunk by ID"""
        for chunk in self.chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        return None

    def search_by_file(self, file_path: str) -> List[KnowledgeChunk]:
        """Get all chunks from a specific file"""
        return [chunk for chunk in self.chunks if chunk.file_path == file_path]


# Backward compatibility alias
MarkdownKnowledgeManager = FAISSKnowledgeManager
