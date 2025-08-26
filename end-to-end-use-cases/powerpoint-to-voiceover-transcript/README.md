# PowerPoint to Voiceover Transcript Generator

> **RAG-powered AI system that converts PowerPoint presentations into professional voiceover transcripts using Llama 4 Maverick**

Transform PowerPoint slides into natural-sounding voiceover scripts optimized for human narration and text-to-speech systems.

## Features

- **Multi-Modal AI**: Analyzes slide visuals and speaker notes with Llama 4 Maverick
- **RAG Enhancement**: FAISS-powered knowledge retrieval from markdown files
- **Narrative Flow**: Maintains context and smooth transitions between slides
- **Speech Ready**: Converts numbers, technical terms, and abbreviations to spoken form

## Use Cases
- **Corporate Presentations:** Internal training, product demos, quarterly reviews
- **Educational Content:** Course materials, conference talks, webinars
- **Marketing Materials:** Product launches, sales presentations, customer demos
- **Technical Documentation:** API walkthroughs, system architecture presentations
## Quick Start

### Prerequisites
- Python 3.12+
- LibreOffice (for PPTX conversion)
- Groq API key

### Installation

#### Option 1: Using uv (Recommended)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup project
git clone <repository-url>
cd powerpoint-to-voiceover-transcript
uv sync

# Activate environment
source .venv/bin/activate  # macOS/Linux
# or .venv\Scripts\activate  # Windows
```

#### Option 2: Using pip

```bash
git clone <repository-url>
cd powerpoint-to-voiceover-transcript
pip install -e .
```

### Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your API key
echo "GROQ_API_KEY=your_api_key_here" >> .env

# Configure your presentation in config.yaml
# Update the pptx_file path to your presentation
```


## Usage

### 🚀 Jupyter Notebooks (Recommended!)
```bash
# Basic workflow with narrative continuity
jupyter notebook narrative_continuity_workflow.ipynb

# Advanced workflow with knowledge enhancement
jupyter notebook knowledge_enhanced_workflow.ipynb
```

### 🐍 Python API
```python
from src.core.pptx_processor import pptx_to_images_and_notes
from src.processors.unified_transcript_generator import UnifiedTranscriptProcessor

# Extract slides and notes
result = pptx_to_images_and_notes("input/presentation.pptx", "output/")

# Generate transcripts with full enhancement
processor = UnifiedTranscriptProcessor(
    use_narrative=True,
    context_window_size=5,
    enable_knowledge=True
)

transcripts = processor.process_slides_dataframe(
    result['notes_df'],
    "output/",
    save_context=True
)

# Save results
transcripts.to_csv("output/transcripts.csv", index=False)
```

## Knowledge Base Setup

1. **Add markdown files** to `knowledge_base/` directory:
   ```bash
   echo "# Product Overview\nOur product..." > knowledge_base/product.md
   echo "# Technical Terms\nAPI means..." > knowledge_base/glossary.md
   ```

2. **The system will automatically**:
   - Index your markdown files with FAISS
   - Search for relevant content during processing
   - Enhance transcripts with domain knowledge

## Configuration

### Main Configuration File (`config.yaml`)

```yaml
# API Configuration
api:
  groq_model: "meta-llama/llama-4-maverick-17b-128e-instruct"
  max_retries: 3
  retry_delay: 1

# Processing Configuration
processing:
  default_dpi: 200
  default_format: "png"
  batch_size: 5

# Current Project Settings
current_project:
  pptx_file: "input/All About Llamas"
  extension: ".pptx"
  output_dir: "output/"

# Knowledge Base Configuration
knowledge:
  # Core settings
  enabled: true
  knowledge_base_dir: "knowledge_base"

  # FAISS Vector Store
  vector_store:
    index_type: "flat"                 # "flat", "ivf", "hnsw"
    use_gpu: false                     # Enable GPU acceleration
    cache_enabled: true                # Persistent caching

  # Embedding Configuration
  embedding:
    model_name: "all-MiniLM-L6-v2"     # Lightweight, fast model
    device: "cpu"                      # Use "cuda" if GPU available

  # Search Configuration
  search:
    top_k: 5                           # Number of chunks to retrieve
    similarity_threshold: 0.3          # Minimum similarity score
    max_chunk_size: 1000              # Maximum characters per chunk

  # Context Integration
  context:
    strategy: "combined"               # "knowledge_only", "narrative_priority", "combined"
    knowledge_weight: 0.3             # Knowledge influence (0.0-1.0)
    integration_method: "system_prompt"

  # Performance Settings
  performance:
    enable_caching: true              # Cache embeddings and search results
    max_memory_mb: 512                # Maximum memory for embeddings
    lazy_loading: true                # Load embeddings on demand

# File Paths
paths:
  cache_dir: "cache"
  logs_dir: "logs"
  temp_dir: "temp"

# Logging
logging:
  level: "INFO"
  file_enabled: true
  console_enabled: true
```

### Environment Variables (`.env`)

```bash
# Required
GROQ_API_KEY=your_groq_api_key_here
```

### Integration Strategies

| Strategy | Configuration | Best For |
|----------|---------------|----------|
| **Knowledge Only** | `strategy: "knowledge_only"` | Technical docs, specifications |
| **Narrative Priority** | `strategy: "narrative_priority"` | Storytelling, educational content |
| **Combined** | `strategy: "combined"` | Most presentations (recommended) |

### Performance Tuning

**For Speed:**
```yaml
processing:
  default_dpi: 150
knowledge:
  search:
    top_k: 3
  performance:
    max_memory_mb: 256
```

**For Quality:**
```yaml
knowledge:
  search:
    top_k: 7
    similarity_threshold: 0.2
  context:
    knowledge_weight: 0.4
```

## System Architecture

### Core Processing Pipeline

The system follows a modular 3-stage pipeline:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PowerPoint    │───▶│     Content     │───▶│    Knowledge    │
│      File       │    │    Extraction   │    │    Retrieval    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     Output      │◀───│    Transcript   │◀───│  LLM Processing │
│      Files      │    │    Generation   │    │  Vision & Text  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```
#### Stage 1: Content Extraction (`pptx_processor.py`)
- Extracts speaker notes and slide text using `python-pptx`
- Converts PPTX → PDF → Images via LibreOffice and PyMuPDF
- Generates structured DataFrame with slide metadata
- Supports configurable DPI (default: 200) and formats (PNG/JPEG)

#### Stage 2: Knowledge Retrieval (`markdown_knowledge.py`)
- Loads and chunks markdown files from knowledge base
- Generates embeddings using sentence-transformers
- Performs semantic search for relevant knowledge chunks
- Integrates knowledge with slide content and speaker notes

#### Stage 3: AI Processing (`groq_client.py` + `unified_transcript_generator.py`)
- Integrates with Groq's vision models via `groq` client
- Base64 encodes images for vision model processing
- Applies narrative continuity with sliding context window
- Handles API retries and comprehensive error management


### Project Structure
```
├── README.md                          # This guide
├── config.yaml                        # Configuration
├── pyproject.toml                     # Dependencies
├── knowledge_enhanced_workflow.ipynb  # Advanced notebook
├── narrative_continuity_workflow.ipynb # Basic notebook
├── input/                             # PowerPoint files
│   └── All About Llamas.pptx         # Example presentation
├── output/                            # Generated transcripts and images
├── knowledge_base/                    # Domain knowledge (.md files)
│   ├── llama diet.md
│   └── llamas.md
└── src/
    ├── config/settings.py             # Configuration management
    ├── core/                          # Core processing
    │   ├── pptx_processor.py          # PPTX extraction
    │   ├── groq_client.py             # Groq API client
    │   └── image_processing.py        # Image encoding
    ├── knowledge/                     # Knowledge management
    │   ├── faiss_knowledge.py         # FAISS vector search
    │   └── context_manager.py         # Context integration
    ├── processors/                    # Main processing
    │   └── unified_transcript_generator.py
    └── utils/                         # Utilities
        ├── visualization.py           # Slide display
        └── transcript_display.py      # Transcript formatting
```

## Processing Your Own Presentations

1. **Add your PowerPoint**:
   ```bash
   cp your_presentation.pptx input/
   ```

2. **Update config.yaml**:
   ```yaml
   current_project:
     pptx_file: "input/your_presentation"
   ```

3. **Add domain knowledge** (optional):
   ```bash
   echo "# Your Domain Info\n..." > knowledge_base/domain.md
   ```

4. **Run processing**:
   ```bash
   jupyter notebook knowledge_enhanced_workflow.ipynb
   ```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| LibreOffice not found | `brew install --cask libreoffice` (macOS) |
| API key error | Set `GROQ_API_KEY` environment variable |
| Memory issues | Reduce `context_window_size` and `top_k` in config |
| Slow processing | Lower `default_dpi` or disable knowledge: `enabled: false` |
| Knowledge base not loading | Check `.md` files exist: `ls knowledge_base/*.md` |

## Output Files

After processing, check the `output/` directory for:
- **Slide images**: `slide-001.png`, `slide-002.png`, etc.
- **Transcripts**: `*_transcripts.csv` and `*_transcripts.json`
- **Context data**: `narrative_context/` (if narrative mode enabled)
- **Knowledge stats**: `knowledge_base_stats.json` (if knowledge enabled)

---
