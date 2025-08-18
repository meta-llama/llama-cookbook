# PowerPoint to Knowledge-Enhanced Voiceover Transcript Generator

> **AI-powered solution for converting PowerPoint presentations into professional, knowledge-enhanced voiceover transcripts using Groq's vision models**


## Overview

This system transforms PowerPoint presentations into natural-sounding voiceover transcripts optimized for human narration and text-to-speech systems. It combines AI-powered content analysis with domain-specific knowledge integration to produce professional-quality transcripts.

### Key Capabilities

- **Multi-Modal AI Processing**: Analyzes both visual slide content and speaker notes
- **Knowledge Base Integration**: Enhances transcripts with domain-specific information
- **Narrative Continuity**: Maintains smooth transitions and consistent terminology
- **Speech Optimization**: Converts technical terms, numbers, and abbreviations to spoken form
- **Flexible Processing Modes**: Standard, narrative-aware, and knowledge-enhanced options

### Use Cases

- **Corporate Presentations**: Internal training, product demos, quarterly reviews
- **Educational Content**: Course materials, conference talks, webinars
- **Marketing Materials**: Product launches, sales presentations, customer demos
- **Technical Documentation**: API walkthroughs, system architecture presentations

## Features

### Core Features
- **AI-Powered Analysis**: Uses Groq's vision models for intelligent content understanding
- **Knowledge Base Integration**: FAISS-powered semantic search through markdown knowledge files
- **Narrative Continuity**: Context-aware processing with configurable sliding window
- **Speech Optimization**: Automatic conversion of numbers, decimals, and technical terms
- **Multi-Format Output**: CSV, JSON, and clean transcript exports
- **Visualization Tools**: Built-in slide preview and analysis utilities

### Advanced Features
- **Unified Processing Pipeline**: Single processor handles all modes
- **Graceful Degradation**: Continues processing even if components fail
- **Performance Optimization**: In-memory vector storage with caching
- **Cross-Platform Support**: Windows, macOS, and Linux compatibility
- **Production Ready**: Comprehensive error handling and retry logic

## Quick Start

### Prerequisites

- **Python 3.12+**
- **LibreOffice** (for PPTX conversion)
- **Groq API Key**

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

### Basic Usage

#### Using Jupyter Notebooks (Recommended)
```bash
# Standard workflow
jupyter notebook narrative_continuity_workflow.ipynb

# Knowledge-enhanced workflow
jupyter notebook knowledge_enhanced_narrative_workflow.ipynb
```

#### Standard Processing
```python
from src.core.pptx_processor import pptx_to_images_and_notes
from src.processors.unified_transcript_generator import UnifiedTranscriptProcessor

# Extract content from PowerPoint
result = pptx_to_images_and_notes("presentation.pptx", "output/")

# Generate transcripts
processor = UnifiedTranscriptProcessor(use_narrative=False)
transcripts = processor.process_slides_dataframe(result['notes_df'], "output/")

# Save results
transcripts.to_csv("transcripts.csv", index=False)
```

#### Knowledge-Enhanced Narrative Processing
```python
# Enable both narrative continuity and knowledge integration
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
powerpoint-to-voiceover-transcript/
├── PROJECT_DOCUMENTATION.md          # This comprehensive guide
├── config.yaml                       # Main configuration
├── pyproject.toml                     # Dependencies and metadata
├── .env.example                       # Environment template
├── knowledge_enhanced_narrative_workflow.ipynb  # Advanced workflow
├── narrative_continuity_workflow.ipynb          # Standard workflow
├── input/                             # PowerPoint files
├── output/                            # Generated content
├── knowledge_base/                    # Domain knowledge files
└── src/
    ├── config/
    │   └── settings.py                # Configuration management
    ├── core/
    │   ├── file_utils.py              # File system utilities
    │   ├── image_processing.py        # Image encoding for API
    │   ├── groq_client.py             # Groq API integration
    │   └── pptx_processor.py          # PPTX extraction and conversion
    ├── knowledge/
    │   ├── markdown_knowledge.py      # Knowledge base management
    │   └── context_manager.py         # Context integration
    ├── processors/
    │   └── unified_transcript_generator.py  # Main processing engine
    └── utils/
        └── visualization.py           # Slide display utilities
```


### Setup Guide

#### 1. Enable Knowledge Base

Edit `config.yaml`:
```yaml
knowledge:
  enabled: true
  knowledge_base_dir: "knowledge_base"
```

#### 2. Create Knowledge Base Structure

```bash
mkdir knowledge_base
cd knowledge_base

# Create domain-specific files
touch company_overview.md
touch technical_glossary.md
touch product_specifications.md
touch presentation_guidelines.md
```
#### 3. Add Knowledge Base Content
For the purposes of the cookbook, we're using local markdown files as the knowledge base. You can use any format you prefer, as long as it can be loaded and processed by the system.

### Processing Workflow

1. **Content Analysis**: System analyzes slide content and speaker notes
2. **Semantic Search**: Finds relevant knowledge chunks using embedding similarity
3. **Context Building**: Combines knowledge with narrative context (if enabled)
4. **Prompt Enhancement**: Integrates context into system prompt or user message
5. **Transcript Generation**: AI generates enhanced transcript with domain knowledge

### Configuration Options

```yaml
knowledge:
  # Core settings
  enabled: true
  knowledge_base_dir: "knowledge_base"

  # Embedding model configuration
  embedding:
    model_name: "all-MiniLM-L6-v2"  # Lightweight, fast model
    device: "cpu"                   # Use "cuda" if GPU available
    batch_size: 32
    max_seq_length: 512

  # Search parameters
  search:
    top_k: 5                        # Number of chunks to retrieve
    similarity_threshold: 0.3       # Minimum similarity score (0.0-1.0)
    enable_keyword_fallback: true   # Fallback to keyword search
    max_chunk_size: 1000           # Maximum characters per chunk
    chunk_overlap: 200             # Overlap between chunks

  # Context integration
  context:
    strategy: "combined"            # "knowledge_only", "narrative_priority", "combined"
    max_context_length: 8000       # Maximum total context length
    knowledge_weight: 0.3          # Knowledge influence (0.0-1.0)
    integration_method: "system_prompt"  # "system_prompt" or "user_message"

  # Performance optimization
  performance:
    enable_caching: true           # Cache embeddings and search results
    cache_dir: "cache/knowledge"   # Cache directory
    cache_expiry_hours: 24         # Cache expiration (0 = never)
    max_memory_mb: 512             # Maximum memory for embeddings
    lazy_loading: true             # Load embeddings on demand

  # Reliability settings
  fallback:
    graceful_degradation: true     # Continue if knowledge base fails
    use_keyword_fallback: true     # Use keyword matching as fallback
    log_errors_only: true          # Log errors but don't fail process
```

### Integration Strategies

#### Knowledge Only
```yaml
context:
  strategy: "knowledge_only"
```
**Best for**: Technical documentation, product specifications, reference materials

#### Narrative Priority
```yaml
context:
  strategy: "narrative_priority"
  knowledge_weight: 0.2
```
**Best for**: Storytelling presentations, educational sequences, marketing narratives

#### Combined (Recommended)
```yaml
context:
  strategy: "combined"
  knowledge_weight: 0.3
```
**Best for**: Most presentations, mixed content types, general use cases

## Configuration

### Main Configuration File (`config.yaml`)

```yaml
# API Configuration
api:
  llama_model: "Llama-4-Maverick-17B-128E-Instruct-FP8"
  max_retries: 3
  retry_delay: 1
  rate_limit_delay: 1

# Processing Configuration
processing:
  default_dpi: 200
  supported_formats: ["png", "jpeg", "jpg"]
  default_format: "png"
  batch_size: 5

# File Paths
paths:
  default_output_dir: "slide_images"
  cache_dir: "cache"
  logs_dir: "logs"
  temp_dir: "temp"

# Current Project Settings
current_project:
  pptx_file: "input/your_presentation"
  extension: ".pptx"
  output_dir: "output/"

# Knowledge Base Configuration (see Knowledge Base section for details)
knowledge:
  enabled: true
  knowledge_base_dir: "knowledge_base"
  # ... additional knowledge settings

# Logging Configuration
logging:
  level: "INFO"
  format: "%(asctime)s - %(levelname)s - %(message)s"
  file_enabled: true
  console_enabled: true
```

### Environment Variables (`.env`)

```bash
# Required
GROQ_API_KEY=your_groq_api_key_here

# Optional
LOG_LEVEL=INFO
CACHE_ENABLED=true
```

## Processing Modes

#### Standard Mode
```python
processor = UnifiedTranscriptProcessor(
    use_narrative=False,
    enable_knowledge=False
)
```
- **Use when**: Simple presentations, time-sensitive processing
- **Benefits**: Fastest processing, no dependencies
- **Limitations**: No context awareness, basic quality

#### Knowledge-Enhanced Mode
```python
processor = UnifiedTranscriptProcessor(
    use_narrative=False,
    enable_knowledge=True
)
```
- **Use when**: Technical presentations requiring domain expertise
- **Benefits**: Enhanced accuracy, domain-specific terminology
- **Limitations**: No narrative flow between slides

#### Narrative Mode
```python
processor = UnifiedTranscriptProcessor(
    use_narrative=True,
    context_window_size=5,
    enable_knowledge=False
)
```
- **Use when**: Educational content, storytelling presentations
- **Benefits**: Smooth transitions, consistent terminology
- **Limitations**: No external knowledge integration

#### Full Enhancement Mode (Recommended)
```python
processor = UnifiedTranscriptProcessor(
    use_narrative=True,
    context_window_size=5,
    enable_knowledge=True
)
```
- **Use when**: Professional presentations requiring highest quality
- **Benefits**: Maximum quality, context awareness, domain expertise
- **Limitations**: Slower processing, requires knowledge base setup

## Deployment

### Development Environment

```bash
# Clone repository
git clone <repository-url>
cd powerpoint-to-voiceover-transcript

# Setup with uv (recommended)
uv sync
source .venv/bin/activate

# Or setup with pip
pip install -e .

# Install system dependencies
# macOS: brew install --cask libreoffice
# Ubuntu: sudo apt-get install libreoffice
# Windows: Download from libreoffice.org
```

### Performance Optimization

#### Memory Management
```yaml
knowledge:
  performance:
    max_memory_mb: 1024        # Adjust based on available RAM
    lazy_loading: true         # Load embeddings on demand
    enable_caching: true       # Cache for repeated processing
```

#### Processing Optimization
```yaml
processing:
  batch_size: 10             # Process slides in batches
  default_dpi: 150           # Lower DPI for faster processing

api:
  max_retries: 5             # Increase retries for production
  retry_delay: 2             # Longer delays for stability
```

## Troubleshooting

### Common Issues and Solutions

#### Installation Issues

**"LibreOffice not found"**
```bash
# macOS
brew install --cask libreoffice

# Ubuntu/Debian
sudo apt-get install libreoffice

# Windows
# Download from https://www.libreoffice.org/download/
```

**"uv sync fails"**
```bash
# Ensure Python 3.12+ is available
uv python install 3.12
uv sync --python 3.12
```

**"sentence_transformers not found"**
```bash
# Install with uv
uv add sentence-transformers

# Or with pip
pip install sentence-transformers

# Restart Jupyter kernel after installation
```

#### Runtime Issues

**"API key not found"**
```bash
# Check .env file exists and contains key
cat .env | grep GROQ_API_KEY

# Or set environment variable directly
export GROQ_API_KEY=your_key_here
```

**"Permission denied on output directory"**
```bash
# Ensure write permissions
chmod 755 output/
mkdir -p output/
```

**"Knowledge base not loading"**
```bash
# Check directory exists and contains .md files
ls -la knowledge_base/
ls knowledge_base/*.md

# Verify configuration
grep -A 5 "knowledge:" config.yaml
```

#### Performance Issues

**"Processing too slow"**
```yaml
# Reduce context window size
context_window_size: 3

# Lower image quality
processing:
  default_dpi: 150

# Disable knowledge base temporarily
knowledge:
  enabled: false
```

**"Memory usage too high"**
```yaml
knowledge:
  performance:
    max_memory_mb: 256
    lazy_loading: true
  search:
    top_k: 3
    max_chunk_size: 500
```

#### Quality Issues

**"Poor transcript quality"**
```yaml
# Increase knowledge retrieval
knowledge:
  search:
    top_k: 7
    similarity_threshold: 0.2

# Increase context window
context_window_size: 7
```

**"Inconsistent terminology"**
- Ensure narrative mode is enabled: `use_narrative=True`
- Add domain-specific terms to knowledge base
- Increase knowledge weight: `knowledge_weight: 0.4`
