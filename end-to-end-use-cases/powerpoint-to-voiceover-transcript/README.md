# PowerPoint to Voiceover Transcript

A Llama 4 powered solution that converts PowerPoint presentations into text-to-speech ready voiceover transcripts. Designed for creating professional narration content from slide decks.

## Overview

This system extracts speaker notes and visual content from PowerPoint files, then uses the Llama 4 Maverick model to generate natural-sounding transcripts optimized for human voiceover or text-to-speech systems. The generated transcripts include proper pronunciation of technical terms, numbers, and model names.

### Key Features

- **AI-Powered Analysis**: Uses Llama 4 Maverick to understand slide content and context
- **Narrative Continuity**: Advanced workflow maintains context across slides for smooth transitions
- **Speech Optimization**: Converts numbers, decimals, and technical terms to spoken form
- **Flexible Processing**: Supports both individual slides and batch processing
- **Cross-Platform**: Works on Windows, macOS, and Linux
- **Production Ready**: Comprehensive error handling, progress tracking, and retry logic

## Quick Start

### Prerequisites

- Python 3.12+
- LibreOffice (for PPTX conversion)
- Llama API key

### Installation

#### Option 1: Using uv (Recommended - Faster)

1. **Install uv (if not already installed):**
   ```bash
   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

   # Or via pip
   pip install uv
   ```

2. **Clone and install dependencies:**
   ```bash
   git clone <repository-url>
   cd powerpoint-to-voiceover-transcript
   uv sync
   ```

3. **Activate the virtual environment:**
   ```bash
   source .venv/bin/activate  # macOS/Linux
   # or
   .venv\Scripts\activate     # Windows
   ```

#### Option 2: Using pip (Traditional)

1. **Clone and install dependencies:**
   ```bash
   git clone https://github.com/meta-llama/llama-cookbook.git
   cd powerpoint-to-voiceover-transcript
   pip install -e .
   ```

2. **Install LibreOffice:**
   - **macOS**: `brew install --cask libreoffice`
   - **Ubuntu**: `sudo apt-get install libreoffice`
   - **Windows**: Download from [libreoffice.org](https://www.libreoffice.org/download/)

3. **Set up environment:**
   ```bash
   cp .env.example .env
   # Edit .env and add your LLAMA_API_KEY
   ```

4. **Configure your presentation:**
   ```bash
   # Edit config.yaml - update the pptx_file path
   current_project:
     pptx_file: "input/your_presentation_name"
     extension: ".pptx"
   ```

### Basic Usage

#### Narrative Continuity Workflow
For presentations requiring smooth narrative flow and consistent terminology:
```bash
jupyter notebook narrative_continuity_workflow.ipynb
```

This workflow uses previous slide transcripts as context to maintain narrative continuity and ensure smooth transitions between slides. Features include:
- **Context-aware processing**: Uses 5 previous slides as context by default
- **Consistent terminology**: Maintains terminology consistency throughout the presentation
- **Smooth transitions**: Generates natural flow between slides
- **Enhanced output**: Includes narrative context analysis and relationship mapping

Or use the Python API:
```python
from src.core.pptx_processor import pptx_to_images_and_notes
from src.processors.transcript_generator import TranscriptProcessor

# Convert PPTX and extract notes
result = pptx_to_images_and_notes("presentation.pptx", "output/")

# Generate transcripts
processor = TranscriptProcessor()
transcripts = processor.process_slides_dataframe(result['notes_df'], "output/")

# Save results
transcripts.to_csv("transcripts.csv", index=False)
```

## Project Structure

```
powerpoint-to-voiceover-transcript/
├── README.md                          # This file
├── config.yaml                        # Main configuration
├── pyproject.toml                     # Dependencies and project metadata
├── uv.lock                            # uv dependency lock file
├── narrative_continuity_workflow.ipynb # Enhanced narrative-aware workflow
├── .env.example                       # Environment template
├── input/                             # Place your PPTX files here
└── src/
    ├── config/
    │   └── settings.py                # Configuration management
    ├── core/
    │   ├── file_utils.py              # File system utilities
    │   ├── image_processing.py        # Image encoding for API
    │   ├── llama_client.py            # Llama API integration
    │   └── pptx_processor.py          # PPTX extraction and conversion
    └── processors/
        ├── transcript_generator.py    # Standard AI transcript generation
        └── narrative_transcript_generator.py # Narrative-aware processing
```

## Configuration

The system uses `config.yaml` for settings:

```yaml
# API Configuration
api:
  llama_model: "Llama-4-Maverick-17B-128E-Instruct-FP8"
  max_retries: 3

# Processing Settings
processing:
  default_dpi: 200
  default_format: "png"
  batch_size: 5

# Your Project
current_project:
  pptx_file: "input/your_presentation"
  extension: ".pptx"
  output_dir: "output/"
```

## API Reference

### Core Functions

#### `pptx_to_images_and_notes(pptx_path, output_dir)`
Converts PowerPoint to images and extracts speaker notes.

**Returns:** Dictionary with `image_files`, `notes_df`, and `output_dir`

#### `TranscriptProcessor()`
Main class for generating AI transcripts.

**Methods:**
- `process_slides_dataframe(df, output_dir)` - Process all slides
- `process_single_slide(image_path, speaker_notes)` - Process one slide

#### `NarrativeTranscriptProcessor(context_window_size=5)`
Enhanced class for narrative-aware transcript generation.

**Methods:**
- `process_slides_dataframe_with_narrative(df, output_dir)` - Process with context
- `process_single_slide_with_context(image_path, speaker_notes, context)` - Process with previous slides

### Speech Optimization

The AI automatically converts technical content for natural speech:

- **Decimals**: `3.2` → "three dot two"
- **Model names**: `LLaMA-3.2` → "LLaMA three dot two"
- **Abbreviations**: `LLM` → "L L M"
- **Large numbers**: `70B` → "seventy billion"

## Requirements

### System Dependencies
- **LibreOffice**: Required for PPTX to PDF conversion
- **Python 3.12+**: Core runtime

### Python Dependencies
- `pandas>=2.3.1` - Data processing
- `python-pptx>=1.0.2` - PowerPoint file handling
- `pymupdf>=1.24.0` - PDF to image conversion
- `llama-api-client>=0.1.0` - AI model access
- `pillow>=11.3.0` - Image processing
- `pyyaml>=6.0.0` - Configuration management

See `pyproject.toml` for complete dependency list.

## Output

### Narrative Continuity Workflow Output
Enhanced output includes:

1. **Narrative-Aware Transcripts**: Context-aware voiceover content with smooth transitions
2. **Context Analysis**: Information about how previous slides influenced each transcript
3. **Narrative Summary**: Overall analysis of presentation flow and consistency
4. **Multiple Formats**: CSV, JSON exports with context information
5. **Context Files**: Detailed narrative context data for each slide

## Troubleshooting

### Common Issues

**"LibreOffice not found"**
- Install LibreOffice or update paths in `config.yaml`

**"API key not found"**
- Set `LLAMA_API_KEY` in your `.env` file

**"Permission denied"**
- Ensure write permissions to output directories

**"Invalid image format"**
- Use supported formats: `png`, `jpeg`, `jpg`

**"uv sync fails"**
- Make sure you have Python 3.12+ installed
- Try `uv python install 3.12` to install Python via uv

**"Context window too large"**
- Reduce `context_window_size` parameter in narrative workflow
- Default is 5 slides, try 3 for shorter presentations

---
