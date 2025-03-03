# Data Prep Toolkit

If you are working on fine-tuning a Large Language Model, the biggest effort is usually preparing the dataset. 

## What does this tool do?

This tool contains a bunch of util functions to make your life easy for loading a dataset to torchtune with helpers:
- Parse files: Prepare any file type (list below) to txt
- Convert to prompt response pairs: Convert the txt file to prompt-response pairs
- Filter: Filter out the low quality prompt response pairs
- Fine-tune: Use the pre-defined configs to fine tune a LLM using torchtune

TODO: Add TT links

## Parsers:

(WIP) We support the following file formats for parsing:

- **PDF** - Extracts text from most PDF files (text-based, not scanned)
- **HTML/Web** - Pulls content from local HTML files or any webpage URL
- **YouTube** - Grabs video transcripts (does not transcribe video)
- **Word (DOCX)** - Extracts text, tables, and structure from Word docs (not images)
- **PowerPoint (PPTX)** - Pulls text from slides, notes, and tables
- **TXT** - For when you already have plain text (just copies it)

## Installation:

TODO: Supply requirements.txt file here instead

### Dependencies

```bash
# Install all dependencies at once
pip install PyPDF2 python-docx beautifulsoup4 requests python-pptx yt-dlp youtube-transcript-api
```

### Steps to run:

TODO: Add links here

1. Clone TODO: Link
2. Install TODO: req.txt
3. Run the parser on your files
4. Create prompt responses
5. Filter low quality
6. Load to TT 

## How to use:

```bash
# Parse a PDF (outputs to data/output/document.txt)
python src/main.py docs/report.pdf

# Parse a website
python src/main.py https://en.wikipedia.org/wiki/Artificial_intelligence

# Get YouTube video transcripts
python src/main.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Custom output location
python src/main.py docs/presentation.pptx -o my_training_data/

# Specify the output filename
python src/main.py docs/contract.docx -n legal_text_001.txt

# Use verbose mode for debugging
python src/main.py weird_file.pdf -v
```

All outputs are saved as UTF-8 txt files in `data/output/` unless otherwise set.

### Rough edges:

- Use quotes around YouTube URLs to avoid shell issues with the `&` character
- HTML parser works with both local files and web URLs
  - Enhanced with session persistence and retry mechanisms
  - Some sites with strong bot protection may still block access
- YouTube parser automatically extracts both manual and auto-generated captions
  - Prioritizes manual captions when available for better quality
  - Formats transcripts cleanly with proper text structure
- PDF extraction works best with digital PDFs, not scanned documents
- All parsers include error handling to gracefully manage parsing failures

## 📁 Project Layout

```
.
├── data/              # Where docs live
│   ├── pdf/           # PDF documents 
│   ├── html/          # HTML files
│   ├── youtube/       # YouTube transcript stuff
│   ├── docx/          # Word documents
│   ├── ppt/           # PowerPoint slides
│   ├── txt/           # Plain text
│   └── output/        # Where the magic happens (output)
│
├── src/               # The code that makes it tick
│   ├── parsers/       # All our parser implementations
│   │   ├── pdf_parser.py     # PDF -> text
│   │   ├── html_parser.py    # HTML/web -> text
│   │   ├── youtube_parser.py # YouTube -> text
│   │   ├── docx_parser.py    # Word -> text
│   │   ├── ppt_parser.py     # PowerPoint -> text
│   │   ├── txt_parser.py     # Text -> text (not much to do here)
│   │   └── __init__.py
│   ├── __init__.py
│   ├── main.py        # CLI entry point
│   └── generate_qa.py # Creates Q&A pairs from text
│
└── README.md          # You are here
```

## 🤖 Generate Q&A Pairs for Fine-tuning

Want to turn your documents into training data? After parsing, use the `generate_qa.py` script to create question-answer pairs using the Cerebras LLM API:

```bash
# Set your API key first
export CEREBRAS_API_KEY="your_key_here"

# Generate QA pairs from a document in three steps:
# 1. Summarize the document
# 2. Generate question-answer pairs
# 3. Evaluate & filter based on relevance
python src/generate_qa.py docs/report.pdf

# Customize the generation
python src/generate_qa.py docs/report.pdf --num-pairs 30 --threshold 7.0

# Skip parsing if you already have text
python src/generate_qa.py docs/report.pdf --text-file data/output/report.txt

# Save output to a specific directory
python src/generate_qa.py docs/report.pdf --output-dir training_data/
```

### 📊 Sample Output

The script outputs a JSON file with:
- A comprehensive summary of the document
- Question-answer pairs in a format ready for fine-tuning
- Quality metrics about the generated pairs

```jsonc
{
  "summary": "This document explains the principles of deep learning...",
  
  "qa_pairs": [
    {"from": "user", "value": "What are the key components of a neural network?"},
    {"from": "assistant", "value": "The key components of a neural network include..."},
    // More Q&A pairs...
  ],
  
  "metrics": {
    "initial_pairs": 25,        // Total generated pairs
    "filtered_pairs": 19,       // Pairs that passed quality check
    "retention_rate": 0.76,     // Percentage kept
    "average_relevance_score": 7.8  // Average quality score
  }
}
```

## Known bugs/sharp edges:

- PDFs: Some PDFs are scanned images and need OCR. This is homework to users :)
- YouTube: We assume videos have captions, if they don't, another task for readers :)


--------

## WIP BFCL FT:

### Instructions to run:

Time to setup: ~20-30 minutes:
- Grab your HF and Wandb.ai API key-you will need those
- Steps to install:

 ```
conda create -n test-ft python=3.10
conda activate test-ft
pip install --pre torch torchvision torchao --index-url https://download.pytorch.org/whl/nightly/cu126
pip install --pre torchtune --extra-index-url https://download.pytorch.org/whl/nightly/cpu --no-cache-dir
pip install transformers datasets wandb
pip install huggingface-cli
huggingface-cli login
wandb login


git clone https://github.com/meta-llama/llama-cookbook/
cd llama-cookbook/
git checkout data-tool
cd end-to-end-use-cases/data-tool/scripts/finetuning
tune download meta-llama/Meta-Llama-3.1-70B-Instruct --output-dir /tmp/Meta-Llama-3.1-70B-Instruct --ignore-patterns "original/consolidated*
tune run --nproc_per_node 8 full_finetune_distributed --config ft-config.yaml
 ```

The end goal for this effort is to serve as fine-tuning data preparation kit.

## Current status:

Currently, I'm (WIP) evaluating the idea to improve tool-calling datasets. 

Setup:
- configs: Has the config prompts for creating synthetic data using `3.3`
- data_prep/scripts: This is what you would like to run to prepare your datasets for annotation
- scripts/annotation-inference: Script for generating synthetic datasets -> Use the vllm script for inference
- fine-tuning: configs for FT using TorchTune
