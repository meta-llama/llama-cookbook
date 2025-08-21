# DocumentLens: Rich Document Parsing with LLMs

A powerful, LLM-based tool for extracting structured data from rich documents (PDFs) with Llama models.

## Overview

This tool uses Llama models to extract text, tables, and images from PDFs, converting unstructured document data into structured, machine-readable formats. It supports:

- **Text extraction**: Extract and structure main text, titles, captions, etc.
- **Table extraction**: Convert complex tables into structured data formats
- **Image extraction**: Extract images with contextual descriptions and captions
- **Multiple output formats**: JSON, CSV, Excel, and SQL database storage
- **Vector search capabilities**: Semantic search across extracted content

The tool is designed to handle complex documents with high accuracy and provides flexible configuration options to tailor extraction tasks to specific needs.

## Installation

### Prerequisites

- Python 3.9+
- [Optional] Local GPU for offline inference

### Setup

1. Clone the repository
```bash
git clone https://github.com/meta-llama/llama-cookbook.git
```
```bash
cd llama-cookbook
```
```bash
pip install -r requirements.txt
```
2. Install project specific dependencies:
```bash
cd end-to-end-use-cases/structured_parser
```
```bash
pip install -r requirements.txt
```

3. Configure the tool (see Configuration section)

## Quick Start

Extract text from a PDF:
```bash
python src/structured_extraction.py path/to/document.pdf --text
```

Extract text and tables, and save tables as CSV files:

```bash
python src/structured_extraction.py path/to/document.pdf --text --tables --save_tables_as_csv
```

Process a directory of PDFs and export tables to Excel:

```bash
python src/structured_extraction.py path/to/pdf_directory --text --tables --export_excel
```

## Configuration

The tool is configured via `config.yaml`. Key configuration options include:

### Model Configuration

```yaml
model:
  backend: openai-compat  # [offline-vllm, openai-compat]

  # For openai-compat
  base_url: "https://api.llama.com/compat/v1"
  api_key: "YOUR_API_KEY"
  model_id: "Llama-4-Maverick-17B-128E-Instruct-FP8"

  # For offline-vllm
  path: "/path/to/checkpoint"
  tensor_parallel_size: 4
  max_model_len: 32000
  max_num_seqs: 32
```

### Inference Parameters

```yaml
extraction_inference:
  temperature: 0.2
  top_p: 0.9
  max_completion_tokens: 17000
  seed: 42
```

### Artifact Configuration

The tool includes configurable prompts and output schemas for each artifact type (text, tables, images). These can be modified in the `config.yaml` file.

## Architecture

### Core Components

1. **RequestBuilder**: Builds inference requests for LLMs
2. **ArtifactExtractor**: Extracts structured data from documents
3. **DatabaseManager**: Manages SQL database operations
4. **VectorIndexManager**: Handles vector indexing and search

### Data Flow

1. PDFs are converted to images (one per page)
2. Images are processed by the LLM to extract structured data
3. Structured data is saved in various formats (JSON, CSV, SQL, etc.)
4. Optional vector indexing for semantic search capabilities

## Extending the Tool

### Adding New Artifact Types

1. Add a new artifact type configuration in `config.yaml`:

```yaml
artifacts:
  my_new_artifact:
    prompts:
      system: "Your system prompt here..."
      user: "Your user prompt with {schema} placeholder..."
    output_schema: {
      # Your JSON schema here
    }
    use_json_decoding: true
```

2. Update the command-line interface in `structured_extraction.py` to include your new artifact type.

### Customizing Extraction Logic

The extraction logic is modular and can be customized by:

1. Modifying prompts in the `config.yaml` file
2. Adjusting output schemas to capture different data structures
3. Extending the `ArtifactExtractor` class for specialized extraction needs

### Using Different Models

The tool supports two backends:

1. **openai-compat**: Any API compatible with the OpenAI API format (including Llama API)
2. **offline-vllm**: Local inference using VLLM for self-hosted deployments

## Database Integration

### SQL Database

The tool can store extracted data in an SQLite database:

```bash
python src/structured_extraction.py path/to/document.pdf --text --tables --save_to_db
```

### Vector Search

When `save_to_db` is enabled and a vector database path is configured, the tool also indexes extracted content for semantic search:

```python
from src.json_to_sql import VectorIndexManager

# Search for relevant content
results = VectorIndexManager.knn_query("What is the revenue growth?", "chroma.db")
```

## Best Practices

1. **Model Selection**: Use larger models for complex documents or when high accuracy is required
2. **Prompt Engineering**: Adjust prompts in `config.yaml` for your specific document types
3. **Output Schema**: Define precise schemas to guide the model's extraction process
4. **Batch Processing**: Use directory processing for efficiently handling multiple documents
5. **Performance Tuning**: Adjust inference parameters based on your accuracy vs. speed requirements

## Limitations

- PDF rendering quality affects extraction accuracy
- Complex multi-column layouts may require specialized prompts
- Very large tables might be truncated due to token limitations

## Advanced Use Cases

### Custom Processing Pipelines

The tool's components can be used programmatically for custom pipelines:

```python
from src.structured_extraction import ArtifactExtractor
from src.utils import PDFUtils

# Extract pages from PDF
pages = PDFUtils.extract_pages("document.pdf")

# Process specific pages
for page in pages[10:20]:  # Process pages 10-19
    artifacts = ArtifactExtractor.from_image(page["image_path"], ["text", "tables"])
    # Custom processing of artifacts...
```

### Export to Other Systems

Extracted data can be exported to various systems:

- **SQL databases**: Using `flatten_json_to_sql`
- **CSV files**: Using `json_to_csv`
- **Excel workbooks**: Using `export_csvs_to_excel_tabs`

## Troubleshooting

- **Model capacity errors**: Reduce max tokens or use a larger model
- **Extraction quality issues**: Adjust prompts or output schemas
- **Performance issues**: Use batch processing or adjust tensor parallelism

## Contributing

Contributions to improve the tool are welcome! Areas for improvement include:

- Additional output formats
- Improved table extraction for complex layouts
- Support for more document types beyond PDFs
- Optimization for specific document domains

## License

[License information here]
