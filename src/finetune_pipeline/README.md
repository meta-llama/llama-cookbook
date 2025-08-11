# Finetune Pipeline

A comprehensive end-to-end pipeline for fine-tuning large language models using TorchTune and running inference with vLLM. This pipeline provides a unified interface for data loading, model fine-tuning, and inference with support for various data formats and distributed training.

## Features

- **Data Loading & Formatting**: Support for multiple formats (TorchTune, vLLM, OpenAI)
- **Flexible Fine-tuning**: LoRA and full fine-tuning with single-device and distributed training
- **Inference**: High-performance inference using vLLM
- **Configuration-driven**: YAML/JSON configuration files for reproducible experiments
- **Modular Design**: Use individual components or run the full pipeline
- **Multi-format Support**: Works with Hugging Face datasets and local data

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage](#usage)
- [Module Structure](#module-structure)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Contributing](#contributing)

## Installation

### Prerequisites

```bash
# Core dependencies
pip install torch torchtune vllm datasets pyyaml tqdm

# Optional dependencies for development
pip install pytest black flake8
```

### Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd finetune_pipeline

# Install in development mode
pip install -e .
```

## Quick Start

1. **Create a configuration file** (`config.yaml`):

```yaml
output_dir: "/path/to/output/"

formatter:
  type: "torchtune"
  data_path: "dz-osamu/IU-Xray"
  is_local: false
  column_mapping:
    input: "query"
    output: "response"
    image: null
  dataset_kwargs:
    split: "validation"

finetuning:
  model_path: "/path/to/Llama-3.1-8B-Instruct"
  tokenizer_path: "/path/to/tokenizer.model"
  strategy: "lora"
  num_epochs: 1
  batch_size: 1
  torchtune_config: "llama3_1/8B_lora"
  num_processes_per_node: 8
  distributed: true

inference:
  model_path: "/path/to/model"
  port: 8000
  host: "0.0.0.0"
  tensor_parallel_size: 1
  max_model_len: 512
  gpu_memory_utilization: 0.95
  inference_data: "dz-osamu/IU-Xray"
```

2. **Run the full pipeline**:

```bash
python run_pipeline.py --config config.yaml
```

3. **Run individual steps**:

```bash
# Data loading only
python run_pipeline.py --config config.yaml --only-data-loading

# Fine-tuning only
python run_pipeline.py --config config.yaml --only-finetuning

# Inference only
python run_pipeline.py --config config.yaml --only-inference
```

## Configuration

The pipeline uses YAML or JSON configuration files with the following main sections:

### Global Configuration

- `output_dir`: Base directory for all outputs

### Data Formatting (`formatter`)

- `type`: Formatter type (`"torchtune"`, `"vllm"`, `"openai"`)
- `data_path`: Path to dataset (Hugging Face ID or local path)
- `is_local`: Whether data is stored locally
- `column_mapping`: Map dataset columns to standard fields
- `dataset_kwargs`: Additional arguments for data loading

### Fine-tuning (`finetuning`)

- `model_path`: Path to base model
- `tokenizer_path`: Path to tokenizer
- `strategy`: Training strategy (`"lora"` or `"fft"`)
- `num_epochs`: Number of training epochs
- `batch_size`: Batch size per device
- `torchtune_config`: TorchTune configuration name
- `distributed`: Enable distributed training
- `num_processes_per_node`: Number of processes for distributed training

### Inference (`inference`)

- `model_path`: Path to model for inference
- `port`: Server port
- `host`: Server host
- `tensor_parallel_size`: Number of GPUs for tensor parallelism
- `max_model_len`: Maximum sequence length
- `gpu_memory_utilization`: GPU memory utilization fraction
- `inference_data`: Dataset for inference

## Usage

### Command Line Interface

The main pipeline script provides several options:

```bash
# Full pipeline
python run_pipeline.py --config config.yaml

# Skip specific steps
python run_pipeline.py --config config.yaml --skip-finetuning --skip-inference

# Run only specific steps
python run_pipeline.py --config config.yaml --only-data-loading
python run_pipeline.py --config config.yaml --only-finetuning
python run_pipeline.py --config config.yaml --only-inference
```

### Individual Components

#### Data Loading and Formatting

```python
from finetune_pipeline.data.data_loader import load_and_format_data, read_config

config = read_config("config.yaml")
formatter_config = config.get("formatter", {})
output_dir = config.get("output_dir", "/tmp/")

formatted_data_paths, conversation_data_paths = load_and_format_data(
    formatter_config, output_dir
)
```

#### Fine-tuning

```python
from finetune_pipeline.finetuning.run_finetuning import run_torch_tune

config = read_config("config.yaml")
finetuning_config = config.get("finetuning", {})

run_torch_tune(finetuning_config, config)
```

#### Inference

```python
from finetune_pipeline.inference.run_inference import run_vllm_batch_inference_on_dataset

results = run_vllm_batch_inference_on_dataset(
    data_path="dataset_name",
    model_path="/path/to/model",
    is_local=False,
    temperature=0.0,
    max_tokens=100,
    # ... other parameters
)
```

## Module Structure

```
finetune_pipeline/
├── __init__.py
├── config.yaml                 # Example configuration
├── run_pipeline.py             # Main pipeline orchestrator
│
├── data/                       # Data loading and formatting
│   ├── __init__.py
│   ├── data_loader.py          # Dataset loading utilities
│   ├── formatter.py            # Data format converters
│   └── augmentation.py         # Data augmentation utilities
│
├── finetuning/                 # Fine-tuning components
│   ├── __init__.py
│   ├── run_finetuning.py       # TorchTune fine-tuning script
│   └── custom_sft_dataset.py   # Custom dataset for supervised fine-tuning
│
├── inference/                  # Inference components
│   ├── __init__.py
│   ├── run_inference.py        # Batch inference utilities
│   ├── start_vllm_server.py    # vLLM server management
│   └── save_inference_results.py # Result saving utilities
│
└── tests/                      # Test suite
    ├── __init__.py
    ├── test_formatter.py
    └── test_finetuning.py
```

## API Reference

### Data Components

#### `Formatter` (Abstract Base Class)
- `format_data(data)`: Format list of conversations
- `format_conversation(conversation)`: Format single conversation
- `format_message(message)`: Format single message

#### `TorchtuneFormatter`
Formats data for TorchTune training with message-based structure.

#### `vLLMFormatter`
Formats data for vLLM inference with optimized structure.

#### `OpenAIFormatter`
Formats data compatible with OpenAI API format.

### Fine-tuning Components

#### `run_torch_tune(training_config, config, args=None)`
Execute TorchTune training with configuration-based parameters.

**Parameters:**
- `training_config`: Training configuration section
- `config`: Full configuration dictionary
- `args`: Additional command-line arguments

### Inference Components

#### `run_vllm_batch_inference_on_dataset(...)`
Run batch inference on a dataset using vLLM.

**Key Parameters:**
- `data_path`: Path to dataset
- `model_path`: Path to model
- `temperature`: Sampling temperature
- `max_tokens`: Maximum tokens to generate
- `gpu_memory_utilization`: GPU memory usage fraction

## Examples

### Example 1: Medical QA Fine-tuning

```yaml
# config_medical_qa.yaml
output_dir: "/workspace/medical_qa_output/"

formatter:
  type: "torchtune"
  data_path: "medical-qa-dataset"
  column_mapping:
    input: "question"
    output: "answer"

finetuning:
  model_path: "/models/Llama-3.1-8B-Instruct"
  strategy: "lora"
  num_epochs: 3
  distributed: true
  num_processes_per_node: 4

inference:
  model_path: "/workspace/medical_qa_output/"
  max_model_len: 1024
  temperature: 0.1
```

```bash
python run_pipeline.py --config config_medical_qa.yaml
```

### Example 2: Multi-modal Fine-tuning

```yaml
# config_multimodal.yaml
formatter:
  type: "torchtune"
  data_path: "multimodal-dataset"
  column_mapping:
    input: "query"
    output: "response"
    image: "image_path"

finetuning:
  strategy: "lora"
  torchtune_config: "llama3_2_vision/11B_lora"
  # ... other config
```

### Example 3: Distributed Training

```yaml
# config_distributed.yaml
finetuning:
  distributed: true
  num_processes_per_node: 8
  strategy: "lora"
  # ... other config
```

```bash
# Run with distributed training
python run_pipeline.py --config config_distributed.yaml
```

### Example 4: Custom Dataset Format

```python
# Custom data loading
from finetune_pipeline.data.formatter import TorchtuneFormatter, Conversation, Message

# Create conversations
conversations = []
conversation = Conversation()
conversation.add_message(Message(role="user", content="What is AI?"))
conversation.add_message(Message(role="assistant", content="AI is..."))
conversations.append(conversation)

# Format for training
formatter = TorchtuneFormatter()
formatted_data = formatter.format_data(conversations)
```

## Advanced Usage

### Custom Arguments

Pass additional arguments to TorchTune:

```bash
python finetuning/run_finetuning.py \
  --config config.yaml \
  --kwargs "dataset.train_on_input=True optimizer.lr=1e-5"
```

### Pipeline Control

Fine-grained control over pipeline execution:

```bash
# Skip certain steps
python run_pipeline.py --config config.yaml --skip-finetuning --skip-server

# Run only data loading and fine-tuning
python run_pipeline.py --config config.yaml --skip-inference --skip-server
```

### Configuration Validation

The pipeline automatically validates configuration parameters and provides helpful error messages for missing or invalid settings.

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce `batch_size`, `max_model_len`, or `gpu_memory_utilization`
2. **Import Errors**: Ensure all dependencies are installed (`torch`, `torchtune`, `vllm`)
3. **Configuration Errors**: Check YAML syntax and required fields
4. **Distributed Training Issues**: Verify `num_processes_per_node` matches available GPUs

### Debug Mode

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run the test suite: `pytest tests/`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
