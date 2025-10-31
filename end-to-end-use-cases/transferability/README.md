# Transferability Research Tool

A Python package for evaluating model transferability across vision-language tasks through systematic fine-tuning and evaluation.

## Directory Structure

```
./
├── config.yaml                    # Main configuration file
├── experiments/                   # Output directory for all experiments
│   └── <experiment_name>/
│       ├── formatted_datasets/    # Processed datasets ready for training
│       ├── finetuned_checkpoints/ # Fine-tuned model checkpoints
│       ├── finetune_logs/         # Training logs
│       ├── grader_logs/           # Evaluation logs per model
│       └── eval_grid_results.json # Final evaluation results
└── transferability/               # Source code package
    ├── __init__.py               # Package entry points
    ├── __main__.py               # Main CLI entry point
    ├── data/                     # Dataset processing
    │   ├── __init__.py
    │   ├── __main__.py           # Module CLI entry point
    │   └── dataset_builder.py
    ├── datasets/                 # Dataset format utilities
    │   ├── __init__.py
    │   └── torchtune_format.py   # TorchTune dataset format
    ├── evals/                    # Evaluation utilities
    │   ├── __init__.py
    │   ├── __main__.py           # Module CLI entry point
    │   ├── eval_grid.py          # Main evaluation grid runner
    │   ├── grader.py             # Task-specific graders
    │   ├── inference.py          # Model inference utilities
    │   ├── json_grading_utils.py # JSON grading utilities
    │   └── shift_analysis.py     # Distribution shift analysis
    ├── finetune/                 # Fine-tuning utilities
    │   ├── __init__.py
    │   ├── __main__.py           # Module CLI entry point
    │   ├── finetune_grid.py      # Main fine-tuning grid runner
    │   ├── 8b_full.yaml          # TorchTune config for full fine-tuning
    │   └── 8b_lora.yaml          # TorchTune config for LoRA fine-tuning
    └── utils.py                  # Shared utilities
```

## Usage

Run individual components as Python modules:

```bash
# Prepare datasets
python -m transferability.data ./experiments/my_experiment

# Run fine-tuning grid
python -m transferability.finetune ./experiments/my_experiment

# Run evaluation grid
python -m transferability.evals ./experiments/my_experiment
```


## Configuration

Edit `config.yaml` to configure your tasks, datasets, and training parameters:

```yaml
task1:
  dataset: your/huggingface/dataset
  system_prompt: "Your system prompt"
  user_prompt: "Your user prompt"
  image_column: image
  assistant_text_column: ground_truth
  grader: JSONGrader
  sample_percent: 0.01

task2:
  # Similar structure for second task

finetuning:
  model_path: /path/to/your/base/model
  tokenizer_path: /path/to/tokenizer
  epochs: 1
  batch_size: 8
  # Fine-tuning strategy flags
  fusion: false
  fusion+encoder: false
  fusion+decoder: false
  fusion+encoder+decoder: true
  lora_ranks: [8, 16, 32]

evals:
  nb_eval_samples: null  # null = use all samples
  checkpoint_to_eval: -1  # -1 = use latest checkpoint
  model_server_args:
    tensor_parallel_size: 2
    max_model_len: 4096
```

## Workflow

1. **Configure**: Edit `config.yaml` with your tasks and model paths
2. **Prepare Data**: Download and format datasets from HuggingFace
3. **Fine-tune**: Train models using different strategies (LoRA, full fine-tuning)
4. **Evaluate**: Test all models on all tasks and generate results

## Key Features

- **Modular Design**: Each component can be run independently
- **Multiple Execution Methods**: Module-level, package-level, or direct imports
- **Configurable Tasks**: Define tasks via YAML configuration
- **Grid Search**: Automatically train multiple model variants
- **Comprehensive Evaluation**: Test transferability across tasks
- **Rich Logging**: Detailed logs and metrics for analysis

## Output Structure

Each experiment creates:
- `formatted_datasets/`: HuggingFace datasets converted to training format
- `finetuned_checkpoints/`: Model checkpoints for each training configuration
- `finetune_logs/`: Training metrics and logs
- `grader_logs/`: Per-model evaluation details
- `eval_grid_results.json`: Summary of all evaluation results

## Next Steps

The package is now properly structured for module execution. You can:

1. Update hardcoded paths in `__main__` sections (as planned)
2. Add more sophisticated CLI argument parsing
3. Add configuration validation
4. Add progress tracking and resumption capabilities
5. Add visualization utilities for results analysis
