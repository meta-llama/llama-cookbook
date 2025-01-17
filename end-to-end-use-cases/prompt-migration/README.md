# Prompt Migration

## Overview

The prompt migration toolkit helps you assess and adapt prompts across different language models, ensuring consistent performance and reliability. It includes benchmarking capabilities and evaluation tools to measure the effectiveness of prompt migrations.

## Project Structure

- `notebooks/`: Contains Jupyter notebooks for interactive prompt migration examples
  - `harness.ipynb`: Main notebook demonstrating the prompt migration workflow
- `benchmarks/`: Tools and scripts for performance evaluation
- `environment.yml`: Conda environment specification with all required dependencies

## Setup Instructions

1. Install dependencies using Conda:
```bash
conda env create -f environment.yml
conda activate prompt-migration
```

## Key Dependencies

- Python 3.10
- DSPy: For prompt engineering and evaluation
- LM-eval: Evaluation framework for language models
- PyTorch and Transformers: For model inference

## Getting Started

1. Activate your environment using Conda as described above
2. Start Jupyter notebook server:
```bash
jupyter notebook
```
3. Navigate to the `notebooks/harness.ipynb` notebook in your browser
4. Use the benchmarking tools in the `benchmarks/` directory to evaluate your migrations

## License

This project is part of the Llama Recipes collection. Please refer to the main repository's license for usage terms.
