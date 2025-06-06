# API Inference

This module provides a command-line interface for interacting with Llama models through the Llama API.

## Overview

The `api_inference.py` script allows you to:
- Connect to Llama's API using your API key
- Launch a Gradio web interface for sending prompts to Llama models
- Get completions from models like Llama-4-Maverick-17B

## Prerequisites

- Python 3.8 or higher
- A valid Llama API key
- Required Python packages:
  - gradio
  - llama_api_client

## Installation

Ensure you have the required packages installed:

```bash
pip install gradio llama_api_client
```

## Usage

You can run the script from the command line using:

```bash
python api_inference.py [OPTIONS]
```

### Command-line Options

- `--api-key`: Your API key (optional)
  - If not provided, the script will look for the appropriate environment variable based on the provider
- `--provider`: API provider to use (optional, default: "Llama")
  - Available options: "Llama", "OpenAI"

### Setting Up Your API Key

You can provide your API key in one of two ways:

1. **Command-line argument**:
   ```bash
   python api_inference.py --api-key YOUR_API_KEY --provider Llama
   ```

2. **Environment variable**:
   The environment variable name depends on the provider you choose:
   ```bash
   # For Llama (default provider)
   export LLAMA_API_KEY=YOUR_API_KEY

   # For OpenAI
   export OPENAI_API_KEY=YOUR_API_KEY


   For Windows:
   ```bash
   # Command Prompt (example for Llama)
   set LLAMA_API_KEY=YOUR_API_KEY

   # PowerShell (example for Llama)
   $env:LLAMA_API_KEY="YOUR_API_KEY"
   ```

## Example

1. Run the script:
   ```bash
   # Using Llama (default provider)
   python api_inference.py --api-key YOUR_API_KEY

   # Using a different provider
   python api_inference.py --api-key YOUR_API_KEY --provider OpenAI
   ```

2. The script will launch a Gradio web interface (typically at http://127.0.0.1:7860)

3. In the interface:
   - Enter your prompt in the text box
   - The default model is "Llama-4-Maverick-17B-128E-Instruct-FP8" but you can change it
   - Click "Submit" to get a response from the model

## Troubleshooting

### API Key Issues

If you see an error like:
```
No API key provided and *_API_KEY environment variable not found
```

Make sure you've either:
- Passed the API key using the `--api-key` argument
- Set the appropriate environment variable for your chosen provider (LLAMA_API_KEY)

## Advanced Usage

You can modify the script to use different models or customize the Gradio interface as needed.

## License

[Include license information here]
