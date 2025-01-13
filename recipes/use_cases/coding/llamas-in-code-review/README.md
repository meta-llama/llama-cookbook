## Llamas in Code Review

<video name="llama-code-review-loop" src="https://github.com/user-attachments/assets/f717889a-e517-4380-a07b-9657319dd189" controls></video>

In this example, we have two agents:

- **Code Author:** Writes the code.
- **Code Reviewer:** Reviews the code and provides constructive feedback.

Together, they'll engage in multiple iterations, and over time improve the code.

This demo demonstrates tool calls, structured outputs and looping with llama.

## Setup

### Prerequisites

- Python 3.10+
- Docker

### Running the demo

We'll be using the fireworks llama-stack distribution to run this example - but you can use most other llama-stack distributions (instructions [here](https://llama-stack.readthedocs.io/en/latest/distributions/index.html)).
(Though note that not all distributions support structured outputs yet e.g., Ollama).

```bash
# You can get this from https://fireworks.ai/account/api-keys - they give out initial free credits
export FIREWORKS_API_KEY=<your-api-key> 

# This runs the llama-stack server
export LLAMA_STACK_PORT=5000
docker run -it \
  -p $LLAMA_STACK_PORT:$LLAMA_STACK_PORT \
  -v ~/.llama:/root/.llama \
  llamastack/distribution-fireworks \
  --port $LLAMA_STACK_PORT \
  --env FIREWORKS_API_KEY=$FIREWORKS_API_KEY
```

Then to run the app:

```bash
# cd to this directory
cd recipes/use_cases/coding/llamas-in-code-review

# Create a virtual environment
# Use your preferred method to create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install llama-stack-client
pip install llama-stack-client

# Run the demo
export LLAMA_STACK_PORT=5000
python app.py
```

The agents will then start writing code in the ./sandbox directory.

### Configuration

You can customize the application's behavior by adjusting parameters in `app.py`:

```python
# The aim of the program
PROGRAM_OBJECTIVE="a web server that has an API endpoint that translates text from English to French."

# Number of code review cycles
CODE_REVIEW_CYCLES = 5

# The model to use
# 3.1 405B works the best, 3.3 70B works really well too, smaller models are a bit hit and miss.
MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
```