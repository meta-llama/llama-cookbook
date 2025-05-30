from __future__ import annotations

import argparse
import logging
import os
import sys

import gradio as gr
from llama_api_client import LlamaAPIClient


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
LOG: logging.Logger = logging.getLogger(__name__)

class LlamaInference:
    def __init__(self, api_key: str):
        self.client = LlamaAPIClient(
            api_key=api_key,
            base_url="https://api.llama.com/v1/",
        )

    def infer(self, user_input: str, model_id: str):
        response = self.client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": user_input}]
        )
        return response.completion_message.content.text

    def launch_interface(self):
        demo = gr.Interface(
            fn=self.infer,
            inputs=[gr.Textbox(), gr.Text("Llama-4-Maverick-17B-128E-Instruct-FP8")],
            outputs=gr.Textbox(),
        )
        print("launching interface")
        demo.launch()

def main() -> None:
    """
    Main function to handle API-based LLM inference.
    Parses command-line arguments, sets they api key, and launches the inference UI.
    """
    print("starting the main function")
    parser = argparse.ArgumentParser(
        description="Perform inference using API-based LLAMA LLMs"
    )

    parser.add_argument(
        "--api-key",
        type=str,
        help="API key for authentication (if not provided, will look for OPENAI_API_KEY or ANYSCALE_API_KEY environment variable)",
    )
    args = parser.parse_args()

    api_key: Optional[str] = args.api_key
    if api_key is not None:
        os.environ["LLAMA_API_KEY"] = api_key
    else:
        env_var_name = f"{args.provider.upper()}_API_KEY"
        api_key = os.environ.get(env_var_name)
        if api_key is None:
            LOG.error(
                f"No API key provided and {env_var_name} environment variable not found"
            )
            sys.exit(1)
    inference = LlamaInference(api_key)
    inference.launch_interface()

if __name__ == "__main__":
    main()
