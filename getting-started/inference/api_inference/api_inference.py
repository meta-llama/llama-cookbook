from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Optional

import gradio as gr
from llama_api_client import LlamaAPIClient
from openai import OpenAI


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
LOG: logging.Logger = logging.getLogger(__name__)


class LlamaInference:
    def __init__(self, api_key: str, provider: str):
        self.provider = provider
        if self.provider == "Llama":
            self.client = LlamaAPIClient(
                api_key=api_key,
                base_url="https://api.llama.com/v1/",
            )
        elif self.provider == "OpenAI":
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://api.llama.com/compat/v1/",
            )

    def infer(self, user_input: str, model_id: str):
        response = self.client.chat.completions.create(
            model=model_id, messages=[{"role": "user", "content": user_input}]
        )
        if self.provider == "Llama":
            return response.completion_message.content.text
        return response.choices[0].message.content

    def launch_interface(self):
        if self.provider == "Llama":
            demo = gr.Interface(
                fn=self.infer,
                inputs=[
                    gr.Textbox(),
                    gr.Text("Llama-4-Maverick-17B-128E-Instruct-FP8"),
                ],
                outputs=gr.Textbox(),
            )
        elif self.provider == "OpenAI":
            demo = gr.Interface(
                fn=self.infer,
                inputs=[gr.Textbox(), gr.Text("Llama-3.3-8B-Instruct")],
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
        help="API key for authentication (if not provided, will look for PROVIDER_API_KEY environment variable)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="OpenAI",
        choices=["Llama", "OpenAI"],
        help="API provider to use (default: Llama)",
    )
    args = parser.parse_args()

    api_key: Optional[str] = args.api_key
    env_var_name = f"LLAMA_API_KEY"

    if api_key is not None:
        os.environ[env_var_name] = api_key
    else:
        api_key = os.environ.get(env_var_name)
        if api_key is None:
            LOG.error(
                f"No API key provided and {env_var_name} environment variable not found"
            )
            sys.exit(1)
    inference = LlamaInference(api_key, args.provider)
    inference.launch_interface()


if __name__ == "__main__":
    main()
