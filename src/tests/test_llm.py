# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import unittest
from unittest.mock import patch

import anthropic
import httpx
import openai

from llama_cookbook.inference.llm import MINIMAX

OPENAI_CLIENT = openai.OpenAI
ANTHROPIC_CLIENT = anthropic.Anthropic


class MiniMaxTest(unittest.TestCase):
    def test_supported_models_and_endpoint_paths(self) -> None:
        cases = [
            (
                "openai",
                "MiniMax-M3",
                None,
                "https://api.minimax.io/v1/chat/completions",
            ),
            (
                "openai",
                "MiniMax-M2.7",
                "https://api.minimaxi.com/v1",
                "https://api.minimaxi.com/v1/chat/completions",
            ),
            (
                "anthropic",
                "MiniMax-M3",
                None,
                "https://api.minimax.io/anthropic/v1/messages",
            ),
            (
                "anthropic",
                "MiniMax-M2.7",
                "https://api.minimaxi.com/anthropic",
                "https://api.minimaxi.com/anthropic/v1/messages",
            ),
        ]

        for api_format, model, base_url, expected_url in cases:
            with self.subTest(api_format=api_format, base_url=base_url):
                requests = []

                def handler(request: httpx.Request) -> httpx.Response:
                    requests.append(request)
                    if api_format == "openai":
                        body = {
                            "id": "chatcmpl-test",
                            "object": "chat.completion",
                            "created": 0,
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": "Test response",
                                    },
                                    "finish_reason": "stop",
                                }
                            ],
                        }
                    else:
                        body = {
                            "id": "msg_test",
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "text", "text": "Test response"}],
                            "model": model,
                            "stop_reason": "end_turn",
                            "stop_sequence": None,
                            "usage": {"input_tokens": 1, "output_tokens": 1},
                        }
                    return httpx.Response(200, json=body, request=request)

                http_client = httpx.Client(transport=httpx.MockTransport(handler))
                kwargs = {"api_format": api_format}
                if base_url is not None:
                    kwargs["base_url"] = base_url

                if api_format == "openai":
                    client = OPENAI_CLIENT(
                        api_key="test-key",
                        base_url=base_url or "https://api.minimax.io/v1",
                        http_client=http_client,
                    )
                    patch_target = "llama_cookbook.inference.llm.openai.OpenAI"
                else:
                    client = ANTHROPIC_CLIENT(
                        api_key="test-key",
                        base_url=base_url or "https://api.minimax.io/anthropic",
                        http_client=http_client,
                    )
                    patch_target = "llama_cookbook.inference.llm.anthropic.Anthropic"

                with patch(patch_target, return_value=client):
                    provider = MINIMAX(model, "test-key", **kwargs)
                    self.assertEqual(provider.query("Test prompt"), "Test response")
                    self.assertEqual(
                        provider.valid_models(), ["MiniMax-M3", "MiniMax-M2.7"]
                    )

                self.assertEqual(len(requests), 1)
                self.assertEqual(str(requests[0].url), expected_url)
                self.assertEqual(json.loads(requests[0].content)["model"], model)
                client.close()


if __name__ == "__main__":
    unittest.main()
