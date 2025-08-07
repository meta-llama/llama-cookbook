import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# Add the parent directory to the path so we can import the modules
sys.path.append(str(Path(__file__).parent.parent))

from data.data_loader import convert_to_conversations, format_data, load_data
from data.formatter import (
    Conversation,
    OpenAIFormatter,
    TorchtuneFormatter,
    vLLMFormatter,
)


class TestFormatter(unittest.TestCase):
    """Test cases for the formatter module."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures, called before any tests are run."""
        # Define a small dataset to use for testing
        cls.dataset_name = "dz-osamu/IU-Xray"
        cls.split = "train[:10]"  # Use only 10 samples for testing

        try:
            # Load the dataset
            cls.dataset = load_data(cls.dataset_name, split=cls.split)

            # Create a column mapping for the dz-osamu/IU-Xray dataset
            cls.column_mapping = {
                "input": "query",
                "output": "response",
                "image": "images",
            }

            # Convert to list for easier processing
            cls.data = list(cls.dataset)

            # Convert to conversations
            cls.conversations = convert_to_conversations(cls.data, cls.column_mapping)

        except Exception as e:
            print(f"Error setting up test fixtures: {e}")
            raise

    def test_conversation_creation(self):
        """Test that conversations are created correctly."""
        self.assertIsNotNone(self.conversations)
        self.assertGreater(len(self.conversations), 0)

        # Check that each conversation has at least two messages (user and assistant)
        for conversation in self.conversations:
            self.assertGreaterEqual(len(conversation.messages), 2)
            self.assertEqual(conversation.messages[0]["role"], "user")
            self.assertEqual(conversation.messages[1]["role"], "assistant")

    def test_torchtune_formatter(self):
        """Test the TorchtuneFormatter."""
        formatter = TorchtuneFormatter()

        # Test format_data
        formatted_data = formatter.format_data(self.conversations)
        self.assertIsNotNone(formatted_data)
        self.assertEqual(len(formatted_data), len(self.conversations))

        # Test format_conversation
        formatted_conversation = formatter.format_conversation(self.conversations[0])
        self.assertIsInstance(formatted_conversation, dict)
        self.assertIn("messages", formatted_conversation)

        # Test format_message
        message = self.conversations[0].messages[0]
        formatted_message = formatter.format_message(message)
        self.assertIsInstance(formatted_message, dict)
        self.assertIn("role", formatted_message)
        self.assertIn("content", formatted_message)

    def test_vllm_formatter(self):
        """Test the vLLMFormatter."""
        formatter = vLLMFormatter()

        # Test format_data
        formatted_data = formatter.format_data(self.conversations)
        self.assertIsNotNone(formatted_data)
        self.assertEqual(len(formatted_data), len(self.conversations))

        # Test format_conversation
        formatted_conversation = formatter.format_conversation(self.conversations[0])
        self.assertIsInstance(formatted_conversation, str)

        # Test format_message
        message = self.conversations[0].messages[0]
        formatted_message = formatter.format_message(message)
        self.assertIsInstance(formatted_message, str)
        self.assertIn(message["role"], formatted_message)

    def test_openai_formatter(self):
        """Test the OpenAIFormatter."""
        formatter = OpenAIFormatter()

        # Test format_data
        formatted_data = formatter.format_data(self.conversations)
        self.assertIsNotNone(formatted_data)
        self.assertEqual(len(formatted_data), len(self.conversations))

        # Test format_conversation
        formatted_conversation = formatter.format_conversation(self.conversations[0])
        self.assertIsInstance(formatted_conversation, dict)
        self.assertIn("messages", formatted_conversation)

        # Test format_message
        message = self.conversations[0].messages[0]
        formatted_message = formatter.format_message(message)
        self.assertIsInstance(formatted_message, dict)
        self.assertIn("role", formatted_message)
        self.assertIn("content", formatted_message)

    def test_format_data_function(self):
        """Test the format_data function from data_loader."""
        # Test with TorchtuneFormatter
        torchtune_data, torchtune_conversations = format_data(
            self.data, "torchtune", self.column_mapping
        )
        self.assertIsNotNone(torchtune_data)
        self.assertEqual(len(torchtune_data), len(self.data))
        self.assertEqual(len(torchtune_conversations), len(self.data))

        # Test with vLLMFormatter
        vllm_data, vllm_conversations = format_data(
            self.data, "vllm", self.column_mapping
        )
        self.assertIsNotNone(vllm_data)
        self.assertEqual(len(vllm_data), len(self.data))
        self.assertEqual(len(vllm_conversations), len(self.data))

        # Test with OpenAIFormatter
        openai_data, openai_conversations = format_data(
            self.data, "openai", self.column_mapping
        )
        self.assertIsNotNone(openai_data)
        self.assertEqual(len(openai_data), len(self.data))
        self.assertEqual(len(openai_conversations), len(self.data))

    def test_with_mock_data(self):
        """Test the formatter pipeline with mock data."""
        # Create mock data that mimics a dataset
        mock_data = [
            {
                "question": "What is the capital of France?",
                "context": "France is a country in Western Europe. Its capital is Paris.",
                "answer": "Paris",
            },
            {
                "question": "Who wrote Hamlet?",
                "context": "Hamlet is a tragedy written by William Shakespeare.",
                "answer": "William Shakespeare",
            },
            {
                "question": "What is the largest planet in our solar system?",
                "context": "Jupiter is the largest planet in our solar system.",
                "answer": "Jupiter",
            },
        ]

        # Create a column mapping for the mock data
        column_mapping = {"input": "context", "output": "answer"}

        # Convert to conversations
        conversations = convert_to_conversations(mock_data, column_mapping)

        # Test that conversations are created correctly
        self.assertEqual(len(conversations), len(mock_data))
        for i, conversation in enumerate(conversations):
            self.assertEqual(len(conversation.messages), 2)
            self.assertEqual(conversation.messages[0]["role"], "user")
            self.assertEqual(conversation.messages[1]["role"], "assistant")

            # Check content of user message
            user_content = conversation.messages[0]["content"]
            self.assertTrue(isinstance(user_content, list))
            self.assertEqual(user_content[0]["type"], "text")
            self.assertEqual(user_content[0]["text"], mock_data[i]["context"])

            # Check content of assistant message
            assistant_content = conversation.messages[1]["content"]
            self.assertTrue(isinstance(assistant_content, list))
            self.assertEqual(assistant_content[0]["type"], "text")
            self.assertEqual(assistant_content[0]["text"], mock_data[i]["answer"])

        # Test each formatter with the mock data
        formatters = {
            "torchtune": TorchtuneFormatter(),
            "vllm": vLLMFormatter(),
            "openai": OpenAIFormatter(),
        }

        for name, formatter in formatters.items():
            formatted_data = formatter.format_data(conversations)
            self.assertEqual(len(formatted_data), len(mock_data))

            # Test the first formatted item
            if name == "vllm":
                # vLLM formatter returns strings
                self.assertTrue(isinstance(formatted_data[0], str))
                self.assertIn("user:", formatted_data[0])
                self.assertIn("assistant:", formatted_data[0])
            else:
                # Torchtune and OpenAI formatters return dicts
                self.assertTrue(isinstance(formatted_data[0], dict))
                self.assertIn("messages", formatted_data[0])
                self.assertEqual(len(formatted_data[0]["messages"]), 2)


if __name__ == "__main__":
    # If run as a script, this allows passing a dataset name as an argument
    import argparse

    parser = argparse.ArgumentParser(
        description="Test the formatter module with a specific dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dz-osamu/IU-Xray",
        help="Name of the Hugging Face dataset to use for testing",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train[:10]",
        help="Dataset split to use (e.g., 'train[:10]', 'validation[:10]')",
    )

    args = parser.parse_args()

    # Override the default dataset in the test class
    TestFormatter.dataset_name = args.dataset
    TestFormatter.split = args.split

    # Run the tests
    unittest.main(argv=["first-arg-is-ignored"])