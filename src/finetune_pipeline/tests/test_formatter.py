# import sys
# import unittest
# from pathlib import Path
# from unittest.mock import MagicMock

# # Add the parent directory to the path so we can import the modules
# sys.path.append(str(Path(__file__).parent.parent))

# from data.data_loader import convert_to_conversations, format_data, load_data
# from data.formatter import (
#     Conversation,
#     OpenAIFormatter,
#     TorchtuneFormatter,
#     vLLMFormatter,
# )


# class TestFormatter(unittest.TestCase):
#     """Test cases for the formatter module."""

#     @classmethod
#     def setUpClass(cls):
#         """Set up test fixtures, called before any tests are run."""
#         # Define a small dataset to use for testing
#         cls.dataset_name = "dz-osamu/IU-Xray"
#         cls.split = "train[:10]"  # Use only 10 samples for testing

#         try:
#             # Load the dataset
#             cls.dataset = load_data(cls.dataset_name, split=cls.split)

#             # Create a column mapping for the dz-osamu/IU-Xray dataset
#             cls.column_mapping = {
#                 "input": "query",
#                 "output": "response",
#                 "image": "images"
#             }

#             # Convert to list for easier processing
#             cls.data = list(cls.dataset)

#             # Convert to conversations
#             cls.conversations = convert_to_conversations(cls.data, cls.column_mapping)

#         except Exception as e:
#             print(f"Error setting up test fixtures: {e}")
#             raise

#     def test_conversation_creation(self):
#         """Test that conversations are created correctly."""
#         self.assertIsNotNone(self.conversations)
#         self.assertGreater(len(self.conversations), 0)

#         # Check that each conversation has at least two messages (user and assistant)
#         for conversation in self.conversations:
#             self.assertGreaterEqual(len(conversation.messages), 2)
#             self.assertEqual(conversation.messages[0]["role"], "user")
#             self.assertEqual(conversation.messages[1]["role"], "assistant")

#     def test_torchtune_formatter(self):
#         """Test the TorchtuneFormatter."""
#         formatter = TorchtuneFormatter()

#         # Test format_data
#         formatted_data = formatter.format_data(self.conversations)
#         self.assertIsNotNone(formatted_data)
#         self.assertEqual(len(formatted_data), len(self.conversations))

#         # Test format_conversation
#         formatted_conversation = formatter.format_conversation(self.conversations[0])
#         self.assertIsInstance(formatted_conversation, dict)
#         self.assertIn("messages", formatted_conversation)

#         # Test format_message
#         message = self.conversations[0].messages[0]
#         formatted_message = formatter.format_message(message)
#         self.assertIsInstance(formatted_message, dict)
#         self.assertIn("role", formatted_message)
#         self.assertIn("content", formatted_message)

#     def test_vllm_formatter(self):
#         """Test the vLLMFormatter."""
#         formatter = vLLMFormatter()

#         # Test format_data
#         formatted_data = formatter.format_data(self.conversations)
#         self.assertIsNotNone(formatted_data)
#         self.assertEqual(len(formatted_data), len(self.conversations))

#         # Test format_conversation
#         formatted_conversation = formatter.format_conversation(self.conversations[0])
#         self.assertIsInstance(formatted_conversation, str)

#         # Test format_message
#         message = self.conversations[0].messages[0]
#         formatted_message = formatter.format_message(message)
#         self.assertIsInstance(formatted_message, str)
#         self.assertIn(message["role"], formatted_message)

#     def test_openai_formatter(self):
#         """Test the OpenAIFormatter."""
#         formatter = OpenAIFormatter()

#         # Test format_data
#         formatted_data = formatter.format_data(self.conversations)
#         self.assertIsNotNone(formatted_data)
#         self.assertEqual(len(formatted_data), len(self.conversations))

#         # Test format_conversation
#         formatted_conversation = formatter.format_conversation(self.conversations[0])
#         self.assertIsInstance(formatted_conversation, dict)
#         self.assertIn("messages", formatted_conversation)

#         # Test format_message
#         message = self.conversations[0].messages[0]
#         formatted_message = formatter.format_message(message)
#         self.assertIsInstance(formatted_message, dict)
#         self.assertIn("role", formatted_message)
#         self.assertIn("content", formatted_message)

#     def test_format_data_function(self):
#         """Test the format_data function from data_loader."""
#         # Test with TorchtuneFormatter
#         torchtune_data = format_data(self.data, "torchtune", self.column_mapping)
#         self.assertIsNotNone(torchtune_data)
#         self.assertEqual(len(torchtune_data), len(self.data))

#         # Test with vLLMFormatter
#         vllm_data = format_data(self.data, "vllm", self.column_mapping)
#         self.assertIsNotNone(vllm_data)
#         self.assertEqual(len(vllm_data), len(self.data))

#         # Test with OpenAIFormatter
#         openai_data = format_data(self.data, "openai", self.column_mapping)
#         self.assertIsNotNone(openai_data)
#         self.assertEqual(len(openai_data), len(self.data))

#     def test_with_mock_data(self):
#         """Test the formatter pipeline with mock data."""
#         # Create mock data that mimics a dataset
#         mock_data = [
#             {
#                 "question": "What is the capital of France?",
#                 "context": "France is a country in Western Europe. Its capital is Paris.",
#                 "answer": "Paris",
#             },
#             {
#                 "question": "Who wrote Hamlet?",
#                 "context": "Hamlet is a tragedy written by William Shakespeare.",
#                 "answer": "William Shakespeare",
#             },
#             {
#                 "question": "What is the largest planet in our solar system?",
#                 "context": "Jupiter is the largest planet in our solar system.",
#                 "answer": "Jupiter",
#             },
#         ]

#         # Create a column mapping for the mock data
#         column_mapping = {"input": "context", "output": "answer"}

#         # Convert to conversations
#         conversations = convert_to_conversations(mock_data, column_mapping)

#         # Test that conversations are created correctly
#         self.assertEqual(len(conversations), len(mock_data))
#         for i, conversation in enumerate(conversations):
#             self.assertEqual(len(conversation.messages), 2)
#             self.assertEqual(conversation.messages[0]["role"], "user")
#             self.assertEqual(conversation.messages[1]["role"], "assistant")

#             # Check content of user message
#             user_content = conversation.messages[0]["content"]
#             self.assertTrue(isinstance(user_content, list))
#             self.assertEqual(user_content[0]["type"], "text")
#             self.assertEqual(user_content[0]["text"], mock_data[i]["context"])

#             # Check content of assistant message
#             assistant_content = conversation.messages[1]["content"]
#             self.assertTrue(isinstance(assistant_content, list))
#             self.assertEqual(assistant_content[0]["type"], "text")
#             self.assertEqual(assistant_content[0]["text"], mock_data[i]["answer"])

#         # Test each formatter with the mock data
#         formatters = {
#             "torchtune": TorchtuneFormatter(),
#             "vllm": vLLMFormatter(),
#             "openai": OpenAIFormatter(),
#         }

#         for name, formatter in formatters.items():
#             formatted_data = formatter.format_data(conversations)
#             self.assertEqual(len(formatted_data), len(mock_data))

#             # Test the first formatted item
#             if name == "vllm":
#                 # vLLM formatter returns strings
#                 self.assertTrue(isinstance(formatted_data[0], str))
#                 self.assertIn("user:", formatted_data[0])
#                 self.assertIn("assistant:", formatted_data[0])
#             else:
#                 # Torchtune and OpenAI formatters return dicts
#                 self.assertTrue(isinstance(formatted_data[0], dict))
#                 self.assertIn("messages", formatted_data[0])
#                 self.assertEqual(len(formatted_data[0]["messages"]), 2)


# if __name__ == "__main__":
#     # If run as a script, this allows passing a dataset name as an argument
#     import argparse

#     parser = argparse.ArgumentParser(
#         description="Test the formatter module with a specific dataset"
#     )
#     parser.add_argument(
#         "--dataset",
#         type=str,
#         default="dz-osamu/IU-Xray",
#         help="Name of the Hugging Face dataset to use for testing",
#     )
#     parser.add_argument(
#         "--split",
#         type=str,
#         default="train[:10]",
#         help="Dataset split to use (e.g., 'train[:10]', 'validation[:10]')",
#     )

#     args = parser.parse_args()

#     # Override the default dataset in the test class
#     TestFormatter.dataset_name = args.dataset
#     TestFormatter.split = args.split

#     # Run the tests
#     unittest.main(argv=["first-arg-is-ignored"])



import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add the parent directory to the path so we can import the modules
sys.path.append(str(Path(__file__).parent.parent))

# Import the module to test
import finetuning


class TestFinetuning(unittest.TestCase):
    """Test cases for the finetuning module."""

    def setUp(self):
        """Set up test fixtures, called before each test."""
        # Create temporary config files for testing
        self.temp_dir = tempfile.TemporaryDirectory()

        # Create a YAML config file
        self.yaml_config_path = os.path.join(self.temp_dir.name, "config.yaml")
        self.create_yaml_config()

        # Create a JSON config file
        self.json_config_path = os.path.join(self.temp_dir.name, "config.json")
        self.create_json_config()

    def tearDown(self):
        """Tear down test fixtures, called after each test."""
        self.temp_dir.cleanup()

    def create_yaml_config(self):
        """Create a YAML config file for testing."""
        yaml_content = """
finetuning:
  strategy: "lora"
  distributed: true
  num_processes_per_node: 4
  torchtune_config: "llama3_2_vision/11B_lora"
"""
        with open(self.yaml_config_path, "w") as f:
            f.write(yaml_content)

    def create_json_config(self):
        """Create a JSON config file for testing."""
        json_content = {
            "finetuning": {
                "strategy": "lora",
                "distributed": True,
                "torchtune_config": "llama3_2_vision/11B_lora",
            }
        }
        with open(self.json_config_path, "w") as f:
            json.dump(json_content, f)

    @patch("subprocess.run")
    def test_run_torch_tune_lora_distributed(self, mock_run):
        """Test running torch tune with LoRA distributed strategy."""
        # Set up the mock
        mock_run.return_value = MagicMock()

        # Get the function from the module
        run_torch_tune = getattr(finetuning, "run_torch_tune")

        # Call the function
        run_torch_tune(self.yaml_config_path)

        # Check that subprocess.run was called with the correct command
        expected_cmd = [
            "tune",
            "run",
            "--nproc_per_node",
            "4",
            "lora_finetune_distributed",
            "--config",
            "llama3_2_vision/11B_lora",
        ]
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        self.assertEqual(args[0], expected_cmd)
        self.assertTrue(kwargs.get("check", False))

    @patch("subprocess.run")
    def test_run_torch_tune_lora_single_device(self, mock_run):
        """Test running torch tune with LoRA single device strategy."""
        # Create a config with single device
        single_device_config_path = os.path.join(
            self.temp_dir.name, "single_device_config.yaml"
        )
        with open(single_device_config_path, "w") as f:
            f.write(
                """
finetuning:
  strategy: "lora"
  distributed: false
  torchtune_config: "llama3_2_vision/11B_lora"
"""
            )

        # Set up the mock
        mock_run.return_value = MagicMock()

        # Get the function from the module
        run_torch_tune = getattr(finetuning, "run_torch_tune")

        # Call the function
        run_torch_tune(single_device_config_path)

        # Check that subprocess.run was called with the correct command
        expected_cmd = [
            "tune",
            "run",
            "lora_finetune_single_device",
            "--config",
            "llama3_2_vision/11B_lora",
        ]
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        self.assertEqual(args[0], expected_cmd)
        self.assertTrue(kwargs.get("check", False))

    @patch("subprocess.run")
    def test_run_torch_tune_invalid_strategy(self, mock_run):
        """Test running torch tune with an invalid strategy."""
        # Create a config with an invalid strategy
        invalid_config_path = os.path.join(self.temp_dir.name, "invalid_config.yaml")
        with open(invalid_config_path, "w") as f:
            f.write(
                """
finetuning:
  strategy: "pretraining"
  distributed: true
  torchtune_config: "llama3_2_vision/11B_lora"
"""
            )

        # Get the function from the module
        run_torch_tune = getattr(finetuning, "run_torch_tune")

        # Call the function and check that it raises a ValueError
        with self.assertRaises(ValueError):
            run_torch_tune(invalid_config_path)

        # Check that subprocess.run was not called
        mock_run.assert_not_called()

    @patch("subprocess.run")
    def test_run_torch_tune_subprocess_error(self, mock_run):
        """Test handling of subprocess errors."""
        # Set up the mock to raise an error
        mock_run.side_effect = subprocess.CalledProcessError(1, ["tune", "run"])

        # Get the function from the module
        run_torch_tune = getattr(finetuning, "run_torch_tune")

        # Call the function and check that it exits with an error
        with self.assertRaises(SystemExit):
            run_torch_tune(self.yaml_config_path)


#     @patch("subprocess.run")
#     def test_run_torch_tune_with_args(self, mock_run):
#         """Test running torch tune with command line arguments."""
#         # Set up the mock
#         mock_run.return_value = MagicMock()

#         # Create mock args
#         args = MagicMock()
#         args.kwargs = "learning_rate=1e-5,batch_size=16"

#         # Modify the finetuning.py file to handle kwargs
#         original_finetuning_py = None
#         with open(finetuning.__file__, "r") as f:
#             original_finetuning_py = f.read()

#         try:
#             # Add code to handle kwargs in run_torch_tune function
#             with open(finetuning.__file__, "a") as f:
#                 f.write(
#                     """
# # Add kwargs to base_cmd if provided
# def add_kwargs_to_cmd(base_cmd, args):
#     if args and hasattr(args, 'kwargs') and args.kwargs:
#         kwargs = args.kwargs.split(',')
#         base_cmd.extend(kwargs)
#     return base_cmd

# # Monkey patch the run_torch_tune function
# original_run_torch_tune = run_torch_tune
# def patched_run_torch_tune(config_path, args=None):
#     # Read the configuration
#     config = read_config(config_path)

#     # Extract parameters from config
#     training_config = config.get("finetuning", {})

#     # Initialize base_cmd to avoid "possibly unbound" error
#     base_cmd = []

#     # Determine the command based on configuration
#     if training_config.get("distributed"):
#         if training_config.get("strategy") == "lora":
#             base_cmd = [
#                 "tune",
#                 "run",
#                 "--nproc_per_node",
#                 str(training_config.get("num_processes_per_node", 1)),
#                 "lora_finetune_distributed",
#                 "--config",
#                 training_config.get("torchtune_config"),
#             ]
#         elif training_config.get("strategy") == "fft":
#             base_cmd = [
#                 "tune",
#                 "run",
#                 "--nproc_per_node",
#                 str(training_config.get("num_processes_per_node", 1)),
#                 "full_finetune_distributed",
#                 "--config",
#                 training_config.get("torchtune_config"),
#             ]
#         else:
#             raise ValueError(f"Invalid strategy: {training_config.get('strategy')}")
#     else:
#         if training_config.get("strategy") == "lora":
#             base_cmd = [
#                 "tune",
#                 "run",
#                 "lora_finetune_single_device",
#                 "--config",
#                 training_config.get("torchtune_config"),
#             ]
#         elif training_config.get("strategy") == "fft":
#             base_cmd = [
#                 "tune",
#                 "run",
#                 "full_finetune_single_device",
#                 "--config",
#                 training_config.get("torchtune_config"),
#             ]
#         else:
#             raise ValueError(f"Invalid strategy: {training_config.get('strategy')}")

#     # Check if we have a valid command
#     if not base_cmd:
#         raise ValueError("Could not determine the appropriate command based on the configuration")

#     # Add kwargs to base_cmd if provided
#     if args and hasattr(args, 'kwargs') and args.kwargs:
#         kwargs = args.kwargs.split(',')
#         base_cmd.extend(kwargs)

#     # Log the command
#     logger.info(f"Running command: {' '.join(base_cmd)}")

#     # Run the command
#     try:
#         subprocess.run(base_cmd, check=True)
#         logger.info("Training complete!")
#     except subprocess.CalledProcessError as e:
#         logger.error(f"Training failed with error: {e}")
#         sys.exit(1)

# # Replace the original function with our patched version
# run_torch_tune = patched_run_torch_tune
# """
#                 )

#             # Call the function with args
#             finetuning.run_torch_tune(self.yaml_config_path, args=args)

#             # Check that subprocess.run was called with the correct command including kwargs
#             expected_cmd = [
#                 "tune",
#                 "run",
#                 "--nproc_per_node",
#                 "4",
#                 "lora_finetune_distributed",
#                 "--config",
#                 "llama3_2_vision/11B_lora",
#                 "learning_rate=1e-5",
#                 "batch_size=16",
#             ]
#             mock_run.assert_called_once()
#             call_args, call_kwargs = mock_run.call_args
#             self.assertEqual(call_args[0], expected_cmd)
#             self.assertTrue(call_kwargs.get("check", False))

#         finally:
#             # Restore the original finetuning.py file
#             if original_finetuning_py:
#                 with open(finetuning.__file__, "w") as f:
#                     f.write(original_finetuning_py)


if __name__ == "__main__":
    unittest.main()
