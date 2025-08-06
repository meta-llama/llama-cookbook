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

from finetuning import run_torch_tune



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

        # Call the function
        finetuning.run_torch_tune(single_device_config_path)

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

        # Call the function and check that it raises a ValueError
        with self.assertRaises(ValueError):
            finetuning.run_torch_tune(invalid_config_path)

        # Check that subprocess.run was not called
        mock_run.assert_not_called()

    @patch("subprocess.run")
    def test_run_torch_tune_subprocess_error(self, mock_run):
        """Test handling of subprocess errors."""
        # Set up the mock to raise an error
        mock_run.side_effect = subprocess.CalledProcessError(1, ["tune", "run"])

        # Call the function and check that it exits with an error
        with self.assertRaises(SystemExit):
            finetuning.run_torch_tune(self.yaml_config_path)


if __name__ == "__main__":
    unittest.main()
