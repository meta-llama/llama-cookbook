import os
import subprocess
import sys

from config import MODEL_CONFIGS  # Import model configurations


def start_vllm(cuda_devices, model_name):
    """Start vLLM server for the selected model with user-defined CUDA settings."""
    if model_name not in MODEL_CONFIGS:
        print(f"Error: Model '{model_name}' not found in config.")
        print("Available models:", ", ".join(MODEL_CONFIGS.keys()))
        sys.exit(1)

    MODEL_SETTINGS = MODEL_CONFIGS[model_name]

    model_path = MODEL_SETTINGS["model"].replace("hosted_vllm/", "")
    api_base = MODEL_SETTINGS["api_base"]
    port = MODEL_SETTINGS["port"]
    tensor_parallel = MODEL_SETTINGS["tensor_parallel"]
    gpu_util = MODEL_SETTINGS["gpu_util"]
    chat_template = MODEL_SETTINGS.get("chat_template", None)

    # Use provided CUDA devices or default from config
    os.environ["CUDA_VISIBLE_DEVICES"] = (
        cuda_devices if cuda_devices != "default" else MODEL_SETTINGS["cuda_devices"]
    )
    print(f"Using CUDA devices: {os.environ['CUDA_VISIBLE_DEVICES']}")

    # Build vLLM serve command
    vllm_command = (
        f"vllm serve {model_path} "
        f"--port {port} "
        f"--tensor-parallel-size {tensor_parallel} "
        f"--gpu-memory-utilization {gpu_util} "
    )

    # Add chat template flag if required
    if chat_template:
        vllm_command += f"--chat-template {chat_template} "

    print(f"Starting vLLM server for model: {model_name}")
    print(f"Running command: {vllm_command}")

    # Run the command in a new process
    subprocess.run(vllm_command, shell=True)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: CUDA_VISIBLE_DEVICES=<devices> python start_vllm.py <cuda_devices> <model_name>"
        )
        print(
            "Example: CUDA_VISIBLE_DEVICES=0,1 python start_vllm.py 0,1 vllm_llama_405b"
        )
        print("Available models:", ", ".join(MODEL_CONFIGS.keys()))
        sys.exit(1)

    cuda_devices = sys.argv[1]
    model_name = sys.argv[2]

    start_vllm(cuda_devices, model_name)
