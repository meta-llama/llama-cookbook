import os

MODEL_CONFIGS = {
    "vllm_llama_70b": {
        "model": "hosted_vllm/meta-llama/Llama-3.3-70B-Instruct",
        "api_base": "http://localhost:8001/v1",
        "api_key": None,
        "port": 8001,
        "cuda_devices": "4,5,6,7",
        "tensor_parallel": 4,
        "gpu_util": 0.90,
        "chat_template": None,
    },
    "vllm_llama_90b": {
        "model": "hosted_vllm/meta-llama/Llama-3.2-90B-Vision-Instruct",
        "api_base": "http://localhost:8090/v1",
        "api_key": None,
        "port": 8090,
        "cuda_devices": "4,5,6,7",
        "tensor_parallel": 4,
        "gpu_util": 0.70,
        "chat_template": None,
    },
    "vllm_llama_405b": {
        "model": "hosted_vllm/meta-llama/Llama-3.1-405B-FP8",
        "api_base": "http://localhost:8405/v1",
        "api_key": None,
        "port": 8405,
        "cuda_devices": "0,1,2,3,4,5,6,7",
        "tensor_parallel": 8,
        "gpu_util": 0.80,
        "chat_template": "./llama3_405b_chat_template.jinja",
    },
    "vllm_llama_8b": {
        "model": "hosted_vllm/meta-llama/Llama-3.1-8B-Instruct",
        "api_base": "http://localhost:8008/v1",
        "api_key": None,
        "port": 8008,
        "cuda_devices": "0",
        "tensor_parallel": 1,
        "gpu_util": 0.95,
        "chat_template": None,
    },
    "openrouter_gpt4o": {
        "model": "openrouter/openai/gpt-4o",
        "api_base": "https://openrouter.ai/api/v1",
        "api_key": os.getenv("OPENROUTER_API_KEY"),
    },
    "openrouter_gpt4o_mini": {
        "model": "openrouter/openai/gpt-4o-mini",
        "api_base": "https://openrouter.ai/api/v1",
        "api_key": os.getenv("OPENROUTER_API_KEY"),
    },
    "openrouter_llama_70b": {
        "model": "openrouter/meta-llama/llama-3.3-70b-instruct",
        "api_base": "https://openrouter.ai/api/v1",
        "api_key": os.getenv("OPENROUTER_API_KEY"),
    },
}
