# Example terraform.tfvars for minimal Amazon SageMaker deployment
# Copy this file to terraform.tfvars and customize as needed

# AWS Configuration
aws_region = "us-west-2"

# Project Configuration
project_name = "my-llama-api"
environment  = "dev"

# Model Configuration
model_image_uri = "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-inference:2.6.0-transformers4.51.3-gpu-py312-cu124-ubuntu22.04"
model_data_s3_path = "s3://llama-model-demo-bucket/model.tar.gz"
model_name = "Llama-3.2-1B-Instruct"

# Instance Configuration
instance_type = "ml.p4d.24xlarge"  # GPU instance for Llama models, will fit larger models
initial_instance_count = 1