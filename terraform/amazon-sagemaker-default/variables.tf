# Variables for minimal Amazon SageMaker deployment

variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Name of the project (used for resource naming)"
  type        = string
  default     = "llama-api"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "model_image_uri" {
  description = "URI of the container image for model inference"
  type        = string
  default     = "763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
}

variable "model_data_s3_path" {
  description = "S3 path to the model artifacts (tar.gz file)"
  type        = string
  default     = ""
}

variable "model_name" {
  description = "Name of the model for inference"
  type        = string
  default     = "llama-3-3-70b-instruct"
}

variable "instance_type" {
  description = "SageMaker instance type for hosting (use ml.m5.xlarge for CPU if GPU quota unavailable)"
  type        = string
  default     = "ml.p4d.24xlarge"
}

variable "initial_instance_count" {
  description = "Initial number of instances for the endpoint"
  type        = number
  default     = 1
}