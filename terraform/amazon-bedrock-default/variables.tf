# Variables for minimal Amazon Bedrock deployment

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

variable "create_user" {
  description = "Whether to create IAM user with access keys for programmatic access"
  type        = bool
  default     = false
}