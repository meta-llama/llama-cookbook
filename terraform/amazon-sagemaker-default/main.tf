# Minimal Amazon SageMaker Terraform configuration for Llama deployment
# This creates only the essential resources for SageMaker model deployment

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# Data sources
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# Local values
locals {
  account_id  = data.aws_caller_identity.current.account_id
  region      = data.aws_region.current.name
  name_prefix = "${var.project_name}-${var.environment}"
}

# S3 bucket for model artifacts (required for SageMaker)
resource "aws_s3_bucket" "model_artifacts" {
  bucket = "${local.name_prefix}-model-artifacts-${random_id.bucket_suffix.hex}"

  tags = {
    Name        = "${local.name_prefix}-model-artifacts"
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}

resource "random_id" "bucket_suffix" {
  byte_length = 4
}

resource "aws_s3_bucket_public_access_block" "model_artifacts" {
  bucket = aws_s3_bucket.model_artifacts.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# IAM role for SageMaker execution
resource "aws_iam_role" "sagemaker_execution_role" {
  name = "${local.name_prefix}-sagemaker-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name        = "${local.name_prefix}-sagemaker-role"
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}

# IAM policy for SageMaker execution (minimal permissions)
resource "aws_iam_role_policy" "sagemaker_policy" {
  name = "${local.name_prefix}-sagemaker-policy"
  role = aws_iam_role.sagemaker_execution_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.model_artifacts.arn,
          "${aws_s3_bucket.model_artifacts.arn}/*",
          "arn:aws:s3:::llama-model-demo-bucket",
          "arn:aws:s3:::llama-model-demo-bucket/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:${local.region}:${local.account_id}:*"
      }
    ]
  })
}

# SageMaker model
resource "aws_sagemaker_model" "llama_model" {
  name               = "${local.name_prefix}-llama-model"
  execution_role_arn = aws_iam_role.sagemaker_execution_role.arn

  primary_container {
    image          = var.model_image_uri
    model_data_url = var.model_data_s3_path

    environment = {
      SAGEMAKER_PROGRAM          = "inference.py"
      SAGEMAKER_SUBMIT_DIRECTORY = "/opt/ml/code"
      MODEL_NAME                 = var.model_name
      HF_TASK                    = "text-generation"
    }
  }

  tags = {
    Name        = "${local.name_prefix}-llama-model"
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}

# SageMaker endpoint configuration
resource "aws_sagemaker_endpoint_configuration" "llama_config" {
  name = "${local.name_prefix}-llama-config-${random_id.config_suffix.hex}"

  production_variants {
    variant_name           = "primary"
    model_name             = aws_sagemaker_model.llama_model.name
    initial_instance_count = var.initial_instance_count
    instance_type          = var.instance_type
    initial_variant_weight = 1
  }

  tags = {
    Name        = "${local.name_prefix}-llama-config"
    Environment = var.environment
    ManagedBy   = "Terraform"
  }

  lifecycle {
    create_before_destroy = true
  }
}

resource "random_id" "config_suffix" {
  byte_length = 4
}

# SageMaker endpoint
resource "aws_sagemaker_endpoint" "llama_endpoint" {
  name                 = "${local.name_prefix}-llama-endpoint"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.llama_config.name

  tags = {
    Name        = "${local.name_prefix}-llama-endpoint"
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}