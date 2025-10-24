# Minimal Amazon Bedrock Terraform configuration for Llama API deployment
# This creates only the essential IAM resources needed to access Bedrock models

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
  account_id = data.aws_caller_identity.current.account_id
  region     = data.aws_region.current.name
}

# IAM role for Bedrock access
resource "aws_iam_role" "bedrock_role" {
  name = "${var.project_name}-bedrock-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = [
            "lambda.amazonaws.com",
            "ec2.amazonaws.com"
          ]
        }
      },
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${local.account_id}:root"
        }
      }
    ]
  })

  tags = {
    Name        = "${var.project_name}-bedrock-role"
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}

# IAM policy for Bedrock model access
resource "aws_iam_policy" "bedrock_access" {
  name        = "${var.project_name}-bedrock-access"
  description = "Minimal policy for accessing Bedrock Llama models"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "bedrock:InvokeModel",
          "bedrock:InvokeModelWithResponseStream"
        ]
        Resource = [
          "arn:aws:bedrock:${local.region}::foundation-model/meta.llama4-scout-17b-instruct-v1:0"
        ]
      }
    ]
  })

  tags = {
    Name        = "${var.project_name}-bedrock-access"
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}

# Attach policy to role
resource "aws_iam_role_policy_attachment" "bedrock_access" {
  role       = aws_iam_role.bedrock_role.name
  policy_arn = aws_iam_policy.bedrock_access.arn
}

# Optional: Create access key for programmatic access
resource "aws_iam_user" "bedrock_user" {
  count = var.create_user ? 1 : 0
  name  = "${var.project_name}-bedrock-user"

  tags = {
    Name        = "${var.project_name}-bedrock-user"
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}

resource "aws_iam_user_policy_attachment" "user_bedrock_access" {
  count      = var.create_user ? 1 : 0
  user       = aws_iam_user.bedrock_user[0].name
  policy_arn = aws_iam_policy.bedrock_access.arn
}

resource "aws_iam_access_key" "bedrock_user" {
  count = var.create_user ? 1 : 0
  user  = aws_iam_user.bedrock_user[0].name
}