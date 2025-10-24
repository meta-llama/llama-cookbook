# Outputs for minimal Amazon SageMaker deployment

output "sagemaker_endpoint_name" {
  description = "Name of the SageMaker endpoint"
  value       = aws_sagemaker_endpoint.llama_endpoint.name
}

output "sagemaker_endpoint_arn" {
  description = "ARN of the SageMaker endpoint"
  value       = aws_sagemaker_endpoint.llama_endpoint.arn
}

output "sagemaker_role_arn" {
  description = "ARN of the SageMaker execution role"
  value       = aws_iam_role.sagemaker_execution_role.arn
}

output "model_artifacts_bucket" {
  description = "S3 bucket name for model artifacts"
  value       = aws_s3_bucket.model_artifacts.bucket
}

output "model_artifacts_bucket_arn" {
  description = "S3 bucket ARN for model artifacts"
  value       = aws_s3_bucket.model_artifacts.arn
}

output "aws_region" {
  description = "AWS region used for deployment"
  value       = var.aws_region
}

output "endpoint_url" {
  description = "SageMaker runtime endpoint URL for inference"
  value       = "https://runtime.sagemaker.${var.aws_region}.amazonaws.com/endpoints/${aws_sagemaker_endpoint.llama_endpoint.name}/invocations"
}