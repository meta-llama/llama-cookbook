# Outputs for minimal Amazon Bedrock deployment

output "bedrock_role_arn" {
  description = "ARN of the IAM role for Bedrock access"
  value       = aws_iam_role.bedrock_role.arn
}

output "bedrock_role_name" {
  description = "Name of the IAM role for Bedrock access"
  value       = aws_iam_role.bedrock_role.name
}

output "bedrock_policy_arn" {
  description = "ARN of the IAM policy for Bedrock access"
  value       = aws_iam_policy.bedrock_access.arn
}

output "aws_region" {
  description = "AWS region used for deployment"
  value       = var.aws_region
}

output "bedrock_endpoint" {
  description = "Bedrock runtime endpoint URL"
  value       = "https://bedrock-runtime.${var.aws_region}.amazonaws.com"
}

output "user_access_key_id" {
  description = "Access key ID for the IAM user (if created)"
  value       = var.create_user ? aws_iam_access_key.bedrock_user[0].id : null
  sensitive   = true
}

output "user_secret_access_key" {
  description = "Secret access key for the IAM user (if created)"
  value       = var.create_user ? aws_iam_access_key.bedrock_user[0].secret : null
  sensitive   = true
}