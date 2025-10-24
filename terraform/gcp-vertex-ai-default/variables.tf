# Variables for minimal GCP Vertex AI deployment

variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP region for deployment"
  type        = string
  default     = "us-central1"
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

variable "artifact_retention_days" {
  description = "Number of days to retain artifacts in the storage bucket"
  type        = number
  default     = 30
}