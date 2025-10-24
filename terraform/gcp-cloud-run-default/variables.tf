# Variables for minimal GCP Cloud Run deployment

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

variable "container_image" {
  description = "Container image URL for the Cloud Run service"
  type        = string
  default     = "gcr.io/cloudrun/hello"
}

variable "cpu_limit" {
  description = "CPU limit for the container"
  type        = string
  default     = "2"
}

variable "memory_limit" {
  description = "Memory limit for the container"
  type        = string
  default     = "2Gi"
}

variable "container_port" {
  description = "Port that the container listens on"
  type        = number
  default     = 8080
}

variable "min_instances" {
  description = "Minimum number of instances"
  type        = number
  default     = 0
}

variable "max_instances" {
  description = "Maximum number of instances"
  type        = number
  default     = 10
}

variable "execution_environment" {
  description = "Execution environment for the service"
  type        = string
  default     = "EXECUTION_ENVIRONMENT_GEN2"
}

variable "environment_variables" {
  description = "Environment variables for the container"
  type        = map(string)
  default     = {}
}

variable "allow_public_access" {
  description = "Whether to allow public access to the service"
  type        = bool
  default     = false
}

variable "allowed_members" {
  description = "List of members allowed to access the service"
  type        = list(string)
  default     = []
}