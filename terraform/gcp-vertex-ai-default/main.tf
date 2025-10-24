# Minimal GCP Vertex AI Terraform configuration for Llama deployment
# This creates only the essential resources for Vertex AI model deployment

terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# Local values
locals {
  name_prefix = "${var.project_name}-${var.environment}"

  # Required APIs for Vertex AI
  required_apis = [
    "aiplatform.googleapis.com",
    "storage.googleapis.com",
    "iam.googleapis.com"
  ]
}

# Enable required Google Cloud APIs
resource "google_project_service" "vertex_apis" {
  for_each = toset(local.required_apis)

  project = var.project_id
  service = each.value

  disable_dependent_services = false
  disable_on_destroy         = false
}

# Service Account for Vertex AI operations
resource "google_service_account" "vertex_ai_sa" {
  account_id   = "${local.name_prefix}-vertex-sa"
  display_name = "Vertex AI Service Account for ${var.project_name}"
  description  = "Service account for Vertex AI Llama model deployment"

  depends_on = [google_project_service.vertex_apis]
}

# IAM roles for the Vertex AI service account
resource "google_project_iam_member" "vertex_ai_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.vertex_ai_sa.email}"
}

resource "google_project_iam_member" "storage_admin" {
  project = var.project_id
  role    = "roles/storage.admin"
  member  = "serviceAccount:${google_service_account.vertex_ai_sa.email}"
}

# Cloud Storage bucket for model artifacts
resource "google_storage_bucket" "vertex_artifacts" {
  name     = "${local.name_prefix}-vertex-artifacts-${random_id.bucket_suffix.hex}"
  location = var.region

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = var.artifact_retention_days
    }
    action {
      type = "Delete"
    }
  }

  labels = {
    project     = var.project_name
    environment = var.environment
    managed-by  = "terraform"
  }

  depends_on = [google_project_service.vertex_apis]
}

# Random ID for bucket naming
resource "random_id" "bucket_suffix" {
  byte_length = 4
}

# Optional: Vertex AI Dataset (uncomment if needed)
# resource "google_vertex_ai_dataset" "llama_dataset" {
#   display_name        = "${local.name_prefix}-dataset"
#   metadata_schema_uri = "gs://google-cloud-aiplatform/schema/dataset/metadata/text_1.0.0.yaml"
#   region              = var.region
#   
#   depends_on = [google_project_service.vertex_apis]
# }

# Optional: Vertex AI Endpoint (uncomment if needed)
# resource "google_vertex_ai_endpoint" "llama_endpoint" {
#   display_name = "${local.name_prefix}-endpoint"
#   location     = var.region
#   description  = "Endpoint for Llama model serving"
#   
#   labels = {
#     project     = var.project_name
#     environment = var.environment
#   }
#   
#   depends_on = [google_project_service.vertex_apis]
# }