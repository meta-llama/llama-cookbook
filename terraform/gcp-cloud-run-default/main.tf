# Minimal GCP Cloud Run Terraform configuration for Llama deployment
# This creates only the essential resources for Cloud Run deployment

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

  # Required APIs for Cloud Run
  required_apis = [
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "iam.googleapis.com"
  ]
}

# Enable required Google Cloud APIs
resource "google_project_service" "cloud_run_apis" {
  for_each = toset(local.required_apis)

  project = var.project_id
  service = each.value

  disable_dependent_services = false
  disable_on_destroy         = false
}

# Artifact Registry repository for container images
resource "google_artifact_registry_repository" "llama_repository" {
  repository_id = "${local.name_prefix}-repo"
  format        = "DOCKER"
  location      = var.region
  description   = "Container repository for Llama inference images"

  labels = {
    project     = var.project_name
    environment = var.environment
    managed-by  = "terraform"
  }

  depends_on = [google_project_service.cloud_run_apis]
}

# Service account for Cloud Run service
resource "google_service_account" "cloud_run_sa" {
  account_id   = "${local.name_prefix}-run-sa"
  display_name = "Cloud Run Service Account for ${var.project_name}"
  description  = "Service account for Llama Cloud Run deployment"

  depends_on = [google_project_service.cloud_run_apis]
}

# IAM role bindings for Cloud Run service account
resource "google_project_iam_member" "cloud_run_sa_roles" {
  for_each = toset([
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
    "roles/artifactregistry.reader"
  ])

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.cloud_run_sa.email}"
}

# Cloud Run service
resource "google_cloud_run_v2_service" "llama_service" {
  name     = "${local.name_prefix}-service"
  location = var.region

  template {
    service_account = google_service_account.cloud_run_sa.email

    containers {
      image = var.container_image

      # Resource allocation
      resources {
        limits = {
          cpu    = var.cpu_limit
          memory = var.memory_limit
        }
      }

      # Environment variables
      dynamic "env" {
        for_each = var.environment_variables
        content {
          name  = env.key
          value = env.value
        }
      }

      # Container port
      ports {
        container_port = var.container_port
        name           = "http1"
      }
    }

    # Service scaling configuration
    scaling {
      min_instance_count = var.min_instances
      max_instance_count = var.max_instances
    }

    # Execution environment
    execution_environment = var.execution_environment
  }

  # Traffic configuration
  traffic {
    percent = 100
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
  }

  labels = {
    project     = var.project_name
    environment = var.environment
    managed-by  = "terraform"
  }

  depends_on = [google_project_service.cloud_run_apis]
}

# IAM policy for public access (optional)
resource "google_cloud_run_v2_service_iam_member" "public_access" {
  count = var.allow_public_access ? 1 : 0

  location = google_cloud_run_v2_service.llama_service.location
  name     = google_cloud_run_v2_service.llama_service.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# IAM policy for authenticated access
resource "google_cloud_run_v2_service_iam_member" "authenticated_access" {
  for_each = toset(var.allowed_members)

  location = google_cloud_run_v2_service.llama_service.location
  name     = google_cloud_run_v2_service.llama_service.name
  role     = "roles/run.invoker"
  member   = each.value
}