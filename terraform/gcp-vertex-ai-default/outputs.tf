# Outputs for minimal GCP Vertex AI deployment

output "project_id" {
  description = "GCP project ID"
  value       = var.project_id
}

output "region" {
  description = "GCP region"
  value       = var.region
}

output "service_account_email" {
  description = "Email of the Vertex AI service account"
  value       = google_service_account.vertex_ai_sa.email
}

output "bucket_name" {
  description = "Name of the artifacts storage bucket"
  value       = google_storage_bucket.vertex_artifacts.name
}

output "bucket_url" {
  description = "URL of the artifacts storage bucket"
  value       = google_storage_bucket.vertex_artifacts.url
}

output "vertex_ai_region_endpoint" {
  description = "Vertex AI API regional endpoint"
  value       = "https://${var.region}-aiplatform.googleapis.com"
}