# GCP Vertex AI deployment

Deploy Llama 4 Scout models using Google Cloud Vertex AI managed service.

## Overview

This Terraform configuration sets up a basic example deployment, demonstrating how to deploy/serve foundation models using GCP Vertex. Vertex AI provides fully managed ML services with Model-as-a-Service (MaaS) endpoints.

This example shows how to use basic services such as:

- IAM roles for permissions management
- Service accounts for fine-grained access control
- Creating Vertex endpoints for model serving

In our [architecture patterns for private cloud guide](/docs/open_source/private_cloud.md) we outline advanced patterns for cloud deployment that you may choose to implement in a more complete deployment. This includes:

- Deployment into multiple regions or clouds
- Managed keys/secrets services
- Comprehensive logging systems for auditing and compliance
- Backup and recovery systems

## Getting started

### Prerequisites

* GCP project with **billing account enabled** (required for API activation)
* Terraform installed
* Gcloud CLI configured
* Application Default Credentials: `gcloud auth application-default login`

### Deploy

1. Configure GCP authentication:
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

2. Edit terraform.tfvars with your project ID.

3. Create configuration:
   ```bash
   cd terraform/gcp-vertex-ai-default
   cp terraform.tfvars.example terraform.tfvars
   ```

4. Deploy:
   ```bash
   terraform init
   terraform plan
   terraform apply
   ```

### Usage

1. Accept Llama Community License in Vertex AI Model Garden
2. Use Llama 4 Scout via MaaS API:

```python
from google.cloud import aiplatform

aiplatform.init(
    project="your-project-id",
    location="us-central1"
)

# Model ID: meta/llama-4-scout-17b-16e-instruct-maas
```

## Next steps

* [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
* [Vertex AI Model Garden](https://console.cloud.google.com/vertex-ai/model-garden)