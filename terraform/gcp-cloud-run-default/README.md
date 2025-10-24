# Google Cloud Platform Cloud Run deployment

Deploy containerized Llama models using Google Cloud Run with auto-scaling.

## Overview

This Terraform configuration sets up a basic example deployment, demonstrating how to deploy/serve foundation models using Google Cloud Run services. Google Cloud Run provides serverless container deployment with automatic scaling. 

This example shows how to use basic services such as:

- IAM roles for permissions management
- Service accounts for fine-grained access control
- Containerization of Llama models on Cloud Run

In our [architecture patterns for private cloud guide](/docs/open_source/private_cloud.md) we outline advanced patterns for cloud deployment that you may choose to implement in a more complete deployment. This includes:

- Deployment into multiple regions or clouds
- Managed keys/secrets services
- Comprehensive logging systems for auditing and compliance
- Backup and recovery systems

## Getting started

### Prerequisites

* GCP project with **billing account enabled** (required for Google Cloud Run and Google Cloud Artifact Registry)
* Terraform installed
* Docker container image with Llama model (see container setup below)
* Google Cloud CLI configured
* Application Default Credentials: `gcloud auth application-default login`

### Deploy

1. Configure GCP authentication:
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

2. Prepare container image with vLLM. For speed and simplicity's sake, we will use a small 1B parameter model. You may choose to use a larger Llama model, and if so should increase the resource requirements in your tfvars file.
   ```bash
   # Create Dockerfile
   cat > Dockerfile << 'EOF'
   FROM vllm/vllm-openai:latest
   ENV MODEL_NAME=meta-llama/Llama-3.2-1B-Instruct
   CMD ["vllm", "serve", "$MODEL_NAME", "--host", "0.0.0.0", "--port", "8080"]
   EOF
   
   # Build and push
   docker build -t llama-inference .
   docker tag llama-inference gcr.io/YOUR_PROJECT_ID/llama-inference:latest
   docker push gcr.io/YOUR_PROJECT_ID/llama-inference:latest
   ```

3. Edit terraform.tfvars with your project ID and container image.

4. Create configuration:
   ```bash
   cd terraform/gcp-cloud-run-default
   cp terraform.tfvars.example terraform.tfvars
   ```

5. Deploy:
   ```bash
   terraform init
   terraform plan
   terraform apply
   ```

### Usage

```bash
# Get service URL
SERVICE_URL=$(terraform output -raw service_url)

# Make request
curl -X POST $SERVICE_URL/predict \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, how are you?", "max_tokens": 100}'
```

## Next steps

* [Google Cloud Run Documentation](https://cloud.google.com/run/docs)
* [Google Container Registry Guide](https://cloud.google.com/container-registry/docs)
