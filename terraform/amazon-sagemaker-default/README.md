# Amazon SageMaker deployment

Deploy Llama models using Amazon SageMaker with GPU instances.

## Overview
This Terraform configuration sets up a basic example deployment, demonstrating how to deploy/serve foundation models using Amazon SageMaker. Amazon SageMaker provides managed inference endpoints with auto-scaling capabilities.

This example shows how to use basic services such as:

- IAM roles for permissions management
- Service accounts for fine-grained access control
- Connecting model artifacts in S3 with SageMaker for deployment

In our [architecture patterns for private cloud guide](/docs/open_source/private_cloud.md) we outline advanced patterns for cloud deployment that you may choose to implement in a more complete deployment. This includes:

- Deployment into multiple regions or clouds
- Managed keys/secrets services
- Comprehensive logging systems for auditing and compliance
- Backup and recovery systems

## Getting started

### Prerequisites

* AWS account with access to Amazon SageMaker
* Terraform installed
* Model artifacts packaged as `tar.gz` (see model setup below)
* Container image (AWS pre-built or custom ECR)
* A Hugging Face account with access to the appropriate models (such as Llama 3.2 1B or Llama 3.3 70B)
* **GPU quota**: Request quota increase for `ml.p4d.24xlarge` instances via AWS Service Quotas (default is 0)

### Deploy

1. Configure AWS credentials:
   ```bash
   aws configure
   ```

2. Prepare Llama model artifacts:
   ```bash
   # Download model using Hugging Face CLI
   pip install huggingface-hub
   huggingface-cli download meta-llama/Llama-3.2-1B-Instruct --local-dir ./model
   
   # Package for Amazon SageMaker
   tar -czf model.tar.gz -C model .
   aws s3 cp model.tar.gz s3://your-bucket/model/
   ```

3. Create configuration:
   ```bash
   cd terraform/amazon-sagemaker-default
   cp terraform.tfvars.example terraform.tfvars
   ```

4. Edit terraform.tfvars with your model S3 path and other variables

5. Deploy:
   ```bash
   terraform init
   terraform plan
   terraform apply
   ```

### Usage

```python
import boto3
import json

client = boto3.client('sagemaker-runtime', region_name='us-east-1')

response = client.invoke_endpoint(
    EndpointName='your-endpoint-name',
    ContentType='application/json',
    Body=json.dumps({
        "inputs": "Hello, how are you?",
        "parameters": {
            "max_new_tokens": 256,
            "temperature": 0.7
        }
    })
)

result = json.loads(response['Body'].read())
print(result)
```

## Next steps

* [Amazon SageMaker Developer Guide](https://docs.aws.amazon.com/sagemaker/)
* [Amazon SageMaker Runtime API](https://docs.aws.amazon.com/sagemaker/latest/APIReference/)
