# Amazon Bedrock deployment

Deploy Llama 4 Scout models using Amazon Bedrock managed service.

## Overview

This Terraform configuration sets up a basic example deployment, demonstrating how to deploy/serve Amazon Bedrock foundation models in Amazon Web Services. Amazon Bedrock provides fully managed AI models without any infrastructure management.

This example shows how to use basic services such as:

- IAM roles for permissions management
- Service accounts for fine-grained access control
- Access to Bedrock Llama models in a minimal policy

In our [architecture patterns for private cloud guide](/docs/open_source/private_cloud.md) we outline advanced patterns for cloud deployment that you may choose to implement in a more complete deployment. This includes:

- Deployment into multiple regions or clouds
- Managed keys/secrets services
- Comprehensive logging systems for auditing and compliance
- Backup and recovery systems

## Getting started

### Prerequisites

* AWS account with access to Amazon Bedrock
* Terraform installed
* AWS CLI configured
* **Model access enabled**: Go to Amazon Bedrock console → Model access → Request access for Meta Llama models

### Deploy

1. Configure AWS credentials:
   ```bash
   aws configure
   ```

2. Edit terraform.tfvars with your values.

3. Create configuration:
   ```bash
   cd terraform/amazon-bedrock-default
   cp terraform.tfvars.example terraform.tfvars
   ```

4. Deploy:
   ```bash
   terraform init
   terraform plan
   terraform apply
   ```

### Usage

```python
import boto3
import json

bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')

response = bedrock.invoke_model(
    modelId='meta.llama4-scout-17b-instruct-v1:0',
    body=json.dumps({
        "prompt": "Hello, how are you?",
        "max_gen_len": 256,
        "temperature": 0.7
    })
)
```

## Next steps

* [Amazon Bedrock User Guide](https://docs.aws.amazon.com/bedrock/)
* [Amazon Bedrock API Reference](https://docs.aws.amazon.com/bedrock/latest/APIReference/)
