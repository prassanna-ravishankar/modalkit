# LLM Deployment with Modalkit

This tutorial demonstrates deploying a Large Language Model (LLM) using Modalkit, showcasing GPU usage, cloud storage, and production-ready features.

## Overview

We'll deploy a text generation model that:
- Uses GPU acceleration (T4/A10G/A100)
- Loads models from S3/GCS using CloudBucketMount
- Implements proper batching for efficiency
- Includes authentication and error handling
- Supports both sync and async inference

## Project Structure

```
llm-service/
├── app.py                 # Modal app definition
├── model.py               # LLM inference implementation
├── modalkit.yaml          # Configuration
├── requirements.txt       # Dependencies
└── models/                # Local model cache (optional)
```

## 1. Model Implementation

Create `model.py`:

```python
from modalkit.inference import InferencePipeline
from pydantic import BaseModel
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os

# Input/Output schemas
class TextGenerationInput(BaseModel):
    prompt: str
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    num_return_sequences: int = 1

class TextGenerationOutput(BaseModel):
    generated_text: str
    prompt: str
    model_name: str
    generation_params: dict

class LLMInference(InferencePipeline):
    def __init__(self, model_name: str, all_model_data_folder: str, common_settings: dict, *args, **kwargs):
        super().__init__(model_name, all_model_data_folder, common_settings)

        # Get model configuration
        self.model_config = common_settings.get(model_name, {})
        self.model_id = self.model_config.get("model_id", "microsoft/DialoGPT-medium")
        self.cache_dir = self.model_config.get("cache_dir", "/tmp/transformers_cache")

        # Load model from mounted cloud storage if available
        model_path = f"/mnt/models/{self.model_id}"
        if os.path.exists(model_path):
            print(f"Loading model from mounted storage: {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                cache_dir=self.cache_dir
            )
        else:
            # Fallback to downloading from HuggingFace
            print(f"Downloading model from HuggingFace: {self.model_id}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                cache_dir=self.cache_dir
            )

        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()
        print(f"Model loaded: {self.model_id}")

    def preprocess(self, input_list: List[TextGenerationInput]) -> dict:
        """Tokenize inputs and prepare for batch inference"""
        prompts = [item.prompt for item in input_list]

        # Tokenize all prompts
        tokenized = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        # Move to GPU if available
        if torch.cuda.is_available():
            tokenized = {k: v.cuda() for k, v in tokenized.items()}

        return {
            "tokenized_inputs": tokenized,
            "original_prompts": prompts,
            "generation_params": [
                {
                    "max_length": item.max_length,
                    "temperature": item.temperature,
                    "top_p": item.top_p,
                    "top_k": item.top_k,
                    "num_return_sequences": item.num_return_sequences
                }
                for item in input_list
            ]
        }

    def predict(self, input_list: List[TextGenerationInput], preprocessed_data: dict) -> dict:
        """Generate text using the LLM"""
        tokenized_inputs = preprocessed_data["tokenized_inputs"]
        generation_params = preprocessed_data["generation_params"]

        # Use the first item's params for batch generation (in real use, you might want to batch by params)
        params = generation_params[0] if generation_params else {}

        with torch.no_grad():
            outputs = self.model.generate(
                **tokenized_inputs,
                max_length=params.get("max_length", 512),
                temperature=params.get("temperature", 0.7),
                top_p=params.get("top_p", 0.9),
                top_k=params.get("top_k", 50),
                num_return_sequences=params.get("num_return_sequences", 1),
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                early_stopping=True
            )

        # Decode generated text
        generated_texts = []
        for i, output in enumerate(outputs):
            # Skip the input tokens to get only generated text
            input_length = tokenized_inputs["input_ids"][i].shape[0]
            generated_tokens = output[input_length:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            generated_texts.append(generated_text)

        return {
            "generated_texts": generated_texts,
            "generation_params": generation_params
        }

    def postprocess(self, input_list: List[TextGenerationInput], raw_output: dict) -> List[TextGenerationOutput]:
        """Format outputs with metadata"""
        generated_texts = raw_output["generated_texts"]
        generation_params = raw_output["generation_params"]

        outputs = []
        for i, (input_item, generated_text, params) in enumerate(zip(input_list, generated_texts, generation_params)):
            outputs.append(TextGenerationOutput(
                generated_text=generated_text,
                prompt=input_item.prompt,
                model_name=self.model_id,
                generation_params=params
            ))

        return outputs
```

## 2. Configuration

Create `modalkit.yaml`:

```yaml
app_settings:
  app_prefix: "llm-service"

  # Authentication
  auth_config:
    ssm_key: "/llm-service/api-key"
    auth_header: "x-api-key"

  # Container configuration
  build_config:
    image: "python:3.11"
    tag: "latest"
    workdir: "/app"
    env:
      TRANSFORMERS_CACHE: "/tmp/transformers_cache"
      PYTORCH_CUDA_ALLOC_CONF: "max_split_size_mb:512"
    extra_run_commands:
      - "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
      - "pip install transformers accelerate"

  # GPU deployment
  deployment_config:
    gpu: "T4"  # Use T4 for cost-effective inference, A10G/A100 for larger models
    concurrency_limit: 5
    container_idle_timeout: 600  # Keep warm for 10 minutes
    retries: 3
    memory: 16384  # 16GB RAM

    # Mount models from cloud storage
    cloud_bucket_mounts:
      - mount_point: "/mnt/models"
        bucket_name: "my-llm-models"
        secret: "aws-credentials"
        key_prefix: "huggingface-models/"
        read_only: true

    # Cache volume for downloads
    volumes:
      "/tmp/transformers_cache": "transformers-cache"
    volume_reload_interval_seconds: 3600  # Reload hourly

  # Batch processing for efficiency
  batch_config:
    max_batch_size: 8  # Process multiple requests together
    wait_ms: 100       # Wait up to 100ms to fill batch

  # Async processing
  queue_config:
    backend: "taskiq"
    broker_url: "redis://redis:6379"

# Model configuration
model_settings:
  local_model_repository_folder: "./models"
  common:
    cache_dir: "/tmp/transformers_cache"
    device: "cuda"
  model_entries:
    llm_model:
      model_id: "microsoft/DialoGPT-medium"  # Change to your preferred model
      cache_dir: "/tmp/transformers_cache"
    # Add more models as needed
    # large_llm:
    #   model_id: "meta-llama/Llama-2-7b-chat-hf"
    #   cache_dir: "/tmp/transformers_cache"
```

## 3. Modal App

Create `app.py`:

```python
import modal
from modalkit.modalapp import ModalService, create_web_endpoints
from modalkit.modalutils import ModalConfig
from model import LLMInference, TextGenerationInput, TextGenerationOutput

# Initialize Modalkit
modal_utils = ModalConfig()
app = modal.App(name=modal_utils.app_name)

# Define Modal app class
@app.cls(**modal_utils.get_app_cls_settings())
class LLMApp(ModalService):
    inference_implementation = LLMInference
    model_name: str = modal.parameter(default="llm_model")
    modal_utils: ModalConfig = modal_utils

# Create endpoints
@app.function(**modal_utils.get_handler_settings())
@modal.asgi_app(**modal_utils.get_asgi_app_settings())
def web_endpoints():
    return create_web_endpoints(
        app_cls=LLMApp,
        input_model=TextGenerationInput,
        output_model=TextGenerationOutput
    )

# Health check endpoint
@app.function()
def health_check():
    return {"status": "healthy", "service": "llm-service"}

if __name__ == "__main__":
    # For local development
    with modal.enable_local_development():
        pass
```

## 4. Dependencies

Create `requirements.txt`:

```txt
torch>=2.0.0
transformers>=4.30.0
accelerate>=0.20.0
sentencepiece>=0.1.99
protobuf>=3.20.0
```

## 5. Setup Cloud Storage

### Option 1: AWS S3
```bash
# Create Modal secret for AWS credentials
modal secret create aws-credentials \
  AWS_ACCESS_KEY_ID=your_access_key \
  AWS_SECRET_ACCESS_KEY=your_secret_key \
  AWS_DEFAULT_REGION=us-east-1

# Upload your models to S3
aws s3 sync ./local_models/ s3://my-llm-models/huggingface-models/
```

### Option 2: Google Cloud Storage
```bash
# Create Modal secret from service account
modal secret create gcp-credentials \
  --from-gcp-service-account path/to/service-account.json

# Upload models to GCS
gsutil -m cp -r ./local_models/ gs://my-llm-models/huggingface-models/
```

## 6. Deployment

### Local Testing
```bash
# Serve locally
modal serve app.py

# Test the endpoint
curl -X POST http://localhost:8000/predict_sync \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-api-key" \
  -d '{
    "prompt": "The future of AI is",
    "max_length": 100,
    "temperature": 0.7
  }'
```

### Production Deployment
```bash
# Deploy to Modal
modal deploy app.py

# The service will be available at:
# https://your-org--llm-service.modal.run
```

## 7. Usage Examples

### Synchronous Generation
```python
import requests

headers = {"x-api-key": "your-api-key"}
response = requests.post(
    "https://your-org--llm-service.modal.run/predict_sync",
    json={
        "prompt": "Write a story about a robot learning to paint:",
        "max_length": 200,
        "temperature": 0.8,
        "top_p": 0.9
    },
    headers=headers
)

result = response.json()
print(result["generated_text"])
```

### Batch Generation
```python
import requests

headers = {"x-api-key": "your-api-key"}
response = requests.post(
    "https://your-org--llm-service.modal.run/predict_batch",
    json=[
        {"prompt": "The meaning of life is", "max_length": 50},
        {"prompt": "In a world where AI rules", "max_length": 50},
        {"prompt": "The last human on Earth", "max_length": 50}
    ],
    headers=headers
)

results = response.json()
for result in results:
    print(f"Prompt: {result['prompt']}")
    print(f"Generated: {result['generated_text']}")
    print("---")
```

### Async Generation
```python
import requests
import time

headers = {"x-api-key": "your-api-key"}

# Submit async request
response = requests.post(
    "https://your-org--llm-service.modal.run/predict_async",
    json={
        "prompt": "Write a detailed essay about climate change:",
        "max_length": 500,
        "temperature": 0.7
    },
    headers=headers
)

message_id = response.json()["message_id"]
print(f"Request submitted: {message_id}")

# Poll for results (in practice, use webhooks)
while True:
    status_response = requests.get(
        f"https://your-org--llm-service.modal.run/status/{message_id}",
        headers=headers
    )

    if status_response.json()["status"] == "completed":
        result = status_response.json()["result"]
        print(result["generated_text"])
        break

    time.sleep(2)
```

## 8. Production Considerations

### Performance Optimization
```yaml
# For high-throughput scenarios
deployment_config:
  gpu: "A10G"  # Better for larger models
  concurrency_limit: 10

batch_config:
  max_batch_size: 16  # Increase batch size
  wait_ms: 200        # Allow more time for batching
```

### Cost Optimization
```yaml
# For cost-sensitive deployments
deployment_config:
  gpu: "T4"  # Most cost-effective
  container_idle_timeout: 300  # Shorter timeout

batch_config:
  max_batch_size: 32  # Maximize batch efficiency
```

### Model Versioning
```yaml
model_settings:
  model_entries:
    llm_v1:
      model_id: "microsoft/DialoGPT-medium"
    llm_v2:
      model_id: "microsoft/DialoGPT-large"
```

## 9. Monitoring and Logging

Add logging to your model:

```python
import logging

logger = logging.getLogger(__name__)

class LLMInference(InferencePipeline):
    def predict(self, input_list: List[TextGenerationInput], preprocessed_data: dict) -> dict:
        logger.info(f"Generating text for {len(input_list)} prompts")

        # ... generation code ...

        logger.info(f"Generated {len(generated_texts)} responses")
        return {"generated_texts": generated_texts}
```

## 10. Error Handling

```python
from modalkit.exceptions import DependencyError

class LLMInference(InferencePipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        try:
            import torch
            if not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
        except ImportError as e:
            raise DependencyError("PyTorch not installed") from e
```

## Key Features Demonstrated

1. **GPU Acceleration**: Efficient GPU usage with proper memory management
2. **Cloud Storage**: Model loading from S3/GCS using CloudBucketMount
3. **Batch Processing**: Intelligent batching for cost-effective inference
4. **Authentication**: Secure API key authentication
5. **Async Processing**: Queue-based async inference for long-running tasks
6. **Error Handling**: Comprehensive error handling and logging
7. **Production Ready**: Proper configuration for production deployment

This example showcases Modalkit's power in deploying production-ready ML services with minimal boilerplate code.
