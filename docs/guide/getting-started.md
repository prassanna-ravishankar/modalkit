# Getting Started

This guide will walk you through creating your first ML deployment with Modalkit on Modal.

## Prerequisites

1. **Install Modal CLI**
   ```bash
   pip install modal
   modal setup
   ```
   Follow the prompts to authenticate with your Modal account.

2. **Install Modalkit**
   ```bash
   pip install modalkit
   ```

## Project Structure

Create a new directory for your project:

```
my-ml-project/
├── app.py              # Modal app definition
├── model.py            # Your ML model code
├── modalkit.yaml       # Configuration
└── requirements.txt    # Dependencies
```

## Step 1: Define Your Model

Create `model.py` with your inference logic:

```python
from modalkit.inference_pipeline import InferencePipeline
from pydantic import BaseModel
from typing import List

# Define input/output schemas
class TextInput(BaseModel):
    text: str
    language: str = "en"

class TextOutput(BaseModel):
    translated_text: str
    confidence: float

# Implement your model
class TranslationModel(InferencePipeline):
    def __init__(self, model_name: str, all_model_data_folder: str, common_settings: dict, *args, **kwargs):
        super().__init__(model_name, all_model_data_folder, common_settings)
        # Load your model here
        # self.model = load_model(...)

    def preprocess(self, input_list: List[TextInput]) -> dict:
        """Prepare batch of inputs for the model"""
        texts = [item.text for item in input_list]
        languages = [item.language for item in input_list]
        return {"texts": texts, "languages": languages}

    def predict(self, input_list: List[TextInput], preprocessed_data: dict) -> dict:
        """Run batch inference"""
        # Your actual model inference
        translations = [f"Translated: {text}" for text in preprocessed_data["texts"]]
        scores = [0.95] * len(translations)
        return {"translations": translations, "scores": scores}

    def postprocess(self, input_list: List[TextInput], raw_output: dict) -> List[TextOutput]:
        """Format outputs"""
        return [
            TextOutput(translated_text=text, confidence=score)
            for text, score in zip(raw_output["translations"], raw_output["scores"])
        ]
```

## Step 2: Configure Your Deployment

Create `modalkit.yaml`:

```yaml
app_settings:
  app_prefix: "translation-demo"

  auth_config:
    # For development - use hardcoded key
    api_key: "dev-key-123"
    auth_header: "x-api-key"

  build_config:
    image: "python:3.11-slim"
    tag: "latest"
    workdir: "/app"

  deployment_config:
    # No GPU for this example
    gpu: null
    concurrency_limit: 10
    container_idle_timeout: 300
    # Use Modal proxy auth in production
    secure: false

  batch_config:
    max_batch_size: 8
    wait_ms: 50

model_settings:
  local_model_repository_folder: "./models"
  common:
    device: "cpu"
  model_entries:
    translation_model:
      version: "1.0"
```

## Step 3: Create Modal App

Create `app.py`:

```python
import modal
from modalkit.modal_service import ModalService, create_web_endpoints
from modalkit.modal_config import ModalConfig
from model import TranslationModel, TextInput, TextOutput

# Initialize Modalkit
modal_config = ModalConfig()
app = modal.App(name=modal_config.app_name)

# Define Modal app class
@app.cls(**modal_config.get_app_cls_settings())
class TranslationApp(ModalService):
    inference_implementation = TranslationModel
    model_name: str = modal.parameter(default="translation_model")
    modal_utils: ModalConfig = modal_config

# Create API endpoints
@app.function(**modal_config.get_handler_settings())
@modal.asgi_app(**modal_config.get_asgi_app_settings())
def web_endpoints():
    return create_web_endpoints(
        app_cls=TranslationApp,
        input_model=TextInput,
        output_model=TextOutput
    )
```

## Step 4: Test Locally

Run your app locally with Modal:

```bash
modal serve app.py
```

Test with curl:

```bash
# Sync endpoint
curl -X POST http://localhost:8000/predict_sync \
  -H "Content-Type: application/json" \
  -H "x-api-key: dev-key-123" \
  -d '{"text": "Hello world", "language": "en"}'

# Batch endpoint
curl -X POST http://localhost:8000/predict_batch \
  -H "Content-Type: application/json" \
  -H "x-api-key: dev-key-123" \
  -d '[{"text": "Hello"}, {"text": "World"}]'
```

## Step 5: Deploy to Modal

Deploy your app:

```bash
modal deploy app.py
```

Your app is now live! Modal will provide the URL:
```
✓ Deployed app to https://your-workspace--translation-demo.modal.run
```

## What's Next?

### Add GPU Support
```yaml
deployment_config:
  gpu: "T4"  # or A10G, A100, etc.
```

### Use Cloud Storage
```yaml
cloud_bucket_mounts:
  - mount_point: "/mnt/models"
    bucket_name: "my-model-bucket"
    secret: "aws-credentials"
```

### Enable Modal Proxy Auth
```yaml
deployment_config:
  secure: true  # Requires Modal-Key and Modal-Secret headers
```

### Add Async Processing
See the [Async Example](../examples/async.md) for queue-based base_inference.

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure all dependencies are in your container image
2. **Auth failures**: Check your API key configuration
3. **GPU issues**: Ensure your Modal workspace has GPU access

### Getting Help

- Check [Modal's documentation](https://modal.com/docs)
- View [Modalkit examples](../examples/basic.md)
- Open an issue on [GitHub](https://github.com/prassanna-ravishankar/modalkit/issues)
