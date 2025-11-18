# Modalkit

<p align="center">
  <a href="https://img.shields.io/github/v/release/prassanna-ravishankar/modalkit">
    <img src="https://img.shields.io/github/v/release/prassanna-ravishankar/modalkit" alt="Release">
  </a>
  <a href="https://codecov.io/gh/prassanna-ravishankar/modalkit">
    <img src="https://codecov.io/gh/prassanna-ravishankar/modalkit/branch/main/graph/badge.svg" alt="codecov">
  </a>
  <a href="https://img.shields.io/github/commit-activity/m/prassanna-ravishankar/modalkit">
    <img src="https://img.shields.io/github/commit-activity/m/prassanna-ravishankar/modalkit" alt="Commit activity">
  </a>
  <a href="https://img.shields.io/github/license/prassanna-ravishankar/modalkit">
    <img src="https://img.shields.io/github/license/prassanna-ravishankar/modalkit" alt="License">
  </a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/prassanna-ravishankar/modalkit/main/docs/modalkit.png" width="400" height="400"/>
</p>

<p align="center">
  Python framework for deploying ML models on Modal
</p>

## What Modalkit Provides

Modalkit adds ML deployment patterns on top of Modal's serverless infrastructure:

### Standardized ML Architecture
- Enforced `preprocess()` → `predict()` → `postprocess()` pattern
- Consistent API endpoints: `/predict_sync`, `/predict_batch`, `/predict_async`
- Pydantic models for data validation

### Configuration-Driven Deployments
- YAML configuration for deployment settings
- Environment-specific configs with override capabilities
- Declarative infrastructure definitions

### Production Features
- Authentication middleware (API key or Modal proxy auth)
- Async processing with multiple queue backend support
- Direct S3/GCS/R2 bucket mounting
- Request batching for GPU efficiency
- Error handling and logging

### Developer Tools
- Pre-configured with ruff, mypy, pre-commit hooks
- Testing patterns for ML deployments
- Reduced boilerplate compared to raw Modal

## Features

- Modal integration for serverless deployment
- Authentication: Modal proxy auth or custom API keys with AWS SSM
- Cloud storage: S3, GCS, and R2 bucket mounting
- Queue integration: Optional TaskIQ, SQS, or custom queue backends
- Batch inference with configurable batch sizes
- Type safety with Pydantic models
- Pre-configured tooling: ruff, mypy, pre-commit
- Error handling and logging

## Quick Start

### Installation

```bash
# Using pip (recommended)
pip install modalkit

# Using uv
uv pip install modalkit

# Development/latest version from GitHub
pip install git+https://github.com/prassanna-ravishankar/modalkit.git
```

### Complete Examples

See the documentation for working examples:
- [Queue Backend Patterns](https://prassanna-ravishankar.github.io/modalkit/examples/queue-patterns/) - Queue backend patterns and dependency injection
- [TaskIQ Integration](https://prassanna-ravishankar.github.io/modalkit/examples/taskiq-integration/) - TaskIQ integration tutorial

### 1. Define Your Model

Create an inference class that inherits from `InferencePipeline`:

```python
from modalkit.inference_pipeline import InferencePipeline
from pydantic import BaseModel
from typing import List

# Define input/output schemas with Pydantic
class TextInput(BaseModel):
    text: str
    language: str = "en"

class TextOutput(BaseModel):
    translated_text: str
    confidence: float

# Implement your model logic
class TranslationModel(InferencePipeline):
    def __init__(self, model_name: str, all_model_data_folder: str, common_settings: dict, *args, **kwargs):
        super().__init__(model_name, all_model_data_folder, common_settings)
        # Load your model here
        # self.model = load_model(...)

    def preprocess(self, input_list: List[TextInput]) -> dict:
        """Prepare inputs for the model"""
        texts = [item.text for item in input_list]
        return {"texts": texts, "languages": [item.language for item in input_list]}

    def predict(self, input_list: List[TextInput], preprocessed_data: dict) -> dict:
        """Run model inference"""
        # Your model prediction logic
        translations = [text.upper() for text in preprocessed_data["texts"]]  # Example
        return {"translations": translations, "scores": [0.95] * len(translations)}

    def postprocess(self, input_list: List[TextInput], raw_output: dict) -> List[TextOutput]:
        """Format model outputs"""
        return [
            TextOutput(translated_text=text, confidence=score)
            for text, score in zip(raw_output["translations"], raw_output["scores"])
        ]
```

### 2. Create Your Modal App

```python
import modal
from modalkit.modal_service import ModalService, create_web_endpoints
from modalkit.modal_config import ModalConfig

# Initialize with your config
modal_config = ModalConfig()
app = modal.App(name=modal_config.app_name)

# Define your Modal app class
@app.cls(**modal_config.get_app_cls_settings())
class TranslationApp(ModalService):
    inference_implementation = TranslationModel
    model_name: str = modal.parameter(default="translation_model")
    modal_utils: ModalConfig = modal_config

    # Optional: Inject custom queue backend
    # def __init__(self, queue_backend=None):
    #     super().__init__(queue_backend=queue_backend)

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

**Note**: Queue backends are optional. Services work without queue configuration. Add TaskIQ or custom queues for async processing. See [documentation examples](https://prassanna-ravishankar.github.io/modalkit/examples/) for implementations.

### 3. Configure Your Deployment

Create a `modalkit.yaml` configuration file:

```yaml
# modalkit.yaml
app_settings:
  app_prefix: "translation-service"

  # Authentication configuration
  auth_config:
    # Option 1: Use API key from AWS SSM
    ssm_key: "/translation/api-key"
    auth_header: "x-api-key"
    # Option 2: Use hardcoded API key (not recommended for production)
    # api_key: "your-api-key-here"
    # auth_header: "x-api-key"

  # Container configuration
  build_config:
    image: "python:3.11-slim"  # or your custom image
    tag: "latest"
    workdir: "/app"
    env:
      MODEL_VERSION: "v1.0"

  # Deployment settings
  deployment_config:
    gpu: "T4"  # Options: T4, A10G, A100, or null for CPU
    concurrency_limit: 10
    container_idle_timeout: 300
    secure: false  # Set to true for Modal proxy auth

    # Cloud storage mounts (optional)
    cloud_bucket_mounts:
      - mount_point: "/mnt/models"
        bucket_name: "my-model-bucket"
        secret: "aws-credentials"
        read_only: true
        key_prefix: "models/"

  # Batch processing settings
  batch_config:
    max_batch_size: 32
    wait_ms: 100  # Wait up to 100ms to fill batch

  # Queue configuration (optional - for async endpoints)
  # Leave empty to disable queues, or configure fallback backend
  queue_config:
    backend: "memory"  # Options: "sqs", "memory", or omit for no queues
    # broker_url: "redis://localhost:6379"  # For TaskIQ via dependency injection

# Model configuration
model_settings:
  local_model_repository_folder: "./models"
  common:
    cache_dir: "./cache"
    device: "cuda"  # or "cpu"
  model_entries:
    translation_model:
      model_path: "path/to/model.pt"
      vocab_size: 50000
```

### 4. Deploy to Modal

```bash
# Test locally
modal serve app.py

# Deploy to production
modal deploy app.py

# View logs
modal logs -f
```

### 5. Use Your API

```python
import requests
import asyncio

# For standard API key auth
headers = {"x-api-key": "your-api-key"}

# Synchronous endpoint
response = requests.post(
    "https://your-org--translation-service.modal.run/predict_sync",
    json={"text": "Hello world", "language": "en"},
    headers=headers
)
print(response.json())
# {"translated_text": "HELLO WORLD", "confidence": 0.95}

# Asynchronous endpoint (returns immediately)
response = requests.post(
    "https://your-org--translation-service.modal.run/predict_async",
    json={"text": "Hello world", "language": "en"},
    headers=headers
)
print(response.json())
# {"message_id": "550e8400-e29b-41d4-a716-446655440000"}

# Batch endpoint
response = requests.post(
    "https://your-org--translation-service.modal.run/predict_batch",
    json=[
        {"text": "Hello", "language": "en"},
        {"text": "World", "language": "en"}
    ],
    headers=headers
)
print(response.json())
# [{"translated_text": "HELLO", "confidence": 0.95}, {"translated_text": "WORLD", "confidence": 0.95}]
```

## Authentication

Modalkit supports multiple authentication options:

### Option 1: Custom API Key (Default)
Configure with `secure: false` in your deployment config.

```yaml
# modalkit.yaml
deployment_config:
  secure: false

auth_config:
  # Store in AWS SSM (recommended)
  ssm_key: "/myapp/api-key"
  # OR hardcode (not recommended)
  # api_key: "sk-1234567890"
  auth_header: "x-api-key"
```

```python
# Client usage
headers = {"x-api-key": "your-api-key"}
response = requests.post(url, json=data, headers=headers)
```

### Option 2: Modal Proxy Authentication
Configure with `secure: true` for Modal's built-in auth:

```yaml
# modalkit.yaml
deployment_config:
  secure: true  # Enables Modal proxy auth
```

```python
# Client usage
headers = {
    "Modal-Key": "your-modal-key",
    "Modal-Secret": "your-modal-secret"
}
response = requests.post(url, json=data, headers=headers)
```

**Note**: Modal proxy auth is recommended for production as it's managed by Modal and requires no additional setup.

## Configuration

### Configuration Structure

Modalkit uses YAML configuration with two main sections:

```yaml
# modalkit.yaml
app_settings:        # Application deployment settings
  app_prefix: str    # Prefix for your Modal app name
  auth_config:       # Authentication configuration
  build_config:      # Container build settings
  deployment_config: # Runtime deployment settings
  batch_config:      # Batch processing settings
  queue_config:      # Async queue settings

model_settings:      # Model-specific settings
  local_model_repository_folder: str
  common: dict       # Shared settings across models
  model_entries:     # Model-specific configurations
    model_name: dict
```

### Environment Variables

Set configuration file location:
```bash
# Default location
export MODALKIT_CONFIG="modalkit.yaml"

# Multiple configs (later files override earlier ones)
export MODALKIT_CONFIG="base.yaml,prod.yaml"

# Other environment variables
export MODALKIT_APP_POSTFIX="-prod"  # Appended to app name
```

### Advanced Configuration Options

```yaml
deployment_config:
  # GPU configuration
  gpu: "T4"  # T4, A10G, A100, H100, or null

  # Resource limits
  concurrency_limit: 10
  container_idle_timeout: 300
  retries: 3

  # Memory/CPU (when gpu is null)
  memory: 8192  # MB
  cpu: 4.0      # cores

  # Volumes and mounts
  volumes:
    "/mnt/cache": "model-cache-vol"
  mounts:
    - local_path: "configs/prod.json"
      remote_path: "/app/config.json"
      type: "file"
```

## Cloud Storage Integration

Modalkit integrates with cloud storage providers through Modal's CloudBucketMount:

### Supported Providers

| Provider | Configuration |
|----------|--------------|
| AWS S3 | Native support with IAM credentials |
| Google Cloud Storage | Service account authentication |
| Cloudflare R2 | S3-compatible API |
| MinIO/Others | Any S3-compatible endpoint |

### Quick Examples

<details>
<summary><b>AWS S3 Configuration</b></summary>

```yaml
cloud_bucket_mounts:
  - mount_point: "/mnt/models"
    bucket_name: "my-ml-models"
    secret: "aws-credentials"  # Modal secret name
    key_prefix: "production/"  # Only mount this prefix
    read_only: true
```

First, create the Modal secret:
```bash
modal secret create aws-credentials \
  AWS_ACCESS_KEY_ID=xxx \
  AWS_SECRET_ACCESS_KEY=yyy \
  AWS_DEFAULT_REGION=us-east-1
```
</details>

<details>
<summary><b>Google Cloud Storage</b></summary>

```yaml
cloud_bucket_mounts:
  - mount_point: "/mnt/datasets"
    bucket_name: "my-datasets"
    bucket_endpoint_url: "https://storage.googleapis.com"
    secret: "gcp-credentials"
```

Create secret from service account:
```bash
modal secret create gcp-credentials \
  --from-gcp-service-account path/to/key.json
```
</details>

<details>
<summary><b>Cloudflare R2</b></summary>

```yaml
cloud_bucket_mounts:
  - mount_point: "/mnt/artifacts"
    bucket_name: "ml-artifacts"
    bucket_endpoint_url: "https://accountid.r2.cloudflarestorage.com"
    secret: "r2-credentials"
```
</details>

### Using Mounted Storage

```python
class MyInference(InferencePipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Load model from mounted bucket
        model_path = "/mnt/models/my_model.pt"
        self.model = torch.load(model_path)

        # Load dataset
        with open("/mnt/datasets/vocab.json") as f:
            self.vocab = json.load(f)
```

### Best Practices

- Use read-only mounts for model artifacts
- Mount only required prefixes with `key_prefix`
- Use separate buckets for models vs. data
- Cache frequently accessed files locally
- Avoid writing logs to mounted buckets
- Don't mount entire buckets if you only need specific files

## Advanced Features

### Queue Processing

Queue processing is optional and supports multiple backends:

#### 1. No Queues (Default)
For sync-only APIs:
```python
class MyService(ModalService):
    inference_implementation = MyModel

# No queue backend - async requests process but don't queue responses
service = MyService()
```

#### 2. TaskIQ Integration (Recommended for Production)
Use dependency injection for full TaskIQ support:
```python
from taskiq_redis import AsyncRedisTaskiqBroker

class TaskIQBackend:
    def __init__(self):
        self.broker = AsyncRedisTaskiqBroker("redis://localhost:6379")

    async def send_message(self, queue_name: str, message: str) -> bool:
        @self.broker.task(task_name=f"process_{queue_name}")
        async def process_result(msg: str) -> str:
            # Your custom processing logic
            return f"Processed: {msg}"

        await process_result.kiq(message)
        return True

# Inject TaskIQ backend
service = MyService(queue_backend=TaskIQBackend())
```

#### 3. Configuration-Based Queues
Use YAML configuration for simple setups:
```yaml
queue_config:
  backend: "sqs"  # or "memory"
  # Additional backend-specific settings
```

#### 4. Custom Queue Systems
Implement any queue system:
```python
class MyCustomQueue:
    async def send_message(self, queue_name: str, message: str) -> bool:
        # Your custom queue implementation (RabbitMQ, Kafka, etc.)
        return True

service = MyService(queue_backend=MyCustomQueue())
```

#### Working Examples
See complete tutorials in the documentation:
- **[Queue Backend Patterns](https://prassanna-ravishankar.github.io/modalkit/examples/queue-patterns/)** - Queue backend patterns
- **[TaskIQ Integration](https://prassanna-ravishankar.github.io/modalkit/examples/taskiq-integration/)** - Full TaskIQ integration

```python
# Async endpoint usage
response = requests.post("/predict_async", json={
    "message": {"text": "Process this"},
    "success_queue": "results",
    "failure_queue": "errors"
})
# {"job_id": "uuid"}
```

### Batch Processing

Configure intelligent batching for better GPU utilization:

```yaml
batch_config:
  max_batch_size: 32
  wait_ms: 100  # Max time to wait for batch to fill
```

### Volume Reloading

Auto-reload Modal volumes for model updates:

```yaml
deployment_config:
  volumes:
    "/mnt/models": "model-volume"
  volume_reload_interval_seconds: 300  # Reload every 5 minutes
```

## Development

### Setup

```bash
# Clone repository
git clone https://github.com/prassanna-ravishankar/modalkit.git
cd modalkit

# Install with uv (recommended)
uv sync

# Install pre-commit hooks
uv run pre-commit install
```

### Testing

```bash
# Run all tests
uv run pytest --cov --cov-config=pyproject.toml --cov-report=xml

# Run specific tests
uv run pytest tests/test_modal_service.py -v

# Run with HTML coverage report
uv run pytest --cov=modalkit --cov-report=html
```

### Code Quality

```bash
# Run all checks
uv run pre-commit run -a

# Run type checking
uv run mypy modalkit/

# Format code
uv run ruff format modalkit/ tests/

# Lint code
uv run ruff check modalkit/ tests/
```

## API Reference

### Endpoints

| Endpoint | Method | Description | Returns |
|----------|---------|-------------|----------|
| `/predict_sync` | POST | Synchronous inference | Model output |
| `/predict_async` | POST | Async inference (queued) | Message ID |
| `/predict_batch` | POST | Batch inference | List of outputs |
| `/health` | GET | Health check | Status |

### InferencePipeline Methods

Your model class must implement:

```python
def preprocess(self, input_list: List[InputModel]) -> dict
def predict(self, input_list: List[InputModel], preprocessed_data: dict) -> dict
def postprocess(self, input_list: List[InputModel], raw_output: dict) -> List[OutputModel]
```

## Contributing

See [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Run tests and linting (`uv run pytest && uv run pre-commit run -a`)
5. Commit your changes (pre-commit hooks will run automatically)
6. Push to your fork and open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Built using:
- [Modal](https://modal.com) - Serverless infrastructure for ML
- [FastAPI](https://fastapi.tiangolo.com) - Modern web framework
- [Pydantic](https://pydantic-docs.helpmanual.io) - Data validation
- [Taskiq](https://taskiq-python.github.io) - Async task processing

---

<p align="center">
  <a href="https://github.com/prassanna-ravishankar/modalkit/issues">Report Bug</a> •
  <a href="https://github.com/prassanna-ravishankar/modalkit/issues">Request Feature</a> •
  <a href="https://prassanna-ravishankar.github.io/modalkit">Documentation</a>
</p>
