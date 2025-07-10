# Async Processing with Queue Backends

ModalKit supports asynchronous processing for long-running inference tasks with flexible queue backend integration. This guide demonstrates how to implement async processing with custom queue systems like TaskIQ.

## Overview

Async processing in ModalKit involves:

1. **Optional Queue Backends**: Use any queue system or none at all
2. **Dependency Injection**: Inject custom queue backends like TaskIQ
3. **Flexible Configuration**: Configure via YAML or code
4. **TaskIQ Native Support**: Proper task definitions and workers

## Queue Backend Options

### 1. No Queues (Default)

Perfect for sync-only APIs or when you don't need async processing:

```python
from modalkit.modal_service import ModalService

class MyService(ModalService):
    inference_implementation = MyInferencePipeline

# No queue backend - async requests are processed but responses aren't queued
service = MyService()
```

### 2. Configuration-Based Queues

Uses settings from `modalkit.yaml`:

```yaml
# modalkit.yaml
app_settings:
  queue_config:
    backend: "sqs"
    # Additional SQS configuration
```

```python
# Automatically uses configured backend
service = MyService()
```

### 3. TaskIQ Integration (Recommended)

For production async processing with TaskIQ:

```python
from taskiq_redis import AsyncRedisTaskiqBroker

class TaskIQBackend:
    def __init__(self, broker_url="redis://localhost:6379"):
        self.broker = AsyncRedisTaskiqBroker(broker_url)

    async def send_message(self, queue_name: str, message: str) -> bool:
        @self.broker.task(task_name=f"process_{queue_name}")
        async def process_ml_result(msg: str) -> str:
            # Your custom processing logic
            import json
            result = json.loads(msg)
            # Process the ML inference result
            # - Send notifications
            # - Update databases
            # - Trigger downstream systems
            return f"Processed {result['status']} result"

        await process_ml_result.kiq(message)
        return True

# Inject TaskIQ backend
taskiq_backend = TaskIQBackend("redis://localhost:6379")
service = MyService(queue_backend=taskiq_backend)
```

### 4. Custom Queue Implementation

Implement any queue system:

```python
class MyCustomQueueBackend:
    async def send_message(self, queue_name: str, message: str) -> bool:
        # Send to your custom queue system (RabbitMQ, Kafka, etc.)
        await my_queue_system.send(queue_name, message)
        return True

service = MyService(queue_backend=MyCustomQueueBackend())
```

## Complete Example

### 1. Define Models

```python
from modalkit.iomodel import AsyncInputModel, InferenceOutputModel
from pydantic import BaseModel

class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(InferenceOutputModel):
    sentiment: str
    confidence: float
```

### 2. Implement Inference Pipeline

```python
from modalkit.inference_pipeline import InferencePipeline
from typing import List

class SentimentPipeline(InferencePipeline):
    def preprocess(self, input_list: List[BaseModel]) -> dict:
        texts = [getattr(req, 'text', str(req)) for req in input_list]
        return {"texts": texts}

    def predict(self, input_list: List[BaseModel], preprocessed_data: dict) -> dict:
        predictions = []
        for text in preprocessed_data["texts"]:
            if "good" in text.lower():
                predictions.append({"sentiment": "positive", "confidence": 0.95})
            elif "bad" in text.lower():
                predictions.append({"sentiment": "negative", "confidence": 0.90})
            else:
                predictions.append({"sentiment": "neutral", "confidence": 0.70})
        return {"predictions": predictions}

    def postprocess(self, input_list: List[BaseModel], raw_output: dict) -> List[InferenceOutputModel]:
        results = []
        for prediction in raw_output["predictions"]:
            results.append(SentimentResponse(
                status="success",
                sentiment=prediction["sentiment"],
                confidence=prediction["confidence"]
            ))
        return results
```

### 3. Configure Modal Service

```python
import modal
from modalkit.modal_service import ModalService, create_web_endpoints
from modalkit.modal_config import ModalConfig

class SentimentService(ModalService):
    inference_implementation = SentimentPipeline

    def __init__(self, queue_backend=None):
        super().__init__(queue_backend=queue_backend)
        self.model_name = "sentiment-analyzer"

# Setup with TaskIQ (optional)
if USE_TASKIQ:
    taskiq_backend = TaskIQBackend("redis://localhost:6379")
    app_cls = lambda: SentimentService(queue_backend=taskiq_backend)
else:
    app_cls = SentimentService

# Modal configuration
app = modal.App(name="sentiment-service")

@app.cls(**modal_config.get_app_cls_settings())
class SentimentApp(app_cls):
    modal_utils: ModalConfig = modal_config

@app.function(**modal_config.get_handler_settings())
@modal.asgi_app()
def web_endpoints():
    return create_web_endpoints(
        app_cls=SentimentApp,
        input_model=SentimentRequest,
        output_model=SentimentResponse
    )
```

## Usage Examples

### Basic Async Request

```python
import requests

# Submit async job with queue processing
response = requests.post(
    "http://localhost:8000/async/predict",
    json={
        "message": {"text": "This movie is great!"},
        "success_queue": "sentiment_results",
        "failure_queue": "sentiment_errors",
        "meta": {"user_id": "12345", "request_id": "req_001"}
    },
    headers={"Authorization": "Bearer your-token"}
)

job_id = response.json()["job_id"]
print(f"Job submitted: {job_id}")
```

### TaskIQ Worker Setup

For production TaskIQ usage, run workers separately:

```bash
# Install TaskIQ
pip install taskiq taskiq-redis

# Run worker
taskiq worker your_module:broker
```

## Working Examples

See complete working examples in the documentation:

- **[Queue Backend Patterns](queue-patterns.md)** - Basic queue backend patterns and dependency injection
- **[TaskIQ Integration](taskiq-integration.md)** - Full TaskIQ integration tutorial

Follow the tutorials to build complete working examples with your own ML models.

## Error Handling

### Queue Send Failures

Queue failures don't break inference processing:

```python
# If queue send fails, inference still completes
# Errors are logged but don't affect the response
async def send_async_response(self, ...):
    try:
        success = await self._send_to_queue(queue_name, message)
        if not success:
            logger.warning(f"Failed to send to queue: {queue_name}")
    except Exception as e:
        logger.error(f"Queue error: {e}")
        # Processing continues normally
```

### TaskIQ Task Failures

Handle task failures in your TaskIQ workers:

```python
@broker.task
async def process_ml_result(message: str) -> str:
    try:
        result = json.loads(message)
        # Process result
        return "success"
    except Exception as e:
        # Log error and optionally retry
        logger.error(f"Task failed: {e}")
        raise  # TaskIQ will handle retries based on configuration
```

## Best Practices

### 1. Queue Backend Selection

- **No Queues**: Simple sync APIs, testing
- **TaskIQ**: Production async processing with workers
- **Configuration-based**: When you need SQS/existing infrastructure
- **Custom**: Specialized queue systems (RabbitMQ, Kafka, etc.)

### 2. Error Resilience

```python
# Always handle queue failures gracefully
async def _send_to_queue(self, queue_name: str, message: str) -> bool:
    if self.queue_backend:
        try:
            return await self.queue_backend.send_message(queue_name, message)
        except Exception as e:
            logger.error(f"Queue backend error: {e}")
            return False
    return False  # No backend configured
```

### 3. TaskIQ Production Setup

```python
# Use proper brokers for production
from taskiq_redis import AsyncRedisTaskiqBroker

# Production Redis setup
broker = AsyncRedisTaskiqBroker(
    url="redis://redis-cluster:6379",
    max_connections=20,
    retry_policy=ExponentialBackoff(max_retries=3)
)
```

### 4. Message Structure

Ensure your queue messages contain all necessary metadata:

```python
{
    "status": "success",
    "result": {...},
    "meta": {
        "user_id": "12345",
        "request_id": "req_001",
        "timestamp": "2024-01-15T10:30:00Z"
    }
}
```

## Configuration

Configure queue settings in `modalkit.yaml`:

```yaml
app_settings:
  # Optional: fallback queue configuration
  queue_config:
    backend: "memory"  # or "sqs"

  # Optional: async processing settings
  async_config:
    timeout: 3600
    result_ttl: 86400
```

## Advanced Features

### Multiple Queue Backends

You can even implement routing to different backends:

```python
class MultiQueueBackend:
    def __init__(self):
        self.taskiq_backend = TaskIQBackend()
        self.sqs_backend = SQSBackend()

    async def send_message(self, queue_name: str, message: str) -> bool:
        if queue_name.startswith("ml_"):
            return await self.taskiq_backend.send_message(queue_name, message)
        else:
            return await self.sqs_backend.send_message(queue_name, message)
```

This design gives you complete flexibility to implement whatever async processing pattern works best for your use case!
