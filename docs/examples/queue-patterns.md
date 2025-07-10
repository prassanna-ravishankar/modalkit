# TaskIQ Queue Integration

This tutorial shows how to integrate TaskIQ with ModalKit for production-ready async ML processing. TaskIQ is the recommended queue system for ModalKit due to its simplicity and powerful features.

## Why TaskIQ?

TaskIQ is a modern async task queue for Python that provides:
- **Native async support**: Built for modern Python async/await patterns
- **Multiple brokers**: Redis, SQS, RabbitMQ, and more
- **Result backends**: Store and retrieve task results
- **Production ready**: Retries, monitoring, and error handling
- **Simple integration**: Clean APIs that work well with dependency injection

## Quick Setup

### 1. Install TaskIQ with Redis

```bash
# Install core TaskIQ
pip install taskiq

# Install Redis broker (recommended)
pip install taskiq-redis
```

### 2. Basic TaskIQ Backend for ModalKit

```python
from taskiq_redis import RedisStreamBroker, RedisAsyncResultBackend
import json

class TaskIQBackend:
    """Simple TaskIQ backend for ModalKit integration"""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        # Create broker and result backend
        self.result_backend = RedisAsyncResultBackend(redis_url=redis_url)
        self.broker = RedisStreamBroker(url=redis_url).with_result_backend(self.result_backend)

        # Define your task processors
        self._setup_tasks()

    def _setup_tasks(self):
        """Setup TaskIQ tasks for processing ML results"""

        @self.broker.task
        async def process_ml_result(message: str) -> str:
            """Process successful ML inference results"""
            try:
                result = json.loads(message)

                # Your custom processing logic here:
                print(f"🎯 Processing ML result:")
                print(f"   Status: {result.get('status')}")
                print(f"   Confidence: {result.get('confidence')}")

                # Examples of what you might do:
                # - Send notifications
                # - Update databases
                # - Trigger downstream ML pipelines
                # - Store results in analytics systems

                return "success"
            except Exception as e:
                print(f"Error processing result: {e}")
                raise  # TaskIQ will handle retries

        @self.broker.task
        async def process_ml_error(message: str) -> str:
            """Process ML inference errors"""
            try:
                error_result = json.loads(message)

                print(f"❌ Processing ML error:")
                print(f"   Error: {error_result.get('error')}")
                print(f"   Request ID: {error_result.get('request_id')}")

                # Error handling:
                # - Log to monitoring systems
                # - Send alerts
                # - Retry failed requests

                return "error_handled"
            except Exception as e:
                print(f"Error processing error: {e}")
                raise

        # Store task references
        self.process_ml_result = process_ml_result
        self.process_ml_error = process_ml_error

    async def send_message(self, queue_name: str, message: str) -> bool:
        """Send message to appropriate TaskIQ task"""
        try:
            # Route to different tasks based on queue name
            if queue_name.endswith("_results"):
                await self.process_ml_result.kiq(message)
            elif queue_name.endswith("_errors"):
                await self.process_ml_error.kiq(message)
            else:
                # Default to result processing
                await self.process_ml_result.kiq(message)

            return True
        except Exception as e:
            print(f"TaskIQ send error: {e}")
            return False
```

## Integration with ModalKit

### 1. Simple ML Pipeline

```python
from modalkit.inference_pipeline import InferencePipeline
from modalkit.iomodel import InferenceOutputModel
from pydantic import BaseModel
from typing import List

class TextRequest(BaseModel):
    text: str

class TextResponse(InferenceOutputModel):
    result: str
    confidence: float

class SimplePipeline(InferencePipeline):
    def preprocess(self, input_list: List[BaseModel]) -> dict:
        texts = [getattr(req, 'text', str(req)) for req in input_list]
        return {"texts": texts}

    def predict(self, input_list: List[BaseModel], preprocessed_data: dict) -> dict:
        # Simple text processing
        results = []
        for text in preprocessed_data["texts"]:
            processed = text.upper()  # Simple transformation
            results.append({"result": processed, "confidence": 0.95})
        return {"predictions": results}

    def postprocess(self, input_list: List[BaseModel], raw_output: dict) -> List[InferenceOutputModel]:
        results = []
        for prediction in raw_output["predictions"]:
            results.append(TextResponse(
                status="success",
                result=prediction["result"],
                confidence=prediction["confidence"]
            ))
        return results
```

### 2. ModalService with TaskIQ

```python
from modalkit.modal_service import ModalService

class TextProcessingService(ModalService):
    inference_implementation = SimplePipeline

    def __init__(self, queue_backend=None):
        super().__init__(queue_backend=queue_backend)
        self.model_name = "text-processor"

# Create TaskIQ backend
taskiq_backend = TaskIQBackend("redis://localhost:6379")

# Create service with TaskIQ integration
service = TextProcessingService(queue_backend=taskiq_backend)
```

### 3. Complete Modal App

```python
import modal
from modalkit.modal_service import ModalService, create_web_endpoints
from modalkit.modal_config import ModalConfig

# TaskIQ backend
taskiq_backend = TaskIQBackend("redis://redis:6379")  # Use Redis hostname in Modal

# Modal configuration
modal_config = ModalConfig()
app = modal.App(name="text-processing-with-taskiq")

@app.cls(**modal_config.get_app_cls_settings())
class TextProcessingApp(ModalService):
    inference_implementation = SimplePipeline

    def __init__(self):
        super().__init__(queue_backend=taskiq_backend)
        self.model_name = "text-processor"

@app.function(**modal_config.get_handler_settings())
@modal.asgi_app()
def web_endpoints():
    return create_web_endpoints(
        app_cls=TextProcessingApp,
        input_model=TextRequest,
        output_model=TextResponse
    )
```

## TaskIQ Worker Setup

### 1. Worker Script

Create `worker.py`:

```python
import asyncio
from your_app import taskiq_backend

async def main():
    """Start TaskIQ worker"""
    print("🚀 Starting TaskIQ worker...")

    # Start the broker
    await taskiq_backend.broker.startup()

    try:
        # Keep worker running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("👋 Shutting down worker...")
    finally:
        await taskiq_backend.broker.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. Run Worker

```bash
# Option 1: Use the worker script
python worker.py

# Option 2: Use TaskIQ CLI (recommended)
taskiq worker your_app:taskiq_backend.broker
```

## Usage Examples

### Async Processing

```python
import requests

headers = {"Authorization": "Bearer your-token"}

# Submit async request with TaskIQ processing
response = requests.post(
    "https://your-org--text-processing-with-taskiq.modal.run/predict_async",
    json={
        "message": {"text": "Process this text asynchronously"},
        "success_queue": "text_results",
        "failure_queue": "text_errors",
        "meta": {"user_id": "123", "request_id": "req_001"}
    },
    headers=headers
)

job_id = response.json()["job_id"]
print(f"Job submitted: {job_id}")
```

When this request is processed:
1. ModalKit processes the ML inference
2. Results are sent to TaskIQ via the `TaskIQBackend`
3. TaskIQ workers process the results asynchronously
4. You can implement custom logic in the TaskIQ tasks

## Advanced TaskIQ Features

### 1. Different Broker Types

The [taskiq-redis](https://github.com/taskiq-python/taskiq-redis) package provides multiple broker types:

```python
from taskiq_redis import (
    RedisStreamBroker,     # Recommended: supports acknowledgments
    RedisListQueueBroker,  # Simple: no acknowledgments
    RedisPubSubBroker      # Broadcasting: no acknowledgments
)

# Stream broker (recommended for production)
broker = RedisStreamBroker(url="redis://localhost:6379")

# List queue broker (simpler, but no guarantees)
broker = RedisListQueueBroker(url="redis://localhost:6379")

# PubSub broker (for broadcasting to multiple workers)
broker = RedisPubSubBroker(url="redis://localhost:6379")
```

### 2. Production Configuration

```python
from taskiq_redis import RedisStreamBroker, RedisAsyncResultBackend

class ProductionTaskIQBackend:
    def __init__(self):
        # Production-ready configuration
        self.result_backend = RedisAsyncResultBackend(
            redis_url="redis://redis-cluster:6379",
            result_ex_time=3600,  # Results expire after 1 hour
            keep_results=False    # Clean up after reading
        )

        self.broker = RedisStreamBroker(
            url="redis://redis-cluster:6379"
        ).with_result_backend(self.result_backend)
```

### 3. Error Handling and Retries

```python
from taskiq import ExponentialBackoff

@broker.task(retry_policy=ExponentialBackoff(max_retries=3))
async def robust_ml_processor(message: str) -> str:
    """Task with automatic retries"""
    try:
        # Process ML result
        return "success"
    except Exception as e:
        print(f"Task failed, will retry: {e}")
        raise  # TaskIQ handles retries automatically
```

## Alternative: Simple Queue Backends

If you don't need TaskIQ's full features, ModalKit also provides simple built-in backends:

```python
from modalkit.task_queue import InMemoryBackend, SQSBackend

# For testing
memory_backend = InMemoryBackend()
service = TextProcessingService(queue_backend=memory_backend)

# For AWS SQS
sqs_backend = SQSBackend(
    queue_url="https://sqs.us-east-1.amazonaws.com/123456789/my-queue"
)
service = TextProcessingService(queue_backend=sqs_backend)
```

## Summary

TaskIQ integration with ModalKit is straightforward:

1. **Install**: `pip install taskiq taskiq-redis`
2. **Create backend**: Implement `TaskIQBackend` with your task logic
3. **Inject**: Pass backend to `ModalService(queue_backend=taskiq_backend)`
4. **Deploy**: Run TaskIQ workers separately from your Modal app

The [taskiq-redis](https://github.com/taskiq-python/taskiq-redis) package makes Redis integration simple and provides production-ready features like acknowledgments, retries, and result storage.

**Key Benefits:**
- ✅ **Native async**: Perfect for modern ML workflows
- ✅ **Production ready**: Retries, monitoring, persistence
- ✅ **Flexible**: Multiple broker types and configurations
- ✅ **Scalable**: Independent worker scaling
- ✅ **Simple**: Clean integration with ModalKit's dependency injection
