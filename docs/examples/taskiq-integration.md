# TaskIQ Integration Tutorial

This tutorial demonstrates how to integrate TaskIQ with ModalKit for production-ready async ML processing. You'll learn how to set up TaskIQ workers, define tasks, and handle ML inference results.

## Overview

TaskIQ is a powerful async task queue system for Python. When combined with ModalKit, it provides:

- **Native TaskIQ patterns**: Proper task definitions and workers
- **Production-ready**: Redis, SQS, and RabbitMQ brokers
- **Dependency injection**: Clean integration with ModalService
- **Error handling**: Retries and failure handling
- **Monitoring**: Task status and metrics

## Prerequisites

Install TaskIQ and broker dependencies:

```bash
# Core TaskIQ
pip install taskiq

# Redis broker (recommended)
pip install taskiq-redis

# AWS SQS broker
pip install taskiq-aws

# RabbitMQ broker
pip install taskiq-rabbitmq
```

## TaskIQ Backend Implementation

### Basic TaskIQ Backend

```python
from taskiq_redis import AsyncRedisTaskiqBroker
from typing import Any
import json

class TaskIQQueueBackend:
    """TaskIQ-based queue backend for ModalKit"""

    def __init__(self, broker_url: str = "redis://localhost:6379"):
        """Initialize TaskIQ broker"""
        self.broker = AsyncRedisTaskiqBroker(broker_url)

        # For AWS SQS:
        # from taskiq_aws import SQSBroker
        # self.broker = SQSBroker(queue_url=broker_url)

        # For RabbitMQ:
        # from taskiq_rabbitmq import RabbitBroker
        # self.broker = RabbitBroker(broker_url)

    async def send_message(self, queue_name: str, message: str) -> bool:
        """Send message to TaskIQ task queue"""
        try:
            # Define a task function for this queue
            @self.broker.task(task_name=f"process_{queue_name}")
            async def process_queue_message(msg: str) -> str:
                """Process ML inference results"""
                result = json.loads(msg)

                # Your custom processing logic here:
                # - Send notifications
                # - Update databases
                # - Trigger downstream systems
                # - Store results

                print(f"📝 Processing result from queue '{queue_name}':")
                print(f"   Status: {result.get('status', 'unknown')}")

                return f"Processed message from {queue_name}"

            # Send the message to the task queue
            await process_queue_message.kiq(message)
            print(f"✅ Message sent to TaskIQ queue: {queue_name}")
            return True

        except Exception as e:
            print(f"❌ Failed to send TaskIQ message: {e}")
            return False
```

### Production TaskIQ Backend

For production use, pre-define your tasks for better performance:

```python
from taskiq_redis import AsyncRedisTaskiqBroker
from taskiq import ExponentialBackoff
import json

# Global broker instance
broker = AsyncRedisTaskiqBroker(
    url="redis://redis-cluster:6379",
    max_connections=20,
    retry_policy=ExponentialBackoff(max_retries=3)
)

# Pre-defined tasks
@broker.task(task_name="process_ml_results")
async def process_ml_results(message: str) -> str:
    """Process successful ML inference results"""
    try:
        result = json.loads(message)

        # Process successful results
        if result.get("status") == "success":
            # Send notifications
            await send_success_notification(result)

            # Update user dashboards
            await update_user_dashboard(result)

            # Store results in database
            await store_result_in_db(result)

        return "success"
    except Exception as e:
        print(f"Error processing ML result: {e}")
        raise  # TaskIQ will handle retries

@broker.task(task_name="process_ml_errors")
async def process_ml_errors(message: str) -> str:
    """Process ML inference errors"""
    try:
        error_result = json.loads(message)

        # Handle errors
        await log_error(error_result)
        await send_error_notification(error_result)

        # Optionally retry the original request
        if should_retry(error_result):
            await retry_ml_request(error_result)

        return "error_handled"
    except Exception as e:
        print(f"Error processing ML error: {e}")
        raise

class ProductionTaskIQBackend:
    """Production-ready TaskIQ backend"""

    def __init__(self):
        self.broker = broker

    async def send_message(self, queue_name: str, message: str) -> bool:
        """Send message to pre-defined TaskIQ tasks"""
        try:
            # Route to appropriate task based on queue name
            if queue_name == "ml_results":
                await process_ml_results.kiq(message)
            elif queue_name == "ml_errors":
                await process_ml_errors.kiq(message)
            else:
                # Generic processing for other queues
                await process_generic_message.kiq(queue_name, message)

            return True
        except Exception as e:
            print(f"TaskIQ send error: {e}")
            return False
```

## ML Pipeline Integration

### 1. Define Your Models

```python
from pydantic import BaseModel
from modalkit.iomodel import InferenceOutputModel

class SentimentRequest(BaseModel):
    text: str
    user_id: str
    request_id: str

class SentimentResponse(InferenceOutputModel):
    sentiment: str
    confidence: float
    processing_time: float
```

### 2. Create Inference Pipeline

```python
from modalkit.inference_pipeline import InferencePipeline
from typing import List, Any
import time

class SentimentPipeline(InferencePipeline):
    """Sentiment analysis pipeline with TaskIQ integration"""

    def __init__(self, model_name: str, all_model_data_folder: str,
                 common_settings: dict, **kwargs: Any):
        super().__init__(model_name, all_model_data_folder, common_settings)
        # Load your model here
        self.model_version = kwargs.get("model_version", "v1.0")

    def preprocess(self, input_list: List[BaseModel]) -> dict:
        """Preprocess text inputs"""
        texts = [getattr(req, 'text', str(req)) for req in input_list]
        return {"texts": texts}

    def predict(self, input_list: List[BaseModel], preprocessed_data: dict) -> dict:
        """Run sentiment prediction"""
        start_time = time.time()
        predictions = []

        for text in preprocessed_data["texts"]:
            if "good" in text.lower() or "great" in text.lower():
                sentiment, confidence = "positive", 0.95
            elif "bad" in text.lower() or "terrible" in text.lower():
                sentiment, confidence = "negative", 0.90
            else:
                sentiment, confidence = "neutral", 0.70

            predictions.append({
                "sentiment": sentiment,
                "confidence": confidence,
                "processing_time": time.time() - start_time
            })

        return {"predictions": predictions}

    def postprocess(self, input_list: List[BaseModel], raw_output: dict) -> List[InferenceOutputModel]:
        """Format outputs"""
        results = []
        for i, prediction in enumerate(raw_output["predictions"]):
            results.append(SentimentResponse(
                status="success",
                sentiment=prediction["sentiment"],
                confidence=prediction["confidence"],
                processing_time=prediction["processing_time"]
            ))
        return results
```

### 3. Create Service with TaskIQ Backend

```python
from modalkit.modal_service import ModalService

class SentimentService(ModalService):
    """Sentiment analysis service with TaskIQ integration"""

    inference_implementation = SentimentPipeline

    def __init__(self, queue_backend=None):
        super().__init__(queue_backend=queue_backend)
        self.model_name = "sentiment-analyzer"

    async def custom_processing(self, result: dict) -> None:
        """Custom processing after inference"""
        # Add custom metadata
        result["service_version"] = "v2.0"
        result["timestamp"] = time.time()

        # Send to appropriate queue based on status
        if result.get("status") == "success":
            await self._send_to_queue("ml_results", json.dumps(result))
        else:
            await self._send_to_queue("ml_errors", json.dumps(result))
```

## Complete Example

Here's a full working example:

```python
"""
Complete TaskIQ + ModalKit Integration Example
"""
import asyncio
import json
import time
from typing import Any, List

from pydantic import BaseModel
from taskiq_redis import AsyncRedisTaskiqBroker

from modalkit.inference_pipeline import InferencePipeline
from modalkit.iomodel import InferenceOutputModel
from modalkit.modal_service import ModalService

# ===== TaskIQ Setup =====

# Create TaskIQ broker
broker = AsyncRedisTaskiqBroker("redis://localhost:6379")

# Define tasks
@broker.task(task_name="process_sentiment_results")
async def process_sentiment_results(message: str) -> str:
    """Process successful sentiment analysis results"""
    try:
        result = json.loads(message)

        # Simulate processing
        print(f"🎯 Processing sentiment result:")
        print(f"   User: {result.get('user_id', 'unknown')}")
        print(f"   Sentiment: {result.get('sentiment', 'unknown')}")
        print(f"   Confidence: {result.get('confidence', 0):.2f}")

        # Your custom logic here:
        # - Send email notifications
        # - Update user preferences
        # - Store in analytics database
        # - Trigger recommendation updates

        return "success"
    except Exception as e:
        print(f"Error processing sentiment result: {e}")
        raise

@broker.task(task_name="process_sentiment_errors")
async def process_sentiment_errors(message: str) -> str:
    """Process sentiment analysis errors"""
    try:
        error_result = json.loads(message)

        print(f"❌ Processing sentiment error:")
        print(f"   Error: {error_result.get('error', 'unknown')}")
        print(f"   Request ID: {error_result.get('request_id', 'unknown')}")

        # Error handling logic:
        # - Log to monitoring system
        # - Send alerts to team
        # - Retry if appropriate

        return "error_handled"
    except Exception as e:
        print(f"Error processing sentiment error: {e}")
        raise

# ===== TaskIQ Backend =====

class TaskIQSentimentBackend:
    """TaskIQ backend for sentiment analysis"""

    def __init__(self, broker_url: str = "redis://localhost:6379"):
        self.broker = broker

    async def send_message(self, queue_name: str, message: str) -> bool:
        """Send message to appropriate TaskIQ task"""
        try:
            if queue_name == "sentiment_results":
                await process_sentiment_results.kiq(message)
            elif queue_name == "sentiment_errors":
                await process_sentiment_errors.kiq(message)
            else:
                print(f"Unknown queue: {queue_name}")
                return False

            return True
        except Exception as e:
            print(f"TaskIQ send error: {e}")
            return False

# ===== Models =====

class SentimentRequest(BaseModel):
    text: str
    user_id: str = "anonymous"
    request_id: str = "req_001"

class SentimentResponse(InferenceOutputModel):
    sentiment: str
    confidence: float
    user_id: str
    request_id: str

# ===== ML Pipeline =====

class SentimentPipeline(InferencePipeline):
    """Sentiment analysis pipeline"""

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
        for i, prediction in enumerate(raw_output["predictions"]):
            input_req = input_list[i]
            results.append(SentimentResponse(
                status="success",
                sentiment=prediction["sentiment"],
                confidence=prediction["confidence"],
                user_id=getattr(input_req, 'user_id', 'anonymous'),
                request_id=getattr(input_req, 'request_id', 'req_001')
            ))
        return results

# ===== Service =====

class SentimentService(ModalService):
    """Sentiment service with TaskIQ integration"""

    inference_implementation = SentimentPipeline

    def __init__(self, queue_backend=None):
        super().__init__(queue_backend=queue_backend)
        self.model_name = "sentiment-analyzer"

# ===== Demo =====

async def main():
    """Demo TaskIQ integration"""

    print("🚀 TaskIQ + ModalKit Integration Demo")
    print("=" * 50)

    # Create TaskIQ backend
    taskiq_backend = TaskIQSentimentBackend()

    # Create service with TaskIQ backend
    service = SentimentService(queue_backend=taskiq_backend)

    # Test direct queue sending
    print("\n📤 Testing queue message sending...")

    # Success result
    success_result = {
        "status": "success",
        "sentiment": "positive",
        "confidence": 0.95,
        "user_id": "user_123",
        "request_id": "req_001"
    }

    await service._send_to_queue("sentiment_results", json.dumps(success_result))

    # Error result
    error_result = {
        "status": "error",
        "error": "Model timeout",
        "request_id": "req_002",
        "user_id": "user_456"
    }

    await service._send_to_queue("sentiment_errors", json.dumps(error_result))

    print("\n✅ Demo completed!")

if __name__ == "__main__":
    asyncio.run(main())
```

## TaskIQ Worker Setup

### 1. Create Worker Script

Create `worker.py`:

```python
"""
TaskIQ Worker for ML Processing
"""
import asyncio
from taskiq_redis import AsyncRedisTaskiqBroker

# Import your tasks
from your_app import broker, process_sentiment_results, process_sentiment_errors

async def main():
    """Start TaskIQ worker"""
    print("🚀 Starting TaskIQ worker...")

    # Start worker
    await broker.startup()

    # Keep running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("👋 Shutting down worker...")
    finally:
        await broker.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. Run Worker

```bash
# Start worker
python worker.py

# Or use TaskIQ CLI
taskiq worker your_app:broker
```

## Configuration

### Redis Configuration

```python
from taskiq_redis import AsyncRedisTaskiqBroker

broker = AsyncRedisTaskiqBroker(
    url="redis://localhost:6379",
    max_connections=20,
    retry_policy=ExponentialBackoff(max_retries=3),
    result_ttl=3600  # Keep results for 1 hour
)
```

### SQS Configuration

```python
from taskiq_aws import SQSBroker

broker = SQSBroker(
    queue_url="https://sqs.us-east-1.amazonaws.com/123456789/ml-queue",
    region="us-east-1",
    visibility_timeout=300,
    message_retention_period=86400
)
```

### RabbitMQ Configuration

```python
from taskiq_rabbitmq import RabbitBroker

broker = RabbitBroker(
    url="amqp://user:password@localhost:5672",
    exchange="ml_exchange",
    routing_key="ml_tasks"
)
```

## Modal App Integration

### Complete Modal App

```python
"""
Complete Modal app with TaskIQ integration
"""
import modal
from modalkit.modal_service import ModalService, create_web_endpoints
from modalkit.modal_config import ModalConfig

# TaskIQ backend
taskiq_backend = TaskIQSentimentBackend("redis://redis:6379")

# Modal configuration
modal_config = ModalConfig()
app = modal.App(name="sentiment-with-taskiq")

@app.cls(**modal_config.get_app_cls_settings())
class SentimentApp(ModalService):
    inference_implementation = SentimentPipeline

    def __init__(self):
        super().__init__(queue_backend=taskiq_backend)
        self.model_name = "sentiment-analyzer"

@app.function(**modal_config.get_handler_settings())
@modal.asgi_app()
def web_endpoints():
    return create_web_endpoints(
        app_cls=SentimentApp,
        input_model=SentimentRequest,
        output_model=SentimentResponse
    )
```

## Best Practices

### 1. Task Organization

```python
# Group related tasks
@broker.task(task_name="ml.sentiment.process_result")
async def process_sentiment_result(message: str) -> str:
    pass

@broker.task(task_name="ml.sentiment.process_error")
async def process_sentiment_error(message: str) -> str:
    pass

@broker.task(task_name="ml.classification.process_result")
async def process_classification_result(message: str) -> str:
    pass
```

### 2. Error Handling

```python
from taskiq import ExponentialBackoff

@broker.task(
    task_name="process_ml_result",
    retry_policy=ExponentialBackoff(max_retries=3)
)
async def process_ml_result(message: str) -> str:
    try:
        # Process message
        return "success"
    except Exception as e:
        # Log error
        logger.error(f"Task failed: {e}")
        raise  # TaskIQ will handle retries
```

### 3. Monitoring

```python
from taskiq import TaskiqMiddleware

class MetricsMiddleware(TaskiqMiddleware):
    """Middleware to track task metrics"""

    async def on_task_start(self, task_name: str, **kwargs):
        # Track task start
        pass

    async def on_task_end(self, task_name: str, result: Any, **kwargs):
        # Track task completion
        pass

    async def on_task_error(self, task_name: str, error: Exception, **kwargs):
        # Track task errors
        pass

# Add middleware
broker.add_middleware(MetricsMiddleware())
```

## Testing

### Unit Tests

```python
import pytest
from your_app import TaskIQSentimentBackend, process_sentiment_results

@pytest.mark.asyncio
async def test_taskiq_backend():
    """Test TaskIQ backend"""
    backend = TaskIQSentimentBackend()

    # Test successful send
    result = await backend.send_message("sentiment_results", '{"test": "data"}')
    assert result is True

@pytest.mark.asyncio
async def test_sentiment_task():
    """Test sentiment processing task"""
    message = '{"sentiment": "positive", "confidence": 0.95}'
    result = await process_sentiment_results(message)
    assert result == "success"
```

### Integration Tests

```python
@pytest.mark.asyncio
async def test_full_pipeline():
    """Test complete ML + TaskIQ pipeline"""
    service = SentimentService(queue_backend=TaskIQSentimentBackend())

    # Test inference with queue processing
    request = SentimentRequest(text="This is great!", user_id="test_user")

    # Process would normally happen via HTTP endpoint
    # This is just for testing the integration
    result = await service.process_request(request)
    assert result.sentiment == "positive"
```

## Deployment

### 1. Modal Deployment

```bash
# Deploy the ML service
modal deploy app.py
```

### 2. Worker Deployment

Deploy TaskIQ workers separately:

```bash
# Using Docker
docker run -d your-taskiq-worker

# Or on separate servers
python worker.py
```

### 3. Monitoring

Set up monitoring for:
- Task queue lengths
- Task processing times
- Error rates
- Worker health

## Troubleshooting

### Common Issues

1. **Connection errors**: Check Redis/broker connectivity
2. **Task not found**: Ensure tasks are imported in worker
3. **Serialization errors**: Validate JSON messages
4. **Memory issues**: Monitor worker memory usage

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# TaskIQ debug
broker = AsyncRedisTaskiqBroker(
    url="redis://localhost:6379",
    debug=True
)
```

This tutorial provides a complete foundation for integrating TaskIQ with ModalKit. The pattern allows you to build sophisticated async ML processing pipelines while maintaining clean separation between inference and queue processing logic.
