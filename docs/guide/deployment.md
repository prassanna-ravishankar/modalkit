# Deployment Guide

ModalKit uses [Modal](https://modal.com/) for deploying ML applications. This guide covers how to deploy your ML applications using ModalKit.

## Overview

ModalKit provides a standardized way to deploy ML models through:

1. A base Modal application class (`ModalService`)
2. Configuration-driven deployment settings
3. Built-in support for both synchronous and asynchronous inference
4. OpenTelemetry integration for monitoring

## Basic Deployment

### 1. Create Your Application

```python
import modal
from modalkit.modal_service import ModalService, create_web_endpoints
from modalkit.modal_config import ModalConfig
from modalkit.settings import Settings

# Load configuration
Settings.model_config["yaml_file"] = "modalkit.yaml"
modal_config = ModalConfig()

# Create Modal app
app = modal.App(name=modal_config.app_name)

@app.cls(**modal_config.get_app_cls_settings())
class MyApp(ModalService):
    inference_implementation = MyInference
    model_name: str = modal.parameter(default="my_model")
    modal_utils: ModalConfig = modal_config

@app.function(**modal_config.get_handler_settings())
@modal.asgi_app()
def web_endpoints():
    return create_web_endpoints(
        app_cls=MyApp,
        input_model=MyInput,
        output_model=MyOutput
    )
```

### 2. Configure Deployment

In `modalkit.yaml`:

```yaml
app_settings:
  deployment_config:
    region: "us-west-2"           # Optional: Deployment region
    gpu: "T4"                     # Optional: GPU type or list of types
    concurrency_limit: 5          # Optional: Maximum concurrent requests
    container_idle_timeout: 180   # Optional: Timeout in seconds
```

### 3. Deploy

```bash
modal deploy app.py
```

## Inference Types

### Synchronous Inference

For immediate results:

```python
@app.function(**modal_config.get_handler_settings())
def predict(input_data: MyInput):
    return MyApp.sync_call(MyApp)(
        model_name="my_model",
        input_data=input_data
    )
```

### Asynchronous Inference

For long-running tasks:

```python
@app.function(**modal_config.get_handler_settings())
def predict_async(input_data: MyAsyncInput):
    return MyApp.async_call(MyApp)(
        model_name="my_model",
        input_data=input_data
    )
```

## Model Loading

Models are loaded automatically when the container starts:

```python
@modal.enter()
def load_artefacts(self):
    """Called when container starts"""
    settings = self.modal_utils.settings
    self._model_inference_kwargs = settings.model_settings.model_entries[self.model_name]
    self._model_inference_instance = self.base_inference_implementation(
        model_name=self.model_name,
        all_model_data_folder=settings.model_settings.local_model_repository_folder,
        common_settings=settings.model_settings.common,
        **self._model_inference_kwargs,
    )
```

## Error Handling

ModalKit provides built-in error handling:

1. **CUDA Errors**: Container is terminated and restarted
2. **Other Errors**: Gracefully handled with appropriate error responses

```python
try:
    result = model.predict(input_data)
except RuntimeError as e:
    if "CUDA error" in str(e):
        # Terminate container for CUDA errors
        modal.experimental.stop_fetching_inputs()
    raise HTTPException(status_code=500, detail=str(e))
except Exception as e:
    # Handle other errors gracefully
    raise HTTPException(status_code=500, detail=str(e))
```

## Resource Management

### Concurrency Settings

Control request handling:

```yaml
app_settings:
  deployment_config:
    concurrency_limit: 5  # Maximum concurrent requests
    allow_concurrent_inputs: 1  # Concurrent inputs per instance
    allow_concurrent_inputs_handler: 10  # Handler concurrency
  batch_config:
    max_batch_size: 1  # Maximum number of messages in an async inference batch
    wait_ms: 1000  # Maximum wait time in milliseconds to trigger a batch
```

## Best Practices

1. **Resource Configuration**
   - Set appropriate concurrency limits
   - Configure suitable timeouts
   - Use GPU resources efficiently

2. **Batch Configuration**
   - Set `max_batch_size` based on model capability and GPU memory constraints
   - Adjust `wait_ms` based on latency and throughput requirements

3. **Error Handling**
   - Implement proper error handling in your inference code
   - Use appropriate HTTP status codes
   - Log errors with context

4. **Model Loading**
   - Keep model loading code in `load_artefacts`
   - Handle initialization errors gracefully
   - Use appropriate caching strategies

5. **Testing**
   - Test locally before deployment
   - Verify resource configurations
   - Test both sync and async endpoints
