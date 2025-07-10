# Async Processing

ModalKit supports asynchronous processing for long-running inference tasks. This guide demonstrates how to implement async processing in your application.

## Overview

Async processing in ModalKit involves:

1. Using `AsyncInputModel` and `AsyncOutputModel`
2. Implementing async versions of inference methods
3. Handling job status and results

## Implementation

### 1. Define Async Models

```python
from modalkit.iomodel import AsyncInputModel, AsyncOutputModel, InferenceOutputModel
from pydantic import BaseModel

class MyInput(BaseModel):
    text: str

class MyOutput(InferenceOutputModel):
    result: str

class MyAsyncInput(AsyncInputModel):
    data: MyInput

class MyAsyncOutput(AsyncOutputModel):
    result: MyOutput
```

### 2. Implement Async Inference

```python
from modalkit.inference import InferencePipeline
from typing import List
import asyncio

class MyAsyncInference(InferencePipeline):
    async def preprocess(self, input_list: List[MyAsyncInput]) -> dict:
        # Async preprocessing logic
        processed_data = [input_data.data.text for input_data in input_list]
        return {"processed": processed_data}

    async def predict(self, input_list: List[MyAsyncInput], preprocessed_data: dict) -> dict:
        # Simulate long-running task
        await asyncio.sleep(5)
        raw_output_data = [text.upper() for text in preprocessed_data["processed"]]
        return {"prediction": raw_output_data}

    async def postprocess(self, input_list: List[MyAsyncInput], raw_output: dict) -> List[MyAsyncOutput]:
        return [
            MyAsyncOutput(
                result=MyOutput(result=pred),
                status="completed"
            )
            for pred in raw_output["prediction"]
        ]
```

### 3. Configure Modal Application

```python
import modal
from modalkit.modalapp import ModalService, create_web_endpoints
from modalkit.modalutils import ModalConfig
from modalkit.settings import Settings

Settings.model_config["yaml_file"] = "modalkit.yaml"
modal_utils = ModalConfig()

app = modal.App(name=modal_utils.app_name)

@app.cls(**modal_utils.get_app_cls_settings())
class MyAsyncApp(ModalService):
    inference_implementation = MyAsyncInference
    model_name: str = modal.parameter(default="my_model")
    modal_utils: ModalConfig = modal_utils

@app.function(**modal_utils.get_handler_settings())
@modal.asgi_app()
def web_endpoints():
    return create_web_endpoints(
        app_cls=MyAsyncApp,
        input_model=MyAsyncInput,
        output_model=MyAsyncOutput
    )
```

## Usage

### 1. Submit Async Job

```python
import requests

# Submit job
response = requests.post(
    "http://localhost:8000/async/predict",
    json={"data": {"text": "hello world"}},
    headers={"x-api-key": "your-api-key"}
)

job_id = response.json()["job_id"]
```

### 2. Check Job Status

```python
# Check status
status_response = requests.get(
    f"http://localhost:8000/async/status/{job_id}",
    headers={"x-api-key": "your-api-key"}
)

print(status_response.json())
# Output: {"status": "processing"}
```

### 3. Get Results

```python
# Get results when complete
result_response = requests.get(
    f"http://localhost:8000/async/result/{job_id}",
    headers={"x-api-key": "your-api-key"}
)

print(result_response.json())
# Output: {"result": {"result": "HELLO WORLD"}, "status": "completed"}
```

## Error Handling

Async jobs can have the following statuses:

- `pending`: Job is queued
- `processing`: Job is being processed
- `completed`: Job completed successfully
- `failed`: Job failed with an error

Handle errors by checking the job status:

```python
if status_response.json()["status"] == "failed":
    error_details = status_response.json()["error"]
    print(f"Job failed: {error_details}")
```

## Configuration

Configure async job settings in `modalkit.yaml`:

```yaml
app_settings:
  deployment_config:
    async_job_timeout: 3600  # Maximum time (seconds) for async job
    async_result_ttl: 86400  # Time to keep results (seconds)
```

## Best Practices

1. **Job Timeouts**: Set appropriate timeouts based on your task
2. **Error Handling**: Always implement proper error handling
3. **Resource Management**: Clean up completed job results
4. **Status Updates**: Provide meaningful status updates
5. **Idempotency**: Ensure job submissions are idempotent
