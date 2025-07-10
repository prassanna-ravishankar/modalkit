# Error Handling Examples

This guide demonstrates how to implement error handling in your ModalKit applications.

## Built-in Error Handling

ModalKit provides built-in error handling for common scenarios:

1. CUDA errors (GPU-related)
2. General inference errors
3. Input validation errors
4. Authentication errors

## Example Implementation

### 1. Basic Error Handling

```python
from modalkit.inference import InferencePipeline
from fastapi import HTTPException

class MyInference(InferencePipeline):
    def predict(self, input_list: list[BaseModel], preprocessed_data: dict) -> dict:
        try:
            # Your prediction logic
            result = self._model.predict(preprocessed_data["processed"])
            return {"prediction": result}
        except Exception as e:
            # Log the error
            self.logger.error(f"Prediction error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
```

### 2. GPU Error Handling

ModalKit automatically handles CUDA errors by terminating and restarting the container:

```python
from modal.experimental import stop_fetching_inputs

class MyInference(InferencePipeline):
    def predict(self, input_list: list[BaseModel], preprocessed_data: dict) -> dict:
        try:
            return self._model.predict(preprocessed_data["processed"])
        except RuntimeError as e:
            if "CUDA error" in str(e):
                self.logger.error("CUDA error detected, terminating container")
                stop_fetching_inputs()
            raise HTTPException(status_code=500, detail=str(e))
```

### 3. Async Error Handling

For asynchronous processing, use the `DelayedFailureOutputModel`:

```python
from modalkit.iomodel import DelayedFailureOutputModel
from modalkit.utils import send_response_queue

class MyAsyncInference(InferencePipeline):
    def process_async(self, wrapped_input_data: MyAsyncInput):
        try:
            result = self.run_inference(wrapped_input_data.message)
            send_response_queue(
                wrapped_input_data.success_queue,
                result.model_dump_json()
            )
        except Exception as e:
            error_response = DelayedFailureOutputModel(
                error=str(e),
                status="error",
                original_message=wrapped_input_data
            )
            send_response_queue(
                wrapped_input_data.failure_queue,
                error_response.model_dump_json()
            )
```

### 4. Input Validation

Use Pydantic models for input validation:

```python
from pydantic import BaseModel, Field, validator

class MyInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    temperature: float = Field(0.7, ge=0.0, le=1.0)

    @validator("text")
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty or whitespace")
        return v
```

### 5. Custom Error Types

Define custom error types for your application:

```python
class ModelNotReadyError(Exception):
    """Raised when model is not initialized"""
    pass

class InvalidInputError(Exception):
    """Raised for invalid input data"""
    pass

class MyInference(InferencePipeline):
    def predict(self, input_list: list[BaseModel], preprocessed_data: dict) -> dict:
        if not hasattr(self, "_model"):
            raise ModelNotReadyError("Model not initialized")

        if not self._validate_input(preprocessed_data):
            raise InvalidInputError("Invalid input format")

        return self._model.predict(preprocessed_data["data"])
```

## Error Response Format

### Synchronous Errors

```python
# HTTP 500 Response
{
    "detail": "Error message",
    "status_code": 500
}

# HTTP 400 Response (Validation Error)
{
    "detail": [
        {
            "loc": ["body", "text"],
            "msg": "Text cannot be empty",
            "type": "value_error"
        }
    ]
}
```

### Asynchronous Errors

```python
# Async Error Response
{
    "error": "Error message",
    "status": "error",
    "original_message": {
        "message": {...},
        "success_queue": "success-queue-url",
        "failure_queue": "failure-queue-url"
    }
}
```

## Best Practices

1. **Logging**
   ```python
   import logging
   logger = logging.getLogger(__name__)

   class MyInference(InferencePipeline):
       def predict(self, input_list: List[MyInput], preprocessed_data: dict) -> dict:
           try:
               result = self._model.predict(preprocessed_data["processed"])
               logger.info("Prediction successful", extra={
                   "batch_size": len(preprocessed_data["processed"]),
                   "model_name": self.model_name
               })
               return {"prediction": result}
           except Exception as e:
               logger.error("Prediction failed", exc_info=True, extra={
                   "model_name": self.model_name,
                   "error_type": type(e).__name__
               })
               raise
   ```

2. **Resource Cleanup**
   ```python
   class MyInference(InferencePipeline):
       def predict(self, input_list: List[MyInput], preprocessed_data: dict) -> dict:
           try:
               # Acquire resources
               self._acquire_resources()
               return self._model.predict(preprocessed_data["processed"])
           finally:
               # Always clean up
               self._cleanup_resources()
   ```

3. **Graceful Degradation**
   ```python
   class MyInference(InferencePipeline):
       def predict(self, input_list: List[MyInput], preprocessed_data: dict) -> dict:
           try:
               return self._primary_prediction(preprocessed_data["processed"])
           except ResourceError:
               # Fall back to simpler model
               return self._fallback_prediction(preprocessed_data["processed"])
   ```

4. **Error Recovery**
   ```python
   class MyInference(InferencePipeline):
       def predict(self, input_list: List[MyInput], preprocessed_data: dict) -> dict:
           for attempt in range(self.max_retries):
               try:
                   return self._model.predict(preprocessed_data["processed"])
               except TemporaryError as e:
                   if attempt == self.max_retries - 1:
                       raise
                   time.sleep(self.retry_delay)
   ```
