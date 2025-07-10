from typing import Generic, Optional, TypeVar

from pydantic import BaseModel, ConfigDict

T = TypeVar("T", bound=BaseModel)


class AsyncOutputModel(BaseModel):
    """
    Model for asynchronous operation outputs.

    Attributes:
        job_id (str): Unique identifier for tracking the asynchronous job
    """

    job_id: str


class AsyncInputModel(BaseModel, Generic[T]):
    """
    Model for asynchronous operation inputs, wrapping the message with queue information.

    Attributes:
        message (T): The actual input data, must be a Pydantic model
        success_queue (str): SQS queue name for successful results
        failure_queue (str): SQS queue name for error messages
        meta (dict): Additional metadata to be passed through the processing pipeline

    Notes:
        The model_config ensures no extra fields are allowed in the input
    """

    model_config = ConfigDict(extra="forbid")
    message: T
    success_queue: str = ""
    failure_queue: str = ""
    meta: dict = {}


class SyncInputModel(BaseModel, Generic[T]):
    """
    Model for synchronous operation inputs.

    Attributes:
        message (T): The actual input data, must be a Pydantic model

    Notes:
        The model_config ensures no extra fields are allowed in the input
    """

    model_config = ConfigDict(extra="forbid")
    message: T


# TracedInputModel removed - no longer needed without OpenTelemetry


class InferenceOutputModel(BaseModel):
    """
    Generic model for responses returned by the inference code.

    Attributes:
        status (str): Status of the response
        error (Optional[str]): Description of the error that occurred

    Notes:
        The model_config allows extra fields to be added to the output
        such as metadata and the original message
    """

    model_config = ConfigDict(extra="allow")
    status: str
    error: Optional[str] = None


class DelayedFailureOutputModel(InferenceOutputModel):
    """
    Model for failure responses in async operations.

    Attributes:
        original_message (AsyncInputModel): The original input message that caused the failure
    """

    original_message: AsyncInputModel
