import asyncio
import time
from collections.abc import Awaitable, Callable

import modal
import modal.experimental
from fastapi import HTTPException
from loguru import logger
from pydantic import BaseModel

from modalkit.inference_pipeline import InferencePipeline
from modalkit.iomodel import (
    AsyncInputModel,
    AsyncOutputModel,
    DelayedFailureOutputModel,
    InferenceOutputModel,
    SyncInputModel,
)
from modalkit.modal_config import ModalConfig
from modalkit.settings import Settings
from modalkit.task_queue import QueueBackend, send_response_queue

settings = Settings()
modal_utils = ModalConfig(settings)


class ModalService:
    """
    Base class for Modal-based ML application deployment.

    This class provides the foundation for deploying ML models using Modal,
    handling model loading, inference, and API endpoint creation. It integrates
    with the InferencePipeline class to standardize model serving.

    The queue backend is fully optional and supports dependency injection for
    maximum flexibility:

    **Usage Examples:**

    1. **No Queues (Default):**
    ```python
    class MyService(ModalService):
        inference_implementation = MyInferencePipeline

    service = MyService()  # No queues, async requests just don't send responses
    ```

    2. **Configuration-Based Queues:**
    ```python
    # Uses modalkit.yaml settings for queue configuration
    service = MyService()  # Automatically uses configured backend (SQS, etc.)
    ```

    3. **Dependency Injection with TaskIQ:**
    ```python
    from taskiq_redis import AsyncRedisTaskiqBroker

    class TaskIQBackend:
        def __init__(self, broker_url="redis://localhost:6379"):
            self.broker = AsyncRedisTaskiqBroker(broker_url)

        async def send_message(self, queue_name: str, message: str) -> bool:
            @self.broker.task(task_name=f"process_{queue_name}")
            async def process_message(msg: str) -> None:
                # Your custom task processing logic
                logger.info(f"Processing: {msg}")

            await process_message.kiq(message)
            return True

    # Inject your TaskIQ backend
    taskiq_backend = TaskIQBackend("redis://localhost:6379")
    service = MyService(queue_backend=taskiq_backend)
    ```

    4. **Custom Queue Implementation:**
    ```python
    class MyCustomQueue:
        async def send_message(self, queue_name: str, message: str) -> bool:
            # Send to your custom queue system (RabbitMQ, Kafka, etc.)
            await my_queue_system.send(queue_name, message)
            return True

    service = MyService(queue_backend=MyCustomQueue())
    ```

    Attributes:
        model_name (str): Name of the model to be served
        inference_implementation (type[InferencePipeline]): Implementation class of the inference pipeline
        modal_utils (ModalConfig): Modal config object, containing the settings and config functions
        queue_backend (Optional[QueueBackend]): Optional queue backend for dependency injection
    """

    model_name: str
    inference_implementation: type[InferencePipeline]
    modal_utils: ModalConfig
    queue_backend: QueueBackend | None = None

    def __init__(self, queue_backend: QueueBackend | None = None):
        """
        Initialize ModalService with optional queue backend.

        Args:
            queue_backend: Optional queue backend for dependency injection.
                          If None, will use configuration-based approach or skip queues.
        """
        self.queue_backend = queue_backend

    @modal.enter()
    def load_artefacts(self) -> None:
        """
        Loads model artifacts and initializes the inference instance.

        This method is called when the Modal container starts up. It:
        1. Retrieves model-specific settings from configuration
        2. Initializes the inference implementation with the model settings
        3. Sets up the model for inference
        4. Initializes volume reloading if configured

        The method is decorated with @modal.enter() to ensure it runs during container startup.
        """
        settings = self.modal_utils.settings
        self._model_inference_kwargs = settings.model_settings.model_entries[self.model_name]

        self._model_inference_instance: InferencePipeline = self.inference_implementation(
            model_name=self.model_name,
            all_model_data_folder=str(settings.model_settings.local_model_repository_folder),
            common_settings=settings.model_settings.common,
            **self._model_inference_kwargs,
        )

        # Initialize volume reloading if configured
        self._last_reload_time = time.time()
        self._reload_interval = settings.app_settings.deployment_config.volume_reload_interval_seconds

    def _reload_volumes_if_needed(self) -> None:
        """
        Reloads all configured volumes if the time since last reload exceeds the configured interval.
        After reloading, calls the on_volume_reload hook on the inference instance.
        If the hook raises an error, it is logged but does not prevent request processing.
        """
        # If reload interval is None, volume reloading is disabled
        if self._reload_interval is None:
            return

        current_time = time.time()
        if current_time - self._last_reload_time >= self._reload_interval:
            logger.info(
                f"Time since last reload {current_time - self._last_reload_time}s exceeded interval {self._reload_interval}s, reloading volumes"
            )
            self.modal_utils.reload_volumes()
            self._last_reload_time = current_time

            # Call the on_volume_reload hook
            try:
                logger.info("Calling on_volume_reload hook")
                self._model_inference_instance.on_volume_reload()
            except Exception:
                logger.exception("Error in on_volume_reload hook, continuing with request processing")

    @modal.batched(
        max_batch_size=modal_utils.settings.app_settings.batch_config.max_batch_size,
        wait_ms=modal_utils.settings.app_settings.batch_config.wait_ms,
    )
    def process_request(self, input_list: list[SyncInputModel | AsyncInputModel]) -> list[InferenceOutputModel]:
        """
        Processes a batch of inference requests.

        Args:
            input_list (list[Union[SyncInputModel, AsyncInputModel]]): The list of input models containing either
                sync or async requests

        Returns:
            list[InferenceOutputModel]: The list of processed outputs conforming to the model's output schema
        """
        batch_size = len(input_list)
        logger.info(f"Received batch of {batch_size} input requests")

        try:
            # Reload volumes if needed before processing the request
            self._reload_volumes_if_needed()

            # Run Inference. Outputs are expected to be in the same order as the inputs
            messages = [input_data.message for input_data in input_list]
            raw_output_list = self._model_inference_instance.run_inference(messages)
            logger.info(
                f"Statuses of the {batch_size} processed requests: {[output.status for output in raw_output_list]}"
            )

            # For any requests that were async, return the response to the appropriate queue
            for message_idx, (input_data, raw_output_data) in enumerate(zip(input_list, raw_output_list, strict=False)):
                if isinstance(input_data, AsyncInputModel):
                    self.send_async_response(message_idx, raw_output_data, input_data)

        # Unhappy path: On internal error, return error outputs to the queues of all async messages
        # and kill the container if a CUDA error was encountered
        except Exception as e:
            if "CUDA error" in str(e):
                logger.error("Exiting container due to CUDA error. This is potentially due to a hardware issue")
                modal.experimental.stop_fetching_inputs()
            err_msg = f"Internal Server Error. Error log: {e}"
            logger.error(f"Error processing batch: {err_msg}")

            for message_idx, input_data in enumerate(input_list):
                if isinstance(input_data, AsyncInputModel):
                    error_response = DelayedFailureOutputModel(
                        status="error", error=err_msg, original_message=input_data
                    )
                    self.send_async_response(message_idx, error_response, input_data)
            raise HTTPException(status_code=500, detail=err_msg) from e
        else:
            return raw_output_list

    async def _send_to_queue(self, queue_name: str, message: str) -> bool:
        """
        Send message to queue using dependency injection or configuration-based approach.

        Args:
            queue_name: Name of the queue to send to
            message: JSON message to send

        Returns:
            bool: True if message was sent successfully, False otherwise
        """
        if self.queue_backend:
            # Use injected queue backend
            try:
                return await self.queue_backend.send_message(queue_name, message)
            except Exception as e:
                logger.error(f"Failed to send message using injected queue backend: {e}")
                return False
        else:
            # Fall back to configuration-based approach
            try:
                return send_response_queue(queue_name, message)
            except Exception as e:
                logger.error(f"Failed to send message using configuration-based queue: {e}")
                return False

    def send_async_response(
        self, message_idx: int, raw_output_data: InferenceOutputModel, input_data: AsyncInputModel
    ) -> None:
        """
        Sends the inference result to the success or failure queues depending on the message status.
        Queue functionality is optional - only attempts to send if queue names are provided.

        Args:
            message_idx: Index of the message in the batch (for logging)
            raw_output_data (InferenceOutputModel): The processed output result
            input_data (AsyncInputModel): Object containing the async input data
        """
        # Only append metadata for regular inference outputs, not DelayedFailureOutputModel
        # DelayedFailureOutputModel already contains the original message with its metadata
        if not isinstance(raw_output_data, DelayedFailureOutputModel):
            # InferenceOutputModel allows extra fields - use setattr to avoid type checker issues
            raw_output_data.meta = input_data.meta

        if raw_output_data.status == "success":
            success_queue = input_data.success_queue
            if success_queue:  # Only send if queue name is provided
                # Use asyncio.create_task to avoid blocking the batch processing
                import asyncio

                try:
                    loop = asyncio.get_running_loop()
                    task = loop.create_task(
                        self._send_to_queue(success_queue, raw_output_data.model_dump_json(exclude={"error"}))
                    )
                    # Don't await - fire and forget to avoid blocking batch processing
                    task.add_done_callback(lambda t: self._log_queue_result(t, success_queue, "success"))
                except RuntimeError:
                    # No running loop, use synchronous fallback
                    success = asyncio.run(
                        self._send_to_queue(success_queue, raw_output_data.model_dump_json(exclude={"error"}))
                    )
                    if not success:
                        logger.warning(f"Failed to send success response to queue: {success_queue}")
            else:
                logger.debug("No success queue specified, skipping queue response")
        else:
            failure_queue = input_data.failure_queue
            if failure_queue:  # Only send if queue name is provided
                # Use asyncio.create_task to avoid blocking the batch processing
                import asyncio

                try:
                    loop = asyncio.get_running_loop()
                    task = loop.create_task(self._send_to_queue(failure_queue, raw_output_data.model_dump_json()))
                    # Don't await - fire and forget to avoid blocking batch processing
                    task.add_done_callback(lambda t: self._log_queue_result(t, failure_queue, "failure"))
                except RuntimeError:
                    # No running loop, use synchronous fallback
                    success = asyncio.run(self._send_to_queue(failure_queue, raw_output_data.model_dump_json()))
                    if not success:
                        logger.warning(f"Failed to send failure response to queue: {failure_queue}")
            else:
                logger.debug("No failure queue specified, skipping queue response")

    def _log_queue_result(self, task: "asyncio.Task", queue_name: str, response_type: str) -> None:
        """Log the result of a queue sending task."""
        try:
            success = task.result()
            if not success:
                logger.warning(f"Failed to send {response_type} response to queue: {queue_name}")
        except Exception as e:
            logger.error(f"Error sending {response_type} response to queue {queue_name}: {e}")

    @staticmethod
    def async_call(cls: type["ModalService"]) -> Callable[[str, BaseModel], Awaitable[AsyncOutputModel]]:
        """
        Creates an asynchronous callable function for processing and returning inference results via queues.

        This method generates a function that spawns an asynchronous task for the `process_request` method.
        It allows triggering an async inference job while returning a job ID for tracking purposes.

        Args:
            cls (type[ModalService]): The class reference for creating an instance of `ModalService`.

        Returns:
            Callable: A function that, when called, spawns an asynchronous task and returns an AsyncOutputModel with job ID.

        Example:
            >>> async_fn = ModalService.async_call(MyApp)
            >>> result = async_fn(model_name="example_model", input_data)
            >>> print(result)
            AsyncOutputModel(job_id="some_job_id")
        """

        async def fn(model_name: str, input_data: BaseModel) -> AsyncOutputModel:
            # input_data should be an AsyncInputModel based on FastAPI usage
            if isinstance(input_data, AsyncInputModel):
                input_list = [input_data]
            else:
                # Fallback for other BaseModel types - shouldn't happen in normal usage
                input_list = [AsyncInputModel(message=input_data)]

            # Create a mock instance for spawning - this is a limitation of the current design
            # In practice, this would need to be refactored to work with Modal's class instantiation
            service_instance = cls()
            service_instance.model_name = model_name
            call = await service_instance.process_request.spawn.aio(input_list)
            return AsyncOutputModel(job_id=call.object_id)

        return fn

    @staticmethod
    def sync_call(cls: type["ModalService"]) -> Callable[[str, BaseModel], Awaitable[BaseModel]]:
        """
        Creates a synchronous callable function for processing inference requests.
        Each request is processed individually to maintain immediate response times.
        For batch processing, use async endpoints.

        This method generates a function that triggers the `process` method of the `ModalService` class.
        It allows synchronous inference processing with input data passed to the model.

        Args:
            cls (type[ModalService]): The class reference for creating an instance of `ModalService`.

        Returns:
            Callable: A function that, when called, executes a synchronous inference call and returns the result.

        Example:
            >>> sync_fn = ModalService.sync_call(MyApp)
            >>> result = sync_fn(model_name="example_model", input_data)
            >>> print(result)
            InferenceOutputModel(status="success", ...)
        """

        async def fn(model_name: str, input_data: BaseModel) -> BaseModel:
            # input_data should be a SyncInputModel based on FastAPI usage
            if isinstance(input_data, SyncInputModel):
                input_list = [input_data]
            else:
                # Fallback for other BaseModel types
                input_list = [SyncInputModel(message=input_data)]

            # Create a mock instance for remote calls - this is a limitation of the current design
            service_instance = cls()
            service_instance.model_name = model_name
            results = await service_instance.process_request.remote.aio(input_list)
            # Return the first (and only) result for sync calls
            return results[0]

        return fn


def create_web_endpoints(
    app_cls: ModalService, input_model: type[BaseModel], output_model: type[BaseModel]
) -> "FastAPI":  # type: ignore # noqa: F821
    """
    Creates and configures a FastAPI web application for the given modal app class.

    This function sets up web endpoints with input/output models for request/response
    validation and authentication middleware.

    Args:
        app_cls (ModalService): The class representing the modal application, which provides
            utilities, settings
        input_model (BaseModel): A Pydantic model that defines the expected structure of
            input data for the web endpoints.
        output_model (BaseModel): A Pydantic model that defines the structure of the response
            data returned by the web endpoints.

    Returns:
        FastAPI: A configured FastAPI application instance with endpoints and middleware.

    Note:
        - Authentication is enforced using Modal proxy auth only.
        - Both synchronous and asynchronous endpoints are added to the web application.
    """
    from modalkit.fast_api import create_app

    app = create_app(
        input_model=input_model,
        output_model=output_model,
        dependencies=[],
        router_dependency=None,
        sync_fn=ModalService.sync_call(type(app_cls)),
        async_fn=ModalService.async_call(type(app_cls)),
    )
    return app
