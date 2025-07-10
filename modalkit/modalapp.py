import time
from typing import Callable, Union

import modal
import modal.experimental
from fastapi import HTTPException
from pydantic import BaseModel

from modalkit.inference import InferencePipeline
from modalkit.iomodel import AsyncInputModel, DelayedFailureOutputModel, InferenceOutputModel, SyncInputModel
from modalkit.logger import LOGGER
from modalkit.modalutils import ModalConfig
from modalkit.settings import Settings
from modalkit.utils import send_response_queue

settings = Settings()
modal_utils = ModalConfig(settings)


class ModalService:
    """
    Base class for Modal-based ML application deployment.

    This class provides the foundation for deploying ML models using Modal,
    handling model loading, inference, and API endpoint creation. It integrates
    with the InferencePipeline class to standardize model serving.

    Attributes:
        model_name (str): Name of the model to be served
        inference_implementation (InferencePipeline): Implementation of the inference pipeline
        modal_utils (ModalConfig): Modal config object, containing the settings and config functions
    """

    model_name: str
    inference_implementation: InferencePipeline
    modal_utils: ModalConfig

    @modal.enter()
    def load_artefacts(self):
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
            all_model_data_folder=settings.model_settings.local_model_repository_folder,
            common_settings=settings.model_settings.common,
            **self._model_inference_kwargs,
        )

        # Initialize volume reloading if configured
        self._last_reload_time = time.time()
        self._reload_interval = settings.app_settings.deployment_config.volume_reload_interval_seconds

    def _reload_volumes_if_needed(self):
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
            LOGGER.info(
                f"Time since last reload {current_time - self._last_reload_time}s exceeded interval {self._reload_interval}s, reloading volumes"
            )
            self.modal_utils.reload_volumes()
            self._last_reload_time = current_time

            # Call the on_volume_reload hook
            try:
                LOGGER.info("Calling on_volume_reload hook")
                self._model_inference_instance.on_volume_reload()
            except Exception:
                LOGGER.exception("Error in on_volume_reload hook, continuing with request processing")

    @modal.batched(**modal_utils.get_batched_method_settings())
    def process_request(self, input_list: list[Union[SyncInputModel, AsyncInputModel]]) -> list[InferenceOutputModel]:
        """
        Processes a batch of inference requests.

        Args:
            input_list (list[Union[SyncInputModel, AsyncInputModel]]): The list of input models containing either
                sync or async requests

        Returns:
            list[InferenceOutputModel]: The list of processed outputs conforming to the model's output schema
        """
        batch_size = len(input_list)
        LOGGER.info(f"Received batch of {batch_size} input requests")

        try:
            # Reload volumes if needed before processing the request
            self._reload_volumes_if_needed()

            # Run Inference. Outputs are expected to be in the same order as the inputs
            messages = [input_data.message for input_data in input_list]
            raw_output_list = self._model_inference_instance.run_inference(messages)
            LOGGER.info(
                f"Statuses of the {batch_size} processed requests: {[output.status for output in raw_output_list]}"
            )

            # For any requests that were async, return the response to the appropriate queue
            for message_idx, (input_data, raw_output_data) in enumerate(zip(input_list, raw_output_list)):
                if isinstance(input_data, AsyncInputModel):
                    self.send_async_response(message_idx, raw_output_data, input_data)

        # Unhappy path: On internal error, return error outputs to the queues of all async messages
        # and kill the container if a CUDA error was encountered
        except Exception as e:
            if "CUDA error" in str(e):
                LOGGER.error("Exiting container due to CUDA error. This is potentially due to a hardware issue")
                modal.experimental.stop_fetching_inputs()
            err_msg = f"Internal Server Error. Error log: {e}"
            LOGGER.error(f"Error processing batch: {err_msg}")

            for message_idx, input_data in enumerate(input_list):
                if isinstance(input_data, AsyncInputModel):
                    error_response = DelayedFailureOutputModel(
                        status="error", error=err_msg, original_message=input_data
                    )
                    self.send_async_response(message_idx, error_response, input_data)
            raise HTTPException(status_code=500, detail=err_msg) from e
        else:
            return raw_output_list

    def send_async_response(self, message_idx: int, raw_output_data: InferenceOutputModel, input_data: AsyncInputModel):
        """
        Sends the inference result to the success or failure queues depending on the message status.
        Queue functionality is optional - only attempts to send if queue names are provided.

        Args:
            raw_output_data (InferenceOutputModel): The processed output result
            input_data (AsyncInputModel): Object containing the async input data
        """
        # Only append metadata for regular inference outputs, not DelayedFailureOutputModel
        # DelayedFailureOutputModel already contains the original message with its metadata
        if not isinstance(raw_output_data, DelayedFailureOutputModel):
            raw_output_data.meta = input_data.meta

        if raw_output_data.status == "success":
            success_queue = input_data.success_queue
            if success_queue:  # Only send if queue name is provided
                success = send_response_queue(success_queue, raw_output_data.model_dump_json(exclude=["error"]))
                if not success:
                    LOGGER.warning(f"Failed to send success response to queue: {success_queue}")
            else:
                LOGGER.debug("No success queue specified, skipping queue response")
        else:
            failure_queue = input_data.failure_queue
            if failure_queue:  # Only send if queue name is provided
                success = send_response_queue(failure_queue, raw_output_data.model_dump_json())
                if not success:
                    LOGGER.warning(f"Failed to send failure response to queue: {failure_queue}")
            else:
                LOGGER.debug("No failure queue specified, skipping queue response")

    @staticmethod
    def async_call(cls: type["ModalService"]) -> Callable[..., dict]:
        """
        Creates an asynchronous callable function for processing and returning inference results via queues.

        This method generates a function that spawns an asynchronous task for the `process_request` method.
        It allows triggering an async inference job while returning a job ID for tracking purposes.

        Args:
            cls (type[ModalService]): The class reference for creating an instance of `ModalService`.

        Returns:
            Callable[..., dict]: A function that, when called, spawns an asynchronous task and returns a dictionary containing the job ID.

        Example:
            >>> async_fn = ModalService.async_call(MyApp)
            >>> result = async_fn(model_name="example_model", input_data, trace_carrier)
            >>> print(result)
            {"job_id": "some_job_id"}
        """

        async def fn(model_name, *args, **kwargs):
            call = await cls(model_name=model_name).process_request.spawn.aio(*args, **kwargs)
            return {"job_id": call.object_id}

        return fn

    @staticmethod
    def sync_call(cls: type["ModalService"]) -> Callable[..., dict]:
        """
        Creates a synchronous callable function for processing inference requests.
        Each request is processed individually to maintain immediate response times.
        For batch processing, use async endpoints.

        This method generates a function that triggers the `process` method of the `ModalService` class.
        It allows synchronous inference processing with input data passed to the model.

        Args:
            cls (type[ModalService]): The class reference for creating an instance of `ModalService`.

        Returns:
            Callable[..., dict]: A function that, when called, executes a synchronous inference call and returns the result.

        Example:
            >>> sync_fn = ModalService.sync_call(MyApp)
            >>> result = sync_fn(model_name="example_model", input_data, trace_carrier)
            >>> print(result)
            {"inference_result": "output_data"}
        """

        async def fn(model_name, *args, **kwargs):
            return await cls(model_name=model_name).process_request.remote.aio(*args, **kwargs)

        return fn


def create_web_endpoints(app_cls: ModalService, input_model: BaseModel, output_model: BaseModel):
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
        - Authentication is enforced using either Modal proxy auth (when secure=True) or
          the `CustomAPIKey` middleware (when secure=False).
        - Both synchronous and asynchronous endpoints are added to the web application.
    """
    from modalkit.fast_api import create_app

    modal_utils = app_cls.modal_utils

    # Use Modal proxy auth when secure=True, otherwise use API key auth
    auth_middleware = None if modal_utils.app_settings.deployment_config.secure else modal_utils.CustomAPIKey

    app = create_app(
        input_model=input_model,
        output_model=output_model,
        dependencies=[],
        auth_middleware=auth_middleware,
        sync_fn=ModalService.sync_call(app_cls),
        async_fn=ModalService.async_call(app_cls),
    )
    return app
