"""
Tests for ModalService behavior.

This module tests the Modal service deployment and orchestration functionality, focusing on:
- Service lifecycle management (initialization, loading, processing)
- Request processing and batch handling behavior
- Error handling and resilience patterns
- Volume reload and hook integration
- Asynchronous and synchronous processing modes
- Queue integration and response handling
"""

import time
from unittest.mock import MagicMock, Mock, patch

import pytest
from fastapi import HTTPException
from pydantic import BaseModel

from modalkit.inference import InferencePipeline
from modalkit.iomodel import (
    AsyncInputModel,
    InferenceOutputModel,
    SyncInputModel,
)
from modalkit.modalapp import ModalService, create_web_endpoints
from modalkit.modalutils import ModalConfig
from modalkit.settings import (
    AppSettings,
    AuthConfig,
    BatchConfig,
    BuildConfig,
    DeploymentConfig,
    ModelSettings,
    Settings,
)


class TestRequest(BaseModel):
    """Test input model for service testing"""

    text: str
    priority: str = "normal"


class TestResponse(InferenceOutputModel):
    """Test output model for service testing"""

    processed_text: str
    processing_time: float


class MockInferencePipeline(InferencePipeline):
    """Mock inference pipeline for testing service behavior"""

    def __init__(self, model_name: str, all_model_data_folder: str, common_settings: dict, **kwargs):
        super().__init__(model_name, all_model_data_folder, common_settings)
        self.processing_delay = kwargs.get("processing_delay", 0.0)
        self.should_fail = kwargs.get("should_fail", False)
        self.failure_message = kwargs.get("failure_message", "Processing failed")
        self.reload_hook_called = False

    def preprocess(self, input_list: list[BaseModel]) -> dict:
        if self.should_fail and "preprocess" in self.failure_message:
            raise ValueError(self.failure_message)

        return {"batch_size": len(input_list), "texts": [item.text for item in input_list]}

    def predict(self, input_list: list[BaseModel], preprocessed_data: dict) -> dict:
        if self.should_fail and (
            "predict" in self.failure_message
            or "CUDA" in self.failure_message
            or "Processing failed" in self.failure_message
        ):
            raise RuntimeError(self.failure_message)

        # Simulate processing time
        if self.processing_delay > 0:
            time.sleep(self.processing_delay)

        predictions = []
        for text in preprocessed_data["texts"]:
            predictions.append({"processed_text": f"Processed: {text}", "processing_time": self.processing_delay})

        return {"predictions": predictions}

    def postprocess(self, input_list: list[BaseModel], raw_output: dict) -> list[InferenceOutputModel]:
        if self.should_fail and "postprocess" in self.failure_message:
            raise ValueError(self.failure_message)

        results = []
        for prediction in raw_output["predictions"]:
            result = TestResponse(
                status="success",
                processed_text=prediction["processed_text"],
                processing_time=prediction["processing_time"],
            )
            results.append(result)

        return results

    def on_volume_reload(self):
        """Hook called after volume reload"""
        self.reload_hook_called = True


class TestModalService(ModalService):
    """Test implementation of ModalService"""

    def __init__(self, model_name: str, tmp_path=None, **inference_kwargs):
        self.model_name = model_name
        self.inference_implementation = MockInferencePipeline
        self.inference_kwargs = inference_kwargs

        # Create test configuration
        app_settings = AppSettings(
            app_prefix="test-service",
            build_config=BuildConfig(image="test", tag="latest"),
            auth_config=AuthConfig(api_key="test-key", auth_header="X-API-Key"),
            deployment_config=DeploymentConfig(volumes={}, volume_reload_interval_seconds=None, secure=False),
            batch_config=BatchConfig(),
        )

        model_folder = str(tmp_path / "models") if tmp_path else "./models"
        model_settings = ModelSettings(
            model_entries={model_name: inference_kwargs},
            local_model_repository_folder=model_folder,
            common={"timeout": 30},
        )

        settings = Settings(app_settings=app_settings, model_settings=model_settings)
        self.modal_utils = ModalConfig(settings)


class TestModalServiceLifecycle:
    """Test suite for ModalService lifecycle management"""

    def test_service_initializes_correctly(self, tmp_path):
        """ModalService should initialize with correct configuration"""
        service = TestModalService("sentiment-model", tmp_path)

        assert service.model_name == "sentiment-model"
        assert service.inference_implementation == MockInferencePipeline
        assert service.modal_utils is not None

    def test_service_loads_artifacts_on_startup(self, tmp_path):
        """ModalService should load model artifacts during startup"""
        service = TestModalService("text-classifier", tmp_path, processing_delay=0.1)

        # Simulate Modal's @modal.enter() behavior
        service.load_artefacts()

        # Verify inference instance was created
        assert hasattr(service, "_model_inference_instance")
        assert isinstance(service._model_inference_instance, MockInferencePipeline)
        assert service._model_inference_instance.model_name == "text-classifier"

    def test_service_configures_volume_reloading(self, tmp_path):
        """ModalService should configure volume reloading based on settings"""
        service = TestModalService("model-with-volumes", tmp_path)
        service.modal_utils.settings.app_settings.deployment_config.volume_reload_interval_seconds = 300

        service.load_artefacts()

        # Verify reload configuration
        assert hasattr(service, "_reload_interval")
        assert service._reload_interval == 300
        assert hasattr(service, "_last_reload_time")


class TestModalServiceRequestProcessing:
    """Test suite for request processing behavior"""

    @pytest.fixture
    def loaded_service(self, tmp_path):
        """Provides a service with loaded artifacts"""
        service = TestModalService("test-model", tmp_path)
        service.load_artefacts()
        return service

    def test_service_processes_single_sync_request(self, loaded_service):
        """ModalService should process single synchronous requests correctly"""
        sync_input = SyncInputModel(message=TestRequest(text="Hello world"))

        results = loaded_service.process_request([sync_input])

        assert len(results) == 1
        assert isinstance(results[0], TestResponse)
        assert results[0].status == "success"
        assert "Processed: Hello world" in results[0].processed_text

    def test_service_processes_batch_requests(self, loaded_service):
        """ModalService should handle batch processing efficiently"""
        batch_inputs = [SyncInputModel(message=TestRequest(text=f"Request {i}")) for i in range(5)]

        results = loaded_service.process_request(batch_inputs)

        assert len(results) == 5
        for i, result in enumerate(results):
            assert result.status == "success"
            assert f"Processed: Request {i}" in result.processed_text

    def test_service_handles_empty_batch(self, loaded_service):
        """ModalService should handle empty input batches gracefully"""
        results = loaded_service.process_request([])

        assert results == []

    def test_service_preserves_request_order(self, loaded_service):
        """ModalService should preserve the order of requests in batch processing"""
        texts = ["First", "Second", "Third", "Fourth"]
        batch_inputs = [SyncInputModel(message=TestRequest(text=text)) for text in texts]

        results = loaded_service.process_request(batch_inputs)

        for _i, (text, result) in enumerate(zip(texts, results)):
            assert f"Processed: {text}" in result.processed_text

    def test_service_processes_async_requests_with_queues(self, loaded_service):
        """ModalService should handle async requests and send responses to queues"""
        async_input = AsyncInputModel(
            message=TestRequest(text="Async request"),
            success_queue="success-topic",
            failure_queue="failure-topic",
            meta={"request_id": "test-123"},
        )

        with patch("modalkit.modalapp.send_response_queue", return_value=True) as mock_send:
            results = loaded_service.process_request([async_input])

            assert len(results) == 1
            assert results[0].status == "success"
            assert results[0].meta == {"request_id": "test-123"}

            # Verify queue response was sent
            mock_send.assert_called_once()
            call_args = mock_send.call_args
            assert call_args[0][0] == "success-topic"  # queue name
            # Response should be JSON-serialized output excluding error field
            response_data = call_args[0][1]
            assert "success" in response_data
            assert "error" not in response_data

    def test_service_handles_mixed_sync_async_batch(self, loaded_service):
        """ModalService should handle mixed batches of sync and async requests"""
        mixed_inputs = [
            SyncInputModel(message=TestRequest(text="Sync request")),
            AsyncInputModel(
                message=TestRequest(text="Async request"),
                success_queue="success-queue",
                failure_queue="failure-queue",
                meta={},
            ),
        ]

        with patch("modalkit.modalapp.send_response_queue", return_value=True):
            results = loaded_service.process_request(mixed_inputs)

            assert len(results) == 2
            assert all(result.status == "success" for result in results)


class TestModalServiceErrorHandling:
    """Test suite for error handling and resilience"""

    def test_service_handles_preprocessing_errors(self, tmp_path):
        """ModalService should handle and propagate preprocessing errors"""
        service = TestModalService("failing-model", tmp_path, should_fail=True, failure_message="preprocess failed")
        service.load_artefacts()

        sync_input = SyncInputModel(message=TestRequest(text="Test"))

        with pytest.raises(HTTPException) as exc_info:
            service.process_request([sync_input])

        assert exc_info.value.status_code == 500
        assert "preprocess failed" in str(exc_info.value.detail)

    def test_service_handles_prediction_errors(self, tmp_path):
        """ModalService should handle and propagate prediction errors"""
        service = TestModalService("failing-model", tmp_path, should_fail=True, failure_message="predict failed")
        service.load_artefacts()

        sync_input = SyncInputModel(message=TestRequest(text="Test"))

        with pytest.raises(HTTPException) as exc_info:
            service.process_request([sync_input])

        assert exc_info.value.status_code == 500
        assert "predict failed" in str(exc_info.value.detail)

    def test_service_handles_cuda_errors_specially(self, tmp_path):
        """ModalService should handle CUDA errors by stopping input fetching"""
        service = TestModalService(
            "failing-model", tmp_path, should_fail=True, failure_message="CUDA error: out of memory"
        )
        service.load_artefacts()

        sync_input = SyncInputModel(message=TestRequest(text="Test"))

        with patch("modal.experimental.stop_fetching_inputs") as mock_stop:
            with pytest.raises(HTTPException):
                service.process_request([sync_input])

            # Verify stop_fetching_inputs was called for CUDA errors
            mock_stop.assert_called_once()

    def test_service_sends_errors_to_async_failure_queues(self, tmp_path):
        """ModalService should send error responses to failure queues for async requests"""
        service = TestModalService("failing-model", tmp_path, should_fail=True, failure_message="Processing failed")
        service.load_artefacts()

        async_inputs = [
            AsyncInputModel(
                message=TestRequest(text=f"Request {i}"),
                success_queue="success-queue",
                failure_queue="failure-queue",
                meta={"id": i},
            )
            for i in range(3)
        ]

        with patch("modalkit.modalapp.send_response_queue", return_value=True) as mock_send:
            with pytest.raises(HTTPException):
                service.process_request(async_inputs)

            # Verify failure responses were sent for all async requests
            assert mock_send.call_count == 3
            for call in mock_send.call_args_list:
                assert call[0][0] == "failure-queue"  # failure queue
                response_data = call[0][1]
                assert "error" in response_data
                assert "Processing failed" in response_data

    def test_service_handles_queue_send_failures_gracefully(self, tmp_path):
        """ModalService should log but not fail when queue sending fails"""
        service = TestModalService("test-model", tmp_path)
        service.load_artefacts()

        async_input = AsyncInputModel(
            message=TestRequest(text="Test"), success_queue="unreachable-queue", failure_queue="failure-queue", meta={}
        )

        # Mock queue sending to fail
        with patch("modalkit.modalapp.send_response_queue", return_value=False):
            # Should not raise exception even if queue sending fails
            results = service.process_request([async_input])
            assert len(results) == 1
            assert results[0].status == "success"


class TestModalServiceVolumeReloading:
    """Test suite for volume reloading behavior"""

    def test_service_skips_reload_when_disabled(self, tmp_path):
        """ModalService should skip volume reloading when interval is None"""
        service = TestModalService("test-model", tmp_path)
        service.modal_utils.settings.app_settings.deployment_config.volume_reload_interval_seconds = None
        service.load_artefacts()

        with patch.object(service.modal_utils, "reload_volumes") as mock_reload:
            sync_input = SyncInputModel(message=TestRequest(text="Test"))
            service.process_request([sync_input])

            # Should not attempt to reload volumes
            mock_reload.assert_not_called()

    def test_service_reloads_volumes_based_on_interval(self, tmp_path):
        """ModalService should reload volumes when interval threshold is exceeded"""
        service = TestModalService("test-model", tmp_path)
        service.modal_utils.settings.app_settings.deployment_config.volume_reload_interval_seconds = 60
        service.load_artefacts()

        # Set last reload time to simulate time passage
        service._last_reload_time = time.time() - 70  # 70 seconds ago

        with patch.object(service.modal_utils, "reload_volumes") as mock_reload:
            sync_input = SyncInputModel(message=TestRequest(text="Test"))
            service.process_request([sync_input])

            # Should reload volumes since 70s > 60s interval
            mock_reload.assert_called_once()

    def test_service_calls_volume_reload_hook(self, tmp_path):
        """ModalService should call inference pipeline volume reload hook"""
        service = TestModalService("test-model", tmp_path)
        service.modal_utils.settings.app_settings.deployment_config.volume_reload_interval_seconds = 60
        service.load_artefacts()

        # Set last reload time to trigger reload
        service._last_reload_time = time.time() - 70

        with patch.object(service.modal_utils, "reload_volumes"):
            sync_input = SyncInputModel(message=TestRequest(text="Test"))
            service.process_request([sync_input])

            # Verify hook was called
            assert service._model_inference_instance.reload_hook_called

    def test_service_handles_volume_reload_hook_errors(self, tmp_path):
        """ModalService should handle errors in volume reload hooks gracefully"""
        service = TestModalService("test-model", tmp_path)
        service.modal_utils.settings.app_settings.deployment_config.volume_reload_interval_seconds = 60
        service.load_artefacts()

        # Make the hook raise an error
        def failing_hook():
            raise RuntimeError("Hook failed")

        service._model_inference_instance.on_volume_reload = failing_hook

        # Set last reload time to trigger reload
        service._last_reload_time = time.time() - 70

        with patch.object(service.modal_utils, "reload_volumes"):
            sync_input = SyncInputModel(message=TestRequest(text="Test"))

            # Should not raise exception even if hook fails
            results = service.process_request([sync_input])
            assert len(results) == 1
            assert results[0].status == "success"


class TestModalServiceStaticMethods:
    """Test suite for static factory methods"""

    def test_async_call_creates_proper_function(self):
        """async_call should create a function that spawns async tasks"""

        class MockAsyncService(ModalService):
            def __init__(self, model_name):
                self.model_name = model_name
                self.process_request = Mock()
                self.process_request.spawn = Mock()
                self.process_request.spawn.aio = Mock()

        async_fn = ModalService.async_call(MockAsyncService)

        # Function should be callable
        assert callable(async_fn)

    def test_sync_call_creates_proper_function(self):
        """sync_call should create a function that makes synchronous calls"""

        class MockSyncService(ModalService):
            def __init__(self, model_name):
                self.model_name = model_name
                self.process_request = Mock()
                self.process_request.remote = Mock()
                self.process_request.remote.aio = Mock()

        sync_fn = ModalService.sync_call(MockSyncService)

        # Function should be callable
        assert callable(sync_fn)


class TestWebEndpointCreation:
    """Test suite for web endpoint creation"""

    def test_create_web_endpoints_generates_fastapi_app(self, tmp_path):
        """create_web_endpoints should generate a properly configured FastAPI app"""
        with patch("modalkit.fast_api.create_app") as mock_create_app:
            mock_app = MagicMock()
            mock_create_app.return_value = mock_app

            service = TestModalService("web-service", tmp_path)

            create_web_endpoints(app_cls=service, input_model=TestRequest, output_model=TestResponse)

            # Verify create_app was called with correct parameters
            mock_create_app.assert_called_once()
            call_kwargs = mock_create_app.call_args[1]

            assert call_kwargs["input_model"] == TestRequest
            assert call_kwargs["output_model"] == TestResponse
            assert "sync_fn" in call_kwargs
            assert "async_fn" in call_kwargs
            assert call_kwargs["dependencies"] == []

    def test_create_web_endpoints_configures_auth_middleware(self, tmp_path):
        """create_web_endpoints should configure authentication middleware correctly"""
        with patch("modalkit.fast_api.create_app") as mock_create_app:
            service = TestModalService("secure-service", tmp_path)

            # Test with secure=False (should use API key auth)
            service.modal_utils.settings.app_settings.deployment_config.secure = False

            create_web_endpoints(app_cls=service, input_model=TestRequest, output_model=TestResponse)

            call_kwargs = mock_create_app.call_args[1]
            assert call_kwargs["auth_middleware"] is not None  # Should have API key middleware

            # Test with secure=True (should use Modal proxy auth)
            service.modal_utils.settings.app_settings.deployment_config.secure = True

            create_web_endpoints(app_cls=service, input_model=TestRequest, output_model=TestResponse)

            call_kwargs = mock_create_app.call_args[1]
            assert call_kwargs["auth_middleware"] is None  # Should use Modal proxy auth


class TestModalServiceIntegration:
    """Test suite for integration scenarios"""

    def test_service_handles_realistic_ml_workflow(self, tmp_path):
        """ModalService should handle realistic ML processing workflows"""
        # Create service with realistic configuration
        service = TestModalService(
            "production-sentiment-analyzer",
            tmp_path,
            processing_delay=0.01,  # Simulate realistic processing time
        )
        service.load_artefacts()

        # Simulate realistic request batch
        requests = [
            SyncInputModel(message=TestRequest(text="This product is amazing!", priority="high")),
            SyncInputModel(message=TestRequest(text="Not satisfied with the quality", priority="normal")),
            AsyncInputModel(
                message=TestRequest(text="Neutral opinion about the service", priority="low"),
                success_queue="analytics-pipeline",
                failure_queue="error-handling",
                meta={"user_id": "user123", "session_id": "sess456"},
            ),
        ]

        start_time = time.time()

        with patch("modalkit.modalapp.send_response_queue", return_value=True):
            results = service.process_request(requests)

        processing_time = time.time() - start_time

        # Verify results
        assert len(results) == 3
        assert all(result.status == "success" for result in results)

        # Verify processing was reasonably fast (should be < 1 second for small batch)
        assert processing_time < 1.0

        # Verify async request has metadata
        assert results[2].meta["user_id"] == "user123"

    def test_service_maintains_performance_under_load(self, tmp_path):
        """ModalService should maintain performance characteristics under load"""
        service = TestModalService("high-throughput-model", tmp_path, processing_delay=0.001)
        service.load_artefacts()

        # Create larger batch to test performance
        large_batch = [SyncInputModel(message=TestRequest(text=f"Request {i}")) for i in range(50)]

        start_time = time.time()
        results = service.process_request(large_batch)
        processing_time = time.time() - start_time

        # Verify all requests processed successfully
        assert len(results) == 50
        assert all(result.status == "success" for result in results)

        # Performance should scale reasonably (not linearly due to batching)
        # With 0.001s per request, 50 requests should take much less than 0.05s due to batching
        assert processing_time < 0.1  # Allow some overhead for test environment
