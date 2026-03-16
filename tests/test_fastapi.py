from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from modalkit.fast_api import create_app
from modalkit.iomodel import AsyncInputModel


class MockInputModel(BaseModel):
    message: str


class MockOutputModel(BaseModel):
    result: str


class TestFastAPI:
    @pytest.fixture
    def setup(self):
        self.auth_mock = MagicMock()

        def auth_middleware():
            self.auth_mock()
            return None  # FastAPI dependency should return something

        self.sync_fn = AsyncMock()
        self.async_fn = AsyncMock()

        self.sync_fn.return_value = {"result": "test"}
        self.async_fn.return_value = {"job_id": "test_job_id"}

        # Create the fastapi handler
        fastapi_app = create_app(
            input_model=MockInputModel,
            output_model=MockOutputModel,
            dependencies=[],
            router_dependency=auth_middleware,  # Don't wrap with Depends here, create_app will handle it
            sync_fn=self.sync_fn,
            async_fn=self.async_fn,
        )
        self.client = TestClient(fastapi_app)

    def test_async_endpoint(self, setup):
        message = MockInputModel(message="test")
        wrapped_input_data = AsyncInputModel(
            message=message, success_queue="success-queue", failure_queue="failure-queue", meta={}
        )
        result = self.client.post(
            "/predict_async", json=wrapped_input_data.model_dump(), params={"model_name": "test_model"}
        )
        print(result.json())
        self.auth_mock.assert_called_once()
        assert result.status_code == 200
        assert result.json() == {"job_id": "test_job_id"}
        self.async_fn.assert_called_once()

    def test_sync_endpoint(self, setup):
        message = MockInputModel(message="test")
        result = self.client.post("/predict_sync", json=message.model_dump(), params={"model_name": "test_model"})
        print(f"Response: {result.json()}")
        print(f"Status: {result.status_code}")
        self.auth_mock.assert_called_once()
        assert result.status_code == 200
        assert result.json() == {"result": "test"}
        self.sync_fn.assert_called_once()

    def test_create_app_no_router_dependency(self):
        """Test creating app without router dependency (for Modal proxy auth)"""
        sync_fn = AsyncMock()
        async_fn = AsyncMock()

        sync_fn.return_value = {"result": "test"}
        async_fn.return_value = {"job_id": "test_job_id"}

        # Create app without router dependency
        fastapi_app = create_app(
            input_model=MockInputModel,
            output_model=MockOutputModel,
            dependencies=[],
            router_dependency=None,  # No router dependency - relies on Modal proxy auth
            sync_fn=sync_fn,
            async_fn=async_fn,
        )

        client = TestClient(fastapi_app)

        # Test that endpoints work without authentication
        message = MockInputModel(message="test")
        result = client.post("/predict_sync", json=message.model_dump(), params={"model_name": "test_model"})
        print(f"Response: {result.json()}")
        print(f"Status: {result.status_code}")
        assert result.status_code == 200
        assert result.json() == {"result": "test"}
        sync_fn.assert_called_once()

    def test_sync_endpoint_invalid_input(self, setup):
        """Invalid input should return 422 with validation details"""
        result = self.client.post("/predict_sync", json={"wrong_field": 123}, params={"model_name": "test_model"})
        assert result.status_code == 422
        assert "detail" in result.json()

    def test_async_endpoint_invalid_input(self, setup):
        """Invalid async input should return 422"""
        result = self.client.post("/predict_async", json={"wrong_field": 123}, params={"model_name": "test_model"})
        assert result.status_code == 422
        assert "detail" in result.json()

    def test_health_endpoint(self, setup):
        """Health endpoint should return ok without authentication"""
        result = self.client.get("/health")
        assert result.status_code == 200
        assert result.json() == {"status": "ok"}

    def test_health_endpoint_no_auth_required(self):
        """Health endpoint should work even without any auth middleware"""
        sync_fn = AsyncMock()
        async_fn = AsyncMock()

        fastapi_app = create_app(
            input_model=MockInputModel,
            output_model=MockOutputModel,
            dependencies=[],
            router_dependency=None,
            sync_fn=sync_fn,
            async_fn=async_fn,
        )

        client = TestClient(fastapi_app)
        result = client.get("/health")
        assert result.status_code == 200
        assert result.json() == {"status": "ok"}
