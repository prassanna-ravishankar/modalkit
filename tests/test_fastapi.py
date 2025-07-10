from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Depends
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

        self.sync_fn = AsyncMock()
        self.async_fn = AsyncMock()

        self.async_fn.return_value = {"job_id": "test_job_id"}
        self.sync_fn.return_value = {"result": "test"}

        # Create the fastapi handler
        fastapi_app = create_app(
            input_model=MockInputModel,
            output_model=MockOutputModel,
            dependencies=[],
            auth_middleware=Depends(auth_middleware),
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
        assert result.status_code == 200
        assert result.json() == {"result": "test"}
        self.sync_fn.assert_called_once()

    def test_create_app_no_auth_middleware(self):
        """Test creating app without auth middleware (for Modal proxy auth)"""
        sync_fn = AsyncMock()
        async_fn = AsyncMock()

        sync_fn.return_value = {"result": "test"}
        async_fn.return_value = {"job_id": "test_job_id"}

        # Create app without auth middleware
        fastapi_app = create_app(
            input_model=MockInputModel,
            output_model=MockOutputModel,
            dependencies=[],
            auth_middleware=None,  # No auth middleware - relies on Modal proxy auth
            sync_fn=sync_fn,
            async_fn=async_fn,
        )

        client = TestClient(fastapi_app)

        # Test that endpoints work without authentication
        message = MockInputModel(message="test")
        result = client.post("/predict_sync", json=message.model_dump(), params={"model_name": "test_model"})
        assert result.status_code == 200
        assert result.json() == {"result": "test"}
        sync_fn.assert_called_once()
