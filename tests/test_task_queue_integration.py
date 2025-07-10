from unittest.mock import AsyncMock, Mock, patch

import pytest

from modalkit.exceptions import TypeValidationError
from modalkit.task_queue import send_response_queue


@pytest.fixture
def mock_backend() -> Mock:
    """Mock queue backend"""
    backend = Mock()
    backend.send_message = AsyncMock(return_value=True)
    return backend


@pytest.fixture
def mock_settings() -> Mock:
    """Mock settings with queue configuration"""
    settings = Mock()
    settings.app_settings.queue_config.backend = "sqs"
    settings.app_settings.queue_config.model_dump.return_value = {"backend": "sqs"}
    return settings


def test_send_response_queue_success(mock_backend: Mock, mock_settings: Mock) -> None:
    """Test successful queue message sending through abstraction layer"""
    with (
        patch("modalkit.settings.Settings", return_value=mock_settings),
        patch("modalkit.task_queue.create_backend", return_value=mock_backend),
    ):
        message = '{"key": "value"}'
        result = send_response_queue("test-queue", message)

        mock_backend.send_message.assert_called_once_with("test-queue", message)
        assert result is True


def test_send_response_queue_backend_failure(mock_backend: Mock, mock_settings: Mock) -> None:
    """Test queue message sending when backend fails"""
    mock_backend.send_message = AsyncMock(return_value=False)

    with (
        patch("modalkit.settings.Settings", return_value=mock_settings),
        patch("modalkit.task_queue.create_backend", return_value=mock_backend),
    ):
        message = '{"key": "value"}'
        result = send_response_queue("test-queue", message)

        assert result is False


def test_send_response_queue_exception_handling(mock_settings: Mock) -> None:
    """Test exception handling in queue sending"""
    with (
        patch("modalkit.settings.Settings", return_value=mock_settings),
        patch("modalkit.task_queue.create_backend", side_effect=Exception("Backend error")),
    ):
        message = '{"key": "value"}'
        result = send_response_queue("test-queue", message)

        assert result is False


def test_send_response_queue_invalid_message_type() -> None:
    """Test sending invalid response type"""
    with pytest.raises(TypeValidationError, match="Expected string"):
        send_response_queue("test-queue", {"invalid": "response"})  # type: ignore


def test_send_response_queue_memory_backend(mock_backend: Mock, mock_settings: Mock) -> None:
    """Test with in-memory backend"""
    mock_settings.app_settings.queue_config.backend = "memory"
    mock_settings.app_settings.queue_config.model_dump.return_value = {"backend": "memory"}

    with (
        patch("modalkit.settings.Settings", return_value=mock_settings),
        patch("modalkit.task_queue.create_backend", return_value=mock_backend),
    ):
        message = '{"key": "value"}'
        result = send_response_queue("test-queue", message)

        mock_backend.send_message.assert_called_once_with("test-queue", message)
        assert result is True
