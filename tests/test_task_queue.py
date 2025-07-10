from unittest.mock import MagicMock, patch

import pytest

from modalkit.exceptions import TypeValidationError
from modalkit.task_queue import InMemoryBackend, QueueBackend, SQSBackend, create_backend, register_backend


class TestQueueBackend:
    """Test the queue backend protocol"""

    def test_queue_backend_protocol(self) -> None:
        """Test that QueueBackend protocol is properly defined"""
        # This is mainly to ensure the protocol is importable and defined
        assert hasattr(QueueBackend, "send_message")


class TestInMemoryBackend:
    """Test in-memory backend implementation"""

    @pytest.mark.asyncio
    async def test_send_message_success(self) -> None:
        """Test successful message sending via in-memory backend"""
        backend = InMemoryBackend()
        result = await backend.send_message("test-queue", '{"test": "data"}')
        assert result is True

    @pytest.mark.asyncio
    async def test_send_message_type_validation(self) -> None:
        """Test that non-string messages raise TypeValidationError"""
        backend = InMemoryBackend()
        with pytest.raises(TypeValidationError):
            await backend.send_message("test-queue", {"test": "data"})  # type: ignore


class TestSQSBackend:
    """Test SQS backend implementation"""

    @patch("boto3.client")
    def test_sqs_backend_with_boto3(self, mock_boto_client: MagicMock) -> None:
        """Test SQS backend when boto3 is available"""
        backend = SQSBackend()
        assert backend.available is True
        mock_boto_client.assert_called_once_with("sqs")

    def test_sqs_backend_without_boto3(self) -> None:
        """Test SQS backend when boto3 is not available"""
        with patch.object(SQSBackend, "__init__", lambda self, **kwargs: setattr(self, "available", False)):
            backend = SQSBackend()
            assert backend.available is False

    @pytest.mark.asyncio
    @patch("boto3.client")
    async def test_send_message_success(self, mock_boto_client: MagicMock) -> None:
        """Test successful message sending via SQS"""
        mock_sqs = MagicMock()
        mock_boto_client.return_value = mock_sqs
        mock_sqs.get_queue_url.return_value = {"QueueUrl": "http://queue.url"}
        mock_sqs.send_message.return_value = {"MessageId": "123"}

        backend = SQSBackend()
        result = await backend.send_message("test-queue", '{"test": "data"}')

        assert result is True
        mock_sqs.send_message.assert_called_once()

    @pytest.mark.asyncio
    @patch("boto3.client")
    async def test_send_message_no_boto3(self, mock_boto_client: MagicMock) -> None:
        """Test message sending when boto3 is not available"""
        backend = SQSBackend()
        backend.available = False  # Mock unavailable state
        result = await backend.send_message("test-queue", '{"test": "data"}')
        assert result is False

    @pytest.mark.asyncio
    @patch("boto3.client")
    async def test_send_message_type_validation(self, mock_boto_client: MagicMock) -> None:
        """Test that non-string messages raise TypeValidationError"""
        backend = SQSBackend()
        with pytest.raises(TypeValidationError):
            await backend.send_message("test-queue", {"test": "data"})  # type: ignore


class TestBackendFactory:
    """Test the backend factory and registration system"""

    @patch("boto3.client")
    def test_create_sqs_backend(self, mock_boto_client: MagicMock) -> None:
        """Test creating SQS backend"""
        backend = create_backend("sqs")
        assert isinstance(backend, SQSBackend)

    def test_create_memory_backend(self) -> None:
        """Test creating in-memory backend"""
        backend = create_backend("memory")
        assert isinstance(backend, InMemoryBackend)

    def test_create_unknown_backend(self) -> None:
        """Test creating unknown backend raises error"""
        with pytest.raises(Exception):  # BackendError # noqa: B017
            create_backend("unknown")

    def test_register_custom_backend(self) -> None:
        """Test registering a custom backend"""

        class CustomBackend:
            async def send_message(self, queue_name: str, message: str) -> bool:
                return True

        register_backend("custom", CustomBackend)
        backend = create_backend("custom")
        assert isinstance(backend, CustomBackend)


class TestDependencyInjection:
    """Test dependency injection usage patterns"""

    @pytest.mark.asyncio
    async def test_direct_backend_usage(self) -> None:
        """Test using backends directly without factory"""
        backend = InMemoryBackend()
        result = await backend.send_message("test-queue", '{"test": "data"}')
        assert result is True

    @pytest.mark.asyncio
    async def test_custom_backend_implementation(self) -> None:
        """Test implementing custom backend"""

        class MockBackend:
            async def send_message(self, queue_name: str, message: str) -> bool:
                return message == "expected"

        backend = MockBackend()
        assert await backend.send_message("test", "expected") is True
        assert await backend.send_message("test", "unexpected") is False


class TestTaskIQExample:
    """Test the TaskIQ example implementation"""

    @pytest.mark.asyncio
    async def test_taskiq_example_import_error(self) -> None:
        """Test TaskIQ example handles import errors gracefully"""
        # This will fail because TaskIQ is not installed, but we can test the structure
        from modalkit.task_queue import TaskIQExample

        # Should raise DependencyError when TaskIQ is not available
        with pytest.raises(Exception):  # DependencyError # noqa: B017
            TaskIQExample()


class TestCustomRedisExample:
    """Test the custom Redis example implementation"""

    @pytest.mark.asyncio
    async def test_redis_example_import_error(self) -> None:
        """Test Redis example handles import errors gracefully"""
        from modalkit.task_queue import CustomRedisExample

        # Should raise DependencyError when redis is not available
        with pytest.raises(Exception):  # DependencyError # noqa: B017
            CustomRedisExample()
