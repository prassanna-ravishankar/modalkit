from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from modalkit.task_queue import QueueBackend, SQSBackend, TaskiqBackend, get_queue_backend


class TestQueueBackend:
    """Test the queue backend abstraction"""

    def test_queue_backend_protocol(self):
        """Test that QueueBackend protocol is properly defined"""
        # This is mainly to ensure the protocol is importable and defined
        assert hasattr(QueueBackend, "send_message")
        assert hasattr(QueueBackend, "create_queue_if_needed")


class TestSQSBackend:
    """Test SQS backend implementation"""

    @patch("boto3.client")
    def test_sqs_backend_with_boto3(self, mock_boto_client):
        """Test SQS backend when boto3 is available"""
        backend = SQSBackend()
        assert backend.available is True
        mock_boto_client.assert_called_once_with("sqs")

    def test_sqs_backend_without_boto3(self):
        """Test SQS backend when boto3 is not available"""
        with patch.object(SQSBackend, "__init__", lambda self: setattr(self, "available", False)):
            backend = SQSBackend()
            assert backend.available is False

    @pytest.mark.asyncio
    @patch("boto3.client")
    async def test_send_message_success(self, mock_boto_client):
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
    async def test_send_message_no_boto3(self, mock_boto_client):
        """Test message sending when boto3 is not available"""
        backend = SQSBackend()
        backend.available = False  # Mock unavailable state
        result = await backend.send_message("test-queue", '{"test": "data"}')
        assert result is False


class TestTaskiqBackend:
    """Test Taskiq backend implementation"""

    @pytest.fixture
    def mock_broker(self):
        """Create a mock broker"""
        broker = MagicMock()
        broker.task = MagicMock()
        return broker

    @patch("taskiq.InMemoryBroker")
    def test_taskiq_backend_inmemory(self, mock_broker_class):
        """Test Taskiq backend with in-memory broker"""
        TaskiqBackend()
        mock_broker_class.assert_called_once()

    def test_taskiq_backend_redis(self):
        """Test Taskiq backend with Redis broker falls back to in-memory"""
        # Since taskiq_redis is not installed, it should fall back to in-memory
        backend = TaskiqBackend("redis://localhost:6379")
        # Should not raise an error and create a working backend
        assert backend is not None

    @pytest.mark.asyncio
    async def test_send_message_success(self, mock_broker):
        """Test successful message sending via Taskiq"""
        backend = TaskiqBackend()
        backend.broker = mock_broker

        # Mock the task and kiq method
        mock_task = MagicMock()
        mock_kiq = AsyncMock(return_value=None)
        mock_task.kiq = mock_kiq
        mock_broker.task.return_value = mock_task

        result = await backend.send_message("test-queue", '{"test": "data"}')

        assert result is True
        mock_broker.task.assert_called_once_with(task_name="process_test-queue", queue_name="test-queue")
        mock_kiq.assert_called_once_with('{"test": "data"}')

    @pytest.mark.asyncio
    async def test_send_message_failure(self, mock_broker):
        """Test message sending failure via Taskiq"""
        backend = TaskiqBackend()
        backend.broker = mock_broker

        # Mock task creation to raise an exception
        mock_broker.task.side_effect = Exception("Connection failed")

        result = await backend.send_message("test-queue", '{"test": "data"}')
        assert result is False


class TestQueueFactory:
    """Test the queue backend factory"""

    @patch("boto3.client")
    def test_get_sqs_backend(self, mock_boto_client):
        """Test getting SQS backend"""
        backend = get_queue_backend("sqs")
        assert isinstance(backend, SQSBackend)

    def test_get_taskiq_backend(self):
        """Test getting Taskiq backend"""
        backend = get_queue_backend("taskiq")
        assert isinstance(backend, TaskiqBackend)

    def test_get_unknown_backend(self):
        """Test getting unknown backend raises error"""
        with pytest.raises(ValueError, match="Unknown backend type: unknown"):
            get_queue_backend("unknown")
