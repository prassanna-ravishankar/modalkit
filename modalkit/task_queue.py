import asyncio
from typing import Protocol

from modalkit.exceptions import BackendError, DependencyError
from modalkit.logger import LOGGER


class QueueBackend(Protocol):
    """Protocol for queue backends"""

    async def send_message(self, queue_name: str, message: str) -> bool:
        """Send a message to the specified queue"""
        ...

    async def create_queue_if_needed(self, queue_name: str) -> bool:
        """Ensure queue exists"""
        ...


class SQSBackend:
    """AWS SQS backend implementation"""

    def __init__(self):
        try:
            import boto3

            self.client = boto3.client("sqs")
            self.available = True
        except ImportError:
            self.available = False
            LOGGER.warning("boto3 not available for SQS backend")

    async def send_message(self, queue_name: str, message: str) -> bool:
        """Send message to SQS queue"""
        if not self.available:
            LOGGER.warning("boto3 not available, cannot send SQS message")
            return False

        try:
            # Import the existing SQS logic
            from modalkit.utils import send_response_queue_impl

            return await asyncio.to_thread(send_response_queue_impl, queue_name, message)
        except Exception as e:
            LOGGER.error(f"Failed to send SQS message: {e}")
            return False

    async def create_queue_if_needed(self, queue_name: str) -> bool:
        """SQS queues are created automatically in send_message"""
        return True


class TaskiqBackend:
    """Taskiq backend implementation"""

    def __init__(self, broker_url: str = "memory://"):
        try:
            from taskiq import InMemoryBroker

            try:
                from taskiq_redis import AsyncRedisTaskiqBroker

                if broker_url.startswith("redis://"):
                    self.broker = AsyncRedisTaskiqBroker(broker_url)
                else:
                    self.broker = InMemoryBroker()
            except ImportError:
                # Fallback to in-memory if redis not available
                self.broker = InMemoryBroker()
                if broker_url.startswith("redis://"):
                    LOGGER.warning("taskiq-redis not available, using in-memory broker")
        except ImportError as e:
            raise DependencyError("taskiq is required for TaskiqBackend. Install with: pip install taskiq") from e

    async def send_message(self, queue_name: str, message: str) -> bool:
        """Send message via Taskiq"""
        try:
            # Create a task for the specific queue
            task = self.broker.task(task_name=f"process_{queue_name}", queue_name=queue_name)

            # Send the message
            await task.kiq(message)
            LOGGER.info(f"Message sent to taskiq queue: {queue_name}")
        except Exception as e:
            LOGGER.error(f"Failed to send message to taskiq: {e}")
            return False
        else:
            return True

    async def create_queue_if_needed(self, queue_name: str) -> bool:
        """Taskiq creates queues dynamically"""
        return True


def get_queue_backend(backend_type: str = "sqs", **kwargs) -> QueueBackend:
    """Factory to get the appropriate queue backend"""
    if backend_type == "sqs":
        return SQSBackend()
    elif backend_type == "taskiq":
        return TaskiqBackend(**kwargs)
    else:
        raise BackendError(f"Unknown backend type: {backend_type}")
