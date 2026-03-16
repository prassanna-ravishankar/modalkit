import asyncio
from typing import Any, Protocol

from loguru import logger

from modalkit.exceptions import BackendError, TypeValidationError


class QueueBackend(Protocol):
    """
    Protocol for queue backends.

    Implement this interface to create custom queue backends.
    The interface is intentionally minimal - just message sending.
    """

    async def send_message(self, queue_name: str, message: str) -> bool:
        """
        Send a message to the specified queue.

        Args:
            queue_name: Name/identifier of the queue
            message: Message content (JSON string)

        Returns:
            True if message was sent successfully, False otherwise
        """
        ...


class InMemoryBackend:
    """
    Simple in-memory backend for testing and development.
    Messages are just logged, not actually queued.
    """

    async def send_message(self, queue_name: str, message: str) -> bool:
        """Send message to in-memory log"""
        if not isinstance(message, str):
            raise TypeValidationError(f"Expected string, got {type(message)}")

        logger.info(f"[IN-MEMORY QUEUE] {queue_name}: {message}")
        return True


class SQSBackend:
    """
    Direct AWS SQS backend implementation.

    This is a basic implementation - for production use, consider
    implementing a custom backend with proper error handling,
    retry logic, etc.
    """

    def __init__(self, **kwargs: Any) -> None:
        try:
            import boto3  # type: ignore

            self.client = boto3.client("sqs")
            self.available = True
        except ImportError:
            self.available = False
            logger.warning("boto3 not available for SQS backend")

    async def send_message(self, queue_name: str, message: str) -> bool:
        """Send message to SQS queue"""
        if not self.available:
            logger.warning("boto3 not available, cannot send SQS message")
            return False

        if not isinstance(message, str):
            raise TypeValidationError(f"Expected string, got {type(message)}")

        # Get or create queue
        try:
            sqs_response = self.client.get_queue_url(QueueName=queue_name)
            queue_url = sqs_response["QueueUrl"]
        except self.client.exceptions.QueueDoesNotExist:
            logger.debug(f"Queue: {queue_name} does not exist. Creating it.")
            try:
                sqs_response = self.client.create_queue(QueueName=queue_name)
                queue_url = sqs_response["QueueUrl"]
                logger.debug(f"Created queue with url: {queue_url}")
            except Exception as create_error:
                logger.error(f"Failed to create SQS queue {queue_name}: {create_error}")
                return False
        except Exception as e:
            logger.error(f"Error while fetching SQS queue URL: {e}")
            return False

        # Send message
        try:
            sqs_response = self.client.send_message(QueueUrl=queue_url, MessageBody=message)
        except Exception as e:
            logger.error(f"Failed to send message to SQS queue: {e}")
            return False
        else:
            logger.info(f"Message ID: {sqs_response['MessageId']} sent to queue: {queue_url}")
            return True


# Backend registry for simple factory pattern (optional)
_BACKEND_REGISTRY: dict[str, type] = {
    "memory": InMemoryBackend,
    "sqs": SQSBackend,
}


def register_backend(name: str, backend_class: type["QueueBackend"]) -> None:
    """
    Register a queue backend class for factory usage.

    Args:
        name: Backend name to use in configuration
        backend_class: Class that implements QueueBackend protocol

    Example:
        class MyCustomBackend:
            async def send_message(self, queue_name: str, message: str) -> bool:
                # Your implementation here
                return True

        register_backend("my-backend", MyCustomBackend)
    """
    _BACKEND_REGISTRY[name] = backend_class
    logger.info(f"Registered queue backend: {name}")


def create_backend(backend_type: str, **kwargs: Any) -> Any:
    """
    Factory function to create queue backends.

    This is optional - users can create backends directly.

    Args:
        backend_type: Type of backend to create
        **kwargs: Backend-specific configuration

    Returns:
        QueueBackend instance

    Raises:
        BackendError: If backend type is not registered
    """
    if backend_type not in _BACKEND_REGISTRY:
        available_backends = ", ".join(_BACKEND_REGISTRY.keys())
        raise BackendError(
            f"Unknown backend type: {backend_type}. "
            f"Available backends: {available_backends}. "
            f"Register custom backends using register_backend()."
        )

    backend_class = _BACKEND_REGISTRY[backend_type]
    return backend_class(**kwargs)


def send_response_queue(queue_name: str, queue_message: str) -> bool:
    """
    Convenience function that uses the configured backend from settings.

    For dependency injection usage, prefer injecting the backend directly.

    Args:
        queue_name: Name of the queue
        queue_message: Message to be sent to queue

    Returns:
        bool: True if message was sent successfully, False otherwise

    Raises:
        TypeValidationError: If queue_message is not a string
    """
    if not isinstance(queue_message, str):
        raise TypeValidationError(f"Expected string, got {type(queue_message)}")

    from modalkit.settings import Settings

    settings = Settings()
    queue_config = settings.app_settings.queue_config

    # Create backend using factory
    try:
        backend = create_backend(queue_config.backend, **queue_config.model_dump())
    except Exception as e:
        logger.error(f"Failed to create backend: {e}")
        return False

    # Execute async send_message in event loop
    # Note: This is a sync function that wraps async backend calls.
    # If calling from async context, use the backend directly instead.
    try:
        result = asyncio.run(backend.send_message(queue_name, queue_message))
    except Exception as e:
        logger.error(f"Failed to send message to queue {queue_name}: {e}")
        return False

    return bool(result)
