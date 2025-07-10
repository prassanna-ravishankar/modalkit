try:
    import boto3
    import botocore

    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

from cachetools import TTLCache, cached

from modalkit.exceptions import AuthConfigError, DependencyError, TypeValidationError
from modalkit.logger import LOGGER


@cached(cache=TTLCache(maxsize=100, ttl=600))
def get_api_key(ssm_key_name=None, hardcoded_key=None):
    """Retrieve the API key from AWS SSM or use a hardcoded key.

    Arguments:
        ssm_key_name (str, optional): SSM parameter name for the API key
        hardcoded_key (str, optional): Hardcoded API key value

    Returns:
        str: The API key

    Raises:
        ValueError: If neither or both arguments are provided
        ImportError: If boto3 is not available when trying to use SSM
    """
    if ssm_key_name and hardcoded_key:
        raise AuthConfigError("Only one of ssm_key_name or hardcoded_key should be provided, not both")
    if not ssm_key_name and not hardcoded_key:
        raise AuthConfigError("Either ssm_key_name or hardcoded_key must be provided")

    if hardcoded_key:
        return hardcoded_key

    # SSM key path - require boto3
    if not BOTO3_AVAILABLE:
        raise DependencyError("boto3 is required for SSM key retrieval. Install it with: pip install boto3")

    ssm_client = boto3.client("ssm")
    api_key = ssm_client.get_parameter(
        Name=ssm_key_name,
        WithDecryption=True,  # Safe to always be true.
    )["Parameter"]["Value"]

    return api_key


def send_response_queue_impl(queue_name, queue_message: str):
    """
    Internal queue implementation (used by SQSBackend)
    """
    if not isinstance(queue_message, str):
        raise TypeValidationError(f"Expected string, got {type(queue_message)}")

    # Check if SQS functionality is available
    if not BOTO3_AVAILABLE:
        LOGGER.warning("boto3 not available, cannot send queue message")
        return False

    sqs = boto3.client("sqs")

    try:
        sqs_response = sqs.get_queue_url(QueueName=queue_name)
        queue_url = sqs_response["QueueUrl"]
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "AWS.SimpleQueueService.NonExistentQueue":
            # Queue does not exist, so create it
            LOGGER.debug(f"Queue: {queue_name} does not exist. Creating it.")
            try:
                sqs_response = sqs.create_queue(QueueName=queue_name)
                queue_url = sqs_response["QueueUrl"]
                LOGGER.debug(f"Created queue with url: {queue_url}")
            except Exception as create_error:
                LOGGER.error(f"Failed to create SQS queue {queue_name}: {create_error}")
                return False
        else:
            # Some other error occurred
            LOGGER.warning(f"Error while fetching SQS queue URL: {e}")
            return False
    except Exception as e:
        LOGGER.warning(f"SQS not available or error occurred: {e}")
        return False

    try:
        sqs_response = sqs.send_message(QueueUrl=queue_url, MessageBody=queue_message)
        LOGGER.info(f"Message ID: {sqs_response['MessageId']} sent to queue: {queue_url}")
    except Exception as e:
        LOGGER.error(f"Failed to send message to SQS queue: {e}")
        return False
    else:
        return True


def send_response_queue(queue_name, queue_message: str):
    """
    Send response to queue (supports both SQS and Taskiq backends)

    Args:
        queue_name(str): Name of the queue
        queue_message(str): Message to be sent to queue.

    Returns:
        bool: True if message was sent successfully, False otherwise

    Raises:
        TypeError: If queue_message is not a string
    """
    if not isinstance(queue_message, str):
        raise TypeValidationError(f"Expected string, got {type(queue_message)}")

    # Get queue backend from settings
    try:
        from modalkit.settings import Settings
        from modalkit.task_queue import get_queue_backend

        settings = Settings()
        queue_config = settings.app_settings.queue_config
        backend_type = queue_config.backend

        backend = get_queue_backend(backend_type, broker_url=queue_config.broker_url)

        # Use async event loop if available, otherwise create one
        import asyncio

        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(backend.send_message(queue_name, queue_message))
            return loop.run_until_complete(task)
        except RuntimeError:
            return asyncio.run(backend.send_message(queue_name, queue_message))
    except Exception as e:
        LOGGER.error(f"Failed to use new queue backend, falling back to original implementation: {e}")
        # Fallback to original implementation
        return send_response_queue_impl(queue_name, queue_message)


# Backward compatibility alias
def send_response_sqs(queue_name, queue_message: str):
    """Backward compatibility function - redirects to send_response_queue"""
    return send_response_queue(queue_name, queue_message)
