from unittest.mock import Mock, patch

import pytest

from modalkit.utils import get_api_key, send_response_sqs


@pytest.fixture
def mock_ssm():
    with patch("boto3.client") as mock_client:
        mock_ssm = Mock()
        mock_ssm.get_parameter.return_value = {"Parameter": {"Value": "test-api-key"}}
        mock_client.return_value = mock_ssm
        yield mock_ssm


@pytest.fixture
def mock_sqs():
    with patch("boto3.client") as mock_client:
        mock_sqs = Mock()
        mock_sqs.get_queue_url.return_value = {"QueueUrl": "test-queue-url"}
        mock_sqs.send_message.return_value = {"MessageId": "test-message-id"}
        mock_client.return_value = mock_sqs
        yield mock_sqs


def test_get_api_key_with_ssm(mock_ssm):
    """Test successful API key retrieval from SSM"""
    with patch("modalkit.utils.BOTO3_AVAILABLE", True):
        result = get_api_key(ssm_key_name="test-service")

        mock_ssm.get_parameter.assert_called_once_with(
            Name="test-service",
            WithDecryption=True,
        )
        assert result == "test-api-key"


def test_get_api_key_with_hardcoded():
    """Test API key retrieval with hardcoded key"""
    result = get_api_key(hardcoded_key="my-hardcoded-key")
    assert result == "my-hardcoded-key"


def test_get_api_key_no_args():
    """Test error when no arguments provided"""
    with pytest.raises(ValueError, match="Either ssm_key_name or hardcoded_key must be provided"):
        get_api_key()


def test_get_api_key_both_args():
    """Test error when both arguments provided"""
    with pytest.raises(ValueError, match="Only one of ssm_key_name or hardcoded_key should be provided"):
        get_api_key(ssm_key_name="test", hardcoded_key="test")


def test_get_api_key_ssm_without_boto3():
    """Test error when trying to use SSM without boto3"""
    # Import locally to avoid caching issues
    from modalkit.utils import get_api_key

    # Clear the cache to ensure fresh evaluation
    get_api_key.cache_clear()

    with (
        patch("modalkit.utils.BOTO3_AVAILABLE", False),
        pytest.raises(ImportError, match="boto3 is required for SSM key retrieval"),
    ):
        get_api_key(ssm_key_name="test-service")


def test_send_response_sqs_success(mock_sqs):
    """Test successful SQS message sending"""
    with patch("modalkit.utils.BOTO3_AVAILABLE", True):
        message = '{"key": "value"}'

        result = send_response_sqs("test-queue", message)

        mock_sqs.get_queue_url.assert_called_once_with(QueueName="test-queue")
        mock_sqs.send_message.assert_called_once_with(QueueUrl="test-queue-url", MessageBody=message)
        assert result is True


def test_send_response_sqs_without_boto3():
    """Test SQS functionality when boto3 is not available"""
    with patch("modalkit.utils.BOTO3_AVAILABLE", False):
        message = '{"key": "value"}'
        result = send_response_sqs("test-queue", message)
        assert result is False


def test_send_response_sqs_invalid_response():
    """Test sending invalid response type"""
    with pytest.raises(TypeError, match="Expected string"):
        send_response_sqs("test-queue", {"invalid": "response"})


def test_send_response_sqs_queue_not_found(mock_sqs):
    """Test handling of non-existent queue"""
    with patch("modalkit.utils.BOTO3_AVAILABLE", True):
        # Mock ClientError
        from unittest.mock import Mock as MockClass

        error_response = {
            "Error": {"Code": "AWS.SimpleQueueService.NonExistentQueue", "Message": "Queue does not exist"}
        }
        client_error = MockClass()
        client_error.response = error_response

        mock_sqs.get_queue_url.side_effect = Exception("ClientError simulation")
        mock_sqs.create_queue.return_value = {"QueueUrl": "new-queue-url"}

        message = '{"key": "value"}'
        result = send_response_sqs("test-queue", message)

        # Should return False due to error handling
        assert result is False


def test_send_response_sqs_send_message_error(mock_sqs):
    """Test handling of send message errors"""
    with patch("modalkit.utils.BOTO3_AVAILABLE", True):
        mock_sqs.send_message.side_effect = Exception("Send error")

        message = '{"key": "value"}'
        result = send_response_sqs("test-queue", message)

        assert result is False
