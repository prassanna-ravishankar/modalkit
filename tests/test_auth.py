from unittest.mock import Mock

import pytest
from fastapi import HTTPException

from modalkit.auth import create_validate_api_key


@pytest.mark.asyncio
async def test_valid_api_key_with_ssm(mocker):
    # Mock get_api_key function
    mocker.patch("modalkit.auth.get_api_key", return_value="valid-key")

    # Create mock auth config with SSM key
    auth_config = Mock()
    auth_config.ssm_key = "test-ssm-key"
    auth_config.api_key = None

    validator = create_validate_api_key("X-API-Key", auth_config)
    result = await validator("valid-key")
    assert result == "valid-key"


@pytest.mark.asyncio
async def test_valid_api_key_with_hardcoded(mocker):
    # Mock get_api_key function
    mocker.patch("modalkit.auth.get_api_key", return_value="hardcoded-key")

    # Create mock auth config with hardcoded key
    auth_config = Mock()
    auth_config.ssm_key = None
    auth_config.api_key = "hardcoded-key"

    validator = create_validate_api_key("X-API-Key", auth_config)
    result = await validator("hardcoded-key")
    assert result == "hardcoded-key"


@pytest.mark.asyncio
async def test_invalid_api_key(mocker):
    # Mock get_api_key function
    mocker.patch("modalkit.auth.get_api_key", return_value="valid-key")

    # Create mock auth config
    auth_config = Mock()
    auth_config.ssm_key = "test-ssm-key"
    auth_config.api_key = None

    validator = create_validate_api_key("X-API-Key", auth_config)
    with pytest.raises(HTTPException) as exc:
        await validator("invalid-key")
    assert exc.value.status_code == 403
