from fastapi import HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN

from modalkit.utils import get_api_key


def create_validate_api_key(header_name, auth_config):
    """Create an API key validator function.

    Args:
        header_name (str): Name of the HTTP header containing the API key
        auth_config: AuthConfig object containing either ssm_key or api_key

    Returns:
        Callable: Async function that validates API keys
    """
    api_key_header = APIKeyHeader(name=header_name, auto_error=True)
    api_key = get_api_key(ssm_key_name=auth_config.ssm_key, hardcoded_key=auth_config.api_key)

    async def validate_api_key(api_key_header: str = Security(api_key_header)):
        if api_key_header == api_key:
            return api_key_header
        else:
            raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Could not validate API KEY")

    return validate_api_key
