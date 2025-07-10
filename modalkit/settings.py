import os
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

from pydantic import BaseModel, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict, YamlConfigSettingsSource
from pydantic_settings_yaml import YamlBaseSettings

from modalkit.exceptions import AuthConfigError


class MountType(str, Enum):
    file = "file"
    dir = "dir"


class Mount(BaseModel):
    """
    Represents a mount point for a local and remote path.

    Attributes:
        local_path (str): Local path to the mount point
        remote_path (str): Remote path to the mount point
    """

    local_path: str
    remote_path: str
    type: MountType = MountType.file


class Volume(BaseModel):
    """
    Represents a volume for a mount point.

    Attributes:
        mount_point (str): Mount point for the volume
        volume_name (str): Name of the volume
    """

    mount_point: str
    volume_name: str


class CloudBucketMount(BaseModel):
    """
    Represents a cloud bucket mount configuration.

    Attributes:
        mount_point (str): Local mount point path
        bucket_name (str): Name of the cloud bucket
        bucket_endpoint_url (Optional[str]): Custom endpoint URL (for Cloudflare R2 or GCS)
        key_prefix (Optional[str]): Prefix for bucket keys (subdirectory)
        secret (Optional[str]): Name of Modal secret for authentication
        oidc_auth_role_arn (Optional[str]): OIDC authentication role ARN
        read_only (bool): Mount bucket as read-only
        requester_pays (bool): Enable requester pays mode
    """

    mount_point: str
    bucket_name: str
    bucket_endpoint_url: Optional[str] = None
    key_prefix: Optional[str] = None
    secret: Optional[str] = None
    oidc_auth_role_arn: Optional[str] = None
    read_only: bool = False
    requester_pays: bool = False


class DeploymentConfig(BaseModel):
    """
    Represents the deployment configuration for a model.

    Attributes:
        gpu (Optional[str, list[str]]): GPU configuration for the deployment
        volumes (Optional[dict[str, str]]): Volume configuration for the deployment
        volume_reload_interval_seconds (Optional[int]): Time interval in seconds after which volumes should be reloaded (None means no auto-reload)
        concurrency_limit (Optional[int]): Concurrency limit for the deployment
        retries (int): Number of retries for the deployment
        secrets (list[str]): List of secrets for the deployment
        mounts (list[Mount]): List of mounts for the deployment
        container_idle_timeout (int): Container idle timeout for the deployment
        allow_concurrent_inputs (int): Allow concurrent inputs for the deployment
        allow_concurrent_inputs_handler (int): Allow concurrent inputs handler for the deployment
        secure (bool): Enable Modal proxy authentication requiring Modal-Key and Modal-Secret headers
        cloud_bucket_mounts (list[CloudBucketMount]): List of cloud bucket mounts for the deployment
    """

    region: Optional[str] = None
    gpu: Union[str, list[str], None] = None
    volumes: Optional[dict[str, str]]
    volume_reload_interval_seconds: Optional[int] = -1
    concurrency_limit: Optional[int] = None
    retries: int = 1
    secrets: list[str] = []
    mounts: list[Mount] = []
    container_idle_timeout: int = 180
    allow_concurrent_inputs: int = 1
    allow_concurrent_inputs_handler: int = 10
    secure: bool = False
    cloud_bucket_mounts: list[CloudBucketMount] = []


class BatchConfig(BaseModel):
    """
    Represents the batching configuration for the endpoints. Note that batching is
    only supported for async endpoints. Sync endpoints process requests individually
    regardless of these settings.

    Attributes:
        max_batch_size (int): limits the number of inputs combined into a single batch
        wait_ms (int): limits the amount of time waiting for more inputs after the first input is received
    """

    max_batch_size: int = 1
    wait_ms: int = 0


class QueueConfig(BaseModel):
    """
    Represents the queue configuration for async message handling.

    Attributes:
        backend (str): Queue backend type ("sqs" or "taskiq")
        broker_url (str): Broker URL for taskiq (e.g., "redis://localhost:6379" or "memory://")
    """

    backend: str = "sqs"
    broker_url: str = "memory://"


class BuildConfig(BaseModel):
    """
    Represents the build configuration for a model.

    Attributes:
        image (str): Image name for the model (can be a registry URL or simple name)
        tag (str): Tag for the model
        env (dict[str, str]): Environment variables for the model
        extra_run_commands (str): Extra run commands for the model
        workdir (str): Working directory in the container
    """

    image: str
    tag: str
    workdir: str = "/root"
    env: dict[str, str] = {}
    extra_run_commands: Union[str, list[str]] = []


class AuthConfig(BaseModel):
    """
    Represents the authentication configuration for a model.

    Attributes:
        ssm_key (Optional[str]): SSM key for the authentication (optional, mutually exclusive with api_key)
        api_key (Optional[str]): Hardcoded API key for authentication (optional, mutually exclusive with ssm_key)
        auth_header (str): Authentication header for the authentication
    """

    ssm_key: Optional[str] = None
    api_key: Optional[str] = None
    auth_header: str

    @model_validator(mode="after")
    def validate_auth_config(self):
        """Validate that exactly one of ssm_key or api_key is provided."""
        if self.ssm_key and self.api_key:
            raise AuthConfigError("Only one of ssm_key or api_key should be provided, not both")
        if not self.ssm_key and not self.api_key:
            raise AuthConfigError("Either ssm_key or api_key must be provided")
        return self


class AppSettings(BaseModel):
    """
    Represents the application settings for a model.

    Attributes:
        app_prefix (str): Application prefix for the model
        build_config (BuildConfig): Build configuration for the model
        auth_config (AuthConfig): Authentication configuration for the model
        deployment_config (DeploymentConfig): Deployment configuration for the model
        batch_config (BatchConfig): Batch endpoint configuration
        queue_config (QueueConfig): Queue configuration for async messaging
    """

    app_prefix: str
    build_config: BuildConfig
    auth_config: AuthConfig
    deployment_config: DeploymentConfig
    batch_config: BatchConfig
    queue_config: QueueConfig = QueueConfig()


class ModelSettings(BaseModel):
    """
    Represents the model settings for a model.

    Attributes:
        local_model_repository_folder (Path): Local model repository folder for the model
        model_entries (dict[str, Any]): Model entries for the model
        common (dict[str, Any]): Common settings for the model
    """

    local_model_repository_folder: Path
    model_entries: dict[str, Any]
    common: dict[str, Any]


class Settings(YamlBaseSettings):
    """
    Main configuration settings for Modalkit applications.

    This class manages all configuration settings for both the application and
    model deployment. It supports loading settings from YAML files and environment
    variables with proper type validation.

    Attributes:
        app_settings (AppSettings): Application-level configuration settings
        model_settings (ModelSettings): Model-specific configuration settings

    Configuration is loaded from:
        - Environment variables with MODALKIT_ prefix
        - modalkit.yaml file. This location can
            be overridden by the
            MODALKIT_CONFIG environment variable.
        - .env file
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        yaml_file="modalkit.yaml",
        env_prefix="MODALKIT_",
        env_nested_delimiter="__",
        protected_namespaces=("settings_",),
        case_sensitive=False,
    )
    app_settings: AppSettings
    model_settings: ModelSettings

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        yaml_files = os.getenv("MODALKIT_CONFIG", "modalkit.yaml").split(",")
        yaml_settings = [YamlConfigSettingsSource(settings_cls, yaml_file=yaml_file) for yaml_file in yaml_files]
        yaml_settings.reverse()
        return (
            init_settings,
            env_settings,
            *yaml_settings,
            file_secret_settings,
        )


if __name__ == "__main__":
    settings = Settings()
    print(settings.model_dump())
    pass
