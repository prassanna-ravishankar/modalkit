import os
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict, YamlConfigSettingsSource
from pydantic_settings_yaml import YamlBaseSettings  # type: ignore[import-untyped]


class MountType(str, Enum):
    FILE = "file"
    DIR = "dir"


class Mount(BaseModel):
    """
    Represents a mount point for a local and remote path.

    Attributes:
        local_path (str): Local path to the mount point
        remote_path (str): Remote path to the mount point
    """

    local_path: str
    remote_path: str
    type: MountType = MountType.FILE


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
    bucket_endpoint_url: str | None = None
    key_prefix: str | None = None
    secret: str | None = None
    oidc_auth_role_arn: str | None = None
    read_only: bool = False
    requester_pays: bool = False


# Keys removed when modal renamed its autoscaling and concurrency options
REMOVED_DEPLOYMENT_KEYS = {
    "concurrency_limit": "max_containers",
    "container_idle_timeout": "scaledown_window",
    "allow_concurrent_inputs": "max_concurrent_inputs",
    "allow_concurrent_inputs_handler": "max_concurrent_inputs_handler",
}


class DeploymentConfig(BaseModel):
    """
    Represents the deployment configuration for a model.

    Attributes:
        gpu (Optional[str, list[str]]): GPU configuration for the deployment
        volumes (Optional[dict[str, str]]): Volume configuration for the deployment
        volume_reload_interval_seconds (Optional[int]): Time interval in seconds after which
            volumes should be reloaded (None means no auto-reload)
        max_containers (Optional[int]): Upper limit on containers for the app class
        retries (int): Number of retries for the deployment
        secrets (list[str]): List of secrets for the deployment
        mounts (list[Mount]): List of mounts for the deployment
        scaledown_window (int): Seconds an idle container stays up before scaling down
        max_concurrent_inputs (int): Inputs one app class container handles at once (@modal.concurrent)
        max_concurrent_inputs_handler (int): Inputs one web handler container handles at once (@modal.concurrent)
        secure (bool): Enable Modal proxy authentication requiring Modal-Key and Modal-Secret headers
        cloud_bucket_mounts (list[CloudBucketMount]): List of cloud bucket mounts for the deployment
    """

    model_config = ConfigDict(extra="forbid")

    region: str | None = None
    gpu: str | list[str] | None = None
    volumes: dict[str, str] | None = None
    volume_reload_interval_seconds: int | None = None
    max_containers: int | None = None
    retries: int = 1
    secrets: list[str] = Field(default_factory=list)
    mounts: list[Mount] = Field(default_factory=list)
    scaledown_window: int = 180
    max_concurrent_inputs: int = 1
    max_concurrent_inputs_handler: int = 10
    secure: bool = False
    cloud_bucket_mounts: list[CloudBucketMount] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def reject_removed_modal_keys(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for old, new in REMOVED_DEPLOYMENT_KEYS.items():
                if old in data:
                    raise ValueError(f"'{old}' was removed with modal's 1.x API; use '{new}' instead")
        return data

    @field_validator("volume_reload_interval_seconds")
    @classmethod
    def validate_reload_interval(cls, v: int | None) -> int | None:
        if v is not None and v < 0:
            return None
        return v


class BatchConfig(BaseModel):
    """
    Represents the batching configuration for the endpoints. Note that batching is
    only supported for async endpoints. Sync endpoints process requests individually
    regardless of these settings.

    Attributes:
        max_batch_size (int): limits the number of inputs combined into a single batch
        wait_ms (int): limits the amount of time waiting for more inputs after the first input is received
    """

    model_config = ConfigDict(extra="forbid")

    max_batch_size: int = 1
    wait_ms: int = 0


class QueueConfig(BaseModel):
    """
    Represents the queue configuration for async message handling.

    Attributes:
        backend (str): Queue backend type ("sqs" or "taskiq")
        broker_url (str): Broker URL for taskiq (e.g., "redis://localhost:6379" or "memory://")
    """

    model_config = ConfigDict(extra="forbid")

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

    model_config = ConfigDict(extra="forbid")

    image: str
    tag: str
    workdir: str = "/root"
    env: dict[str, str] = Field(default_factory=dict)
    extra_run_commands: str | list[str] = Field(default_factory=list)


class AppSettings(BaseModel):
    """
    Represents the application settings for a model.

    Attributes:
        app_prefix (str): Application prefix for the model
        build_config (BuildConfig): Build configuration for the model
        deployment_config (DeploymentConfig): Deployment configuration for the model
        batch_config (BatchConfig): Batch endpoint configuration
        queue_config (QueueConfig): Queue configuration for async messaging
    """

    model_config = ConfigDict(extra="forbid")

    app_prefix: str
    build_config: BuildConfig
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

    model_config = ConfigDict(extra="forbid")

    local_model_repository_folder: Path
    model_entries: dict[str, Any]
    common: dict[str, Any]


class Settings(YamlBaseSettings):  # type: ignore[no-any-unimported]
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

    def __init__(self, **kwargs: Any) -> None:
        """Initialize Settings with optional direct values or load from config sources."""
        super().__init__(**kwargs)

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
