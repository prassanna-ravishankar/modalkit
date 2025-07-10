import os
from pathlib import Path
from typing import Annotated, Any, Union

import modal
from fastapi import Depends

from modalkit.auth import create_validate_api_key
from modalkit.logger import LOGGER
from modalkit.settings import AppSettings, ModelSettings, MountType, Settings


class ModalConfig:
    """
    Configuration class for handling Modal-specific operations.
    Provide many helper methods to permit shorthand usage of Modal, in app code

    Additionally, has some staticmethods that can be used without instantiating the class.
    """

    def __init__(self, settings=None):
        if settings is None:
            self.settings = Settings()
        else:
            self.settings = settings

        # Volumes
        self._volumes = None

    @property
    def cloud_bucket_mounts(self) -> dict[str, modal.CloudBucketMount]:
        """
        Gets the Modal cloud bucket mounts based on configuration.

        Returns:
            dict: A dictionary mapping mount points to Modal CloudBucketMount objects
        """
        cloud_mounts = {}
        for mount_config in self.app_settings.deployment_config.cloud_bucket_mounts:
            # Build CloudBucketMount kwargs
            kwargs = {
                "bucket_name": mount_config.bucket_name,
                "read_only": mount_config.read_only,
                "requester_pays": mount_config.requester_pays,
            }

            # Add optional parameters if specified
            if mount_config.bucket_endpoint_url:
                kwargs["bucket_endpoint_url"] = mount_config.bucket_endpoint_url
            if mount_config.key_prefix:
                kwargs["key_prefix"] = mount_config.key_prefix
            if mount_config.secret:
                kwargs["secret"] = modal.Secret.from_name(mount_config.secret)
            if mount_config.oidc_auth_role_arn:
                kwargs["oidc_auth_role_arn"] = mount_config.oidc_auth_role_arn

            cloud_mounts[mount_config.mount_point] = modal.CloudBucketMount(**kwargs)

        return cloud_mounts

    @property
    def volumes(self) -> dict[str, modal.Volume]:
        """
        Gets the Modal volumes based on the deployment config.
        Returns cached volumes if already computed.

        Returns:
            dict[str, modal.Volume]: Dictionary of Modal volumes
        """
        if not self._volumes:
            LOGGER.info("Initializing Modal volumes")
            if not self.settings.app_settings.deployment_config.volumes:
                LOGGER.info("No volumes configured in deployment config")
                self._volumes = {}
            else:
                try:
                    self._volumes = {
                        Path(key).as_posix(): modal.Volume.from_name(val)
                        for key, val in self.settings.app_settings.deployment_config.volumes.items()
                    }
                    LOGGER.info(f"Successfully initialized {len(self._volumes)} volumes")
                    for mount_point in self._volumes:
                        LOGGER.debug(f"Volume initialized at mount point: {mount_point}")
                except Exception:
                    LOGGER.exception("Failed to initialize volume")
                    raise
        return self._volumes

    @property
    def all_volumes(self) -> dict[str, Union[modal.Volume, modal.CloudBucketMount]]:
        """
        Gets all volume mounts including both regular volumes and cloud bucket mounts.

        Returns:
            dict: Combined dictionary of Modal volumes and CloudBucketMounts
        """
        combined_volumes = {}

        # Add regular volumes
        combined_volumes.update(self.volumes)

        # Add cloud bucket mounts
        combined_volumes.update(self.cloud_bucket_mounts)

        return combined_volumes

    @property
    def app_settings(self) -> "AppSettings":
        """
        Gets the application-specific settings.

        Returns:
            AppSettings: The application configuration settings
        """
        return self.settings.app_settings

    @property
    def model_settings(self) -> "ModelSettings":
        """
        Gets the model-specific settings.

        Returns:
            ModelSettings: The model configuration settings
        """
        return self.settings.model_settings

    @property
    def app_name(self) -> str:
        """
        Gets the complete application name.

        Returns:
            str: The application name with prefix and postfix
        """
        return self.app_settings.app_prefix + self.app_postfix

    @property
    def app_postfix(self) -> str:
        """
        Gets the application postfix from environment.

        Returns:
            str: The application postfix, defaults to "-dev"
        """
        return os.getenv("MODALKIT_APP_POSTFIX", "-dev")

    @property
    def CustomAPIKey(self) -> Annotated:
        """
        Creates an API key validation dependency.

        Returns:
            Annotated[str, Depends]: FastAPI dependency for API key validation

        Notes:
            Requires AWS access during Modal serve or deploy operations
        """
        auth_middleware = create_validate_api_key(
            self.app_settings.auth_config.auth_header, self.app_settings.auth_config
        )
        return Depends(auth_middleware)

    @property
    def mounts(self) -> list[modal.Mount]:
        """
        Gets the Modal mount configurations.

        Returns:
            list[modal.Mount]: List of Modal mount objects for container filesystem access
        """
        return [
            (
                modal.Mount.from_local_file(mnt.local_path, mnt.remote_path)
                if mnt.type == MountType.file
                else modal.Mount.from_local_dir(mnt.local_path, remote_path=mnt.remote_path)
            )
            for mnt in self.app_settings.deployment_config.mounts
        ]

    @property
    def region(self) -> str:
        """
        Gets the Modal deployment region.

        Returns:
            str: String of the modal deployment region.
        """
        return self.app_settings.deployment_config.region

    def get_image(self) -> modal.Image:
        """
        Creates a Modal container image configuration.

        Returns:
            modal.Image: Configured Modal container image with:
                - Base image (either from registry or debian_slim)
                - Build commands
                - Environment variables
                - Working directory
        """
        extra_run_commands = []
        if isinstance(self.app_settings.build_config.extra_run_commands, str):
            extra_run_commands.append(self.app_settings.build_config.extra_run_commands)

        envvars = self.app_settings.build_config.env.copy()
        modalkit_config_path = os.getenv("MODALKIT_CONFIG")
        if modalkit_config_path:
            envvars["MODALKIT_CONFIG"] = modalkit_config_path

        # Use Modal's debian_slim as default, or from_registry for custom registries
        if self.app_settings.build_config.image and self.app_settings.build_config.tag:
            # If image looks like a registry URL, use from_registry
            image_ref = f"{self.app_settings.build_config.image}:{self.app_settings.build_config.tag}"
            if "/" in self.app_settings.build_config.image:
                image = modal.Image.from_registry(image_ref)
            else:
                # Default to debian_slim for simple image names
                image = modal.Image.debian_slim()
        else:
            # Default to debian_slim if no image specified
            image = modal.Image.debian_slim()

        image = image.env(envvars).run_commands(extra_run_commands).workdir(self.app_settings.build_config.workdir)
        return image

    def get_app_cls_settings(self) -> dict[str, Any]:
        """
        Gets Modal application class settings.

        Returns:
            dict: Application settings with None values removed, including:
                - Container image configuration
                - GPU requirements
                - File system mounts
                - Secrets and concurrency settings
                - Volume configurations
        """
        settings = {
            "region": self.region,
            "image": self.get_image(),
            "gpu": self.app_settings.deployment_config.gpu,
            "mounts": self.mounts,
            "secrets": [modal.Secret.from_name(secret) for secret in self.app_settings.deployment_config.secrets],
            "concurrency_limit": self.app_settings.deployment_config.concurrency_limit,
            "allow_concurrent_inputs": self.app_settings.deployment_config.allow_concurrent_inputs,
            "container_idle_timeout": self.app_settings.deployment_config.container_idle_timeout,
            "retries": self.app_settings.deployment_config.retries,
            "volumes": self.all_volumes,
        }
        return {k: v for k, v in settings.items() if v is not None}

    def get_handler_settings(self) -> dict[str, Any]:
        """
        Gets Modal request handler settings.

        Returns:
            dict: Handler settings including:
                - Application image
                - Required secrets
                - File system mounts
                - Concurrency settings
        """
        settings = {
            "image": self.get_image(),
            "secrets": [modal.Secret.from_name(secret) for secret in self.app_settings.deployment_config.secrets],
            "mounts": self.mounts,
            "allow_concurrent_inputs": self.app_settings.deployment_config.allow_concurrent_inputs_handler,
        }

        return settings

    def get_batched_method_settings(self) -> dict[str, Any]:
        """
        Gets Modal batched method settings.

        Returns:
            dict: batched method including:
                - max_batch_size
                - wait_ms
        """
        return self.app_settings.batch_config.model_dump()

    def get_asgi_app_settings(self) -> dict[str, Any]:
        """
        Gets Modal ASGI app settings for web endpoints.

        Returns:
            dict: ASGI app settings including:
                - requires_proxy_auth: Whether to enable Modal proxy authentication
        """
        return {"requires_proxy_auth": self.app_settings.deployment_config.secure}

    def reload_volumes(self):
        """
        Reloads the Modal volumes.
        Handles errors gracefully and provides detailed logging of the process.
        """
        if not self._volumes:
            LOGGER.info("No volumes initialized yet, skipping reload")
            return

        LOGGER.info(f"Starting volume reload for {len(self._volumes)} volumes")
        for mount_point, volume in self._volumes.items():
            try:
                LOGGER.debug(f"Reloading volume at mount point: {mount_point}")
                volume.reload()
                LOGGER.debug(f"Successfully reloaded volume at mount point: {mount_point}")
            except Exception:
                LOGGER.exception(f"Failed to reload volume at mount point {mount_point}")
                # We don't re-raise as we want to continue with other volumes
                continue
        LOGGER.info("Volume reload completed")
