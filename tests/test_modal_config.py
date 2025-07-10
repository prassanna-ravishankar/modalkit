"""
Tests for ModalConfig behavior.

This module tests the Modal deployment configuration management functionality, focusing on:
- Configuration loading and validation behavior
- Image and deployment settings generation
- Volume and storage mount configuration
- Authentication and security settings
- Error handling and edge cases
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from modalkit.modalutils import ModalConfig
from modalkit.settings import (
    AppSettings,
    AuthConfig,
    BatchConfig,
    BuildConfig,
    CloudBucketMount,
    DeploymentConfig,
    ModelSettings,
    Settings,
)


class TestModalConfigurationBehavior:
    """Test suite focusing on ModalConfig configuration management behavior"""

    @pytest.fixture
    def basic_settings(self, tmp_path):
        """Provides basic configuration settings for testing"""
        app_settings = AppSettings(
            app_prefix="sentiment-analyzer",
            build_config=BuildConfig(
                image="python", tag="3.11-slim", env={"MODEL_PATH": "/models", "CACHE_SIZE": "1000"}, workdir="/app"
            ),
            auth_config=AuthConfig(ssm_key="/ml/api-key", auth_header="X-API-Key"),
            deployment_config=DeploymentConfig(
                gpu="T4", volumes={}, concurrency_limit=5, container_idle_timeout=300, secure=False
            ),
            batch_config=BatchConfig(max_batch_size=10, wait_ms=100),
        )

        model_settings = ModelSettings(
            model_entries={"sentiment-v1": {"version": "1.0"}},
            local_model_repository_folder=str(tmp_path / "models"),
            common={"cache_folder": str(tmp_path / "cache"), "timeout": 30},
        )

        return Settings(app_settings=app_settings, model_settings=model_settings)

    @pytest.fixture
    def settings_with_storage(self, tmp_path):
        """Provides settings with various storage configurations"""
        cloud_mounts = [
            CloudBucketMount(
                mount_point="/data/s3", bucket_name="ml-training-data", secret="aws-s3-credentials", read_only=True
            ),
            CloudBucketMount(
                mount_point="/models/gcs",
                bucket_name="ml-models-store",
                bucket_endpoint_url="https://storage.googleapis.com",
                key_prefix="production/",
                secret="gcp-storage-credentials",
            ),
        ]

        app_settings = AppSettings(
            app_prefix="ml-service",
            build_config=BuildConfig(image="ml-base", tag="latest"),
            auth_config=AuthConfig(api_key="test-key", auth_header="Authorization"),
            deployment_config=DeploymentConfig(
                volumes={"/shared/models": "model-cache"}, cloud_bucket_mounts=cloud_mounts
            ),
            batch_config=BatchConfig(),
        )

        model_settings = ModelSettings(
            model_entries={}, local_model_repository_folder=str(tmp_path / "models"), common={}
        )

        return Settings(app_settings=app_settings, model_settings=model_settings)

    def test_config_initializes_with_provided_settings(self, basic_settings):
        """ModalConfig should initialize with provided settings"""
        config = ModalConfig(basic_settings)

        assert config.settings == basic_settings
        assert config.app_settings == basic_settings.app_settings
        assert config.model_settings == basic_settings.model_settings

    def test_config_can_initialize_with_default_settings(self):
        """ModalConfig should initialize with default settings when none provided"""
        with patch("modalkit.modalutils.Settings") as mock_settings:
            mock_instance = MagicMock()
            mock_settings.return_value = mock_instance

            config = ModalConfig()

            assert config.settings == mock_instance
            mock_settings.assert_called_once()

    def test_config_generates_correct_app_name(self, basic_settings):
        """ModalConfig should generate app names with prefix and environment postfix"""
        config = ModalConfig(basic_settings)

        # Default postfix is -dev
        expected_name = "sentiment-analyzer-dev"
        assert config.app_name == expected_name

    @patch.dict(os.environ, {"MODALKIT_APP_POSTFIX": "-production"})
    def test_config_respects_environment_app_postfix(self, basic_settings):
        """ModalConfig should use environment variable for app postfix"""
        config = ModalConfig(basic_settings)

        expected_name = "sentiment-analyzer-production"
        assert config.app_name == expected_name

    def test_config_manages_volume_lifecycle(self, settings_with_storage):
        """ModalConfig should manage Modal volume initialization and access"""
        with patch("modal.Volume.from_name") as mock_volume:
            mock_volume_instance = MagicMock()
            mock_volume.return_value = mock_volume_instance

            config = ModalConfig(settings_with_storage)

            # First access should initialize volumes
            volumes = config.volumes

            assert "/shared/models" in volumes
            assert volumes["/shared/models"] == mock_volume_instance
            mock_volume.assert_called_once_with("model-cache")

            # Second access should return cached volumes
            volumes2 = config.volumes
            assert volumes2 == volumes
            # Should not call Modal again
            mock_volume.assert_called_once()

    def test_config_handles_empty_volume_configuration(self, basic_settings):
        """ModalConfig should handle configurations with no volumes gracefully"""
        config = ModalConfig(basic_settings)

        volumes = config.volumes

        assert volumes == {}

    def test_config_handles_volume_initialization_errors(self, settings_with_storage):
        """ModalConfig should propagate volume initialization errors"""
        with patch("modal.Volume.from_name", side_effect=Exception("Volume not found")):
            config = ModalConfig(settings_with_storage)

            with pytest.raises(Exception, match="Volume not found"):
                _ = config.volumes

    def test_config_manages_cloud_bucket_mounts(self, settings_with_storage):
        """ModalConfig should configure cloud bucket mounts with appropriate settings"""
        with patch("modal.CloudBucketMount") as mock_cloud_mount, patch("modal.Secret.from_name") as mock_secret:
            mock_mount_instance = MagicMock()
            mock_cloud_mount.return_value = mock_mount_instance
            mock_secret_instance = MagicMock()
            mock_secret.return_value = mock_secret_instance

            config = ModalConfig(settings_with_storage)
            cloud_mounts = config.cloud_bucket_mounts

            assert len(cloud_mounts) == 2
            assert "/data/s3" in cloud_mounts
            assert "/models/gcs" in cloud_mounts

            # Verify cloud mount configurations
            assert mock_cloud_mount.call_count == 2

            # Check that secrets were properly configured
            mock_secret.assert_any_call("aws-s3-credentials")
            mock_secret.assert_any_call("gcp-storage-credentials")

    def test_config_combines_volumes_and_cloud_mounts(self, settings_with_storage):
        """ModalConfig should provide unified access to all volume types"""
        with (
            patch("modal.Volume.from_name") as mock_volume,
            patch("modal.CloudBucketMount") as mock_cloud_mount,
            patch("modal.Secret.from_name"),
        ):
            mock_volume_instance = MagicMock()
            mock_volume.return_value = mock_volume_instance
            mock_mount_instance = MagicMock()
            mock_cloud_mount.return_value = mock_mount_instance

            config = ModalConfig(settings_with_storage)
            all_volumes = config.all_volumes

            # Should include both regular volumes and cloud mounts
            assert len(all_volumes) == 3  # 1 volume + 2 cloud mounts
            assert "/shared/models" in all_volumes  # regular volume
            assert "/data/s3" in all_volumes  # cloud mount
            assert "/models/gcs" in all_volumes  # cloud mount

    @patch("modal.Image.debian_slim")
    def test_config_generates_modal_images_correctly(self, mock_debian_slim, basic_settings):
        """ModalConfig should generate Modal images with correct configuration"""
        mock_image = MagicMock()
        mock_env_chain = MagicMock()
        mock_run_chain = MagicMock()
        mock_workdir_chain = MagicMock()

        mock_debian_slim.return_value = mock_image
        mock_image.env.return_value = mock_env_chain
        mock_env_chain.run_commands.return_value = mock_run_chain
        mock_run_chain.workdir.return_value = mock_workdir_chain

        config = ModalConfig(basic_settings)
        config.get_image()

        # Verify image construction flow
        mock_debian_slim.assert_called_once()
        mock_image.env.assert_called_once()
        mock_env_chain.run_commands.assert_called_once_with([])
        mock_run_chain.workdir.assert_called_once_with("/app")

        # Verify environment variables were passed
        env_call_args = mock_image.env.call_args[0][0]
        assert "MODEL_PATH" in env_call_args
        assert env_call_args["MODEL_PATH"] == "/models"
        assert env_call_args["CACHE_SIZE"] == "1000"

    @patch("modal.Image.from_registry")
    def test_config_handles_custom_registry_images(self, mock_from_registry, basic_settings):
        """ModalConfig should use from_registry for custom image URLs"""
        # Configure for registry image
        basic_settings.app_settings.build_config.image = "ghcr.io/company/ml-base"

        mock_image = MagicMock()
        mock_env_chain = MagicMock()
        mock_run_chain = MagicMock()

        mock_from_registry.return_value = mock_image
        mock_image.env.return_value = mock_env_chain
        mock_env_chain.run_commands.return_value = mock_run_chain

        config = ModalConfig(basic_settings)
        config.get_image()

        expected_image_ref = "ghcr.io/company/ml-base:3.11-slim"
        mock_from_registry.assert_called_once_with(expected_image_ref)

    @patch.dict(os.environ, {"MODALKIT_CONFIG": "/custom/config.yaml"})
    def test_config_includes_environment_variables_in_images(self, basic_settings):
        """ModalConfig should include environment variables in image configuration"""
        with patch("modal.Image.debian_slim") as mock_debian_slim:
            mock_image = MagicMock()
            mock_debian_slim.return_value = mock_image

            config = ModalConfig(basic_settings)
            config.get_image()

            # Verify MODALKIT_CONFIG was included in environment
            env_call_args = mock_image.env.call_args[0][0]
            assert "MODALKIT_CONFIG" in env_call_args
            assert env_call_args["MODALKIT_CONFIG"] == "/custom/config.yaml"

    def test_config_generates_app_class_settings(self, basic_settings):
        """ModalConfig should generate complete Modal app class settings"""
        with patch.object(ModalConfig, "get_image") as mock_get_image, patch("modal.Secret.from_name") as mock_secret:
            mock_image_instance = MagicMock()
            mock_get_image.return_value = mock_image_instance
            mock_secret_instance = MagicMock()
            mock_secret.return_value = mock_secret_instance

            config = ModalConfig(basic_settings)
            app_settings = config.get_app_cls_settings()

            # Verify required settings are present
            assert "image" in app_settings
            assert "gpu" in app_settings
            assert "concurrency_limit" in app_settings
            assert "container_idle_timeout" in app_settings
            assert "retries" in app_settings

            # Verify values match configuration
            assert app_settings["gpu"] == "T4"
            assert app_settings["concurrency_limit"] == 5
            assert app_settings["container_idle_timeout"] == 300

    def test_config_filters_none_values_from_settings(self, basic_settings):
        """ModalConfig should exclude None values from generated settings"""
        # Set a field to None
        basic_settings.app_settings.deployment_config.gpu = None

        config = ModalConfig(basic_settings)
        app_settings = config.get_app_cls_settings()

        # None values should be filtered out
        assert "gpu" not in app_settings

    def test_config_generates_handler_settings(self, basic_settings):
        """ModalConfig should generate handler-specific settings"""
        with patch.object(ModalConfig, "get_image") as mock_get_image, patch("modal.Secret.from_name"):
            mock_image_instance = MagicMock()
            mock_get_image.return_value = mock_image_instance

            config = ModalConfig(basic_settings)
            handler_settings = config.get_handler_settings()

            # Verify handler-specific settings
            assert "image" in handler_settings
            assert "secrets" in handler_settings
            assert "mounts" in handler_settings
            assert "allow_concurrent_inputs" in handler_settings

    def test_config_generates_batch_settings(self, basic_settings):
        """ModalConfig should provide batch processing configuration"""
        config = ModalConfig(basic_settings)
        batch_settings = config.get_batched_method_settings()

        assert "max_batch_size" in batch_settings
        assert "wait_ms" in batch_settings
        assert batch_settings["max_batch_size"] == 10
        assert batch_settings["wait_ms"] == 100

    def test_config_generates_asgi_settings_for_secure_deployment(self, basic_settings):
        """ModalConfig should configure ASGI app for secure deployments"""
        basic_settings.app_settings.deployment_config.secure = True

        config = ModalConfig(basic_settings)
        asgi_settings = config.get_asgi_app_settings()

        assert asgi_settings["requires_proxy_auth"] is True

    def test_config_generates_asgi_settings_for_insecure_deployment(self, basic_settings):
        """ModalConfig should configure ASGI app for insecure deployments"""
        basic_settings.app_settings.deployment_config.secure = False

        config = ModalConfig(basic_settings)
        asgi_settings = config.get_asgi_app_settings()

        assert asgi_settings["requires_proxy_auth"] is False

    def test_config_provides_custom_api_key_dependency(self, basic_settings):
        """ModalConfig should provide FastAPI dependency for API key validation"""
        with patch("modalkit.auth.get_api_key", return_value="test-api-key"):
            config = ModalConfig(basic_settings)

            # Should not raise exceptions when accessing the dependency
            api_key_dep = config.CustomAPIKey
            assert api_key_dep is not None

    def test_config_reloads_volumes_safely(self, settings_with_storage):
        """ModalConfig should reload volumes with proper error handling"""
        with patch("modal.Volume.from_name") as mock_volume:
            mock_volume_instance = MagicMock()
            mock_volume.return_value = mock_volume_instance

            config = ModalConfig(settings_with_storage)

            # Initialize volumes first
            _ = config.volumes

            # Reload should work without errors
            config.reload_volumes()
            mock_volume_instance.reload.assert_called_once()

    def test_config_handles_volume_reload_errors_gracefully(self, settings_with_storage):
        """ModalConfig should handle volume reload errors without failing completely"""
        with patch("modal.Volume.from_name") as mock_volume:
            mock_volume_instance = MagicMock()
            mock_volume_instance.reload.side_effect = Exception("Reload failed")
            mock_volume.return_value = mock_volume_instance

            config = ModalConfig(settings_with_storage)

            # Initialize volumes first
            _ = config.volumes

            # Reload should not raise exception even if individual volumes fail
            config.reload_volumes()  # Should complete without raising

    def test_config_skips_reload_when_no_volumes_initialized(self, basic_settings):
        """ModalConfig should skip reload when no volumes have been initialized"""
        config = ModalConfig(basic_settings)

        # Should not raise any errors when no volumes exist
        config.reload_volumes()


class TestModalConfigurationIntegration:
    """Test suite for integration scenarios and edge cases"""

    def test_config_handles_complex_deployment_scenario(self, tmp_path):
        """ModalConfig should handle complex real-world deployment configurations"""
        # Complex configuration with multiple features
        cloud_mounts = [
            CloudBucketMount(
                mount_point="/training-data",
                bucket_name="ml-training-datasets",
                key_prefix="preprocessed/",
                read_only=True,
                secret="aws-readonly-credentials",
            ),
            CloudBucketMount(
                mount_point="/model-artifacts",
                bucket_name="ml-model-registry",
                bucket_endpoint_url="https://storage.googleapis.com",
                oidc_auth_role_arn="arn:aws:iam::123456789:role/ml-service-role",
            ),
        ]

        app_settings = AppSettings(
            app_prefix="production-ml-service",
            build_config=BuildConfig(
                image="gcr.io/company/ml-runtime",
                tag="v2.1.0",
                env={"ENVIRONMENT": "production", "LOG_LEVEL": "INFO"},
                extra_run_commands=["pip install --upgrade torch", "apt-get update"],
                workdir="/opt/ml",
            ),
            auth_config=AuthConfig(ssm_key="/production/ml-service/api-key", auth_header="X-ML-API-Key"),
            deployment_config=DeploymentConfig(
                gpu="A100",
                volumes={"/cache": "ml-cache-vol", "/var/storage": "temp-vol"},
                cloud_bucket_mounts=cloud_mounts,
                concurrency_limit=20,
                retries=3,
                secrets=["ml-service-secrets", "database-credentials"],
                container_idle_timeout=600,
                secure=True,
            ),
            batch_config=BatchConfig(max_batch_size=50, wait_ms=200),
        )

        model_settings = ModelSettings(
            model_entries={
                "text-classifier-v3": {"checkpoint": "latest"},
                "sentiment-analyzer-v2": {"checkpoint": "stable"},
            },
            local_model_repository_folder=str(tmp_path / "models"),
            common={"cache_size": 5000, "timeout": 120},
        )

        settings = Settings(app_settings=app_settings, model_settings=model_settings)

        with (
            patch("modal.Volume.from_name"),
            patch("modal.CloudBucketMount"),
            patch("modal.Secret.from_name"),
            patch("modal.Image.from_registry"),
        ):
            config = ModalConfig(settings)

            # Should handle all configuration aspects without errors
            assert config.app_name == "production-ml-service-dev"

            app_settings = config.get_app_cls_settings()
            assert app_settings["gpu"] == "A100"
            assert app_settings["concurrency_limit"] == 20
            assert len(app_settings["secrets"]) == 2

            batch_settings = config.get_batched_method_settings()
            assert batch_settings["max_batch_size"] == 50

            asgi_settings = config.get_asgi_app_settings()
            assert asgi_settings["requires_proxy_auth"] is True

    def test_config_maintains_consistency_across_multiple_accesses(self, tmp_path):
        """ModalConfig should maintain consistent state across multiple property accesses"""
        # Create settings with storage for this test
        cloud_mounts = [CloudBucketMount(mount_point="/data/s3", bucket_name="test-bucket", read_only=True)]

        app_settings = AppSettings(
            app_prefix="test-service",
            build_config=BuildConfig(image="test", tag="latest"),
            auth_config=AuthConfig(api_key="test-key", auth_header="Authorization"),
            deployment_config=DeploymentConfig(
                volumes={"/shared/models": "model-cache"}, cloud_bucket_mounts=cloud_mounts
            ),
            batch_config=BatchConfig(),
        )

        model_settings = ModelSettings(
            model_entries={}, local_model_repository_folder=str(tmp_path / "models"), common={}
        )

        settings_with_storage = Settings(app_settings=app_settings, model_settings=model_settings)

        with (
            patch("modal.Volume.from_name") as mock_volume,
            patch("modal.CloudBucketMount") as mock_cloud_mount,
            patch("modal.Secret.from_name"),
        ):
            mock_volume_instance = MagicMock()
            mock_volume.return_value = mock_volume_instance
            mock_mount_instance = MagicMock()
            mock_cloud_mount.return_value = mock_mount_instance

            config = ModalConfig(settings_with_storage)

            # Multiple accesses should return consistent results
            volumes1 = config.volumes
            volumes2 = config.volumes
            cloud_mounts1 = config.cloud_bucket_mounts
            cloud_mounts2 = config.cloud_bucket_mounts
            all_volumes1 = config.all_volumes
            all_volumes2 = config.all_volumes

            assert volumes1 is volumes2  # Should be the same cached instance
            assert cloud_mounts1 == cloud_mounts2
            assert all_volumes1 == all_volumes2

            # Underlying Modal calls should only happen once for volumes
            mock_volume.assert_called_once()
