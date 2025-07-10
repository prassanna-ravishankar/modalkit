import os

import pytest
from pydantic import ValidationError

from modalkit.settings import CloudBucketMount, Settings


@pytest.fixture
def set_config_path():
    os.environ["MODALKIT_CONFIG"] = "modalkit.yaml"
    yield
    del os.environ["MODALKIT_CONFIG"]


@pytest.fixture
def set_env_vars():
    os.environ["MODALKIT_CONFIG"] = "modalkit.yaml"
    os.environ["MODALKIT_APP_SETTINGS__APP_PREFIX"] = "test_prefix"
    yield
    del os.environ["MODALKIT_CONFIG"]
    del os.environ["MODALKIT_APP_SETTINGS__APP_PREFIX"]


def test_load_settings(set_config_path):
    settings = Settings()

    # Test that the settings are loaded correctly from the realistic example config
    assert settings.app_settings.app_prefix == "my-ml-service"
    assert settings.app_settings.auth_config.ssm_key == "/my-service/api-key"
    assert settings.app_settings.build_config.image == "python:3.11-slim"


def test_change_settings_file_location(monkeypatch, tmp_path):
    # Create a temporary YAML file with test settings
    test_yaml = tmp_path / "test_modalkit.yaml"
    test_yaml.write_text(
        """
app_settings:
  app_prefix: "new_prefix"
  auth_config:
    ssm_key: "/test/key"
    auth_header: "x-test-header"
  build_config:
    image: "test-image"
    tag: "test"
  deployment_config:
    region: "test"
    gpu: "test-gpu"
    volumes: {}
    concurrency_limit: 1
  batch_config:
    max_batch_size: 1
    wait_ms: 0
model_settings:
  local_model_repository_folder: "/tmp/test"
  model_entries: {}
  common: {}
"""
    )

    # Change the settings file location to our temporary file
    monkeypatch.setenv("MODALKIT_CONFIG", str(test_yaml))
    settings = Settings()

    # Test with the new file content
    assert settings.app_settings.app_prefix == "new_prefix"


def test_multiple_settings_file(monkeypatch, tmp_path):
    # Create a temporary YAML file with test settings
    main_yaml = tmp_path / "test_modalkit.yaml"
    main_yaml.write_text(
        """
app_settings:
  app_prefix: "new_prefix"
  auth_config:
    ssm_key: "/test/key"
    auth_header: "x-test-header"
  build_config:
    image: "test-image"
    tag: "test"
  deployment_config:
    region: "test"
    gpu: "test-gpu"
    volumes: {}
    concurrency_limit: 1
  batch_config:
    max_batch_size: 1
    wait_ms: 0
model_settings:
  local_model_repository_folder: "/tmp/test"
  model_entries: {}
  common: {}
"""
    )

    override_yaml = tmp_path / "override.yaml"
    override_yaml.write_text(
        """
app_settings:
  deployment_config:
    concurrency_limit: 2
"""
    )

    # Change the settings file location to our temporary file
    monkeypatch.setenv("MODALKIT_CONFIG", f"{main_yaml},{override_yaml}")
    settings = Settings()

    # Test with the new file content
    assert settings.app_settings.app_prefix == "new_prefix"
    assert settings.app_settings.deployment_config.concurrency_limit == 2


def test_env_overrides(set_env_vars):
    settings = Settings()

    # Test that the environment variable overrides the YAML file
    assert settings.app_settings.app_prefix == "test_prefix"


class TestCloudBucketMount:
    """Test cases for CloudBucketMount model validation"""

    def test_cloud_bucket_mount_minimal(self):
        """Test CloudBucketMount with minimal required fields"""
        mount = CloudBucketMount(mount_point="/mnt/bucket", bucket_name="my-bucket")

        assert mount.mount_point == "/mnt/bucket"
        assert mount.bucket_name == "my-bucket"
        assert mount.bucket_endpoint_url is None
        assert mount.key_prefix is None
        assert mount.secret is None
        assert mount.oidc_auth_role_arn is None
        assert mount.read_only is False
        assert mount.requester_pays is False

    def test_cloud_bucket_mount_s3_full(self):
        """Test CloudBucketMount with S3 configuration"""
        mount = CloudBucketMount(
            mount_point="/mnt/s3-bucket",
            bucket_name="my-s3-bucket",
            secret="aws-credentials",
            read_only=True,
            requester_pays=False,
        )

        assert mount.mount_point == "/mnt/s3-bucket"
        assert mount.bucket_name == "my-s3-bucket"
        assert mount.secret == "aws-credentials"
        assert mount.read_only is True
        assert mount.requester_pays is False

    def test_cloud_bucket_mount_gcs_full(self):
        """Test CloudBucketMount with Google Cloud Storage configuration"""
        mount = CloudBucketMount(
            mount_point="/mnt/gcs-bucket",
            bucket_name="my-gcs-bucket",
            bucket_endpoint_url="https://storage.googleapis.com",
            key_prefix="data/",
            secret="gcp-credentials",
            read_only=False,
        )

        assert mount.mount_point == "/mnt/gcs-bucket"
        assert mount.bucket_name == "my-gcs-bucket"
        assert mount.bucket_endpoint_url == "https://storage.googleapis.com"
        assert mount.key_prefix == "data/"
        assert mount.secret == "gcp-credentials"
        assert mount.read_only is False

    def test_cloud_bucket_mount_r2_with_oidc(self):
        """Test CloudBucketMount with Cloudflare R2 and OIDC authentication"""
        mount = CloudBucketMount(
            mount_point="/mnt/r2-bucket",
            bucket_name="my-r2-bucket",
            bucket_endpoint_url="https://account.r2.cloudflarestorage.com",
            oidc_auth_role_arn="arn:aws:iam::123456789:role/r2-role",
            requester_pays=True,
        )

        assert mount.mount_point == "/mnt/r2-bucket"
        assert mount.bucket_name == "my-r2-bucket"
        assert mount.bucket_endpoint_url == "https://account.r2.cloudflarestorage.com"
        assert mount.oidc_auth_role_arn == "arn:aws:iam::123456789:role/r2-role"
        assert mount.requester_pays is True
        assert mount.secret is None

    def test_cloud_bucket_mount_missing_required_fields(self):
        """Test CloudBucketMount validation with missing required fields"""
        with pytest.raises(ValidationError) as exc_info:
            CloudBucketMount(
                mount_point="/mnt/bucket"
                # Missing bucket_name
            )

        assert "bucket_name" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            CloudBucketMount(
                bucket_name="my-bucket"
                # Missing mount_point
            )

        assert "mount_point" in str(exc_info.value)
