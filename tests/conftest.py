from pathlib import Path

import pytest

from modalkit.settings import Settings


@pytest.fixture
def sample_settings(tmp_path: Path):
    # Create a temporary settings file for testing using pytest's tmp_path fixture
    return Settings(
        app_settings={
            "app_prefix": "test-app",
            "build_config": {
                "image": "test-image",
                "tag": "latest",
                "env": {"TEST_ENV": "value"},
            },
            "auth_config": {"ssm_key": "test-ssm-key", "auth_header": "X-API-Key"},
            "deployment_config": {
                "region": None,
                "gpu": None,
                "volumes": None,
                "concurrency_limit": 10,
                "retries": 3,
                "secrets": ["test-secret"],
                "mounts": [],
                "container_idle_timeout": 300,
                "allow_concurrent_inputs": 5,
                "allow_concurrent_inputs_handler": 5,
            },
            "queue_config": {"backend": "sqs", "broker_url": "memory://"},
            "batch_config": {"max_batch_size": 1, "wait_ms": 0},
        },
        model_settings={
            "local_model_repository_folder": tmp_path / "models",
            "model_entries": {"test-model": {"param": "value"}},
            "common": {"common_param": "value"},
        },
    )
