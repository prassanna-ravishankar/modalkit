from pathlib import Path

import pytest

from modalkit.settings import (
    AppSettings,
    BatchConfig,
    BuildConfig,
    DeploymentConfig,
    ModelSettings,
    QueueConfig,
    Settings,
)


@pytest.fixture
def sample_settings(tmp_path: Path) -> Settings:
    # Create a temporary settings file for testing using pytest's tmp_path fixture
    build_config = BuildConfig(
        image="test-image",
        tag="latest",
        env={"TEST_ENV": "value"},
    )

    deployment_config = DeploymentConfig(
        region=None,
        gpu=None,
        volumes=None,
        concurrency_limit=10,
        retries=3,
        secrets=["test-secret"],
        mounts=[],
        container_idle_timeout=300,
        allow_concurrent_inputs=5,
        allow_concurrent_inputs_handler=5,
    )

    queue_config = QueueConfig(backend="sqs", broker_url="memory://")
    batch_config = BatchConfig(max_batch_size=1, wait_ms=0)

    app_settings = AppSettings(
        app_prefix="test-app",
        build_config=build_config,
        deployment_config=deployment_config,
        queue_config=queue_config,
        batch_config=batch_config,
    )

    model_settings = ModelSettings(
        local_model_repository_folder=tmp_path / "models",
        model_entries={"test-model": {"param": "value"}},
        common={"common_param": "value"},
    )

    return Settings(
        app_settings=app_settings,
        model_settings=model_settings,
    )
