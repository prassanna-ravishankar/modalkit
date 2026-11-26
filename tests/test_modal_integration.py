"""
Runs modalkit classes through Modal's real decorators, with no mocks.

Unit tests elsewhere mock Modal, so code that Modal rejects at decoration time still
passes them. Importing a module decorated with the real API fails the same way a user's
`modal deploy` would. Decoration is local: no Modal credentials or network are needed.
Modal only accepts classes and functions defined at module scope, so each case writes a
small module and imports it.
"""

import importlib.util
import sys
import textwrap
from pathlib import Path
from types import ModuleType

import pytest

from modalkit.cli import init_project
from modalkit.modal_service import ModalService


def import_module_from(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_modal_service_has_no_custom_constructor() -> None:
    """Modal constructs service instances itself, so ModalService must not define __init__"""
    assert ModalService.__init__ is object.__init__


def test_service_with_parameter_decorates_with_real_modal(tmp_path: Path) -> None:
    """The documented ModalService subclass with modal.parameter() must pass app.cls"""
    app_file = tmp_path / "param_app.py"
    app_file.write_text(
        textwrap.dedent(
            """
            import modal

            from modalkit.modal_service import ModalService

            app = modal.App("it-param")


            @app.cls()
            class ParamService(ModalService):
                model_name: str = modal.parameter(default="m")
            """
        )
    )

    module = import_module_from(app_file, "it_param_app")

    assert "ParamService" in module.app.registered_classes


FULL_DEPLOYMENT_CONFIG = """\
app_settings:
  app_prefix: "it-full"
  build_config:
    image: "python"
    tag: "3.12-slim"
  deployment_config:
    region: "us-east"
    gpu: "T4"
    volumes:
      /models: "it-models"
    cloud_bucket_mounts:
      - mount_point: "/bucket"
        bucket_name: "it-bucket"
        secret: "aws"
    max_containers: 3
    scaledown_window: 120
    max_concurrent_inputs: 2
    max_concurrent_inputs_handler: 8
    retries: 2
    secrets: ["it-secret"]
    secure: true
  batch_config:
    max_batch_size: 4
    wait_ms: 10
model_settings:
  local_model_repository_folder: "./models"
  common: {}
  model_entries:
    it_project_model: {}
"""


@pytest.mark.parametrize("config", ["scaffold", "full"])
def test_scaffolded_app_decorates_with_real_modal(config: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`modalkit init` output must import against the installed modal, for every deployment option"""
    init_project(tmp_path, "it-project")
    if config == "full":
        (tmp_path / "modalkit.yaml").write_text(FULL_DEPLOYMENT_CONFIG)
    monkeypatch.setenv("MODALKIT_CONFIG", str(tmp_path / "modalkit.yaml"))
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "model", raising=False)

    module = import_module_from(tmp_path / "app.py", f"it_project_app_{config}")

    assert "MyApp" in module.app.registered_classes
    assert "web_endpoints" in module.app.registered_functions
