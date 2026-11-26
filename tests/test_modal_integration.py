"""
Runs modalkit classes through Modal's real decorators, with no mocks.

Unit tests elsewhere mock Modal, so code that Modal rejects at decoration time still
passes them. Importing a module decorated with the real API fails the same way a user's
`modal deploy` would. Decoration is local: no Modal credentials or network are needed.
Modal only accepts classes and functions defined at module scope, so each case writes a
small module and imports it.
"""

import importlib.util
import textwrap
from pathlib import Path
from types import ModuleType

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
