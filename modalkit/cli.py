"""Modalkit CLI — project scaffolding and utilities."""

import argparse
import sys
from pathlib import Path

MODALKIT_YAML = """\
app_settings:
  app_prefix: "{project_name}"

  build_config:
    image: "python:3.11-slim"
    tag: "latest"
    workdir: "/app"

  deployment_config:
    gpu: null
    concurrency_limit: 10
    container_idle_timeout: 300
    secure: false

  batch_config:
    max_batch_size: 8
    wait_ms: 50

model_settings:
  local_model_repository_folder: "./models"
  common:
    device: "cpu"
  model_entries:
    {model_name}:
      version: "1.0"
"""

MODEL_PY = """\
from typing import Any

from pydantic import BaseModel

from modalkit.inference_pipeline import InferencePipeline
from modalkit.iomodel import InferenceOutputModel


class PredictInput(BaseModel):
    text: str


class PredictOutput(InferenceOutputModel):
    result: str


class MyModel(InferencePipeline):
    def __init__(
        self,
        model_name: str,
        all_model_data_folder: str,
        common_settings: dict,
        **kwargs: Any,
    ):
        super().__init__(model_name, all_model_data_folder, common_settings)
        # Load your model artifacts here
        # e.g. self.model = torch.load(f"{{all_model_data_folder}}/model.pt")

    def preprocess(self, input_list: list[BaseModel]) -> dict[str, Any]:
        return {{"texts": [item.text for item in input_list]}}

    def predict(
        self, input_list: list[BaseModel], preprocessed_data: dict[str, Any]
    ) -> dict[str, Any]:
        # Replace with your model inference
        results = [text.upper() for text in preprocessed_data["texts"]]
        return {{"results": results}}

    def postprocess(
        self, input_list: list[BaseModel], raw_output: dict[str, Any]
    ) -> list[InferenceOutputModel]:
        return [
            PredictOutput(status="success", result=r)
            for r in raw_output["results"]
        ]
"""

APP_PY = """\
import modal

from modalkit.modal_config import ModalConfig
from modalkit.modal_service import ModalService, create_web_endpoints

from model import MyModel, PredictInput, PredictOutput

modal_config = ModalConfig()
app = modal.App(name=modal_config.app_name)


@app.cls(**modal_config.get_app_cls_settings())
class MyApp(ModalService):
    inference_implementation = MyModel
    model_name: str = modal.parameter(default="{model_name}")
    modal_utils: ModalConfig = modal_config


@app.function(**modal_config.get_handler_settings())
@modal.asgi_app(**modal_config.get_asgi_app_settings())
def web_endpoints():
    return create_web_endpoints(
        app_cls=MyApp,
        input_model=PredictInput,
        output_model=PredictOutput,
    )
"""


def init_project(target_dir: Path, project_name: str) -> None:
    """Scaffold a new modalkit project."""
    target_dir.mkdir(parents=True, exist_ok=True)
    model_name = project_name.replace("-", "_") + "_model"

    files = {
        "modalkit.yaml": MODALKIT_YAML.format(project_name=project_name, model_name=model_name),
        "model.py": MODEL_PY,
        "app.py": APP_PY.format(model_name=model_name),
    }

    created = []
    skipped = []
    for filename, content in files.items():
        filepath = target_dir / filename
        if filepath.exists():
            skipped.append(filename)
        else:
            filepath.write_text(content)
            created.append(filename)

    if created:
        print(f"Created: {', '.join(created)}")
    if skipped:
        print(f"Skipped (already exist): {', '.join(skipped)}")

    print("\nNext steps:")
    print(f"  cd {target_dir}")
    print("  modal serve app.py    # test locally")
    print("  modal deploy app.py   # deploy to Modal")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="modalkit", description="Modalkit CLI")
    subparsers = parser.add_subparsers(dest="command")

    init_parser = subparsers.add_parser("init", help="Scaffold a new modalkit project")
    init_parser.add_argument("name", nargs="?", default="my-ml-service", help="Project name (default: my-ml-service)")
    init_parser.add_argument("--dir", default=".", help="Target directory (default: current)")

    args = parser.parse_args(argv)

    if args.command == "init":
        init_project(Path(args.dir), args.name)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
