"""
Every deployment_config example in the README and docs must validate against DeploymentConfig.

DeploymentConfig forbids unknown keys, so an example with a key it does not define
(e.g. a setting that was renamed or never existed) fails for anyone who copies it.
"""

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from modalkit.settings import DeploymentConfig

ROOT = Path(__file__).resolve().parent.parent
DOC_FILES = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
YAML_BLOCK = re.compile(r"```ya?ml\n(.*?)```", re.S)


def deployment_configs(node: Any) -> Iterator[dict[str, Any]]:
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "deployment_config" and isinstance(value, dict):
                yield value
            yield from deployment_configs(value)


def collect_examples() -> list[Any]:
    examples = []
    for doc in DOC_FILES:
        text = doc.read_text()
        for block in YAML_BLOCK.finditer(text):
            line = text[: block.start()].count("\n") + 2
            for config in deployment_configs(yaml.safe_load(block.group(1))):
                examples.append(pytest.param(config, id=f"{doc.relative_to(ROOT)}:{line}"))
    return examples


EXAMPLES = collect_examples()


def test_docs_contain_deployment_config_examples() -> None:
    assert EXAMPLES


@pytest.mark.parametrize("config", EXAMPLES)
def test_deployment_config_example_validates(config: dict[str, Any]) -> None:
    DeploymentConfig(**config)
