from __future__ import annotations

from pathlib import Path
import tomllib

import jp_learning_platform
from jp_learning_platform.__main__ import main


def test_package_exposes_release_version() -> None:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    assert pyproject["project"]["version"] == "1.0.0"
    assert jp_learning_platform.__version__ == pyproject["project"]["version"]


def test_module_entrypoint_returns_success() -> None:
    assert main() == 0
