from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PATH = ROOT / "examples" / "operator-usage" / "example.py"


def _load_example():
    spec = importlib.util.spec_from_file_location(
        "operator_usage_example", EXAMPLE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_operator_usage_example_imports_without_hardware_runtime():
    module = _load_example()

    assert module.ROOT == ROOT
    assert callable(module.main)
    assert callable(module._run_mapping)
    assert callable(module._spawn)


def test_operator_usage_example_help_signature_is_available():
    result = subprocess.run(
        [sys.executable, str(EXAMPLE_PATH), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    help_text = result.stdout
    assert "--mapping" in help_text
    assert "--device" in help_text
    assert "a1-mixed" in help_text
