"""Every module has exactly one importable name.

The router core used to be importable both as bare `schemas` (via sys.path) and as
`code.schemas`, so each pydantic model existed twice and isinstance checks across
the two halves silently failed. These tests fail if either alias comes back.
"""

import importlib
import pkgutil
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

import notification_router

ROOT = Path(__file__).resolve().parent.parent

CORE = [
    "agent_tools", "config", "context_builder", "data_store", "deciders", "embeddings",
    "finalize", "media_analysis", "pipeline", "retrieval", "router", "routing",
    "safety_gate", "schemas", "trace",
]  # fmt: skip


def _all_modules() -> list[str]:
    return [
        m.name
        for m in pkgutil.walk_packages(notification_router.__path__, "notification_router.")
        if m.name != "notification_router.__main__"
    ]


def _assert_no_aliases() -> None:
    aliases = [m for m in sys.modules if m in CORE or m == "app" or m.startswith(("code.", "app."))]
    assert not aliases, f"modules loaded under a second name: {aliases}"


@pytest.mark.parametrize("name", _all_modules())
def test_module_imports(name: str) -> None:
    importlib.import_module(name)
    _assert_no_aliases()


@pytest.mark.parametrize("script", ["route_one.py", "compare_deciders.py"])
def test_script_imports(script: str) -> None:
    runpy.run_path(str(ROOT / "scripts" / script), run_name="not_main")
    _assert_no_aliases()


def test_one_class_per_model() -> None:
    from notification_router.data_store import DataStore
    from notification_router.schemas import IncomingMessage

    assert DataStore._to_incoming.__globals__["IncomingMessage"] is IncomingMessage


def test_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "notification_router", "--help"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert "route" in result.stdout
