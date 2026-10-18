"""Packaging: one version everywhere, and the files an installed package needs."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from importlib import resources
from pathlib import Path

import dji_thermal_cli as dtc

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def test_version_single_source():
    declared = tomllib.loads(PYPROJECT.read_text())["project"]["version"]
    assert dtc.__version__ == declared
    out = subprocess.run([sys.executable, "-m", "dji_thermal_cli", "--version"], capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.strip() == f"dji-thermal-cli {declared}"


def test_package_files():
    pkg = resources.files("dji_thermal_cli")
    assert pkg.joinpath("py.typed").is_file()
    assert pkg.joinpath("tsdk", "data", "dji_tsdk_v18.npz").is_file()
