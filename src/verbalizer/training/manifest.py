"""Run-manifest collection for reproducible local training."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch


PACKAGES = ("torch", "transformers", "sentencepiece", "pyarrow", "PyYAML", "num2words")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def package_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for package in PACKAGES:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "missing"
    return result


def hardware_manifest() -> dict[str, Any]:
    cuda = torch.cuda.is_available()
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cuda_available": cuda,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory if cuda else None,
        "bf16_supported": torch.cuda.is_bf16_supported() if cuda else False,
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

