"""Serving-manifest creation and compatibility checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .numeric import (
    NUMERIC_POLICY_VERSION,
    numeric_component_hashes,
    numeric_policy_hash,
    preprocessing_hash,
)


SERVING_MANIFEST = "numeric_pipeline_manifest.json"
TOKENIZER_FILES = ("source.spm", "target.spm", "vocab.json", "tokenizer_config.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tokenizer_hash(path: Path) -> str:
    values = [(name, _sha256(path / name)) for name in TOKENIZER_FILES if (path / name).is_file()]
    if not values:
        raise ValueError(f"no tokenizer files found in {path}")
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode("utf-8")).hexdigest()


def serving_manifest(path: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "numeric_policy_version": NUMERIC_POLICY_VERSION,
        "numeric_policy_hash": numeric_policy_hash(),
        "preprocessing_hash": preprocessing_hash(),
        "component_hashes": numeric_component_hashes(),
        "tokenizer_hash": tokenizer_hash(path),
        "input_preprocessing": NUMERIC_POLICY_VERSION,
    }


def write_serving_manifest(path: Path) -> Path:
    output = path / SERVING_MANIFEST
    output.write_text(
        json.dumps(serving_manifest(path), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def validate_serving_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path / SERVING_MANIFEST
    if not manifest_path.is_file():
        raise ValueError(f"incompatible checkpoint: missing {SERVING_MANIFEST}")
    actual = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = serving_manifest(path)
    mismatches = [
        key
        for key in ("numeric_policy_version", "numeric_policy_hash", "preprocessing_hash", "component_hashes", "tokenizer_hash", "input_preprocessing")
        if actual.get(key) != expected[key]
    ]
    if mismatches:
        raise ValueError(f"incompatible checkpoint manifest fields: {', '.join(mismatches)}")
    return actual
