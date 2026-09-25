from pathlib import Path

import pytest

from src.verbalizer.numeric import NUMERIC_POLICY_VERSION
from src.verbalizer.serving import validate_serving_manifest, write_serving_manifest


def _tokenizer_files(path: Path) -> None:
    for name in ("source.spm", "target.spm", "vocab.json", "tokenizer_config.json"):
        (path / name).write_text(name, encoding="utf-8")


def test_serving_manifest_rejects_missing_and_changed_policy_assets(tmp_path: Path) -> None:
    _tokenizer_files(tmp_path)
    with pytest.raises(ValueError, match="missing numeric_pipeline_manifest"):
        validate_serving_manifest(tmp_path)
    write_serving_manifest(tmp_path)
    assert validate_serving_manifest(tmp_path)["input_preprocessing"] == NUMERIC_POLICY_VERSION
    (tmp_path / "vocab.json").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="tokenizer_hash"):
        validate_serving_manifest(tmp_path)
