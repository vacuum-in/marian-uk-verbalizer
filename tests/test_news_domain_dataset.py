import json
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]


def test_news_domain_manifest_proves_disjoint_official_test() -> None:
    output = ROOT / "data/news_domain"
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["sources"]["main_train_95k"]["rows"] == 95_000
    assert manifest["sources"]["golden_5k"]["rows"] == 5_000
    assert manifest["rows"] == {"test": 5_000, "train": 89_996, "validation": 5_000}
    assert manifest["drops"]["conflicting_duplicate_source_rows"] == 4
    assert all(value == 0 for value in manifest["leakage"].values())
    for split, rows in manifest["rows"].items():
        path = output / f"{split}.parquet"
        assert pq.read_metadata(path).num_rows == rows
        assert set(pq.read_schema(path).names) >= {"id", "source", "target"}
