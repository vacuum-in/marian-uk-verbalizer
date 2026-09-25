import json
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]


def test_news_replay_is_balanced_and_excludes_final_evaluation() -> None:
    output = ROOT / "data/news_domain_replay"
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    rows = manifest["rows"]
    assert rows["train_news"] == 89_996
    assert rows["train_structured_replay"] == rows["train_news"]
    assert rows["validation_news"] == 5_000
    assert rows["validation_structured_replay"] == 5_000
    assert rows["train"] == 179_992
    assert rows["validation"] == 10_000
    assert all(value == 0 for value in manifest["leakage"].values())
    for split in ("train", "validation"):
        assert pq.read_metadata(output / f"{split}.parquet").num_rows == rows[split]
