import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.materialize_numeric_training import materialize


def test_materializer_keeps_raw_formula_supervision_without_rule_quarantine(tmp_path: Path) -> None:
    source_path = tmp_path / "input.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "id": "good",
                    "source": "Формула H₂SO₄.",
                    "target": "Формула аш два ес о чотири.",
                    "categories": ["formula"],
                },
                {
                    "id": "bad",
                    "source": "Формула H₂SO₄.",
                    "target": "Формула два ес о два.",
                    "categories": ["formula"],
                },
            ]
        ),
        source_path,
    )
    output = tmp_path / "output.parquet"
    quarantine = tmp_path / "quarantine.jsonl"
    report = materialize(source_path, output, quarantine)
    rows = pq.read_table(output).to_pylist()
    assert report["output"]["rows"] == 2
    assert [row["id"] for row in rows] == ["good", "bad"]
    assert all("₂" in row["source"] for row in rows)
    assert all("англомаркер" not in row["source"] for row in rows)
    assert quarantine.read_text(encoding="utf-8") == ""
