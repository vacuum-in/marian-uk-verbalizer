from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.evaluate_marian import load_rows


def test_load_rows_adapts_inputs_labels_schema(tmp_path: Path) -> None:
    path = tmp_path / "golden.parquet"
    pq.write_table(
        pa.Table.from_pylist([{"inputs": "31 Липня 2026", "labels": "очікувано"}]),
        path,
    )
    assert load_rows(path, "inputs", "labels") == [
        {
            "id": "row-000000",
            "source": "31 Липня 2026",
            "target": "очікувано",
            "categories": [],
        }
    ]


def test_load_rows_supports_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "regressions.jsonl"
    path.write_text(
        '{"id":"terminal-d","source":"Термінал D","target":"Термінал де","categories":["letter_context"]}\n',
        encoding="utf-8",
    )
    assert load_rows(path, "source", "target") == [
        {
            "id": "terminal-d",
            "source": "Термінал D",
            "target": "Термінал де",
            "categories": ["letter_context"],
        }
    ]
