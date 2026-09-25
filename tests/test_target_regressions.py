import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_user_target_regressions_are_unique_and_fully_verbalized() -> None:
    path = ROOT / "data/challenge/user_target_regressions.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 4
    assert len({row["id"] for row in rows}) == len(rows)
    assert len({row["source"] for row in rows}) == len(rows)
    assert all(not any(character.isdigit() for character in row["target"]) for row in rows)
    assert rows[0]["target"].startswith("Координати точки: сорок сім цілих")
    assert rows[1]["target"] == "Потрібно прочитати двадцятий розділ до понеділка."
    assert rows[2]["target"].startswith(
        "Температура вночі від мінус вісім до плюс три"
    )
    assert rows[3]["target"].endswith(
        "одна з яких була вбита у дві тисячі двадцять четвертому році."
    )
