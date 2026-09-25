"""Wait for golden 5k capture, run strict no-worse parity, then audit evidence."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CAPTURE = ROOT / "data/reference/verbalizer_skravchenko_gemma3_news_golden5k.jsonl"
CAPTURE_PID = ROOT / "logs/reference-gemma3-news-golden5k.pid"
LOCK = ROOT / "logs/reference-gemma3-news-finalizer.lock"
STATUS = ROOT / "logs/reference-gemma3-news-finalizer-status.json"
EXPECTED_ROWS = 5_000


def line_count() -> int:
    if not CAPTURE.exists():
        return 0
    with CAPTURE.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def status(state: str, **extra: object) -> None:
    payload = {
        "state": state,
        "captured_rows": line_count(),
        "expected_rows": EXPECTED_ROWS,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **extra,
    }
    temporary = STATUS.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(STATUS)


def main() -> None:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("w", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        handle.write(str(os.getpid()) + "\n")
        handle.flush()
        while line_count() < EXPECTED_ROWS:
            capture_pid = int(CAPTURE_PID.read_text(encoding="utf-8").strip())
            if not alive(capture_pid):
                status("capture_stopped_early", capture_pid=capture_pid)
                raise RuntimeError("news reference capture stopped before 5,000 rows")
            status("waiting", capture_pid=capture_pid, finalizer_pid=os.getpid())
            time.sleep(30)
        if line_count() != EXPECTED_ROWS:
            raise RuntimeError("news reference capture row count exceeded 5,000")

        status("comparing", finalizer_pid=os.getpid())
        comparator = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/compare_reference_parity.py"),
                "--capture",
                str(CAPTURE),
                "--local-report",
                str(ROOT / "reports/e8-news-golden5k.json"),
                "--output-json",
                str(ROOT / "reports/e8-news-golden5k-reference-parity.json"),
                "--output-markdown",
                str(ROOT / "reports/e8-news-golden5k-reference-parity.md"),
                "--output-disagreements",
                str(ROOT / "reports/e8-news-golden5k-reference-disagreements.jsonl"),
                "--allow-nonexact-local",
                "--require-complete",
                "--require-gate-pass",
            ],
            cwd=ROOT,
            check=False,
        )
        audit_returncode: int | None = None
        if comparator.returncode == 0:
            audit_returncode = subprocess.run(
                [sys.executable, str(ROOT / "scripts/audit_news_reference_completion.py")],
                cwd=ROOT,
                check=False,
            ).returncode
        final_state = (
            "passed"
            if comparator.returncode == 0 and audit_returncode == 0
            else "failed"
        )
        status(
            final_state,
            comparator_returncode=comparator.returncode,
            audit_returncode=audit_returncode,
            finalizer_pid=os.getpid(),
        )
        if comparator.returncode or audit_returncode:
            raise SystemExit(comparator.returncode or audit_returncode or 1)


if __name__ == "__main__":
    main()
