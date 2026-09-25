"""Wait for the active reference capture and run the strict final parity gate."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAPTURE = ROOT / "data/reference/verbalizer_skravchenko_gemma3_15k.jsonl"
DEFAULT_CAPTURE_PID = ROOT / "logs/reference-gemma3-15k.pid"
DEFAULT_LOCK = ROOT / "logs/reference-finalizer.lock"
DEFAULT_STATUS = ROOT / "logs/reference-finalizer-status.json"


def line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def write_status(path: Path, **values: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **values,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def monitor(args: argparse.Namespace) -> None:
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    with args.lock.open("w", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("a reference finalizer is already running") from exc
        lock_handle.write(str(os.getpid()) + "\n")
        lock_handle.flush()

        while True:
            captured = line_count(args.capture)
            if captured > args.expected_rows:
                raise RuntimeError(
                    f"capture has {captured} rows, expected at most {args.expected_rows}"
                )
            if captured == args.expected_rows:
                break
            capture_pid = int(args.capture_pid.read_text(encoding="utf-8").strip())
            if not process_alive(capture_pid):
                write_status(
                    args.status,
                    state="capture_stopped_early",
                    captured_rows=captured,
                    expected_rows=args.expected_rows,
                    capture_pid=capture_pid,
                )
                raise RuntimeError(
                    f"capture process {capture_pid} stopped at {captured}/{args.expected_rows}"
                )
            write_status(
                args.status,
                state="waiting",
                captured_rows=captured,
                expected_rows=args.expected_rows,
                capture_pid=capture_pid,
                finalizer_pid=os.getpid(),
            )
            time.sleep(args.poll_seconds)

        write_status(
            args.status,
            state="comparing",
            captured_rows=args.expected_rows,
            expected_rows=args.expected_rows,
            finalizer_pid=os.getpid(),
        )
        comparator_command = [
            sys.executable,
            str(ROOT / "scripts/compare_reference_parity.py"),
            "--require-complete",
            "--require-gate-pass",
        ]
        comparator = subprocess.run(comparator_command, cwd=ROOT, check=False)
        audit_returncode: int | None = None
        if comparator.returncode == 0:
            audit_command = [
                sys.executable,
                str(ROOT / "scripts/audit_external_parity_completion.py"),
            ]
            audit = subprocess.run(audit_command, cwd=ROOT, check=False)
            audit_returncode = audit.returncode
        state = (
            "passed"
            if comparator.returncode == 0 and audit_returncode == 0
            else "failed"
        )
        write_status(
            args.status,
            state=state,
            captured_rows=args.expected_rows,
            expected_rows=args.expected_rows,
            comparator_returncode=comparator.returncode,
            audit_returncode=audit_returncode,
            finalizer_pid=os.getpid(),
        )
        if comparator.returncode:
            raise SystemExit(comparator.returncode)
        if audit_returncode:
            raise SystemExit(audit_returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--capture-pid", type=Path, default=DEFAULT_CAPTURE_PID)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--expected-rows", type=int, default=15_000)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()
    if args.expected_rows < 1 or args.poll_seconds <= 0:
        parser.error("expected-rows and poll-seconds must be positive")
    return args


if __name__ == "__main__":
    monitor(parse_args())
