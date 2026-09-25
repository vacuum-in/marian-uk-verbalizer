"""LLM judge for verbalizer output (offline evaluation only).

Credentials come from the environment; nothing secret is printed or written.
Verdicts are cached in reports/judge_cache.jsonl keyed by rubric, model, source
and prediction, so re-scoring a checkpoint only pays for new outputs.

  calibrate: compare each judge model with the in-session verdicts in
             reports/hf-uktn-e40-judged.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUBRIC_PATH = ROOT / "configs/evaluate/judge_rubric.md"
# gpt-5.4's deployment allows 1,000 tokens a minute; the full rubric alone exceeds that.
RUBRIC_OVERRIDE = {"gpt-5.4": ROOT / "configs/evaluate/judge_rubric_compact.md"}
CACHE_PATH = ROOT / "reports/judge_cache.jsonl"

MODELS = {
    "deepseek-v4-pro": ("AZURE_FOUNDRY_OPENAI_BASE_URL", "AZURE_FOUNDRY_OPENAI_API_KEY", "AZURE_FOUNDRY_OPENAI_MODEL"),
    "deepseek-v3.2": ("AZURE_OPENAI_ENDPOINT_VERIFY", "AZURE_OPENAI_API_KEY_VERIFY", "AZURE_OPENAI_DEPLOYMENT_VERIFY"),
    "gpt-5.4": ("AZURE_OPENAI_ENDPOINT_INITAITESTING", "AZURE_OPENAI_API_KEY_INITAITESTING", "AZURE_OPENAI_DEPLOYMENT_INITAITESTING"),
}
VERDICTS = ("ok", "minor", "wrong")
_cache_lock = threading.Lock()
MIN_INTERVAL: dict[str, float] = {}
MAX_COMPLETION_TOKENS: dict[str, int] = {"gpt-5.4": 600}
_pace_lock = threading.Lock()
_last_call: dict[str, float] = {}


def pace(model: str) -> None:
    """Keep at least MIN_INTERVAL[model] seconds between request starts."""

    interval = MIN_INTERVAL.get(model, 0.0)
    if not interval:
        return
    with _pace_lock:
        wait = _last_call.get(model, 0.0) + interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_call[model] = time.monotonic()


def rubric(model: str | None = None) -> str:
    return RUBRIC_OVERRIDE.get(model, RUBRIC_PATH).read_text(encoding="utf-8")


def rubric_version(model: str | None = None) -> str:
    return hashlib.sha256(rubric(model).encode()).hexdigest()[:12]


def cache_key(model: str, source: str, prediction: str) -> str:
    payload = json.dumps([rubric_version(model), model, source, prediction], ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def load_cache() -> dict[str, dict]:
    if not CACHE_PATH.exists():
        return {}
    cache = {}
    for line in CACHE_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            cache[row["key"]] = row
    return cache


WRONG_ISSUES = {"value", "unread", "misread", "dropped", "added", "damaged"}
MINOR_ISSUES = {"ending", "style"}


def derive_verdict(checks: list[dict]) -> str:
    """The verdict follows from the issue codes, not from the model's own label."""

    issues = {str(check.get("issue", "")).strip().lower() for check in checks if isinstance(check, dict)}
    if issues & WRONG_ISSUES:
        return "wrong"
    if issues & MINOR_ISSUES:
        return "minor"
    return "ok"


def parse_verdict(text: str) -> tuple[str | None, str, list[dict]]:
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict) and isinstance(data.get("checks"), list):
            return derive_verdict(data["checks"]), str(data.get("reason", ""))[:300], data["checks"][:20]
        if isinstance(data, dict) and str(data.get("verdict", "")).strip().lower() in VERDICTS:
            return str(data["verdict"]).strip().lower(), str(data.get("reason", ""))[:300], []
    return None, text.strip()[:300], []


def call_model(model: str, source: str, reference: str, prediction: str) -> dict:
    url_var, key_var, deployment_var = MODELS[model]
    user = f"SOURCE: {source}\nOUTPUT: {prediction}\nREFERENCE: {reference}"
    body = json.dumps({
        "model": os.environ[deployment_var],
        "messages": [{"role": "system", "content": rubric(model)}, {"role": "user", "content": user}],
        "max_completion_tokens": MAX_COMPLETION_TOKENS.get(model, 4000),
    }).encode()
    request = urllib.request.Request(
        os.environ[url_var].rstrip("/") + "/chat/completions", data=body,
        headers={"Content-Type": "application/json", "api-key": os.environ[key_var]},
    )
    attempts = 10
    for attempt in range(attempts):
        pace(model)
        try:
            response = json.load(urllib.request.urlopen(request, timeout=180))
            text = response["choices"][0]["message"].get("content") or ""
            verdict, reason, checks = parse_verdict(text)
            return {"verdict": verdict, "reason": reason, "checks": checks,
                    "tokens": response.get("usage", {}).get("total_tokens")}
        except urllib.error.HTTPError as error:
            if MIN_INTERVAL.get(model):
                remaining = error.headers.get("x-ratelimit-remaining-tokens") if error.headers else None
                print(f"  {model}: attempt {attempt + 1} HTTP {error.code}, tokens left {remaining}",
                      file=sys.stderr, flush=True)
            if error.code in (429, 500, 502, 503, 504) and attempt < attempts - 1:
                retry_after = error.headers.get("Retry-After") if error.headers else None
                wait = float(retry_after) if retry_after and retry_after.replace(".", "").isdigit() else min(60, 2 ** attempt)
                time.sleep(wait + random.random())
                continue
            return {"verdict": None, "reason": f"HTTP {error.code}", "tokens": None}
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt < attempts - 1:
                time.sleep(min(60, 2 ** attempt))
                continue
            return {"verdict": None, "reason": type(error).__name__, "tokens": None}
    return {"verdict": None, "reason": "retries exhausted", "tokens": None}


def judge_rows(model: str, rows: list[dict], workers: int) -> dict[str, dict]:
    """Return {row id: verdict record}, using and extending the cache."""

    cache = load_cache()
    results: dict[str, dict] = {}
    todo = []
    for row in rows:
        key = cache_key(model, row["source"], row["prediction"])
        if key in cache and cache[key]["verdict"]:
            results[row["id"]] = cache[key]
        else:
            todo.append((row, key))

    def work(item):
        row, key = item
        record = {"key": key, "model": model, "rubric": rubric_version(model),
                  "source": row["source"], "prediction": row["prediction"],
                  **call_model(model, row["source"], row["reference"], row["prediction"])}
        if record["verdict"]:
            with _cache_lock, CACHE_PATH.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        else:
            print(f"  {model}: no verdict for {row['id']}: {record['reason'][:120]}", file=sys.stderr, flush=True)
        return row["id"], record

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done, (row_id, record) in enumerate(pool.map(work, todo), 1):
            results[row_id] = record
            if done % 50 == 0:
                print(f"  {model}: {done}/{len(todo)}", file=sys.stderr, flush=True)
    return results


def binary(verdict: str | None) -> str | None:
    return None if verdict is None else ("wrong" if verdict == "wrong" else "correct")


def agreement(gold: dict[str, str], predicted: dict[str, str | None]) -> dict:
    ids = [i for i in gold if predicted.get(i)]
    three = sum(gold[i] == predicted[i] for i in ids)
    two = sum(binary(gold[i]) == binary(predicted[i]) for i in ids)
    confusion = Counter((binary(gold[i]), binary(predicted[i])) for i in ids)
    gold_wrong = confusion[("wrong", "wrong")] + confusion[("wrong", "correct")]
    said_wrong = confusion[("wrong", "wrong")] + confusion[("correct", "wrong")]
    return {
        "rows": len(ids), "missing": len(gold) - len(ids),
        "binary_agreement_pct": round(100 * two / len(ids), 2) if ids else 0,
        "three_class_agreement_pct": round(100 * three / len(ids), 2) if ids else 0,
        "wrong_recall_pct": round(100 * confusion[("wrong", "wrong")] / gold_wrong, 2) if gold_wrong else None,
        "wrong_precision_pct": round(100 * confusion[("wrong", "wrong")] / said_wrong, 2) if said_wrong else None,
        "correct_flagged_wrong": confusion[("correct", "wrong")],
        "wrong_passed_as_correct": confusion[("wrong", "correct")],
    }


def calibration_rows(judged: Path, predictions: Path, exact_sample: int, seed: int) -> tuple[list[dict], dict[str, str]]:
    verdicts = json.loads(judged.read_text(encoding="utf-8"))["verdicts"]
    examples = {e["id"]: e for e in json.loads(predictions.read_text(encoding="utf-8"))["examples"]}
    loose = lambda s: re.sub(r"\s+", " ", re.sub(r"[^\w\s']", "", s.replace("’", "'").casefold())).strip()
    misses = [i for i, e in examples.items() if loose(e["prediction"]) != loose(e["reference"])]
    matches = [i for i, e in examples.items() if loose(e["prediction"]) == loose(e["reference"])]
    chosen = misses + random.Random(seed).sample(matches, min(exact_sample, len(matches)))
    rows = [{"id": i, "source": examples[i]["source"], "reference": examples[i]["reference"],
             "prediction": examples[i]["prediction"]} for i in chosen]
    return rows, {i: verdicts[i]["verdict"] for i in chosen}


def calibrate(args: argparse.Namespace) -> None:
    rows, gold = calibration_rows(args.judged, args.predictions, args.exact_sample, args.seed)
    if args.limit:
        rows = random.Random(args.seed).sample(rows, args.limit)
        gold = {row["id"]: gold[row["id"]] for row in rows}
    if args.priority:
        import csv
        with args.priority.open(encoding="utf-8-sig") as handle:
            first = {row["id"] for row in csv.DictReader(handle)}
        rows.sort(key=lambda row: row["id"] not in first)
    print(f"calibration rows: {len(rows)}  gold: {dict(Counter(gold.values()))}", file=sys.stderr)
    by_model = {}
    for model in args.models:
        workers = args.gpt_workers if model == "gpt-5.4" else args.workers
        by_model[model] = judge_rows(model, rows, workers)
    verdicts = {model: {i: r["verdict"] for i, r in res.items()} for model, res in by_model.items()}
    report = {"rubric": rubric_version(), "rows": len(rows), "gold": dict(Counter(gold.values())),
              "models": {m: agreement(gold, v) for m, v in verdicts.items()}}
    if len(args.models) == 3:
        a, b, c = args.models
        majority = {}
        cascade = {}
        for i in gold:
            votes = [binary(verdicts[m].get(i)) for m in args.models]
            if None in votes:
                continue
            top = Counter(votes).most_common(1)[0][0]
            majority[i] = "wrong" if top == "wrong" else "ok"
            first, second = binary(verdicts[a][i]), binary(verdicts[b][i])
            final = first if first == second else binary(verdicts[c][i])
            cascade[i] = "wrong" if final == "wrong" else "ok"
        report["majority_of_3"] = agreement(gold, majority)
        report[f"cascade_{a}+{b}->{c}"] = agreement(gold, cascade)
    disagreements = []
    for row in rows:
        votes = {m: verdicts[m].get(row["id"]) for m in args.models}
        if any(v and binary(v) != binary(gold[row["id"]]) for v in votes.values()):
            disagreements.append({**row, "gold": gold[row["id"]], "votes": votes,
                                  "reasons": {m: by_model[m][row["id"]]["reason"] for m in args.models}})
    report["disagreement_rows"] = len(disagreements)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with args.output.with_name(args.output.stem + "-disagreements.jsonl").open("w", encoding="utf-8") as handle:
        for row in disagreements:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=1))


def cached_votes(rows: list[dict], models: list[str], rubric_versions: list[str]) -> dict[str, dict[str, dict]]:
    """Latest cached verdict per row and model for any of the given rubric versions."""

    by_key: dict[tuple, dict] = {}
    if CACHE_PATH.exists():
        for line in CACHE_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                if record["rubric"] in rubric_versions and record.get("verdict"):
                    by_key[(record["model"], record["source"], record["prediction"])] = record
    return {row["id"]: {m: by_key[(m, row["source"], row["prediction"])] for m in models
                        if (m, row["source"], row["prediction"]) in by_key} for row in rows}


def consensus(args: argparse.Namespace) -> None:
    """Accept rows where every vote agrees on correct vs wrong; queue the rest for review."""

    examples = json.loads(args.predictions.read_text(encoding="utf-8"))["examples"]
    human = json.loads(args.human.read_text(encoding="utf-8"))["verdicts"] if args.human else {}
    loose = lambda s: re.sub(r"\s+", " ", re.sub(r"[^\w\s']", "", s.replace("’", "'").casefold())).strip()
    rows = [{"id": e["id"], "source": e["source"], "reference": e["reference"],
             "prediction": e["prediction"], "categories": e.get("categories", [])} for e in examples]
    votes = cached_votes(rows, args.models, args.rubrics)
    accepted, review = [], []
    for row in rows:
        if loose(row["prediction"]) == loose(row["reference"]):
            accepted.append({**row, "label": "correct", "basis": "matches_reference"})
            continue
        ballot = {m: record["verdict"] for m, record in votes[row["id"]].items()}
        if row["id"] in human:
            ballot["human"] = human[row["id"]]["verdict"]
        required = set(args.required) | ({"human"} if human else set())
        labels = {binary(v) for v in ballot.values()}
        if required <= set(ballot) and len(labels) == 1:
            accepted.append({**row, "label": labels.pop(), "basis": "unanimous", "votes": ballot})
        else:
            review.append({**row, "votes": ballot,
                           "reasons": {m: r.get("reason", "") for m, r in votes[row["id"]].items()}})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "accepted.jsonl").open("w", encoding="utf-8") as handle:
        for row in accepted:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    import csv
    with (args.output_dir / "manual_review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "decision (ok/minor/wrong)", "source", "output", "reference",
                         *[f"vote_{m}" for m in ["human", *args.models]], *[f"reason_{m}" for m in args.models]])
        for row in review:
            writer.writerow([row["id"], "", row["source"], row["prediction"], row["reference"],
                             *[row["votes"].get(m, "") for m in ["human", *args.models]],
                             *[row["reasons"].get(m, "") for m in args.models]])
    summary = {"rows": len(rows), "accepted": len(accepted), "manual_review": len(review),
               "accepted_by_basis": dict(Counter(r["basis"] for r in accepted)),
               "accepted_labels": dict(Counter(r["label"] for r in accepted))}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))


def judge_new(args: argparse.Namespace) -> None:
    """Judge a checkpoint's benchmark outputs that no earlier verdict covers.

    Rows matching the reference are correct; rows whose output equals the
    prior checkpoint's keep the prior final verdict; the rest are judged by
    every model and split into accepted (unanimous) and manual review.
    """

    examples = json.loads(args.predictions.read_text(encoding="utf-8"))["examples"]
    prior = json.loads(args.prior.read_text(encoding="utf-8"))["verdicts"]
    loose = lambda s: re.sub(r"\s+", " ", re.sub(r"[^\w\s']", "", s.replace("’", "'").casefold())).strip()
    verdicts, todo = {}, []
    for e in examples:
        row = {"id": e["id"], "source": e["source"], "reference": e["reference"], "prediction": e["prediction"]}
        old = prior.get(e["id"])
        if old and old.get("excluded"):
            verdicts[e["id"]] = {"label": old["label"], "basis": "excluded", "excluded": True}
        elif loose(e["prediction"]) == loose(e["reference"]):
            verdicts[e["id"]] = {"label": "ok", "basis": "matches_reference"}
        elif old and old["prediction"] == e["prediction"]:
            verdicts[e["id"]] = {"label": old["label"], "basis": "prior_" + old["basis"]}
        else:
            todo.append(row)
    print(f"to judge: {len(todo)}", file=sys.stderr)
    votes = {m: judge_rows(m, todo, args.gpt_workers if m == "gpt-5.4" else args.workers) for m in args.models}
    review = []
    for row in todo:
        ballot = {m: votes[m].get(row["id"], {}).get("verdict") for m in args.models}
        labels = {binary(v) for v in ballot.values()}
        if None not in ballot.values() and len(labels) == 1:
            three = [v for v in ballot.values()]
            label = "wrong" if labels == {"wrong"} else ("minor" if "minor" in three else "ok")
            verdicts[row["id"]] = {"label": label, "basis": "unanimous", "votes": ballot}
        else:
            verdicts[row["id"]] = {"label": "review", "basis": "manual_review", "votes": ballot}
            review.append({**row, "votes": ballot,
                           "reasons": {m: votes[m].get(row["id"], {}).get("reason", "") for m in args.models}})
    for e in examples:
        verdicts[e["id"]].update({"source": e["source"], "prediction": e["prediction"], "reference": e["reference"]})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "verdicts.json").write_text(json.dumps({"verdicts": verdicts}, ensure_ascii=False, indent=1), encoding="utf-8")
    with (args.output_dir / "manual_review.jsonl").open("w", encoding="utf-8") as handle:
        for row in review:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": len(verdicts), "judged": len(todo), "manual_review": len(review),
                      "basis": dict(Counter(v["basis"] for v in verdicts.values()))}, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    cal = sub.add_parser("calibrate")
    cal.add_argument("--judged", type=Path, default=ROOT / "reports/hf-uktn-e40-judged.json")
    cal.add_argument("--predictions", type=Path, default=ROOT / "reports/hf-uktn-e40-full-raw-marian.json")
    cal.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    cal.add_argument("--exact-sample", type=int, default=200)
    cal.add_argument("--seed", type=int, default=20260923)
    cal.add_argument("--workers", type=int, default=12)
    cal.add_argument("--gpt-workers", type=int, default=1)
    cal.add_argument("--gpt-interval", type=float, default=12.0,
                     help="minimum seconds between gpt-5.4 requests (its deployment quota is small)")
    cal.add_argument("--limit", type=int, default=0)
    cal.add_argument("--priority", type=Path, default=None, help="CSV with an id column to judge first")
    cal.add_argument("--output", type=Path, default=ROOT / "reports/judge-calibration-e40.json")
    con = sub.add_parser("consensus")
    con.add_argument("--predictions", type=Path, default=ROOT / "reports/hf-uktn-e40-full-raw-marian.json")
    con.add_argument("--human", type=Path, default=None, help="in-session verdicts to count as a vote")
    con.add_argument("--models", nargs="+", default=["deepseek-v4-pro", "gpt-5.4"], choices=list(MODELS),
                     help="voters; deepseek-v3.2 is excluded by default (65%% agreement, mostly false 'wrong')")
    con.add_argument("--required", nargs="*", default=["deepseek-v4-pro", "gpt-5.4"],
                     help="models whose vote must be present for a row to be accepted")
    con.add_argument("--rubrics", nargs="+", default=sorted({rubric_version(m) for m in MODELS}))
    con.add_argument("--output-dir", type=Path, required=True)
    new = sub.add_parser("judge-new")
    new.add_argument("--predictions", type=Path, required=True)
    new.add_argument("--prior", type=Path, required=True, help="final verdicts of the previous checkpoint")
    new.add_argument("--models", nargs="+", default=["deepseek-v4-pro", "gpt-5.4"], choices=list(MODELS))
    new.add_argument("--workers", type=int, default=5)
    new.add_argument("--gpt-workers", type=int, default=2)
    new.add_argument("--gpt-interval", type=float, default=3.5)
    new.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "judge-new":
        MIN_INTERVAL["gpt-5.4"] = args.gpt_interval
        judge_new(args)
    if args.command == "calibrate":
        MIN_INTERVAL["gpt-5.4"] = args.gpt_interval
        calibrate(args)
    elif args.command == "consensus":
        consensus(args)


if __name__ == "__main__":
    main()
