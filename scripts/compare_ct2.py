"""Compare a CTranslate2 export with its PyTorch checkpoint on the same evaluations.

Scores the frozen benchmark (exact match, like evaluate_marian.py) and the
generated domain sets (domain_check), and reports throughput. Every input is
sent to the model (no routing), matching the PyTorch reports it compares to.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.verbalizer.evaluation.domain_check import score_row  # noqa: E402
from src.verbalizer.evaluation.metrics import normalize_evaluation_text  # noqa: E402


def translate(model_dir: Path, sources: list[str], device: str, compute_type: str, threads: int, batch: int) -> tuple[list[str], float]:
    import ctranslate2
    from transformers import MarianTokenizer

    tokenizer = MarianTokenizer.from_pretrained(model_dir)
    translator = ctranslate2.Translator(str(model_dir), device=device, compute_type=compute_type,
                                        inter_threads=1, intra_threads=threads)
    token_lists = [tokenizer.convert_ids_to_tokens(tokenizer(s, truncation=True, max_length=384)["input_ids"]) for s in sources]
    order = sorted(range(len(sources)), key=lambda i: len(token_lists[i]))
    outputs: list[str] = [""] * len(sources)
    started = time.monotonic()
    for start in range(0, len(order), batch):
        indices = order[start:start + batch]
        longest = max(len(token_lists[i]) for i in indices)
        results = translator.translate_batch([token_lists[i] for i in indices], beam_size=1,
                                             max_decoding_length=min(384, max(128, longest * 3)),
                                             suppress_sequences=[["<s>"]])
        for i, result in zip(indices, results):
            ids = tokenizer.convert_tokens_to_ids(result.hypotheses[0])
            outputs[i] = tokenizer.decode(ids, skip_special_tokens=True)
    return outputs, time.monotonic() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--pytorch-benchmark", type=Path, required=True, help="evaluate_marian.py report of the source checkpoint")
    parser.add_argument("--pytorch-domains", type=Path, required=True, help="evaluate_domains.py report of the source checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    bench = json.loads(args.pytorch_benchmark.read_text(encoding="utf-8"))["examples"]
    domain_rows = [json.loads(line) for path in sorted((ROOT / "reports/domain-eval").glob("*.jsonl"))
                   for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    sources = [e["source"] for e in bench] + [r["source"] for r in domain_rows]
    outputs, seconds = translate(args.model, sources, args.device, args.compute_type, args.threads, args.batch)
    bench_out, domain_out = outputs[:len(bench)], outputs[len(bench):]

    n = normalize_evaluation_text
    ct2_exact = sum(n(o) == n(e["reference"]) for o, e in zip(bench_out, bench))
    pt_exact = sum(n(e["prediction"]) == n(e["reference"]) for e in bench)
    same_as_pt = sum(n(o) == n(e["prediction"]) for o, e in zip(bench_out, bench))

    by_domain = defaultdict(Counter)
    for row, out in zip(domain_rows, domain_out):
        by_domain[row["domain"]][score_row(row, out)["ok"]] += 1
    pt_domains = json.loads(args.pytorch_domains.read_text(encoding="utf-8"))["domains"]
    report = {
        "model": str(args.model), "device": args.device, "compute_type": args.compute_type, "threads": args.threads,
        "sentences": len(sources), "seconds": round(seconds, 1), "sentences_per_second": round(len(sources) / seconds, 1),
        "benchmark_exact_pct": {"ct2": round(100 * ct2_exact / len(bench), 2), "pytorch": round(100 * pt_exact / len(bench), 2)},
        "benchmark_identical_to_pytorch_pct": round(100 * same_as_pt / len(bench), 2),
        "domains": {d: {"ct2": round(100 * c[True] / sum(c.values()), 2), "pytorch": pt_domains.get(d, {}).get("correct_pct")}
                    for d, c in sorted(by_domain.items())},
        "benchmark_differences": [{"source": e["source"], "pytorch": e["prediction"], "ct2": o}
                                  for o, e in zip(bench_out, bench) if n(o) != n(e["prediction"])][:200],
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "benchmark_differences"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
