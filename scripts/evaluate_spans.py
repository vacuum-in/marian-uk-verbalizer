#!/usr/bin/env python3
"""Score the span-to-span verbalizer against whole-sentence systems.

    evaluate_spans.py --model outputs/e55-spans-pilot/final --synthetic DIR --real FILE [--out reports/x.json]

--synthetic: the held-out splits of skypro1111/uk-text-normalization (*.jsonl with original / verbalized /
verbalized_alts). --real: book sentences with a reference written by the judge and the E54 / Gemma outputs
(fields sentence, reference, marian, gemma). Scores: exact match, spoken match (case, punctuation and hyphens are not
heard), and for real text the share of sentences where an ordinary word was altered.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import time
from pathlib import Path

import torch
from transformers import MarianMTModel, MarianTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.verbalizer.evaluation.metrics import spoken_form  # noqa: E402
from src.verbalizer.spans import assemble, detect_spans, model_input  # noqa: E402

WORD = re.compile(r"[А-Яа-яІіЇїЄєҐґ'’ʼ]+")


def altered_words(source: str, output: str) -> list[str]:
    """Ordinary words (3+ letters, not capitals) outside every detected span that are missing from the output: prose
    the system had no reason to touch. Words inside spans may legitimately change ("571 доларів" -> "долар")."""
    spans = detect_spans(source)
    have = collections.Counter(w.lower().replace("’", "'") for w in WORD.findall(output))
    lost = []
    for m in WORD.finditer(source):
        w = m.group()
        if len(w) < 3 or w.isupper() or any(sp.start < m.end() and m.start() < sp.end for sp in spans):
            continue
        k = w.lower().replace("’", "'")
        if have[k]:
            have[k] -= 1
        else:
            lost.append(k)
    return lost


class SpanVerbalizer:
    def __init__(self, path: Path, device: str, batch: int = 64):
        self.tok = MarianTokenizer.from_pretrained(path)
        self.device, self.batch, self.translator = device, batch, None
        if (path / "model.bin").is_file():  # a CTranslate2 export
            import ctranslate2
            self.translator = ctranslate2.Translator(str(path), device=device, compute_type="int8")
            return
        self.model = MarianMTModel.from_pretrained(path).to(device).eval()
        if device == "cuda":
            self.model = self.model.to(torch.bfloat16)

    @torch.inference_mode()
    def _generate(self, inputs: list[str]) -> list[str]:
        if self.translator is not None:
            out = []
            for i in range(0, len(inputs), self.batch):
                toks = [self.tok.convert_ids_to_tokens(self.tok(x, truncation=True, max_length=192)["input_ids"])
                        for x in inputs[i:i + self.batch]]
                res = self.translator.translate_batch(toks, beam_size=1, max_decoding_length=96, min_decoding_length=0,
                                                      suppress_sequences=[["<s>"]])
                out.extend(self.tok.decode(self.tok.convert_tokens_to_ids(r.hypotheses[0]), skip_special_tokens=True)
                           for r in res)
            return out
        out = []
        for i in range(0, len(inputs), self.batch):
            enc = self.tok(inputs[i:i + self.batch], return_tensors="pt", padding=True, truncation=True, max_length=192)
            gen = self.model.generate(**{k: v.to(self.device) for k, v in enc.items()}, num_beams=1, max_new_tokens=96)
            out.extend(self.tok.batch_decode(gen, skip_special_tokens=True))
        return out

    def verbalize(self, sentences: list[str]) -> list[str]:
        spans = [detect_spans(s) for s in sentences]
        flat = [model_input(s, sp) for s, sps in zip(sentences, spans) for sp in sps]
        outs = iter(self._generate(flat))
        return [assemble(s, sps, [next(outs) for _ in sps]) for s, sps in zip(sentences, spans)]


def score(rows, key):
    n = len(rows)
    exact = sum(any(r[key] == ref for ref in r["refs"]) for r in rows)
    spoken = sum(any(spoken_form(r[key]) == spoken_form(ref) for ref in r["refs"]) for r in rows)
    altered = sum(bool(altered_words(r["src"], r[key])) for r in rows)
    return {"rows": n, "exact": round(100 * exact / n, 2), "spoken": round(100 * spoken / n, 2),
            "altered_prose_rows": round(100 * altered / n, 2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--synthetic", type=Path)
    ap.add_argument("--real", type=Path)
    ap.add_argument("--baseline-errors", type=Path, help="errs_marian-ct2_cpu_all.json: E54's wrong rows on --synthetic")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--show", type=int, default=12)
    args = ap.parse_args()
    v = SpanVerbalizer(args.model, args.device)
    report = {}
    if args.synthetic:
        rows = []
        for f in sorted(args.synthetic.glob("*.jsonl")):
            for line in f.open(encoding="utf-8"):
                r = json.loads(line)
                rows.append({"split": f.stem, "src": r["original"], "refs": [r["verbalized"], *(r.get("verbalized_alts") or [])]})
        t0 = time.time()
        for r, o in zip(rows, v.verbalize([r["src"] for r in rows])):
            r["span"] = o
        report["synthetic_ms_per_sentence"] = round((time.time() - t0) / len(rows) * 1000, 1)
        report["synthetic"] = {"span model": score(rows, "span")}
        per = collections.defaultdict(list)
        for r in rows:
            per[r["split"]].append(r)
        report["synthetic_by_split"] = {k: score(rs, "span") for k, rs in per.items()}
        if args.baseline_errors:
            wrong = {(e["split"], e["in"]): e["out"] for e in json.loads(args.baseline_errors.read_text(encoding="utf-8"))}
            for r in rows:
                r["e54"] = wrong.get((r["split"], r["src"]), r["refs"][0])
            report["synthetic"]["E54 whole sentence"] = score(rows, "e54")
        bad = [r for r in rows if not any(spoken_form(r["span"]) == spoken_form(x) for x in r["refs"])]
        report["synthetic_errors"] = [{"in": r["src"], "out": r["span"], "ref": r["refs"][0]} for r in bad]
    if args.real:
        rows = []
        for line in args.real.open(encoding="utf-8"):
            r = json.loads(line)
            ref = r.get("reference")
            if not ref or len(ref) < 0.5 * len(r["sentence"]):
                continue
            rows.append({"src": r["sentence"], "refs": [ref], "marian": r["marian"], "gemma": r["gemma"],
                         "classes": r.get("classes", ["?"])})
        for r, o in zip(rows, v.verbalize([r["src"] for r in rows])):
            r["span"] = o
        report["real"] = {"E54 whole sentence": score(rows, "marian"), "Gemma whole sentence": score(rows, "gemma"),
                          "span model": score(rows, "span")}
        per = collections.defaultdict(list)
        for r in rows:
            per[r["classes"][0]].append(r)
        report["real_by_class"] = {k: {s: score(rs, s)["spoken"] for s in ("marian", "gemma", "span")}
                                   for k, rs in sorted(per.items())}
        bad = [r for r in rows if spoken_form(r["span"]) != spoken_form(r["refs"][0])]
        report["real_errors"] = [{"in": r["src"], "out": r["span"], "ref": r["refs"][0]} for r in bad]
    shown = {k: v for k, v in report.items() if not k.endswith("_errors")}
    print(json.dumps(shown, indent=1, ensure_ascii=False))
    for k in ("synthetic_errors", "real_errors"):
        for e in report.get(k, [])[: args.show]:
            print(f"\n[{k}] IN : {e['in'][:200]}\n  OUT: {e['out'][:200]}\n  REF: {e['ref'][:200]}")
    if args.out:
        args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
