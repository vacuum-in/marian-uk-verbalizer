"""Train and verify the shared 16k Marian SentencePiece tokenizer."""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import sentencepiece as spm
from transformers import MarianTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.verbalizer.config import load_yaml  # noqa: E402
from src.verbalizer.training.manifest import sha256_file, write_json  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/tokenizer/spm-16k.yaml"


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def write_corpus(input_path: Path, corpus_path: Path) -> int:
    rows = 0
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    with corpus_path.open("w", encoding="utf-8") as handle:
        parquet = pq.ParquetFile(input_path)
        for batch in parquet.iter_batches(columns=["source", "target"], batch_size=4096):
            for source, target in zip(
                batch.column("source").to_pylist(),
                batch.column("target").to_pylist(),
                strict=True,
            ):
                handle.write(str(source).replace("\n", " ") + "\n")
                handle.write(str(target).replace("\n", " ") + "\n")
                rows += 1
    return rows


def train(config_path: Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    input_path = ROOT / config["input"]
    output_dir = ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / "corpus.txt"
    rows = write_corpus(input_path, corpus_path)
    model_prefix = output_dir / "spm"
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(model_prefix),
        vocab_size=int(config["vocab_size"]),
        model_type=str(config["model_type"]),
        character_coverage=float(config["character_coverage"]),
        normalization_rule_name=str(config["normalization_rule_name"]),
        pad_id=0,
        pad_piece="<pad>",
        unk_id=1,
        unk_piece="<unk>",
        bos_id=-1,
        eos_id=2,
        eos_piece="</s>",
        user_defined_symbols=list(config["user_defined_symbols"]),
        shuffle_input_sentence=False,
        hard_vocab_limit=False,
        max_sentence_length=8192,
        num_threads=8,
    )
    processor = spm.SentencePieceProcessor(model_file=str(model_prefix.with_suffix(".model")))
    vocab = {processor.id_to_piece(index): index for index in range(processor.get_piece_size())}
    vocab_path = output_dir / "vocab.json"
    vocab_path.write_text(json.dumps(vocab, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tokenizer = MarianTokenizer(
        source_spm=str(model_prefix.with_suffix(".model")),
        target_spm=str(model_prefix.with_suffix(".model")),
        vocab=str(vocab_path),
        model_max_length=int(config["model_max_length"]),
    )
    tokenizer.save_pretrained(output_dir)

    lengths_source: list[int] = []
    lengths_target: list[int] = []
    unknown_pieces = 0
    parquet = pq.ParquetFile(input_path)
    for batch in parquet.iter_batches(columns=["source", "target"], batch_size=4096):
        sources = [str(value) for value in batch.column("source").to_pylist()]
        targets = [str(value) for value in batch.column("target").to_pylist()]
        source_ids = tokenizer(sources, add_special_tokens=True)["input_ids"]
        target_ids = tokenizer(targets, add_special_tokens=True)["input_ids"]
        lengths_source.extend(map(len, source_ids))
        lengths_target.extend(map(len, target_ids))
        unknown_pieces += sum(ids.count(tokenizer.unk_token_id) for ids in source_ids + target_ids)

    required_tokens = ["<pad>", "<unk>", "</s>", *config["user_defined_symbols"]]
    missing = [token for token in required_tokens if tokenizer.convert_tokens_to_ids(token) == tokenizer.unk_token_id and token != "<unk>"]
    if missing:
        raise ValueError(f"required tokenizer symbols are missing: {missing}")
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        raise ValueError("pad and EOS IDs must be distinct")

    report = {
        "config": config,
        "rows": rows,
        "vocab_size": len(tokenizer),
        "special_token_ids": {
            "pad": tokenizer.pad_token_id,
            "unk": tokenizer.unk_token_id,
            "eos": tokenizer.eos_token_id,
            "decoder_start": tokenizer.pad_token_id,
        },
        "source_lengths": {
            "mean": round(statistics.mean(lengths_source), 3),
            "p95": percentile(lengths_source, 0.95),
            "max": max(lengths_source),
        },
        "target_lengths": {
            "mean": round(statistics.mean(lengths_target), 3),
            "p95": percentile(lengths_target, 0.95),
            "max": max(lengths_target),
        },
        "unknown_pieces": unknown_pieces,
        "files": {
            path.name: sha256_file(path)
            for path in output_dir.iterdir()
            if path.is_file() and path.name != "corpus.txt"
        },
    }
    write_json(output_dir / "tokenizer_report.json", report)
    shutil.copy2(config_path, output_dir / "training_config.yaml")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(train(args.config), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

