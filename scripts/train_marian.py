"""Train a Marian model from scratch on the prepared Ukrainian pilot dataset."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import MarianMTModel, MarianTokenizer, get_linear_schedule_with_warmup


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, os.environ.get("MLOPS_HELPERS_PATH", "/home/devops/mlops/python"))

from src.verbalizer.config import load_yaml  # noqa: E402
from src.verbalizer.model.config import build_marian_config, parameter_report  # noqa: E402
from src.verbalizer.numeric import NUMERIC_POLICY_VERSION, prepare_training_pair  # noqa: E402
from src.verbalizer.text import prepare_model_input  # noqa: E402
from src.verbalizer.serving import write_serving_manifest  # noqa: E402
from src.verbalizer.training.batching import TokenBudgetBatchSampler, estimated_pair_tokens  # noqa: E402
from src.verbalizer.training.manifest import (  # noqa: E402
    git_revision,
    hardware_manifest,
    package_versions,
    sha256_file,
    write_json,
)
from mlops_tracking import end_run, log_directory, log_file, log_json, log_metrics, log_params, start_run  # noqa: E402


TRACKING_ENABLED = True


def tracking_call(function: Any, *args: Any, **kwargs: Any) -> None:
    """Keep local training usable when the optional MLflow extras are absent."""

    global TRACKING_ENABLED
    if not TRACKING_ENABLED:
        return
    try:
        function(*args, **kwargs)
    except RuntimeError as error:
        TRACKING_ENABLED = False
        print(json.dumps({"tracking_disabled": str(error)}), flush=True)


DEFAULT_CONFIG = ROOT / "configs/train/pilot-e1.yaml"
class PairDataset(Dataset[dict[str, str]]):
    def __init__(self, path: Path, limit: int = 0) -> None:
        rows = pq.read_table(path, columns=["id", "source", "target"]).to_pylist()
        if limit > 0:
            rows = rows[:limit]
        self.rows = [
            {"id": str(row["id"]), "source": str(row["source"]), "target": str(row["target"])}
            for row in rows
        ]

    def __getitem__(self, index: int) -> dict[str, str]:
        return self.rows[index]

    def __len__(self) -> int:
        return len(self.rows)


class Collator:
    def __init__(
        self,
        tokenizer: MarianTokenizer,
        source_limit: int,
        target_limit: int,
        input_preprocessing: str,
    ) -> None:
        self.tokenizer = tokenizer
        self.source_limit = source_limit
        self.target_limit = target_limit
        self.input_preprocessing = input_preprocessing

    def prepare_source(self, value: str) -> str:
        if self.input_preprocessing in {"raw", NUMERIC_POLICY_VERSION}:
            return prepare_model_input(value)
        raise ValueError(
            "input_preprocessing must be 'raw' or " f"'{NUMERIC_POLICY_VERSION}'"
        )

    def __call__(self, rows: list[dict[str, str]]) -> dict[str, Any]:
        if self.input_preprocessing == NUMERIC_POLICY_VERSION:
            pairs = [prepare_training_pair(row["source"], row["target"]) for row in rows]
            sources = [pair[0] for pair in pairs]
            targets = [pair[1] for pair in pairs]
        else:
            sources = [self.prepare_source(row["source"]) for row in rows]
            targets = [row["target"] for row in rows]
        encoded = self.tokenizer(
            sources,
            max_length=self.source_limit,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        target = self.tokenizer(
            text_target=targets,
            max_length=self.target_limit,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        labels = target["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "ids": [row["id"] for row in rows],
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": labels,
        }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def precision_context(precision: str):
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def compute_loss(outputs: Any, labels: torch.Tensor, smoothing: float) -> torch.Tensor:
    if smoothing <= 0:
        return outputs.loss
    return F.cross_entropy(
        outputs.logits.reshape(-1, outputs.logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
        label_smoothing=smoothing,
    )


@torch.no_grad()
def evaluate(
    model: MarianMTModel,
    loader: DataLoader,
    device: torch.device,
    precision: str,
    max_batches: int,
) -> dict[str, float | int]:
    model.eval()
    losses: list[float] = []
    tokens = 0
    for batch_index, batch in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        labels = batch["labels"].to(device, non_blocking=True)
        with precision_context(precision):
            outputs = model(
                input_ids=batch["input_ids"].to(device, non_blocking=True),
                attention_mask=batch["attention_mask"].to(device, non_blocking=True),
                labels=labels,
            )
        losses.append(float(outputs.loss.detach().float()))
        tokens += int((labels != -100).sum())
    model.train()
    mean_loss = sum(losses) / max(1, len(losses))
    return {
        "eval_loss": round(mean_loss, 6),
        "eval_perplexity": round(math.exp(min(20, mean_loss)), 6),
        "eval_batches": len(losses),
        "eval_target_tokens": tokens,
    }


def save_checkpoint(
    path: Path,
    model: MarianMTModel,
    tokenizer: MarianTokenizer,
    optimizer: AdamW,
    scheduler: Any,
    state: dict[str, Any],
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    write_serving_manifest(path)
    torch.save(optimizer.state_dict(), path / "optimizer.pt")
    torch.save(scheduler.state_dict(), path / "scheduler.pt")
    write_json(path / "trainer_state.json", state)


@torch.no_grad()
def sample_predictions(
    model: MarianMTModel,
    tokenizer: MarianTokenizer,
    dataset: PairDataset,
    device: torch.device,
    input_preprocessing: str,
    count: int = 8,
) -> list[dict[str, str]]:
    model.eval()
    previous_cache = model.config.use_cache
    model.config.use_cache = True
    results: list[dict[str, str]] = []
    for row in dataset.rows[:count]:
        source = prepare_model_input(row["source"])
        encoded = tokenizer(
            source,
            max_length=256,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        generated = model.generate(
            **encoded,
            max_new_tokens=128,
            num_beams=1,
            do_sample=False,
        )
        results.append(
            {
                "id": row["id"],
                "source": row["source"],
                "reference": row["target"],
                "prediction": tokenizer.decode(generated[0], skip_special_tokens=True),
            }
        )
    model.config.use_cache = previous_cache
    model.train()
    return results


def train(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    config = load_yaml(config_path)
    if args.output_dir:
        config["output_dir"] = args.output_dir
    output_dir = (ROOT / config["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = (ROOT / config["train_data"]).resolve()
    validation_path = (ROOT / config["validation_data"]).resolve()
    tokenizer_path = (ROOT / config["tokenizer"]).resolve()
    model_config_path = (ROOT / config["model_config"]).resolve()

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA GPU is required; pass --allow-cpu only for a smoke test")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = str(config["precision"]) if device.type == "cuda" else "fp32"
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        precision = "fp16"
    seed = int(config["seed"])
    seed_everything(seed)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    tokenizer = MarianTokenizer.from_pretrained(tokenizer_path)
    train_dataset = PairDataset(train_path, args.train_limit)
    validation_dataset = PairDataset(validation_path, args.validation_limit)
    source_limit = int(config["max_source_tokens"])
    target_limit = int(config["max_target_tokens"])
    train_lengths = [
        estimated_pair_tokens(row["source"], row["target"], source_limit, target_limit)
        for row in train_dataset.rows
    ]
    validation_lengths = [
        estimated_pair_tokens(row["source"], row["target"], source_limit, target_limit)
        for row in validation_dataset.rows
    ]
    train_sampler = TokenBudgetBatchSampler(
        train_lengths,
        int(config["max_batch_tokens"]),
        int(config["max_batch_examples"]),
        seed,
        shuffle=True,
    )
    validation_sampler = TokenBudgetBatchSampler(
        validation_lengths,
        int(config["max_batch_tokens"]),
        int(config["max_batch_examples"]),
        seed,
        shuffle=False,
    )
    input_preprocessing = str(config.get("input_preprocessing", NUMERIC_POLICY_VERSION))
    collator = Collator(tokenizer, source_limit, target_limit, input_preprocessing)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collator,
        num_workers=int(config["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_sampler=validation_sampler,
        collate_fn=collator,
        num_workers=int(config["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    # TokenBudgetBatchSampler builds its length-sorted batches on demand.  The
    # count is invariant within an epoch, so calculating it inside every
    # optimizer step would repeatedly sort the complete training corpus.
    train_batches = len(train_sampler)

    if args.resume and args.initialize_from:
        raise ValueError("--resume and --initialize-from are mutually exclusive")
    if args.resume:
        resume_path = args.resume.resolve()
        model = MarianMTModel.from_pretrained(resume_path)
        state = json.loads((resume_path / "trainer_state.json").read_text(encoding="utf-8"))
    elif args.initialize_from:
        model = MarianMTModel.from_pretrained(args.initialize_from.resolve())
        state = {"global_step": 0, "epoch": 0, "batch_in_epoch": 0}
    else:
        model_values = load_yaml(model_config_path)
        model = MarianMTModel(build_marian_config(model_values, tokenizer))
        state = {"global_step": 0, "epoch": 0, "batch_in_epoch": 0}
    model.to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    accumulation = int(config["gradient_accumulation_steps"])
    steps_per_epoch = math.ceil(train_batches / accumulation)
    configured_steps = steps_per_epoch * int(config["epochs"])
    total_steps = args.max_steps if args.max_steps > 0 else configured_steps
    warmup_steps = round(total_steps * float(config["warmup_ratio"]))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    if args.resume:
        optimizer.load_state_dict(torch.load(args.resume / "optimizer.pt", map_location=device))
        scheduler.load_state_dict(torch.load(args.resume / "scheduler.pt", map_location=device))

    model_report = parameter_report(model)
    tied_embeddings = model.model.shared.weight.data_ptr() == model.lm_head.weight.data_ptr()
    if bool(model.config.tie_word_embeddings) and not tied_embeddings:
        raise RuntimeError("Marian output embeddings are not tied")
    manifest = {
        "experiment": config["experiment_name"],
        "resolved_config": config,
        "config_path": str(config_path),
        "git_revision": git_revision(ROOT),
        "packages": package_versions(),
        "hardware": hardware_manifest(),
        "seed": seed,
        "data": {
            "train": {"path": str(train_path), "sha256": sha256_file(train_path), "rows": len(train_dataset)},
            "validation": {
                "path": str(validation_path),
                "sha256": sha256_file(validation_path),
                "rows": len(validation_dataset),
            },
        },
        "tokenizer": {
            "path": str(tokenizer_path),
            "vocab_size": len(tokenizer),
            "report_sha256": sha256_file(tokenizer_path / "tokenizer_report.json"),
        },
        "model": {**model_report, "tied_embeddings": tied_embeddings},
        "training": {
            "resume_from": str(args.resume.resolve()) if args.resume else None,
            "initialize_from": (
                str(args.initialize_from.resolve()) if args.initialize_from else None
            ),
            "batches_per_epoch": train_batches,
            "optimizer_steps_per_epoch": steps_per_epoch,
            "total_optimizer_steps": total_steps,
            "warmup_steps": warmup_steps,
            "precision": precision,
            "input_preprocessing": input_preprocessing,
        },
    }
    write_json(output_dir / "run_manifest.json", manifest)
    tokenizer.save_pretrained(output_dir / "tokenizer")
    tracking_call(start_run,
        "ukrainian-marian-verbalizer",
        run_name=output_dir.name,
        tags={"project": "marian-uk-verbalizer", "task": "text-verbalization"},
    )
    tracking_call(log_params,
        {
            "config": config,
            "output_dir": output_dir,
            "train_limit": args.train_limit,
            "validation_limit": args.validation_limit,
            "max_steps": args.max_steps,
            "device": device,
        }
    )
    tracking_call(log_json, manifest, "lineage/run_manifest.json")

    metrics_path = output_dir / "metrics.jsonl"
    global_step = int(state["global_step"])
    start_epoch = int(state["epoch"])
    resume_batch = int(state["batch_in_epoch"])
    optimizer.zero_grad(set_to_none=True)
    start_time = time.monotonic()
    log_loss = 0.0
    log_batches = 0
    stop = False

    for epoch in range(start_epoch, int(config["epochs"])):
        train_sampler.set_epoch(epoch)
        model.train()
        for batch_index, batch in enumerate(train_loader):
            if epoch == start_epoch and batch_index < resume_batch:
                continue
            labels = batch["labels"].to(device, non_blocking=True)
            with precision_context(precision):
                outputs = model(
                    input_ids=batch["input_ids"].to(device, non_blocking=True),
                    attention_mask=batch["attention_mask"].to(device, non_blocking=True),
                    labels=labels,
                )
                raw_loss = compute_loss(outputs, labels, float(config["label_smoothing"]))
                loss = raw_loss / accumulation
            loss.backward()
            log_loss += float(raw_loss.detach().float())
            log_batches += 1
            is_boundary = (
                (batch_index + 1) % accumulation == 0
                or batch_index + 1 == train_batches
            )
            if not is_boundary:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clipping"]))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            state = {
                "global_step": global_step,
                "epoch": epoch,
                "batch_in_epoch": batch_index + 1,
            }

            if global_step % int(config["log_steps"]) == 0 or global_step == 1:
                elapsed = time.monotonic() - start_time
                metric = {
                    "step": global_step,
                    "epoch": epoch,
                    "batch": batch_index + 1,
                    "train_loss": round(log_loss / max(1, log_batches), 6),
                    "learning_rate": scheduler.get_last_lr()[0],
                    "elapsed_seconds": round(elapsed, 2),
                    "optimizer_steps_per_second": round(global_step / max(elapsed, 1e-6), 4),
                    "gpu_memory_allocated_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 2)
                    if device.type == "cuda"
                    else 0,
                }
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(metric, ensure_ascii=False) + "\n")
                print(json.dumps(metric, ensure_ascii=False), flush=True)
                tracking_call(log_metrics, metric, global_step)
                log_loss = 0.0
                log_batches = 0

            if global_step % int(config["eval_steps"]) == 0:
                metric = {"step": global_step, **evaluate(
                    model,
                    validation_loader,
                    device,
                    precision,
                    int(config["eval_max_batches"]),
                )}
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(metric, ensure_ascii=False) + "\n")
                print(json.dumps(metric, ensure_ascii=False), flush=True)
                tracking_call(log_metrics, metric, global_step)

            if global_step % int(config["save_steps"]) == 0:
                save_checkpoint(
                    output_dir / f"checkpoint-{global_step}",
                    model,
                    tokenizer,
                    optimizer,
                    scheduler,
                    state,
                )
            if global_step >= total_steps:
                stop = True
                break
        resume_batch = 0
        state = {"global_step": global_step, "epoch": epoch + 1, "batch_in_epoch": 0}
        if stop:
            break

    final_dir = output_dir / "final"
    save_checkpoint(final_dir, model, tokenizer, optimizer, scheduler, state)
    predictions = sample_predictions(
        model,
        tokenizer,
        validation_dataset,
        device,
        input_preprocessing,
    )
    write_json(output_dir / "sample_predictions.json", {"predictions": predictions})
    result = {
        "output_dir": str(output_dir),
        "global_step": global_step,
        "state": state,
        "model": model_report,
        "sample_predictions": len(predictions),
    }
    write_json(output_dir / "result.json", result)
    tracking_call(log_metrics, {"global_step": global_step, "sample_predictions": len(predictions)}, global_step)
    tracking_call(log_file, output_dir / "run_manifest.json", "lineage")
    tracking_call(log_file, metrics_path, "reports")
    tracking_call(log_file, output_dir / "sample_predictions.json", "reports")
    tracking_call(log_file, output_dir / "result.json", "reports")
    tracking_call(log_directory, final_dir, "model")
    tracking_call(end_run)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir")
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--initialize-from",
        type=Path,
        help="Load model weights/tokenizer-compatible config but reset optimizer and training state.",
    )
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--validation-limit", type=int, default=0)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    print(json.dumps(train(args), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
