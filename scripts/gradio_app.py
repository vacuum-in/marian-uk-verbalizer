"""Serve the selected Ukrainian Marian verbalizer with Gradio."""

from __future__ import annotations

import argparse
import sys
import time
import unicodedata
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any
import torch
from transformers import MarianMTModel, MarianTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.verbalizer.text import (  # noqa: E402
    TextRoute,
    enforce_output_fidelity_detailed,
    prepare_model_input,
    route_text,
)
from src.verbalizer.serving import validate_serving_manifest  # noqa: E402

if TYPE_CHECKING:
    import gradio as gr


DEFAULT_CHECKPOINT = ROOT / "outputs/e19-marian-small-clean-ivx-convergence/checkpoint-8000"
MAX_SOURCE_TOKENS = 384
MAX_NEW_TOKENS = 384


def device_options(
    force_cpu: bool,
    cuda_available: bool | None = None,
) -> tuple[list[str], str]:
    """Return UI device choices and the default without loading a model."""

    if cuda_available is None:
        cuda_available = torch.cuda.is_available()
    if force_cpu or not cuda_available:
        return ["CPU"], "CPU"
    return ["GPU", "CPU"], "GPU"


class MarianVerbalizer:
    def __init__(self, checkpoint: Path, force_cpu: bool = False, force_marian: bool = False) -> None:
        # force_marian sends every input to the selected model instead of
        # consulting the router. It only changes which model receives the
        # unchanged text; no rule, dictionary or correction is involved.
        self.force_marian = force_marian
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
        self.checkpoint = checkpoint.resolve()
        self.serving_manifest = validate_serving_manifest(self.checkpoint)
        self.device_choices, self.default_device_choice = device_options(force_cpu)
        self.default_device = torch.device(
            "cuda" if self.default_device_choice == "GPU" else "cpu"
        )
        self.tokenizer = MarianTokenizer.from_pretrained(self.checkpoint)
        self.models: dict[str, MarianMTModel] = {}
        self.models[self.default_device.type] = self._load_model(self.default_device)
        if self.default_device.type == "cuda":
            torch.set_float32_matmul_precision("high")

    @property
    def device_label(self) -> str:
        return self._device_label(self.default_device)

    @staticmethod
    def _device_label(device: torch.device) -> str:
        if device.type == "cuda":
            return torch.cuda.get_device_name(device)
        return "CPU"

    def _load_model(self, device: torch.device) -> MarianMTModel:
        model = MarianMTModel.from_pretrained(self.checkpoint).to(device).eval()
        model.config.use_cache = True
        return model

    def _device_for_choice(self, choice: str | None) -> torch.device:
        choice = choice or self.default_device_choice
        if choice not in self.device_choices:
            raise ValueError(
                f"Пристрій {choice!r} недоступний; доступні: {', '.join(self.device_choices)}."
            )
        return torch.device("cuda" if choice == "GPU" else "cpu")

    def _model_for(self, device: torch.device) -> MarianMTModel:
        model = self.models.get(device.type)
        if model is None:
            model = self._load_model(device)
            self.models[device.type] = model
        return model

    @staticmethod
    def _precision_context(device: torch.device):
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cpu":
            return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
        return nullcontext()

    @torch.inference_mode()
    def verbalize(
        self,
        text: str,
        decoding: str,
        device_choice: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        text = unicodedata.normalize("NFC", (text or "").strip())
        if not text:
            return "", {"ok": False, "error": "Введіть текст для вербалізації."}

        routed = route_text(text)
        if not self.force_marian and routed.route is not TextRoute.MARIAN:
            return routed.text, {"ok": True, "route": routed.route.value, "checkpoint": "not-required"}

        try:
            device = self._device_for_choice(device_choice)
        except ValueError as exc:
            return "", {"ok": False, "error": str(exc)}
        model = self._model_for(device)

        marian_source = text if self.force_marian else routed.text
        model_inputs = [prepare_model_input(marian_source)]
        source_lengths = [
            len(self.tokenizer(model_input, add_special_tokens=True)["input_ids"])
            for model_input in model_inputs
        ]
        if max(source_lengths) > MAX_SOURCE_TOKENS:
            return "", {
                "ok": False,
                "error": (
                    f"Одне речення має {max(source_lengths)} токенів; "
                    f"межа моделі — {MAX_SOURCE_TOKENS}. "
                    "Розділіть текст на коротші речення."
                ),
                "source_tokens_max": max(source_lengths),
            }

        beams = int(decoding.split()[0])
        started = time.perf_counter()
        decoded_segments: list[str] = []
        generated_rows: list[list[int]] = []
        # Generate each sentence independently. This is numerically stable across
        # single- and multi-sentence requests and avoids padding-dependent ties.
        for model_input, source_length in zip(model_inputs, source_lengths, strict=True):
            encoded = self.tokenizer(model_input, return_tensors="pt").to(device)
            dynamic_max = min(MAX_NEW_TOKENS, max(128, source_length * 3))
            with self._precision_context(device):
                generated = model.generate(
                    **encoded,
                    max_new_tokens=dynamic_max,
                    num_beams=beams,
                    do_sample=False,
                    early_stopping=beams > 1,
                )
            generated_rows.extend(generated.tolist())
            decoded_segments.extend(
                self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            )
        elapsed = time.perf_counter() - started
        output = " ".join(
            value.strip() for value in decoded_segments
        )
        fidelity = enforce_output_fidelity_detailed(text, output)
        output = fidelity.text
        terminated = all(
            self.tokenizer.eos_token_id in token_ids for token_ids in generated_rows
        )
        ok = bool(output)
        diagnostics = {
            "ok": ok,
            "device": self._device_label(device),
            "device_choice": "GPU" if device.type == "cuda" else "CPU",
            "loaded_devices": sorted(
                "GPU" if device_type == "cuda" else "CPU"
                for device_type in self.models
            ),
            "checkpoint": str(self.checkpoint),
            "decoding": f"beam-{beams}" if beams > 1 else "greedy",
            "compute_precision": "bf16" if device.type in {"cuda", "cpu"} else "fp32",
            "latency_ms": round(elapsed * 1000, 2),
            "route": TextRoute.MARIAN.value,
            "sentence_segments": 1,
            "source_tokens_total": sum(source_lengths),
            "source_tokens_max": max(source_lengths),
            "generated_tokens": sum(
                token_id != self.tokenizer.pad_token_id
                for token_ids in generated_rows
                for token_id in token_ids
            ),
            "terminated_with_eos": terminated,
            "numeric_policy_version": self.serving_manifest["numeric_policy_version"],
            "raw_model_output": True,
        }
        return output, diagnostics


class CT2Verbalizer:
    """Same interface as MarianVerbalizer, backed by a CTranslate2 export."""

    def __init__(self, model_dir: Path, force_cpu: bool = False, force_marian: bool = False,
                 compute_type: str = "int8", cpu_threads: int = 8) -> None:
        import ctranslate2

        self.force_marian = force_marian
        self.checkpoint = model_dir.resolve()
        self.serving_manifest = validate_serving_manifest(self.checkpoint)
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self.device_choices, self.default_device_choice = device_options(
            force_cpu, cuda_available=ctranslate2.get_cuda_device_count() > 0
        )
        self.tokenizer = MarianTokenizer.from_pretrained(self.checkpoint)
        self.translators: dict[str, Any] = {}
        self._translator_for(self.default_device_choice)

    @property
    def device_label(self) -> str:
        return f"{self.default_device_choice} (CTranslate2 {self.compute_type})"

    def _translator_for(self, choice: str):
        import ctranslate2

        if choice not in self.translators:
            device = "cuda" if choice == "GPU" else "cpu"
            self.translators[choice] = ctranslate2.Translator(
                str(self.checkpoint), device=device, compute_type=self.compute_type,
                inter_threads=1, intra_threads=self.cpu_threads if device == "cpu" else 1,
            )
        return self.translators[choice]

    def verbalize(self, text: str, decoding: str, device_choice: str | None = None) -> tuple[str, dict[str, Any]]:
        text = unicodedata.normalize("NFC", (text or "").strip())
        if not text:
            return "", {"ok": False, "error": "Введіть текст для вербалізації."}
        routed = route_text(text)
        if not self.force_marian and routed.route is not TextRoute.MARIAN:
            return routed.text, {"ok": True, "route": routed.route.value, "checkpoint": "not-required"}
        choice = device_choice or self.default_device_choice
        if choice not in self.device_choices:
            return "", {"ok": False, "error": f"Пристрій {choice!r} недоступний; доступні: {', '.join(self.device_choices)}."}
        token_ids = self.tokenizer(prepare_model_input(routed.text), truncation=True, max_length=MAX_SOURCE_TOKENS)["input_ids"]
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        beams = int(decoding.split()[0])
        started = time.perf_counter()
        # Token 0 ("<s>", the pad row) is the decoder start and is never generated,
        # matching Transformers' Marian generate().
        result = self._translator_for(choice).translate_batch(
            [tokens], beam_size=beams, max_decoding_length=min(MAX_NEW_TOKENS, max(128, len(tokens) * 3)),
            suppress_sequences=[["<s>"]],
        )[0]
        elapsed = time.perf_counter() - started
        hypothesis = result.hypotheses[0]
        output = self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(hypothesis), skip_special_tokens=True).strip()
        output = enforce_output_fidelity_detailed(text, output).text
        return output, {
            "ok": bool(output), "device": self.device_label if choice == self.default_device_choice else f"{choice} (CTranslate2 {self.compute_type})",
            "device_choice": choice, "loaded_devices": sorted(self.translators), "checkpoint": str(self.checkpoint),
            "decoding": f"beam-{beams}" if beams > 1 else "greedy", "compute_precision": self.compute_type,
            "latency_ms": round(elapsed * 1000, 2), "route": TextRoute.MARIAN.value,
            "source_tokens_total": len(tokens), "generated_tokens": len(hypothesis),
            "numeric_policy_version": self.serving_manifest["numeric_policy_version"], "raw_model_output": True,
        }


def build_demo(engine: MarianVerbalizer | CT2Verbalizer) -> gr.Blocks:
    import gradio as gr

    title = "Український Marian-вербалізатор"
    description = (
        "Роутер передає моделі без змін усе, що може потребувати вербалізації "
        "(числа, латиницю, абревіатури, символи), а звичайний український текст "
        "повертає без обробки. Вивід моделі не виправляється і не відхиляється правилами."
    )
    examples = [
        ["Координати точки: 47.1024, 32.2757.", "1 — greedy", engine.default_device_choice],
        ["Потрібно прочитати XX розділ до понеділка.", "1 — greedy", "CPU"],
        ["Подія 16.08.2030 починається о 08:45 і триває 90 хв.", "1 — greedy", engine.default_device_choice],
        ["Телефон для зв’язку: +380 (97) 309-22-05.", "1 — greedy", engine.default_device_choice],
        ["Це звичайне українське речення без чисел.", "1 — greedy", "CPU"],
    ]

    with gr.Blocks(title=title) as demo:
        gr.Markdown(
            f"# {title}\n\n{description}\n\n"
            f"**Типовий пристрій:** {engine.device_label}. "
            "CPU-модель завантажується під час першого CPU-запиту."
        )
        source = gr.Textbox(
            label="Вхідний текст",
            lines=6,
            max_lines=12,
            placeholder="Наприклад: У 2025 році витрати зросли на 15,5%.",
            autofocus=True,
        )
        decoding = gr.Radio(
            choices=["1 — greedy", "2 — beam", "4 — beam"],
            value="1 — greedy",
            label="Декодування",
        )
        device = gr.Radio(
            choices=engine.device_choices,
            value=engine.default_device_choice,
            label="Пристрій",
            info="GPU швидший; CPU не використовує відеопам’ять.",
        )
        run = gr.Button("Вербалізувати", variant="primary")
        output = gr.Textbox(label="Результат", lines=6, buttons=["copy"])
        diagnostics = gr.JSON(label="Діагностика та safety-перевірки")
        gr.Examples(examples=examples, inputs=[source, decoding, device])

        event_args = {
            "fn": engine.verbalize,
            "inputs": [source, decoding, device],
            "outputs": [output, diagnostics],
            "api_name": "verbalize",
            "concurrency_limit": 1,
        }
        run.click(**event_args)
        source.submit(**event_args)

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--server-name", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--force-marian",
        action="store_true",
        help="send every input to the loaded checkpoint instead of consulting the router",
    )
    parser.add_argument("--ct2", action="store_true", help="--checkpoint is a CTranslate2 export")
    parser.add_argument("--ct2-compute-type", default="int8")
    args = parser.parse_args()

    if args.ct2:
        engine = CT2Verbalizer(args.checkpoint, force_cpu=args.cpu, force_marian=args.force_marian,
                               compute_type=args.ct2_compute_type)
    else:
        engine = MarianVerbalizer(args.checkpoint, force_cpu=args.cpu, force_marian=args.force_marian)
    demo = build_demo(engine)
    demo.queue(default_concurrency_limit=1, max_size=16).launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
