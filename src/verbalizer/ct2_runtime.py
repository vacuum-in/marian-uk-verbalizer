"""CTranslate2 Marian runtime using the shared correctness-first pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .serving import validate_serving_manifest
from .text import TextRoute, enforce_output_fidelity_detailed, prepare_model_input, route_text


class CTranslate2MarianVerbalizer:
    """Route one sentence to pass-through or Marian."""

    def __init__(
        self,
        model_path: Path,
        *,
        device: str = "cpu",
        compute_type: str = "int8",
        intra_threads: int = 4,
        max_source_tokens: int = 384,
        max_new_tokens: int = 384,
    ) -> None:
        import ctranslate2  # type: ignore[import-not-found]
        from transformers import MarianTokenizer

        self.model_path = model_path.resolve()
        self.manifest = validate_serving_manifest(self.model_path)
        self.tokenizer = MarianTokenizer.from_pretrained(self.model_path)
        self.translator = ctranslate2.Translator(
            str(self.model_path),
            device=device,
            compute_type=compute_type,
            inter_threads=1,
            intra_threads=intra_threads,
        )
        self.max_source_tokens = max_source_tokens
        self.max_new_tokens = max_new_tokens

    def verbalize(self, source: str) -> tuple[str, dict[str, Any]]:
        routed = route_text(source)
        if routed.route is not TextRoute.MARIAN:
            return routed.text, {
                "route": routed.route.value,
                "numeric_policy_version": self.manifest["numeric_policy_version"],
            }
        model_input = prepare_model_input(routed.text)
        token_ids = self.tokenizer(
            model_input,
            truncation=True,
            max_length=self.max_source_tokens,
            add_special_tokens=True,
        )["input_ids"]
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        # Token 0 ("<s>", the pad row) is the decoder start and is never generated,
        # matching Transformers' Marian generate().
        result = self.translator.translate_batch(
            [tokens],
            beam_size=1,
            num_hypotheses=1,
            max_decoding_length=self.max_new_tokens,
            suppress_sequences=[["<s>"]],
        )[0]
        prediction_tokens = result.hypotheses[0]
        prediction = self.tokenizer.decode(
            self.tokenizer.convert_tokens_to_ids(prediction_tokens),
            skip_special_tokens=True,
        )
        fidelity = enforce_output_fidelity_detailed(source, prediction)
        return fidelity.text, {
            "route": TextRoute.MARIAN.value,
            "numeric_policy_version": self.manifest["numeric_policy_version"],
        }
