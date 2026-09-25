#!/usr/bin/env python3
"""Export this project's pad-at-zero Marian checkpoints to CTranslate2."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import ctranslate2
from ctranslate2.converters import transformers as ct2_transformers

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.verbalizer.serving import SERVING_MANIFEST, validate_serving_manifest  # noqa: E402


class PadFirstMarianLoader(ct2_transformers.MarianMTLoader):
    """Adapt CT2's Marian exporter to a tokenizer with ``<pad>`` at ID zero.

    CT2 starts Marian decoding from a zero embedding, so its first vocabulary item
    is labelled ``<s>`` instead of HF's padding token.  The existing E22 export
    uses this convention.  Retaining every learned row except the final unused
    SentencePiece item keeps all IDs aligned with the original weights.
    """

    def set_config(self, config, model, tokenizer):
        super().set_config(config, model, tokenizer)
        # Decode from token 0 like Transformers does (decoder_start_token_id = pad = 0).
        config.decoder_start_token = "<s>"

    def set_decoder(self, spec, decoder):
        super().set_decoder(spec, decoder)
        # CT2's Marian loader assumes a zero start embedding (true for Helsinki-NLP
        # models). This project's pad row is learned (norm ~0.75, x22.6 scale), so a
        # zero start changes the first decoding step and drops first words.
        spec.start_from_zero_embedding = False

    def get_vocabulary(self, model, tokenizer):
        tokens = super(ct2_transformers.MarianMTLoader, self).get_vocabulary(
            model, tokenizer
        )
        if tokens[0] != "<pad>":
            raise ValueError("expected a Marian tokenizer with <pad> at ID zero")
        tokens[0] = "<s>"
        return tokens[:-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--quantization", default="int8")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_serving_manifest(args.source)
    ct2_transformers._MODEL_LOADERS["MarianConfig"] = PadFirstMarianLoader()
    converter = ctranslate2.converters.TransformersConverter(str(args.source))
    converter.convert(str(args.output), quantization=args.quantization, force=True)
    for filename in (
        "source.spm",
        "target.spm",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "vocab.json",
        "generation_config.json",
        SERVING_MANIFEST,
    ):
        shutil.copy2(args.source / filename, args.output / filename)


if __name__ == "__main__":
    main()
