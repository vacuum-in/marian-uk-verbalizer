"""Build explicit Marian configurations from resolved experiment settings."""

from __future__ import annotations

from typing import Any

from transformers import MarianConfig


MODEL_KEYS = {
    "encoder_layers",
    "decoder_layers",
    "d_model",
    "encoder_attention_heads",
    "decoder_attention_heads",
    "encoder_ffn_dim",
    "decoder_ffn_dim",
    "dropout",
    "attention_dropout",
    "activation_dropout",
    "activation_function",
    "max_position_embeddings",
    "share_encoder_decoder_embeddings",
    "tie_word_embeddings",
    "scale_embedding",
}


def build_marian_config(model_values: dict[str, Any], tokenizer: Any) -> MarianConfig:
    missing = MODEL_KEYS - model_values.keys()
    if missing:
        raise ValueError(f"missing Marian model settings: {sorted(missing)}")
    pad_id = int(tokenizer.pad_token_id)
    eos_id = int(tokenizer.eos_token_id)
    if pad_id < 0 or eos_id < 0 or pad_id == eos_id:
        raise ValueError(f"invalid special token IDs: pad={pad_id}, eos={eos_id}")
    config = MarianConfig(
        vocab_size=len(tokenizer),
        pad_token_id=pad_id,
        eos_token_id=eos_id,
        decoder_start_token_id=pad_id,
        forced_eos_token_id=eos_id,
        # Transformers 4.30 rejects an empty list during generation. None has
        # the same intended meaning: no prohibited token sequences.
        bad_words_ids=None,
        use_cache=False,
        **{key: model_values[key] for key in MODEL_KEYS},
    )
    return config


def parameter_report(model: Any) -> dict[str, int | float]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "parameters": total,
        "trainable_parameters": trainable,
        "fp32_parameter_mib": round(total * 4 / 1024**2, 2),
        "bf16_parameter_mib": round(total * 2 / 1024**2, 2),
    }
