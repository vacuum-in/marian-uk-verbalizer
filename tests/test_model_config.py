from transformers import MarianMTModel

from src.verbalizer.model.config import build_marian_config, parameter_report


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __len__(self) -> int:
        return 128


def test_marian_config_sets_special_tokens_and_ties_embeddings() -> None:
    values = {
        "encoder_layers": 1,
        "decoder_layers": 1,
        "d_model": 32,
        "encoder_attention_heads": 4,
        "decoder_attention_heads": 4,
        "encoder_ffn_dim": 64,
        "decoder_ffn_dim": 64,
        "dropout": 0.1,
        "attention_dropout": 0.0,
        "activation_dropout": 0.0,
        "activation_function": "gelu",
        "max_position_embeddings": 64,
        "share_encoder_decoder_embeddings": True,
        "tie_word_embeddings": True,
        "scale_embedding": False,
    }
    config = build_marian_config(values, TinyTokenizer())
    assert config.pad_token_id == 0
    assert config.eos_token_id == 2
    assert config.decoder_start_token_id == 0
    assert config.bad_words_ids is None
    model = MarianMTModel(config)
    assert model.model.shared.weight.data_ptr() == model.lm_head.weight.data_ptr()
    assert parameter_report(model)["parameters"] > 0
