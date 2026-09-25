"""Lazy runtime adapter for the trained English-to-Ukrainian transducer."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path


MODEL_VERSION = "en-uk-combined-v1-medium-sched-ct2-int8"
WORD_RE = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*\Z")
SPAN_RE = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*")


def _workspace_assets() -> tuple[Path, Path] | None:
    """Locate the sibling transducer release in the shared verbalizer workspace."""

    verbalizer_root = Path(__file__).resolve().parents[3]
    transducer_root = verbalizer_root / "en-uk-transducer"
    model_root = transducer_root / "models/en-uk-combined-v1-medium-sched"
    paths = (
        model_root / "ct2-int8",
        model_root / "vocab.json",
    )
    return paths if all(path.exists() for path in paths) else None


def _entity_type(word: str) -> str | None:
    """Return the trained-model tag for a Latin token, including abbreviations."""

    if not WORD_RE.fullmatch(word):
        return None
    letters = [character for character in word if character.isalpha()]
    if all(character.isupper() for character in letters):
        return "english_word"
    return "person" if word[:1].isupper() else "english_word"


class EnglishTransducer:
    def __init__(self, model_path: Path, vocabulary_path: Path) -> None:
        import ctranslate2

        if not vocabulary_path.is_file():
            raise ValueError("English transducer vocabulary is malformed")
        self.translator = ctranslate2.Translator(
            str(model_path), device="cpu", inter_threads=1, intra_threads=1
        )

    def render(self, word: str) -> str | None:
        entity_type = _entity_type(word)
        if entity_type is None:
            return None
        case = _case_pattern(word)
        serialized = f"<TYPE:{entity_type}> <LANG:en> <CASE:{case}> " + " ".join(word.casefold())
        result = self.translator.translate_batch(
            [serialized.split() + ["</s>"]], beam_size=1, max_decoding_length=128
        )[0]
        reading = "".join(" " if token == "<SPACE>" else token for token in result.hypotheses[0]).strip()
        return reading or None


def _case_pattern(word: str) -> str:
    letters = "".join(character for character in word if character.isalpha())
    if letters.islower():
        return "lower"
    if letters[:1].isupper() and letters[1:].islower():
        return "title"
    return "mixed"


@lru_cache(maxsize=1)
def trained_english_transducer() -> EnglishTransducer | None:
    assets = _workspace_assets()
    return EnglishTransducer(*assets) if assets else None


def render_english_word(word: str) -> str | None:
    transducer = trained_english_transducer()
    return transducer.render(word) if transducer else None


def render_english_spans(value: str) -> tuple[str, bool]:
    """Render eligible Latin spans through the trained transducer in-place."""

    converted = False

    def replace(match: re.Match[str]) -> str:
        nonlocal converted
        reading = render_english_word(match.group(0))
        if reading is None:
            return match.group(0)
        converted = True
        return reading

    return SPAN_RE.sub(replace, value), converted
