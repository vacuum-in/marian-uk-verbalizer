# marian-uk-verbalizer

A small (52.7M-parameter) Marian sequence-to-sequence model that converts
written Ukrainian text into the form it should be **spoken** — the
text-normalization front end for a Ukrainian TTS pipeline. It expands roman
numerals, dates, currencies, coordinates, phone numbers, IBANs, card numbers,
math, symbols, abbreviations and acronyms into the words a reader would say
aloud, while leaving ordinary prose untouched.

```
Потрібно прочитати XX розділ до понеділка.
  -> Потрібно прочитати двадцятий розділ до понеділка.

Я пропрацював у ФБР двадцять років.
  -> Я пропрацював у еф-бе-ер двадцять років.

73°33′58″ пн. ш.
  -> сімдесят три градуси тридцять три хвилини п'ятдесят вісім секунд північної широти
```

## Why a seq2seq model instead of rules

Ukrainian verbalization is heavily inflected: a Roman ordinal, a number, or an
abbreviation takes a different ending depending on the case and gender of the
noun that governs it ("двадцятий розділ" vs. "двадцятого розділу" vs.
"двадцятим розділом"). A rule table for this quickly turns into an
unmaintainable dictionary that still gets the agreement wrong. This project
takes the opposite approach and enforces it structurally:

> **No hardcoded verbalization rules, dictionaries, or word lists anywhere in
> the runtime pipeline.** The only runtime logic is *routing* an unmodified
> input to a trained model, or *passing it through* unchanged when it needs no
> verbalization. All verbalization quality comes from the model and its
> training data — see [`AGENT.md`](AGENT.md).

Offline **data generation** is allowed to use linguistic tools
(`num2words`, `pymorphy3`) to build training examples, and every generated
target is cross-checked by parsing the numeral words back to a value and
comparing it against the source — but none of that runs at inference time.

## Architecture

```
input text --> router (regex, shape-only) --> passthrough   (plain Cyrillic prose)
                                          \--> Marian model  (everything else)
```

The router (`src/verbalizer/text.py`) makes its decision purely from the
*shape* of the text — character classes, capitalization, position — never
from a word list. It sends anything that is not plain Cyrillic prose (digits,
Latin letters, symbols, all-caps runs longer than a normal abbreviation, …) to
the Marian model, and lets ordinary sentences pass straight through. The
model itself is a standard 6+6 layer Marian encoder-decoder
(`d_model=512`, 52.7M params) with a shared 16k-token SentencePiece
vocabulary, trained on parallel (written, spoken) sentence pairs.

## Results

Measured on the frozen 3,597-row benchmark derived from
[`skypro1111/uk-text-normalization`](https://huggingface.co/datasets/skypro1111/uk-text-normalization)
(held out of training) and a suite of generated per-domain evaluation sets
(each independently frame-and-slot scored — see
[`src/verbalizer/evaluation/domain_check.py`](src/verbalizer/evaluation/domain_check.py)),
current checkpoint **E54**:

| Metric (frozen benchmark, 3,597 rows) | Score |
|---|---|
| Exact string match | 85.2% |
| Word error rate | 3.0% |
| Character error rate | 1.2% |
| **Numeric value fidelity** (every digit sequence, read back and compared to source) | **100.0%** |
| Digit sequence fidelity | 100.0% |
| Identity accuracy (plain prose left untouched) | 100.0% |

| Domain (generated eval set) | Correct | Rows |
|---|---|---|
| Coordinates | 100.0% | 500 |
| Phone numbers | 100.0% | 500 |
| Roman numeral + noun case agreement | 99.0% | 301 |
| Context (held-out book sentences) | 97.4% | 500 |
| English words embedded in Ukrainian | 88.4% | 500 |
| Card numbers / IBAN | 86.8% | 500 |
| Acronyms (letter-by-letter reading) | 85.5% | 207 |

"Correct" is a lenient, frame-and-slot match: the fixed part of the sentence
must reproduce exactly, and each slot must be read in one of its accepted
forms (transliteration variants for English, any valid digit grouping for
numbers, etc.) — see the module docstring linked above for exactly what each
domain accepts.

## Quickstart (PyTorch)

```bash
pip install torch transformers sentencepiece
```

```python
from transformers import MarianMTModel, MarianTokenizer

checkpoint = "aloudreader/marian-uk-verbalizer"  # or a local path
tokenizer = MarianTokenizer.from_pretrained(checkpoint)
model = MarianMTModel.from_pretrained(checkpoint)

text = "Потрібно прочитати XX розділ до понеділка."
batch = tokenizer([text], return_tensors="pt", padding=True)
out = model.generate(**batch, max_new_tokens=128, num_beams=1)
print(tokenizer.batch_decode(out, skip_special_tokens=True)[0])
```

Route plain prose around the model with the same shape-only rule this project
uses in production (`src/verbalizer/text.py`) — sentences the router marks
`passthrough` need no model call at all.

## Quickstart (CTranslate2, int8, CPU or GPU)

A quantized int8 export is published at
`aloudreader/marian-uk-verbalizer-ct2-int8` — about 52 MB, and
99.1–99.2% output-identical to the full-precision PyTorch model on the
benchmark above.

```bash
pip install ctranslate2 transformers sentencepiece huggingface_hub
```

```python
import ctranslate2
from huggingface_hub import snapshot_download
from transformers import MarianTokenizer

path = snapshot_download("aloudreader/marian-uk-verbalizer-ct2-int8")
tokenizer = MarianTokenizer.from_pretrained(path)
translator = ctranslate2.Translator(path, device="cpu", compute_type="int8")

text = "Потрібно прочитати XX розділ до понеділка."
ids = tokenizer(text, truncation=True, max_length=384, add_special_tokens=True)["input_ids"]
tokens = tokenizer.convert_ids_to_tokens(ids)

# suppress_sequences is required: this export uses a non-default decoder
# start token ("<s>", token id 0) that must never itself be generated.
result = translator.translate_batch([tokens], beam_size=1, suppress_sequences=[["<s>"]])
out_ids = tokenizer.convert_tokens_to_ids(result[0].hypotheses[0])
print(tokenizer.decode(out_ids, skip_special_tokens=True))
```

(The `<s>` decoder-start / suppress-sequence handling matters: this Marian
export uses a non-default decoder start token, and a naive CTranslate2
conversion silently drops the first generated word without it — see
[`scripts/convert_marian_ct2.py`](scripts/convert_marian_ct2.py).)

## Try it in the browser

```bash
pip install -e ".[ui]"
PYTHONPATH=src python scripts/gradio_app.py --checkpoint <path-to-checkpoint>
# or, for the int8 export:
PYTHONPATH=src python scripts/gradio_app.py --ct2 --checkpoint <path-to-ct2-export>
```

## Repository layout

```
src/verbalizer/            # library code: router, numeric helpers, evaluation
  evaluation/               # frame+slot domain checker, book-domain identity/numeric checks
  data/                     # offline dataset builders (books, English readings)
  _audit_numparse.py        # QA-only: parses numeral words back to integers
scripts/                    # training, conversion, evaluation, data-generation entry points
configs/
  train/                    # one YAML per training run (E1..E54), each with a header
                             # comment explaining what changed and why
  data/, model/, evaluate/  # dataset, architecture and judging configs
tests/                      # pytest suite (router, domain checker, numeric audit)
openspec/                   # the original OpenSpec change proposal for this experiment
AGENT.md                    # the no-hardcoded-rules policy, verbatim
```

`data/`, `outputs/` (trained checkpoints) and `reports/` (evaluation runs) are
not checked into this repository — see **Data and reproducibility** below.

## Data and reproducibility

Training combines:

- the public [`skypro1111/uk-text-normalization`](https://huggingface.co/datasets/skypro1111/uk-text-normalization)
  dataset (fully used in training; a 3,597-row slice is frozen out as the
  benchmark above and never trained on);
- synthetic rows generated offline with `num2words` and `pymorphy3` for
  domains and grammatical cases the base dataset covers thinly (Roman
  numeral + noun agreement, coordinates, acronyms, ranges, …) — every target
  is cross-checked by parsing it back to a value with
  [`src/verbalizer/_audit_numparse.py`](src/verbalizer/_audit_numparse.py);
  see `scripts/build_e*.py` for each batch's generator and its config's
  header comment for what problem it was built to fix.

**This repository does not include text extracted from books.** Some
generator scripts accept a `--books-dir` of held-out prose used, in earlier
iterations, to build a context/identity eval set (checking that the model
leaves ordinary sentences alone and generalizes beyond templated training
data) — that data is not redistributed here because the books' copyright
status varies. Point `--books-dir` at your own corpus of public-domain or
licensed Ukrainian prose to reproduce that part of the pipeline.

To retrain from scratch:

```bash
python scripts/train_marian.py --config configs/train/e54-roman-balance.yaml \
  --initialize-from outputs/e53-roman-rich/final   # or omit to train from the tokenizer up
```

Each `configs/train/eN-*.yaml` is a complete, runnable config; the header
comment says what changed from the previous run and what it fixed. Evaluate
with:

```bash
python scripts/evaluate_domains.py <checkpoint> --output reports/domain-eval.json
python scripts/evaluate_marian.py <checkpoint> --data reports/hf-uktn-frozen-3597.jsonl \
  --output reports/benchmark.json --num-beams 1 --force-marian
```

## License

Code is released under the [MIT License](LICENSE). Model weights on Hugging
Face carry their own model card and license — see the links above.

The `skypro1111/uk-text-normalization` dataset used in training has its own
license on Hugging Face; check it before redistributing data derived from
this model.
