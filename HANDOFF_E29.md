# E28/E29 training handoff

This document records the work completed in WSL and is the starting point for
the next agent. The repository root is:

`/home/devops/verbolizer/marian-uk-verbalizer-openspec`

The sibling English transducer repository is:

`/home/devops/verbolizer/en-uk-transducer`

## Non-negotiable product policy

Read [AGENT.md](AGENT.md) before changing code. The runtime must not contain
hardcoded verbalization rules. Do not add dictionaries, word exceptions,
numeric conversion tables, candidate inventories, masking/restoration,
post-processing, output correction, validation, or rejection. Runtime logic may
only route unchanged input to a selected model or pass through text that needs
no verbalization. Improve output with model architecture, training data, and
training—not hand-written readings.

Numeric/formula detection in `src/verbalizer/numeric.py` is routing metadata
only. It does not produce a spoken form. The current evaluator writes raw model
output; it does not repair or reject predictions.

## Current runtime routing

`src/verbalizer/text.py` currently routes:

1. A standalone 1–4-character all-capital Latin/Cyrillic token to Marian.
2. A Latin word with trailing digits to Marian.
3. Any recognized numeric/formula/Roman/identifier span to Marian.
4. Other Latin spans to the trained English transducer.
5. Other text through unchanged.

The English adapter is in `src/verbalizer/english_transducer.py`. It uses the
sibling CTranslate2 model and supplies only the model's serialized input tags;
there is no pronunciation dictionary in this repository. Do not change routing
to promote E29 until the full regression suite passes.

## Environment and verification

Training was run in WSL2 on an NVIDIA GeForce RTX 3070 Laptop GPU. Use the
existing environment; do not install host packages:

```bash
cd /home/devops/verbolizer/marian-uk-verbalizer-openspec
export PYTHONPATH=src
.venv-ct2/bin/python -m pytest -q
```

The last repository check passed: **61 tests passed**.

## English transducer source data

The reviewed source corpus is:

`/home/devops/verbolizer/en-uk-transducer/data/splits/en-uk-combined-v1`

It contains raw `src`/`tgt` pairs and metadata. Counts are 40,231 train,
2,177 validation, and 2,158 test. The test split was kept untouched for all
Marian transfer evaluations. The specialist's documented baseline is 77.53%
exact match (`en-uk-combined-v1-medium-test.json`); the scheduled specialist
report in the sibling repository records 80.03% (`en-uk-combined-v1-medium-sched-test.json`).

## Dataset construction completed

`scripts/build_english_transfer_data.py` performs a literal raw-pair transfer:
`src` becomes `source` and `tgt` becomes `target`. It does not recreate any
pronunciation logic. It writes:

- `data/e28_english_transfer/train.parquet` — 162,334 rows: the E27 Marian
  corpus plus English-transducer train rows.
- `data/e28_english_transfer/validation.parquet` — 22,181 rows: the E27
  validation corpus plus English-transducer validation rows.
- `data/e28_english_transfer/english-train.parquet` — 40,231 rows.
- `data/e28_english_transfer/english-validation.parquet` — 2,177 rows.
- `data/e28_english_transfer/english-test.parquet` — 2,158 held-out rows.
- `data/e28_english_transfer/manifest.json` — source paths, row counts, and
  SHA-256 hashes.

The E27 base data came from:

- `data/e27_abbreviation_routing/train.parquet` — 122,103 rows.
- `data/e27_abbreviation_routing/validation.parquet` — 20,004 rows.
- `data/curated/abbreviation-supervision.jsonl` — reviewed training examples,
  not runtime rules.

## Training completed

### Earlier checkpoints retained

These earlier experiments remain available for comparison, but they are not
the current handoff candidate:

- `outputs/e25-numeric-correctness/final` — 3,250 steps. Its old report was
  `reports/hf-uktn-numeric-release-3597-e25-wsl.json`.
- `outputs/e26-model-numeric/final` — 7,052 steps. Its old report was
  `reports/hf-uktn-numeric-release-3597-e26-wsl.json`; it had severe numeric
  regressions and must not be used as a release model.
- `outputs/e27-abbreviation-routing/final` — 7,632 steps. It added reviewed
  abbreviation supervision for HDD/LTE/BRT/ДТП as training data only. The
  routed report `reports/hf-uktn-routing-e27-3597.json` scored 1,058/3,597
  exact (29.41%) with 100% numeric-value and identifier-group fidelity. E27 is
  the strongest existing Ukrainian routed baseline in this handoff.

### E28 mixed transfer

Config: `configs/train/e28-english-transfer.yaml`

Command:

```bash
PYTHONPATH=src .venv-ct2/bin/python scripts/train_marian.py \
  --config configs/train/e28-english-transfer.yaml \
  --initialize-from outputs/e27-abbreviation-routing/final
```

Output: `outputs/e28-english-transfer/final`.

E28 ran one mixed epoch (10,146 steps). Direct Marian evaluation on the
English test split was only 540/2,158 = **25.02% exact match**. Report:
`reports/e28-english-heldout.json`.

### E29 English specialization

Config: `configs/train/e29-english-specialization.yaml`.

Command used:

```bash
PYTHONPATH=src .venv-ct2/bin/python scripts/train_marian.py \
  --config configs/train/e29-english-specialization.yaml \
  --initialize-from outputs/e28-english-transfer/final
```

E29 trained for 20 epochs / 50,300 optimizer steps on the English-only
transfer split. It completed successfully. Final checkpoint:

`outputs/e29-english-specialization/final`

Important files in that directory include `pytorch_model.bin`, tokenizer files,
`run_manifest.json` (under the output directory), `result.json`, and trainer
state. E29 English-only final test score:

- 1,282 / 2,158 exact = **59.41%**.
- CER: **11.04%**.
- Report: `reports/e29-english-heldout-final.json`.
- CSV: `reports/e29-english-heldout-final.csv`.

E29 improved English but remains below the existing specialist baseline.

## Full Marian-only regression completed

The complete frozen evaluation set is:

`reports/hf-uktn-frozen-3597.jsonl`

It was evaluated with every row forced through raw E29 Marian (`--force-marian`),
so the dedicated English transducer was not used.

Command:

```bash
PYTHONPATH=src .venv-ct2/bin/python scripts/evaluate_marian.py \
  outputs/e29-english-specialization/final \
  --data reports/hf-uktn-frozen-3597.jsonl \
  --output reports/hf-uktn-e29-full-raw-marian.json \
  --csv reports/reference-vs-e29-full-raw-marian.csv \
  --limit 0 --batch-size 32 --num-beams 1 --force-marian
```

Results:

- 95 / 3,597 exact = **2.64%**.
- CER: **48.47%**.
- Numeric-value fidelity: **100%** for 2,406 recognized numeric rows.
- Identifier-group fidelity: **100%** for 275 rows.
- Digit fidelity: **100%** for 306 formula/identifier rows.
- Identity accuracy: 3 / 36 = **8.33%**.
- The model catastrophically forgot Ukrainian sentence verbalization during
  English-only specialization.

Reports and CSV:

- `reports/hf-uktn-e29-full-raw-marian.json`
- `reports/hf-uktn-e29-full-raw-marian-numeric-failures.jsonl` (normally empty
  under the routing-only evaluator)
- `reports/reference-vs-e29-full-raw-marian.csv`
- `/mnt/c/lmfiles/reference-vs-e29-full-raw-marian.csv`

The CSV columns are `id`, `source`, `reference`, `prediction`, `exact`, and
`categories`.

## Evaluation code changes

`scripts/evaluate_marian.py` now supports:

- `--force-marian`: evaluate raw Marian even when normal routing would select
  the English transducer.
- `--csv PATH`: write source/reference/prediction comparison rows.

The evaluator intentionally records the selected model's raw output. Do not
reintroduce correction or rejection logic.

`scripts/train_marian.py` was made tolerant of missing optional MLflow extras:
tracking calls disable themselves after an optional tracking runtime error, so
local WSL training can proceed without installing host dependencies.

## What the next agent should do

Do **not** switch live routing to E29. The next training objective is a
balanced consolidation model that recovers Ukrainian verbalization while
retaining the English capability. A sensible experiment is to construct a new
versioned replay/mixed dataset from the existing E27 Ukrainian corpus and the
English transfer corpus, with an explicitly measured English/Ukrainian sampling
ratio, then initialize from E29 and evaluate both held-out English and the full
3,597-row frozen set. Keep `english-test.parquet` and
`reports/hf-uktn-frozen-3597.jsonl` untouched.

Promotion requires evidence on both domains. At minimum, report:

- English transducer holdout exact/CER against the 77.53% / 80.03% baselines.
- Full 3,597-row exact/CER and the separate numeric/identifier fidelity fields.
- A CSV containing source, reference, and raw model output.
- `PYTHONPATH=src .venv-ct2/bin/python -m pytest -q`.

If consolidation does not pass, keep the existing routed architecture. Do not
paper over failures with runtime exceptions or output substitutions.

## Copy-paste prompt for the next agent

```text
Continue the Marian English-transfer work in
/home/devops/verbolizer/marian-uk-verbalizer-openspec.

Read AGENT.md and HANDOFF_E29.md completely before acting. Never add any
hardcoded verbalization rule, dictionary, exception, numeric conversion,
candidate list, post-processing, correction, validation, or rejection. Runtime
code may only route unchanged input to a model or pass it through. Improve
quality only through data/model training.

E29 is complete at outputs/e29-english-specialization/final. It was trained on
40,231 raw English-transducer pairs for 20 epochs. It scores 59.41% exact on
the held-out 2,158-row English test, below the existing English specialist
(77.53% / 80.03% reports). When forced through Marian on the frozen 3,597-row
Ukrainian set it scores only 2.64% exact, although numeric and identifier
fidelity metrics are 100%; English-only specialization caused catastrophic
Ukrainian forgetting.

Construct a new versioned balanced replay/consolidation dataset from
data/e27_abbreviation_routing/{train,validation}.parquet and the literal raw
English transfer files in data/e28_english_transfer/. Keep
data/e28_english_transfer/english-test.parquet and
reports/hf-uktn-frozen-3597.jsonl untouched. Train a new experiment from E29,
measure several English/Ukrainian sampling ratios, and evaluate raw Marian on
both frozen suites. Use scripts/evaluate_marian.py with --force-marian and
--csv. Do not change live routing or promote a checkpoint until the two-domain
results and pytest are reported. Preserve all prior artifacts and document the
new experiment, hashes, commands, metrics, and CSV paths in a new handoff.
```
