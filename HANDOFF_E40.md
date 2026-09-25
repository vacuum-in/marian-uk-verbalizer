# E30–E40 training handoff

Supersedes `HANDOFF_E29.md`, which is retained for history but is wrong on two
counts recorded below. Repository root:
`/home/devops/verbolizer/marian-uk-verbalizer-openspec`

## Non-negotiable product policy (unchanged)

Read [AGENT.md](AGENT.md). The runtime must not contain hardcoded verbalization
rules: no dictionaries, word exceptions, numeric conversion tables, candidate
inventories, masking/restoration, post-processing, output correction,
validation, or rejection. Runtime logic may only route unchanged input to a
selected model or pass through text needing no verbalization.

Everything below improves quality through dataset construction and training
only. `src/verbalizer/text.py` routing is **unchanged**; no checkpoint has been
promoted to live routing.

## Current state

| Checkpoint | UK EM | CER | WER | Latin | non-Latin | pass-through | EN holdout |
|---|---|---|---|---|---|---|---|
| E28 (session start) | 34.83% | 6.32% | 17.02% | 28.79% | 38.76% | 34/36 | 25.02% |
| E30 orthography | 59.52% | 4.84% | 10.73% | 35.29% | 75.28% | 35/36 | 33.97% |
| E31 corpus recovery | 66.33% | 3.71% | 8.52% | 47.78% | 78.39% | 36/36 | 34.89% |
| E32 capacity 52.7M | 79.48% | 2.52% | 4.99% | 73.25% | 83.53% | 31/36 | 77.11% |
| E34b pass-through | 81.26% | 2.22% | 4.51% | 74.24% | 85.83% | 35/36 | 71.73% |
| E35 balanced growth | 81.79% | 1.81% | 4.06% | 75.44% | 85.92% | 35/36 | 75.76% |
| E36b clean long | 82.15% | 1.80% | 4.08% | 75.86% | 86.24% | 35/36 | 74.19% |
| **E39 people counts** | **82.90%** | **1.59%** | **3.72%** | 75.94% | **87.43%** | **36/36** | **74.37%** |
| E40 oblique counts | 82.43% | 1.68% | 3.84% | 75.37% | 87.02% | 35/36 | 73.73% |

Numeric-value, identifier-group and digit fidelity are **100% on every row of
every checkpoint above**. 61 tests pass.

**Two candidates, no promotion decision made:**

- `outputs/e39-people-counts/final` — highest benchmark score.
- `outputs/e40-oblique-counts/final` — 0.47 pts lower, but the only checkpoint
  that reads `на 1001 людину` correctly (9/9 targeted numeric cases vs E39's 5/9).

All models are `configs/model/marian-base-scaled.yaml`: 6+6 layers, d_model 512,
52.7M parameters, spm-16k tokenizer.

## What HANDOFF_E29 got wrong

1. It records E27 as the strongest Ukrainian baseline at 29.41%. E28 raw-Marian
   scores **34.83%**; E28 was judged on its English number alone and its
   Ukrainian result was never compared like-for-like.
2. It recommends keeping the routed architecture. Routing to the sibling English
   transducer **costs 5.4 points** (E28 routed 29.47% vs forced 34.83%), and E32
   onward scores 73–77% on the specialist's own English test while doing
   Ukrainian. The two-model split is obsolete.

## The four findings that produced the gain

1. **Apostrophe convention (+24.69).** Training targets used curly `’`
   exclusively (111,053 occurrences); frozen references use straight `'`
   exclusively. 1,285 rows (35.7%) contained one, capping exact match at
   **64.28%** regardless of model quality. Both glyphs were already distinct
   tokens in the vocab, so this was pure dataset hygiene.
2. **The E24 build discarded 49,886 rows and 160 category labels (+6.81).**
   `data/hf_uktn_e23/train.parquet` has 167,893 rows / 169 categories;
   `e24_pipeline_hf_plus_synthetic` has 118,007 / 9. 96,760 of the dropped rows
   were excluded as `deterministic_pipeline_sentence` — sentences *the router*
   decided did not need Marian, baking the routed architecture into the corpus.
3. **Capacity (+13.15).** 23M → 52.7M params moved Latin-bearing rows from
   47.78% to 73.25% with the corpus held fixed. A coverage analysis had argued
   this was impossible; it was wrong (see "Corrections" below).
4. **Signals must grow together (+0.53).** E34b traded English for Ukrainian.
   E35 added 15,879 previously-unused reviewed transducer pairs alongside the
   pass-through rows and beat E34b on both domains.

## Corrections to earlier reasoning in this session

- **"90% is unreachable because coverage tops out at 67.6%"** — wrong. Coverage
  really does cap there, but E32 showed parameters buy compositional
  grapheme→Cyrillic generalization that vocabulary cannot. Do not treat token
  coverage as predictive of score.
- **"112M will beat 52.7M"** — wrong. E33 tracked ~33% worse eval loss at every
  checkpoint and never converged. Capacity is exhausted at this data budget.
- **The numeric audit reported 14.4%, then 5.2%, then 0.2% error** — the first
  two were tooling bugs (prefix matching read `сторони`/`становила` as `ст`=100;
  the extractor split `30 000 000` into `[30,0,0]`). Trust only the final 99.21%,
  which was reached after hand-triaging every residual mismatch.

## Dataset audit

`scripts/audit_numeric_targets.py` parses targets back to integers and compares
against the source, bucketing digit-wise/decimal/range/time rows that legitimately
do not encode a single cardinal. Offline QA only; never runs in the pipeline.

```bash
PYTHONPATH=src .venv-ct2/bin/python scripts/audit_numeric_targets.py \
  data/e40_oblique_counts/train.parquet --sample 1000
```

Result on the shipped corpus: **98.99% value accuracy**, and every residual
mismatch is an audit-tool artefact (model names such as `Gemma 3 8B`, read
digit-wise by design). **Zero genuine numeric value errors.**

All 28,497 generated rows across E38/E39/E40 were verified to round-trip:
**zero incorrect**.

## Datasets

| Path | Rows | What it adds |
|---|---|---|
| `data/e30_canonical_orthography` | 164,344 | apostrophe canonicalized; both-glyph source augmentation |
| `data/e31_unified_labelled` | 265,509 | full labelled HF split + 2,099-row stratified dev set (179 categories) |
| `data/e32_transliteration_scale` | 266,801 | bronze two-provider-agreement pairs |
| `data/e34_passthrough_balance` | 281,801 | 15,000 reviewed `requires_change=False` rows |
| `data/e35_balanced_growth` | 297,680 | 15,879 unused reviewed transducer pairs (v18-clean) |
| `data/e36_cmudict_growth` | 298,894 | cmudict two-provider pairs |
| `data/e37_numeric_fixes` | 304,746 | 3 ungrammatical rows dropped; 5,855 separator-stripped variants |
| `data/e38_bare_cardinals` | 316,512 | 11,766 bare-cardinal pairs |
| `data/e39_people_counts` | 324,243 | 7,731 people-count readouts |
| `data/e40_oblique_counts` | 333,243 | 9,000 accusative animate-count rows |

`data/e31_unified_labelled/dev-stratified.parquet` (2,099 rows, all 179
categories) is the selection set. Verified disjoint from train and from the
frozen benchmark. It correctly flagged the overfitting point in E31 and E34b.

## Generated-data provenance

E38/E39 use `num2words` in **nominative frames only**, where citation form is
correct and no case or gender agreement is at stake. E40 additionally inflects
**only the trailing unit word** with `pymorphy3` + the Ukrainian VESUM
dictionary; inflecting every word breaks numeral-scale agreement, producing
`три тисячу` and `п'ять тисячі` instead of `три тисячі` / `п'ять тисяч`. E40 is
scoped to values ending in 1 (excluding teens) with **animate** nouns, because
masculine inanimate accusative takes `один день`, not `одного день`.

Every generated pair is cross-checked against `src/verbalizer/_audit_numparse.py`,
an independently written parser, and dropped unless the value round-trips.
num2words and that parser were validated against each other over 12,000 values
with **zero disagreements**.

`pymorphy3` and `pymorphy3-dicts-uk` were installed into `.venv-ct2`.
`gradio==6.19.0` was also installed — a declared `ui` extra that was missing.

## Known defects (not fixed)

1. **The frozen benchmark contains 5 ungrammatical references.** The template
   `У черзі # людей чекають на аналіз гепатиту С.` pairs a random number with a
   fixed `людей` + plural verb, which is only grammatical when the number ends
   in 5–9 or 0. Rows `291`, `381`, `61` need `одна людина`; `123`, `334` need
   `людини`. These rows are **unwinnable by any model**. The 3 equivalent
   training rows were dropped in E37. Editing
   `reports/hf-uktn-frozen-3597.jsonl` would break comparability with every
   report from E28 onward — **open decision, deliberately not taken.**
2. **~96 remaining misses are pure letter-case**, on references that contradict
   themselves (`Де Пе Ес` vs `де пе ес`). Ignoring case raises E36b from 82.15%
   to 84.82%; ignoring case and punctuation, 85.18%.
3. **Beam search stopped helping** once the model passed ~80% (E35 greedy 81.79%
   vs beam-4 81.62%). Use greedy.

## Why 90% is likely the wrong target

Of the 642 misses at E36b, 96 are case-only and 13 punctuation-only. Of the
remaining true content errors, 58.9% differ by a single word, and those words
are largely defensible alternatives rather than mistakes: `ютуб`/`ютюб`,
`кідз`/`кідс`, `готлайн`/`хотлайн` (the standard Ukrainian h→г vs h→х dispute),
`з`/`із`. The two generation providers disagree with each other on **36.5% of
ordinary words** for the same reason.

Reaching 90% single-string exact match would require guessing one arbitrary side
of every convention dispute, including ones the reference is inconsistent about.
The honest routes are scoring against an accepted-variants set, or making the
references case-consistent — either puts the true figure near **85% today**.

## Environment

WSL2, NVIDIA RTX 3070 Laptop 8 GB.

**The GPU faults under memory pressure and it is not a hardware problem.**
`dmesg` shows `dxgk: dxgkio_make_resident: Ioctl failed: -12` (ENOMEM) — WSL2's
GPU paravirtualization layer failing against Windows-host VRAM pressure. The card
reports 50 °C, no throttling, no ECC errors, and Linux had 28 GB free. Three
training runs died mid-flight this way, and one evaluation wedged for 3h52m
producing nothing.

Every run that stayed near 3 GB VRAM completed. Use
`max_batch_tokens: 4096`, `max_batch_examples: 16`, `gradient_accumulation_steps: 2`,
`num_workers: 0` when the card is shared. Evaluation runs fine on CPU at
~17 sentences/s (full 3,597-row pass in ~3.5 min) — prefer that when another job
holds the GPU:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src OMP_NUM_THREADS=14 \
  .venv-ct2/bin/python scripts/evaluate_marian.py CHECKPOINT \
  --data reports/hf-uktn-frozen-3597.jsonl --output OUT.json --csv OUT.csv \
  --limit 0 --batch-size 16 --num-beams 1 --force-marian --allow-cpu
```

## Reproducing

```bash
cd /home/devops/verbolizer/marian-uk-verbalizer-openspec
export PYTHONPATH=src
.venv-ct2/bin/python -m pytest -q                      # 61 passed

.venv-ct2/bin/python scripts/train_marian.py \
  --config configs/train/e40-oblique-counts.yaml \
  --initialize-from outputs/e39-people-counts/final

.venv-ct2/bin/python scripts/gradio_app.py \
  --checkpoint outputs/e40-oblique-counts/final --force-marian --port 7860
```

`scripts/gradio_app.py` gained a `--force-marian` flag mirroring the evaluator's.
It is off by default and only selects which model receives the unchanged text.

## Suggested next steps

1. Decide E39 vs E40, or rebuild E40 with ~2–3k oblique rows and more frame
   diversity to keep the numeric fix while shrinking the 0.47-point regression.
2. Decide whether to correct the 5 ungrammatical frozen references.
3. If the benchmark metric is revisited, an accepted-variants scorer is worth
   more than further training — the bronze shards already record the alternative
   readings.
4. Do not retry capacity above 52.7M without substantially more data.
