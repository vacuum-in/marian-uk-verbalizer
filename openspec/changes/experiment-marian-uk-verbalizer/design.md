# Design: MarianMT Experiment for Ukrainian Text Verbalization

## 1. Design Principles

1. **Measure semantics, not only string similarity.**
   A fluent output with the wrong number or unit is a critical failure.
2. **Prevent train/test leakage.**
   News corpora often contain duplicates, syndicated text, and repeated numeric
   templates.
3. **Separate model capability from deterministic preservation.**
   Compare a raw model against an identical model trained with protected spans.
4. **Use a shared source/target tokenizer.**
   This is a monolingual transformation task with substantial copying.
5. **Keep stress placement separate.**
   Combining stress and verbalization would make failures difficult to attribute.
6. **Treat automatically generated labels as noisy.**
   Audit, filter, and maintain a human-reviewed test set.
7. **Benchmark the actual deployment path.**
   PyTorch quality alone is insufficient; export and measure CTranslate2 INT8.
8. **Separate external parity from policy imitation.**
   Compare both systems on identical frozen rows and report target quality and
   output-to-output agreement independently.

### 1.1 External-comparator protocol

- Freeze the 15,000 row IDs, sources, and dataset targets before API capture.
- Capture `https://verbalizer.skravchenko.dev` `/normalize` outputs in source
  order with endpoint and displayed model metadata.
- Make capture append-only and resumable, and verify every resumed ID/source.
- Submit one queued request at a time; recursively split a newline batch when
  output line count does not equal input line count.
- Never use captured 15k outputs for training, checkpoint selection, prompt
  construction, deterministic exceptions, or preprocessing derived from row IDs.
- Report exact match, CER, WER, semantic flags, catastrophic-output checks, and
  category-level paired wins for Marian and the external comparator.
- Report a secondary apostrophe-canonical view, while retaining the primary
  Unicode-sensitive metrics.
- Classify policy differences separately from semantic or morphological errors.
- Generate any corrective training examples from disjoint seeds and verify zero
  canonical-source overlap with the frozen 15k before training.
- Run the full comparison after selecting a checkpoint without using frozen
  test outcomes for checkpoint selection.

## 2. Assumptions

- The first experiment uses `skypro1111/ubertext-2-news-verbalized`.
- Latin-script words and abbreviations are copied unchanged.
- The target runtime is CPU-first.
- Sentences longer than the selected sequence limit may be segmented upstream.
- Production will retain deterministic validation and fallback logic.

These assumptions are inputs to the experiment, not established facts about all
future product requirements.

## 3. Repository Layout

```text
.
├── configs/
│   ├── data/
│   │   ├── pilot.yaml
│   │   └── full.yaml
│   ├── tokenizer/
│   │   ├── spm-8k.yaml
│   │   ├── spm-16k.yaml
│   │   └── spm-32k.yaml
│   ├── model/
│   │   ├── marian-small.yaml
│   │   └── marian-base.yaml
│   ├── train/
│   │   ├── pilot.yaml
│   │   └── full.yaml
│   └── evaluate/default.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   ├── manifests/
│   └── challenge/
├── src/verbalizer/
│   ├── data/
│   │   ├── download.py
│   │   ├── audit.py
│   │   ├── clean.py
│   │   ├── deduplicate.py
│   │   ├── categorize.py
│   │   ├── split.py
│   │   └── collator.py
│   ├── protection/
│   │   ├── detect.py
│   │   ├── replace.py
│   │   └── restore.py
│   ├── tokenizer/
│   │   ├── train_sentencepiece.py
│   │   └── build_marian_tokenizer.py
│   ├── model/
│   │   ├── config.py
│   │   ├── train.py
│   │   └── generate.py
│   ├── evaluation/
│   │   ├── normalize.py
│   │   ├── metrics.py
│   │   ├── semantic_checks.py
│   │   ├── error_taxonomy.py
│   │   └── report.py
│   ├── export/
│   │   ├── export_ct2.py
│   │   └── verify_export.py
│   └── benchmark/
│       ├── pytorch_cpu.py
│       └── ctranslate2_cpu.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── reports/
├── pyproject.toml
└── README.md
```

## 4. Dataset Pipeline

### 4.1 Input schema

Canonical processed row:

```json
{
  "id": "stable-content-id",
  "source": "У 2025 році витрати зросли на 15,5%.",
  "target": "У дві тисячі двадцять п’ятому році витрати зросли на п’ятнадцять цілих п’ять десятих відсотка.",
  "source_nfc": "...",
  "target_nfc": "...",
  "categories": ["year", "decimal", "percent"],
  "requires_change": true,
  "protected_spans": [],
  "group_id": "dedup-or-document-group",
  "quality_flags": [],
  "dataset_version": "..."
}
```

### 4.2 Unicode policy

- Convert text to Unicode NFC.
- Normalize only explicitly approved whitespace variants.
- Preserve Ukrainian apostrophes, quotation marks, dashes, non-breaking spaces,
  and case unless a documented normalization rule changes them.
- Do not silently apply broad NFKC normalization to the training labels.
- Record the original row for traceability.

### 4.3 Automatic audit

Produce counts and examples for:

- null or empty source/target;
- source equal to target;
- target still containing Arabic digits;
- target introducing digits not present in source;
- unexpected Latin-script changes;
- character-set anomalies;
- repeated words or phrases;
- extreme source/target length ratios;
- truncated sentences;
- mismatched punctuation;
- likely numerical value changes;
- invalid Ukrainian characters or broken apostrophes;
- exact duplicates;
- conflicting targets for the same source;
- near-duplicate sources with different targets.

Rows are not deleted silently. Every exclusion receives a reason code.

### 4.4 Quality tiers

- **Tier A:** passes all automatic checks.
- **Tier B:** minor formatting anomaly; usable only after deterministic repair.
- **Tier C:** ambiguous or suspicious semantic transformation; excluded from
  training but retained for audit.
- **Tier D:** clearly corrupt; excluded.

The first pilot trains only on Tier A plus explicitly repaired Tier B rows.

### 4.5 Deduplication and leakage prevention

Create:

1. exact source hash;
2. canonical source hash with normalized whitespace and punctuation;
3. template fingerprint replacing verbalizable spans with typed placeholders;
4. optional near-duplicate fingerprint using character n-grams or MinHash.

All rows sharing a duplicate or document group must remain in one split.
Template fingerprints used in the OOD test must be absent from training.

### 4.6 Splits

Build immutable manifests with stable IDs and checksums.

Recommended initial split:

- `train`: remaining accepted data;
- `validation`: 10,000 grouped examples;
- `test_id`: 10,000 grouped in-domain examples;
- `test_template_ood`: 5,000 examples whose numeric templates are absent from
  training;
- `test_challenge`: 2,000 curated difficult examples;
- `test_human_gold`: at least 500 manually reviewed examples.

If there is insufficient data for a category, use fixed counts rather than
forcing these sizes.

### 4.7 Identity examples

Add Ukrainian sentences that require no transformation.

Purpose:

- measure over-normalization;
- teach the model to copy ordinary text;
- reduce the tendency to paraphrase.

Pilot target: 15–25% identity examples. This is a proposed experimental range,
not a guaranteed optimum. Run an ablation at 0%, 15%, and 25%.

### 4.8 Category labels

At minimum:

- integer;
- negative number;
- decimal;
- fraction;
- percentage;
- year;
- full date;
- numeric date;
- time;
- time range;
- numeric range;
- currency;
- measurement unit;
- area and volume;
- temperature;
- ordinal;
- cardinal in oblique case;
- phone number;
- identifier;
- version;
- Roman numeral;
- abbreviation;
- mixed Cyrillic/Latin text;
- multiple interacting spans;
- no-change sentence.

A row may have multiple categories.

## 5. Protected-Span Variant

### 5.1 Protected content

By default protect:

- URLs;
- email addresses;
- file paths;
- code fragments;
- hashes and opaque identifiers;
- Latin words;
- mixed-case product identifiers;
- configured abbreviations that must remain unchanged.

The policy must be configurable because some future abbreviations may need
verbalization rather than preservation.

### 5.2 Replacement

Before tokenization:

```text
CPU Intel i7-14700K коштує 15 000 грн.
```

becomes:

```text
<KEEP_0> <KEEP_1> <KEEP_2> коштує 15 000 грн.
```

The protected values are stored out of band and restored after decoding.

Reserve a fixed set of sentinel tokens, initially `<KEEP_0>` through
`<KEEP_31>`. Reject or segment inputs requiring more sentinels.

### 5.3 Restoration invariants

The output is invalid when:

- a required sentinel is missing;
- a sentinel is duplicated;
- an unknown sentinel appears;
- sentinel order changes when order preservation is required;
- restoration changes any protected byte sequence.

Invalid output must trigger a deterministic fallback or return a structured
error. It must never be silently accepted.

## 6. Tokenizer

### 6.1 Type

Train SentencePiece Unigram tokenizers on accepted Ukrainian source and target
text combined.

### 6.2 Shared vocabulary

Use one shared vocabulary and tie:

- encoder input embeddings;
- decoder input embeddings;
- output projection when supported by the implementation.

This reduces parameters and supports copying.

### 6.3 Normalization

- External pipeline performs NFC normalization.
- SentencePiece uses an identity normalization rule where practical.
- Enable explicit user-defined symbols for sentinel tokens.
- Ensure combining acute accent remains representable even though stress is out
  of scope, so later pipeline experiments do not require rebuilding the tokenizer.

### 6.4 Vocabulary ablation

Compare:

- 8,000 pieces;
- 16,000 pieces;
- 32,000 pieces.

Default candidate: 16,000.

Select using:

- mean and p95 tokens per sentence;
- character coverage;
- fragmentation of Ukrainian inflections;
- fragmentation of numbers and unit forms;
- model parameter count;
- validation quality;
- CPU latency.

## 7. Model Configurations

Both models use `MarianMTModel` with a custom `MarianConfig`, shared vocabulary,
and training from scratch.

### 7.1 `marian-small`

```yaml
encoder_layers: 4
decoder_layers: 4
d_model: 384
encoder_attention_heads: 6
decoder_attention_heads: 6
encoder_ffn_dim: 1536
decoder_ffn_dim: 1536
dropout: 0.1
attention_dropout: 0.0
activation_function: gelu
max_position_embeddings: 384
share_encoder_decoder_embeddings: true
tie_word_embeddings: true
```

Purpose: low-latency pilot and CPU candidate.

### 7.2 `marian-base`

```yaml
encoder_layers: 6
decoder_layers: 6
d_model: 512
encoder_attention_heads: 8
decoder_attention_heads: 8
encoder_ffn_dim: 2048
decoder_ffn_dim: 2048
dropout: 0.1
attention_dropout: 0.0
activation_function: gelu
max_position_embeddings: 384
share_encoder_decoder_embeddings: true
tie_word_embeddings: true
```

Purpose: quality ceiling within a still-compact Marian design.

### 7.3 Special token configuration

Set explicitly and test:

```yaml
pad_token_id: ...
eos_token_id: ...
decoder_start_token_id: <pad token id>
bad_words_ids: []
forced_eos_token_id: <eos token id>
```

Never rely on accidental tokenizer defaults.

## 8. Training

### 8.1 Pilot stage

Use a stratified subset of approximately 100,000–200,000 accepted examples.

Objectives:

- validate the entire pipeline;
- identify tokenizer or label problems;
- estimate convergence;
- compare raw and protected-span variants;
- reject clearly unsuitable model configurations early.

### 8.2 Full stage

Train on all accepted training rows after pilot review.

Initial hyperparameter search:

```yaml
optimizer: AdamW
learning_rate: [0.0003, 0.0005]
weight_decay: [0.0, 0.01]
warmup_ratio: 0.05
lr_scheduler: linear
label_smoothing: [0.0, 0.1]
effective_batch_tokens: [16384, 32768]
max_source_tokens: 256
max_target_tokens: 384
epochs: 2-5
gradient_clipping: 1.0
precision: bf16_if_supported_else_fp16
gradient_checkpointing: false_for_small_true_if_required_for_base
seed: [17, 29, 43]
```

The ranges are experiment candidates. Final values must be selected from pilot
results rather than assumed.

### 8.3 Batching

Use dynamic token-based batching. Track source tokens, target tokens, examples,
and optimizer steps so runs are comparable across hardware.

### 8.4 Early stopping

Do not early-stop only on cross-entropy loss.

Primary selection score:

```text
composite =
    semantic_span_accuracy
  + identity_accuracy
  + protected_span_accuracy
  - critical_semantic_error_penalty
  - hallucination_penalty
```

The exact weights must be declared in the evaluation config and frozen before
the final test run.

### 8.5 Reproducibility

Each run records:

- Git commit;
- dataset revision and checksums;
- split manifest checksum;
- tokenizer checksum;
- full resolved config;
- package lock;
- CUDA, PyTorch, Transformers, SentencePiece, and CTranslate2 versions;
- hardware;
- seed;
- checkpoints;
- metrics;
- sample predictions.

## 9. Decoding

Evaluate:

1. greedy decoding;
2. beam size 2;
3. beam size 4.

Initial generation constraints:

```yaml
max_new_tokens: dynamic_based_on_input_with_hard_cap
length_penalty: 1.0
early_stopping: true_for_beam_search
no_repeat_ngram_size: 0
```

Do not enable generic anti-repetition constraints without evidence; they can
damage legitimate repeated forms.

Choose the lowest-cost decoding mode that does not materially worsen critical
semantic errors.

## 10. Evaluation

### 10.1 String metrics

Report:

- exact match after approved evaluation normalization;
- character error rate;
- word error rate;
- chrF or equivalent character n-gram score.

String metrics are secondary to semantic checks.

### 10.2 Semantic metrics

For each verbalizable span:

- numeric value preservation;
- sign preservation;
- decimal precision preservation;
- range boundary preservation;
- currency preservation;
- unit preservation;
- date component preservation;
- time component preservation;
- ordinal/cardinal intent;
- grammatical-case correctness where annotated or reliably checked.

Report macro and micro averages by category.

### 10.3 Safety metrics

- identity accuracy;
- over-normalization rate;
- protected-span exact preservation;
- source-content deletion rate;
- unsupported content insertion rate;
- punctuation corruption rate;
- sentinel restoration failure rate;
- output truncation rate;
- invalid Unicode rate.

### 10.4 Critical error definition

A critical error includes:

- changed numeric value;
- changed sign;
- changed date or time;
- changed currency or unit;
- dropped required numeric span;
- invented numeric span;
- corrupted protected content;
- output that cannot be restored safely.

Critical error rate must be reported separately and must not be hidden inside an
average text metric.

### 10.5 Human evaluation

At least two native Ukrainian reviewers inspect the final gold/challenge sample
when resources allow.

Label dimensions:

- semantic correctness;
- grammatical correctness;
- naturalness for speech;
- preservation of non-verbalized text;
- severity: pass, minor, major, critical;
- corrected target.

Measure reviewer agreement and adjudicate disagreements for critical cases.

## 11. Experimental Matrix

Minimum matrix:

| ID | Model | Tokenizer | Protection | Identity mix | Decoding |
|---|---|---:|---|---:|---|
| E1 | small | 16k | off | 15% | greedy |
| E2 | small | 16k | on | 15% | greedy |
| E3 | base | 16k | on | 15% | greedy |
| E4 | best model | 8k | best | 15% | greedy |
| E5 | best model | 32k | best | 15% | greedy |
| E6 | best model | best | best | 0% | greedy |
| E7 | best model | best | best | 25% | greedy |
| E8 | best checkpoint | best | best | best | beam 2 |
| E9 | best checkpoint | best | best | best | beam 4 |
| E10 | best checkpoint | best | best | best | CT2 INT8 |

Use pilot runs to prune obviously inferior branches before full training.

## 12. Benchmarking

### 12.1 Reference inputs

Create fixed suites:

- short: <= 64 Unicode characters;
- medium: 65–160 characters;
- long: 161–320 characters;
- one verbalizable span;
- multiple spans;
- protected mixed-script input;
- identity input.

### 12.2 Measurements

For batch sizes 1, 4, 16:

- warm and cold latency;
- p50, p95, p99 latency;
- sentences per second;
- source and generated tokens per second;
- peak RSS;
- model size on disk;
- startup time;
- thread count and CPU utilization.

Record exact CPU model, instruction set, operating system, and runtime options.

### 12.3 CTranslate2

Convert the selected model and test:

- FP32;
- INT8 or INT8-FP32 CPU compute;
- greedy and selected beam mode.

Run the complete quality suite after conversion. Quantization is rejected if it
causes an unacceptable increase in critical errors even when aggregate text
metrics remain similar.

## 13. Proposed Decision Gates

These are starting targets to make the experiment falsifiable. They are not
claims about expected results and may be revised once the deterministic
baseline is measured.

### Quality gate

- protected-span exact accuracy >= 99.9%;
- identity exact accuracy >= 99.5%;
- critical semantic error rate <= 0.5% on `test_id`;
- critical semantic error rate <= 1.0% on `test_template_ood`;
- no category with fewer than the agreed minimum examples is used to claim
  production readiness;
- INT8 critical error increase <= 0.1 percentage point absolute versus the same
  checkpoint in PyTorch.

### Performance gate

Set from product constraints before the final benchmark. Placeholder target:

- batch-1 p95 <= 75 ms for a medium sentence on the declared reference desktop
  CPU;
- peak RSS <= 500 MB for the selected INT8 runtime;
- model artifact <= 150 MB.

If the product target is browser/mobile CPU, run a separate benchmark; desktop
numbers must not be extrapolated.

### Recommendation logic

- **Proceed with Marian:** quality and performance gates pass.
- **Proceed as hybrid only:** model is useful but requires deterministic routing,
  validation, or fallback for specific categories.
- **Reject:** critical semantic reliability or deployment cost is unacceptable.

## 14. Baselines

Required baselines:

1. Original deterministic verbalizer, if available.
2. Identity output.
3. Rule-only output for categories supported by current rules.
4. Marian-small raw.
5. Marian-small protected.
6. Marian-base protected.

Optional comparison:

- a similarly sized custom T5 encoder-decoder using the same data and tokenizer
  budget.

The optional T5 comparison should be added only after the Marian experiment is
reproducible; otherwise it expands scope prematurely.

## 15. Risks and Mitigations

### Risk: Automatically generated targets contain wrong morphology or values

Mitigation:

- semantic filters;
- conflict detection;
- human gold set;
- corrected hard-example replay set;
- do not report training-corpus metrics as gold quality.

### Risk: News-only domain does not generalize to books or application UI

Mitigation:

- template OOD test;
- challenge set from intended product inputs;
- later domain adaptation dataset;
- explicit domain label in reports.

### Risk: Model paraphrases text instead of only verbalizing it

Mitigation:

- identity examples;
- copy-sensitive evaluation;
- protected spans;
- reject outputs with unexplained edits;
- fallback path.

### Risk: Split leakage inflates exact match

Mitigation:

- group-based splits;
- template fingerprints;
- near-duplicate analysis;
- immutable manifests.

### Risk: Tokenizer fragments Ukrainian morphology excessively

Mitigation:

- 8k/16k/32k ablation;
- shared vocabulary;
- token-length reports;
- inspect suffix fragmentation and rare symbols.

### Risk: INT8 changes rare-category behavior

Mitigation:

- full post-export evaluation;
- category-level deltas;
- critical-error gate.

## 16. Final Report

The final report must contain:

1. dataset audit and exclusions;
2. split methodology and leakage checks;
3. tokenizer comparison;
4. run table with resolved configs;
5. category metrics;
6. critical error examples;
7. human evaluation;
8. latency and memory measurements;
9. PyTorch versus CTranslate2 comparison;
10. clear recommendation and remaining unknowns.

No result may be described as production-ready without the human gold set,
critical-error analysis, and deployment-runtime benchmark.

## 17. Structured Spoken-Domain Adaptation

The user target is represented by the 30-family verified synthetic generator,
covering interacting coordinates, dates and times, money, percentages, ranges,
measurements, storage units, formulae, grouped counts, addresses, phones,
identifiers, product versions, files, legal references, URLs, email addresses,
and Roman numerals.

Use three disjoint data roles:

1. a newly generated adaptation train seed plus identity examples;
2. a separate generated validation seed for checkpoint selection;
3. the existing immutable 15k diverse seed and user-authored regressions for
   final acceptance only.

The dataset builder must fail on any canonical source overlap between these
roles. User-authored regression sources must never enter train or validation.
The first explicit regressions are the coordinate pair `47.1024, 32.2757` and
`XX розділ`. Roman data is a finite 1–39 task and may contain only the glyphs
`I`, `V`, and `X`; explicit regression source strings remain excluded.

Before tokenization only, map NBSP/narrow-NBSP to ordinary space and map
semantically equivalent unsupported glyphs (`℃`, `≈`, `÷`, `≥`, `≤`) to the
tokenizer-covered forms (`°C`, `~`, `/`, `>=`, `<=`). Reports retain the raw
source. The same mapping must run in training, evaluation, and Gradio.
Roman glyphs remain unchanged before tokenization and are verbalized by Marian.
Active Roman train, validation, and test splits use values 1 through 39 and
contain only `I`, `V`, and `X`. All supported values occur in training, while
validation and test hold out sentence templates. Neither inference nor the
fidelity layer contains a Roman parser, source-row lookup, or `XX` exception.
Compact explicitly signed ranges are canonicalized to `від … до …` so the
boundary signs cannot merge with neighboring values. Four-digit years in the
construction `у/в YYYY році` are converted with a generic Ukrainian ordinal
inflector to the masculine locative form before tokenization. These are
contextual value rules, not source-row exceptions.
Standalone integers with at least five digits are converted to Ukrainian
cardinal words before tokenization (`200619` becomes `двісті тисяч шістсот
дев’ятнадцять`); alphanumeric IDs are excluded. This value-preserving
representation removes long-number decoding ambiguity.
Evaluation exact-match also ignores whitespace immediately before punctuation,
because the oracle contains generated forms such as `літрів .` that are not a
meaningful TTS distinction.

The profile's exact-match gate is 100% on the finite regression set and a stated
measured result on the frozen 15k acceptance set. A failure to reach 100% on the
15k suite is reported as evidence for hybrid routing, not concealed by aggregate
CER/WER. The model-only output remains experimental even if the finite suite
passes because an encoder-decoder cannot prove arithmetic or grammar invariants.

### 17.1 Selected E5 evidence

Residual hard curricula were generated from disjoint seeds without reading
frozen targets or predictions. Their manifests report zero canonical-source
overlap with validation, the frozen acceptance sources, and user regressions.
Checkpoint selection used only the corresponding generated validation split.

The progression on the immutable 15,000-row acceptance suite was E2b
14,934 exact, E3 14,971, E4 14,990, and E5 14,999 before segmentation.
Splitting independent sentences removed the sole cross-clause interference;
the integrated E5 path records 15,000/15,000 exact, CER 0, WER 0, zero output
truncations, and 4/4 exact current user regressions. This finite result does not
replace human-gold review or deterministic production validation.

On the separate residual 5,000-row validation split, E5 records 4,970 exact
with whole-row inference and 4,853 with sentence segmentation. The lower
segmented stress score is retained as counter-evidence against a universal
100% claim; the selected runtime result is scoped to the requested frozen suite.

## 18. Main 95k News-Domain Adaptation

`main_train_95k` is a training pool and `golden_5k` is its official disjoint
test. Four rows from two ambiguous duplicate-source groups are quarantined.
The remaining 94,996 rows are deterministically split into 89,996 train and
5,000 validation rows. Golden sources and user regressions are exclusion-only;
all canonical-source overlap checks must remain zero.

News-only E6 demonstrates domain learnability but is not selectable because it
forgets structured policy. E7 therefore interleaves every news train row with
one deterministic structured replay row. E8 applies a bounded low-learning-rate
structured repair to E7. Gradient updates and loss-based checkpoint selection
read only the internal news and structured validation partitions, never golden
labels or public API outputs. Aggregate golden metrics were observed after each
completed E6–E8 stage, so golden is evaluation evidence but no longer an
untouched single-use test after E6.

The public Gemma 3 270M output is an evaluation comparator, not training data.
Completion requires all 5,000 ordered remote responses and a no-worse gate:
local target exact must be at least remote exact, local CER/WER must be no
higher, and excess semantic flags and critical failures must be no greater.
Unlike the synthetic 15k gate, news parity does not require 100% target exact
because the news labels contain lexical paraphrases beyond TTS normalization.

Repeated phone digits receive a deterministic post-generation fidelity check.
The source phone pattern determines the complete digit sequence; only a matched
spoken phone span is replaced. This is necessary because seq2seq decoding can
collapse repeated identical digit words even when they are supplied verbatim.
The same fidelity layer runs in evaluation and Gradio.

## 19. Learned Roman-Numeral Adaptation

E9 through E12 continue from E8 and train Roman morphology rather than adding
an inference rewrite. The final E12 mix contains 150,000 general Roman rows,
45,024 extra small-value replay rows, 120,000 structured rows, 89,996 news rows,
and 20,000 signed-temperature-range rows. Frozen Roman holdout and explicit
user regressions are exclusion-only, with zero source and Roman-value overlap.

E12 reaches 2,440/2,500 exact (97.60%) on unseen Roman values/templates and 4/4
on explicit user regressions, including `XX розділ`. It reaches 14,984/15,000
exact on the broad structured suite. On news golden 5k it reaches 21.46% exact,
8.44% CER, and 15.86% WER, passing all no-worse comparisons against the complete
public Gemma 3 270M capture. Greedy BF16 is selected because beam search reduced
structured exactness. CPU and GPU serving use the same BF16 generation path and
infer sentence segments independently.

## 20. Deterministic Financial Identifiers and Large Integers

Decade forms ending in `-х` are converted to Ukrainian plural ordinals before
tokenization (`1990-х` becomes `тисяча дев’ятсот дев’яностих`). Standalone
integers containing at least five digits, including space-grouped values, are
converted to Ukrainian cardinal words. Four-digit grouped values retain the
previous model behavior.

Sixteen-digit payment-card numbers grouped in fours and `UA` IBAN-like values
are treated as exact numeric sequences. Every digit is exposed separately to
Marian. The shared post-generation fidelity layer restores the complete source
digit sequence when seq2seq decoding drops or repeats a digit. It does not log,
store, or look up any particular card or account value. The same handling is
used by evaluation and Gradio on CPU and GPU.

## 21. Model-Learned Latin-Letter Disambiguation and IVX Policy

There is no inference-time lexical rule for `C` or `D`. Marian sees the original
glyph and learns its spoken letter name from varied terminal, class, category,
designation, vehicle, document, server, and medical contexts. The disjoint
letter validation set reaches 1,760/1,760 exact, including templates absent from
training. Explicit user regressions remain exclusion-only.

The active Roman generator is mechanically restricted to values 1 through 39,
so every Roman token consists only of `I`, `V`, and `X`. It rejects a requested
maximum above 39, and an automated test audits train, validation, and test
Parquet rows. The trainer independently audits both configured datasets and
fails before model loading if Roman supervision violates that policy. E14/E15
are rejected negative evidence because their 1–99 corpus
included `L`. The clean E16–E19 lineage starts from pre-Roman E8 and never loads
E9–E15 weights. E19 checkpoint 8000 is selected: 421/468 exact on disjoint IVX
templates, 3/3 explicit letter regressions, and 4/4 user regressions. It records
1,072/5,000 exact on news golden 5k, 1,805/1,813 on the focused single-clause
suite, and 14,860/15,000 on the frozen broad structured acceptance file.
