# Proposal: Experiment with MarianMT for Ukrainian Text Verbalization

## Intent

Build a reproducible experiment that determines whether a compact Marian-style
encoder-decoder model can accurately verbalize Ukrainian text for a TTS pipeline
while remaining fast enough for CPU inference.

The experiment must answer four questions:

1. Can Marian learn Ukrainian number and symbol verbalization with correct
   morphology and local context?
2. Does deterministic protection of Latin words, abbreviations, URLs, emails,
   and code-like spans improve fidelity?
3. Which compact Marian configuration gives the best quality/latency trade-off?
4. Does INT8 CTranslate2 inference preserve sufficient quality for production
   consideration?

## Context

Primary candidate dataset:

- `skypro1111/ubertext-2-news-verbalized`
- Approximately 889,000 Ukrainian news sentence pairs
- Source contains written forms with numbers and symbols
- Target contains verbalized equivalents
- Targets were generated automatically, so dataset quality must be audited before
  training and must not be treated as gold human annotation

The existing TTS pipeline should keep Latin words and abbreviations unchanged.
Stress placement is a separate downstream task and is intentionally excluded
from this experiment.

## Scope

### In scope

- Dataset inspection, cleaning, deduplication, and leakage-safe splitting
- Ukrainian-focused shared SentencePiece tokenizer
- MarianMT models trained from scratch with a custom vocabulary
- Two model-size experiments
- Pure end-to-end and protected-span preprocessing variants
- Category-aware evaluation of verbalization
- Human-reviewed challenge set
- PyTorch inference benchmark
- CTranslate2 conversion and INT8 CPU benchmark
- Reproducible configs, manifests, metrics, and model cards
- A final go/no-go recommendation

### Out of scope

- Stress prediction or insertion
- Grapheme-to-phoneme conversion
- Acoustic TTS training
- Browser or mobile runtime integration
- Full production API deployment
- Supporting languages other than Ukrainian
- Verbalizing arbitrary Latin words that the pipeline contract says to preserve
- Replacing deterministic safety checks with model-only generation

## Proposed Approach

Treat verbalization as monolingual sequence-to-sequence transformation:

```text
source: У 2025 році витрати зросли на 15,5%.
target: У дві тисячі двадцять п’ятому році витрати зросли на п’ятнадцять цілих п’ять десятих відсотка.
```

Train a Marian encoder-decoder with a shared Ukrainian source/target vocabulary.
Compare:

1. `marian-small`: lower-latency pilot model
2. `marian-base`: standard six-layer quality model
3. Raw end-to-end preprocessing
4. Protected-span preprocessing for text that must be copied exactly
5. Greedy and beam-search decoding
6. FP32/BF16 PyTorch and INT8 CTranslate2 inference

## Deliverables

- Data audit report
- Dataset build script and immutable split manifest
- Ukrainian SentencePiece tokenizer
- Training and evaluation pipeline
- Two trained model configurations
- Category-aware benchmark report
- Error taxonomy with representative examples
- CTranslate2 INT8 export and CPU benchmark
- Final recommendation:
  - proceed with Marian,
  - proceed only as part of a hybrid system, or
  - reject Marian for this use case

## Success Definition

The experiment succeeds when it produces a reproducible, evidence-based answer,
not merely when a model finishes training.

The recommended candidate must:

- preserve protected spans exactly;
- avoid changing sentences that require no verbalization;
- correctly preserve the semantic value of numbers, dates, times, currencies,
  percentages, ranges, and units;
- demonstrate acceptable morphology in the target context;
- remain stable after INT8 conversion;
- meet the configurable CPU latency and memory budget on declared reference
  hardware.

Numeric acceptance thresholds are experimental targets and must be reviewed
after the baseline evaluation. They are defined in the design and must not be
reported as achieved until measured.

## Structured Spoken-Domain Target

The user-provided structured examples define an additional `spoken-structured-domain`
profile. In this profile, numeric values, Latin product names, identifiers,
filenames, URLs, and email addresses are rendered for speech according to the
verified synthetic oracle. This profile is intentionally separate from the
default protected-span profile, which preserves configured spans verbatim.

Acceptance uses generated values and full inputs that are absent from training.
The finite acceptance objective is 100% exact match on the frozen target suite,
including explicit coordinate and Roman-numeral regressions. This objective is
not a claim that a neural model is mathematically correct for every possible
future input; production still requires deterministic validation and fallback.

## External Quality Comparator

The public `https://verbalizer.skravchenko.dev` API, using its default
`/normalize` endpoint and displayed Gemma 3 270M model, is an external quality
comparator for all 15,000 frozen structured-domain inputs. Its outputs are
model-generated evidence, not human gold labels and not training data.

The comparison must report three distinct views on the identical row IDs:

1. Marian output against the frozen dataset target;
2. external API output against the frozen dataset target; and
3. Marian output against the external API output.

Website imitation and target quality are not interchangeable because valid
spoken policies can differ, for example `крапка` digit-group rendering versus
formal decimal fractions. The candidate reaches external quality parity only
after the complete 15,000-row capture shows that Marian is no worse than the
external comparator on target exact match, CER, WER, semantic error rate, and
critical failure counts. The original structured-domain objective of 100% exact
match remains in force; external parity does not weaken it.
