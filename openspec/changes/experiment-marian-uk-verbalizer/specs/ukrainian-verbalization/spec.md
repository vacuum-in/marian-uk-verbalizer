# Delta for Ukrainian Verbalization

## ADDED Requirements

### Requirement: Ukrainian text verbalization

The system MUST transform supported written Ukrainian numeric and symbolic forms
into spoken Ukrainian text while preserving their semantic value.

#### Scenario: Integer in context

- GIVEN the input `Компанія відкрила 15 магазинів.`
- WHEN the verbalizer processes the sentence
- THEN the output expresses the value fifteen in Ukrainian words
- AND the noun phrase remains grammatically compatible with the sentence
- AND no unrelated words are paraphrased

#### Scenario: Decimal and percentage

- GIVEN the input `Показник зріс на 15,5%.`
- WHEN the verbalizer processes the sentence
- THEN the output preserves the value `15.5`
- AND the output preserves the percent meaning
- AND the result is suitable for spoken Ukrainian

#### Scenario: Date and year

- GIVEN an input containing a numeric date or year
- WHEN the verbalizer processes the input
- THEN every date component retains its original value
- AND the grammatical form is selected from sentence context
- AND no new date component is invented

#### Scenario: Currency and unit

- GIVEN an input containing a number, currency, or measurement unit
- WHEN the verbalizer processes the input
- THEN the numeric value is preserved
- AND the currency or unit identity is preserved
- AND the generated grammatical form is compatible with the local context

### Requirement: Identity behavior

The system MUST leave ordinary text unchanged when no supported verbalization is
required, except for explicitly documented Unicode or whitespace normalization.

#### Scenario: No verbalizable spans

- GIVEN a Ukrainian sentence with no supported numeric or symbolic span
- WHEN the verbalizer processes the sentence
- THEN the output equals the approved normalized input
- AND the system does not paraphrase, summarize, or add content

### Requirement: Protected-span preservation

The system MUST preserve configured Latin words, abbreviations, URLs, emails,
paths, code fragments, hashes, and opaque identifiers exactly.

#### Scenario: Latin product name

- GIVEN the input `CPU Intel i7-14700K коштує 15 000 грн.`
- AND Latin and product identifier spans are configured as protected
- WHEN the verbalizer processes the input
- THEN `CPU`, `Intel`, and `i7-14700K` are unchanged
- AND the price is verbalized
- AND every protected span is restored exactly once

#### Scenario: Invalid sentinel output

- GIVEN protected spans were replaced by internal sentinels
- WHEN model output omits, duplicates, reorders illegally, or invents a sentinel
- THEN the output is marked invalid
- AND the system does not silently return the corrupted text
- AND the configured fallback behavior is invoked

### Requirement: Deterministic preprocessing

The system MUST apply a versioned and reproducible preprocessing policy before
training, evaluation, and inference.

#### Scenario: Unicode normalization

- GIVEN canonically equivalent Ukrainian text
- WHEN preprocessing runs
- THEN the result follows the configured Unicode NFC policy
- AND the original row remains traceable
- AND apostrophes and combining characters are not silently corrupted

#### Scenario: Reproducible row identity

- GIVEN the same raw row and preprocessing version
- WHEN the dataset is rebuilt
- THEN the row receives the same stable identifier
- AND its inclusion or exclusion reason is reproducible

### Requirement: Dataset quality audit

The training pipeline MUST audit automatically generated targets and MUST retain
exclusion reasons.

#### Scenario: Conflicting targets

- GIVEN two accepted rows with the same canonical source and materially different
  targets
- WHEN the audit runs
- THEN the conflict is reported
- AND the rows are excluded from the highest-quality training tier until resolved

#### Scenario: Possible numeric corruption

- GIVEN a target that may express a different number, date, time, currency, or
  unit from the source
- WHEN semantic validation runs
- THEN the row receives a quality flag
- AND it is excluded or manually reviewed according to configuration

### Requirement: Leakage-safe evaluation

The experiment MUST prevent duplicate, near-duplicate, document-group, and held
out template leakage between training and evaluation splits.

#### Scenario: Duplicate group

- GIVEN multiple rows in the same duplicate or document group
- WHEN dataset splits are generated
- THEN all rows in the group are assigned to one split only

#### Scenario: Template out-of-distribution test

- GIVEN a numeric template selected for the template-OOD test
- WHEN training data is finalized
- THEN that template fingerprint is absent from the training split

### Requirement: Category-aware metrics

The evaluation pipeline MUST report semantic and safety metrics by verbalization
category in addition to aggregate string metrics.

#### Scenario: Wrong numeric value in fluent output

- GIVEN a generated sentence that is linguistically fluent but changes a numeric
  value
- WHEN evaluation runs
- THEN the prediction is counted as a critical semantic error
- AND a high string-similarity score does not hide the failure

#### Scenario: No-change over-normalization

- GIVEN an identity test sentence
- WHEN the model changes unrelated words
- THEN the change is counted as over-normalization
- AND the identity exact-match metric decreases

### Requirement: Reproducible model training

Every training run MUST record sufficient information to reproduce its dataset,
tokenizer, model configuration, optimization, and evaluation.

#### Scenario: Run manifest

- GIVEN a completed or interrupted training run
- WHEN its manifest is inspected
- THEN it contains code revision, dataset revision, split checksum, tokenizer
  checksum, resolved configuration, package versions, hardware, seed, and
  checkpoint references

### Requirement: Deployment-runtime verification

The selected checkpoint MUST be exported to CTranslate2 and evaluated after
quantization before a production recommendation is made.

#### Scenario: INT8 quality regression

- GIVEN the same checkpoint in PyTorch and CTranslate2 INT8
- WHEN both are evaluated on the frozen test suites
- THEN aggregate and category-level metric differences are reported
- AND critical semantic error changes are reported separately

#### Scenario: CPU benchmark disclosure

- GIVEN a latency or memory result
- WHEN it is published
- THEN the report identifies CPU model, operating system, runtime version,
  compute type, thread settings, batch size, input bucket, and decoding mode

### Requirement: Explicit recommendation

The experiment MUST end with an explicit `proceed`, `hybrid only`, or `reject`
recommendation supported by measured evidence.

#### Scenario: Unmet quality gate

- GIVEN the selected model misses the critical semantic reliability gate
- WHEN the final report is produced
- THEN the model is not described as production-ready
- AND the report identifies whether deterministic routing or fallback can make a
  hybrid design acceptable

#### Scenario: Missing evidence

- GIVEN the human-reviewed test set or deployment-runtime benchmark is incomplete
- WHEN the final report is produced
- THEN the missing evidence is stated
- AND the conclusion remains provisional

### Requirement: Structured-domain finite acceptance

The selected structured-domain runtime MUST use one generic inference policy
and achieve exact target output for every frozen 15,000-row acceptance example
and every explicit user regression, without row-ID or frozen-source lookup.

#### Scenario: Unseen Roman ordinal

- GIVEN `Потрібно прочитати XX розділ до понеділка.` as a held-out regression
- WHEN the selected runtime verbalizes it
- THEN the output is `Потрібно прочитати двадцятий розділ до понеділка.`
- AND the behavior is learned from disjoint Roman training data rather than an
  inference parser or memorized source-string exception

#### Scenario: Multi-sentence cross-clause isolation

- GIVEN an input containing independent structured sentences
- WHEN the selected runtime generates its output
- THEN each sentence is inferred independently and rejoined in source order
- AND values or units from one sentence do not contaminate another sentence

### Requirement: Financial numeric sequence fidelity

The runtime MUST verbalize grouped payment-card numbers and `UA` IBAN-like
values digit by digit and MUST preserve every source digit in source order.

#### Scenario: Card and IBAN in one input

- GIVEN a sentence containing a 16-digit card number and a grouped `UA` IBAN
- WHEN the selected runtime verbalizes it
- THEN each identifier digit is spoken individually
- AND no digit is omitted, duplicated, or interpreted as a cardinal group

### Requirement: Decades and large cardinal values

The runtime MUST render decade forms ending in `-х` as Ukrainian plural
ordinals and MUST render standalone integers with at least five digits as
cardinal words.

#### Scenario: Decade and two large money values

- GIVEN `У 1990-х батько заробляв 12000000 доларів, тепер оренда — 18 500 грн.`
- WHEN the selected runtime verbalizes it
- THEN `1990-х` is rendered as `тисяча дев’ятсот дев’яностих`
- AND both money values are rendered as complete cardinal values

### Requirement: Learned Latin-letter versus Roman-number disambiguation

The model MUST learn from varied contexts that standalone Latin `C` and `D` are
letter labels. The runtime MUST NOT use lexical inference rules to make this
decision. Roman-number training, validation, and test data MUST contain only
`I`, `V`, and `X`, with supported values limited to 1 through 39.

#### Scenario: Vitamin and hepatitis labels

- GIVEN `Пацієнту призначили вітамін D, аналіз на гепатит C — негативний.`
- WHEN the selected runtime verbalizes it
- THEN `D` is rendered as `де`
- AND `C` is rendered as `це`
- AND the result comes from Marian inference without a medical-context rewrite

#### Scenario: Airport terminal letter

- GIVEN `Виліт з терміналу D`
- WHEN the selected runtime verbalizes it
- THEN the output is `Виліт з терміналу де`
- AND the same learned behavior generalizes to unseen terminal, class,
  category, and designation contexts

### Requirement: External quality parity

The selected Marian runtime MUST be compared with the public reference service
on every frozen 15,000-row source. Reference outputs MUST remain evaluation-only
evidence and MUST NOT influence training or checkpoint selection.

#### Scenario: Resumable complete reference capture

- GIVEN an interrupted external capture
- WHEN capture resumes
- THEN every existing row ID and source is validated in frozen order
- AND new rows are appended without replacing prior responses
- AND completion requires exactly one response for every frozen row

#### Scenario: No-worse quality decision

- GIVEN a complete 15,000-row capture and the independently selected checkpoint
- WHEN parity metrics are computed
- THEN target exact match, CER, WER, semantic-review flags, and critical failures
  are reported for both systems
- AND Marian is accepted only if it is no worse than the external comparator on
  every declared quality measure
