# Tasks

## 1. Experiment Scaffolding

- [x] 1.1 Create the Python project, package layout, configuration loader, and CLI
- [x] 1.2 Pin dependencies and generate a reproducible lock file
- [x] 1.3 Add run manifests with Git revision, package versions, hardware, and seeds
- [x] 1.4 Add structured logging and experiment output directories
- [ ] 1.5 Add CI checks for formatting, typing, unit tests, and config validation

## 2. Dataset Acquisition and Audit

- [x] 2.1 Use the existing local checked-in artifacts for `skypro1111/ubertext-2-news-verbalized` (download intentionally disabled)
- [x] 2.2 Map raw columns into the canonical source/target schema
- [x] 2.3 Implement NFC and approved whitespace normalization
- [x] 2.4 Generate stable row identifiers and source checksums
- [x] 2.5 Implement empty-row, digit-residue, length-ratio, punctuation, Unicode,
      and truncation checks
- [x] 2.6 Detect exact duplicate sources and conflicting targets
- [x] 2.7 Implement canonical and template fingerprints
- [x] 2.8 Add near-duplicate analysis
- [x] 2.9 Implement preliminary numeric/date/time/currency/unit semantic checks
- [x] 2.10 Produce the first audit report with exclusion reason counts and samples
- [ ] 2.11 Manually review a stratified sample and refine audit rules
- [ ] 2.12 Freeze the dataset quality-tier policy

## 3. Verbalization Categories and Challenge Data

- [ ] 3.1 Implement category detection for all categories listed in `design.md`
- [ ] 3.2 Produce category distribution and multi-category interaction reports
- [ ] 3.3 Identify rare and missing categories
- [ ] 3.4 Create a challenge set from intended TTS product inputs
- [x] 3.5 Add identity/no-change Ukrainian sentences
- [ ] 3.6 Create a human-review annotation guide
- [ ] 3.7 Build and adjudicate the initial human gold set
- [ ] 3.8 Version and checksum all evaluation sets

## 4. Leakage-Safe Splits

- [x] 4.1 Implement group-based split assignment
- [ ] 4.2 Ensure exact and near duplicates cannot cross splits
- [x] 4.3 Reserve template fingerprints for `test_template_ood`
- [ ] 4.4 Generate train, validation, test-ID, template-OOD, challenge, and
      human-gold manifests
- [x] 4.5 Add automated leakage tests
- [ ] 4.6 Freeze and checksum the final manifests before model comparison

## 5. Protected-Span Pipeline

- [ ] 5.1 Define the configurable protected-span policy
- [ ] 5.2 Implement URL, email, path, code, hash, identifier, and Latin-span detection
- [ ] 5.3 Implement `<KEEP_0>` through `<KEEP_31>` replacement
- [ ] 5.4 Implement exact restoration and validation
- [ ] 5.5 Implement duplicate, missing, unknown, and illegal-order sentinel failures
- [ ] 5.6 Add deterministic fallback behavior
- [ ] 5.7 Add property-based and adversarial tests for preservation
- [ ] 5.8 Produce statistics showing how often protection is used

## 6. Tokenizer Experiments

- [x] 6.1 Build the combined accepted source/target tokenizer corpus
- [ ] 6.2 Train 8k SentencePiece Unigram tokenizer
- [x] 6.3 Train 16k SentencePiece Unigram tokenizer
- [ ] 6.4 Train 32k SentencePiece Unigram tokenizer
- [x] 6.5 Register sentinel and required special tokens explicitly
- [x] 6.6 Build compatible Marian tokenizer artifacts
- [x] 6.7 Verify pad, EOS, unknown, and decoder-start IDs
- [ ] 6.8 Measure token lengths, coverage, suffix fragmentation, and vocabulary cost
- [ ] 6.9 Select the tokenizer for the first pilot and document the rationale

## 7. Marian Model Implementation

- [x] 7.1 Implement `marian-small` custom `MarianConfig`
- [x] 7.2 Implement `marian-base` custom `MarianConfig`
- [x] 7.3 Enable shared encoder/decoder embeddings and tied output embeddings
- [x] 7.4 Add dynamic token-based batching and sequence-length validation
- [x] 7.5 Add checkpoint save/resume support
- [ ] 7.6 Add deterministic generation fixtures
- [ ] 7.7 Verify that raw and protected variants use identical model settings
- [x] 7.8 Add parameter-count and memory-estimation reports

## 8. Pilot Training

- [x] 8.1 Build a stratified 100k–200k pilot dataset
- [x] 8.2 Train E1: small, 16k, protection off, 15% identity, greedy
- [ ] 8.3 Train E2: small, 16k, protection on, 15% identity, greedy
- [ ] 8.4 Compare learning curves, semantic errors, identity accuracy, and preservation
- [x] 8.5 Inspect at least 200 representative pilot errors
- [ ] 8.6 Refine data filters and challenge cases without modifying frozen final tests
- [x] 8.7 Select learning rate, label smoothing, batch size, and epoch budget
- [x] 8.8 Decide whether the full experiment remains justified

## 9. Evaluation Framework

- [x] 9.1 Implement approved evaluation-text normalization
- [x] 9.2 Implement exact match, CER, WER, and character n-gram metrics
- [x] 9.3 Implement semantic extraction and source/output value comparison
- [x] 9.4 Implement identity and over-normalization metrics
- [ ] 9.5 Implement protected-span preservation metrics
- [x] 9.6 Implement deletion, insertion, truncation, punctuation, and Unicode checks
- [x] 9.7 Implement category-level macro and micro reports
- [ ] 9.8 Implement critical error classification
- [x] 9.9 Freeze the composite model-selection score before final testing
- [ ] 9.10 Generate HTML/Markdown/JSON evaluation reports
- [ ] 9.11 Add regression tests using known difficult Ukrainian examples

## 10. Full Training and Ablations

- [ ] 10.1 Train the selected small protected model on the full accepted dataset
- [ ] 10.2 Train the base protected model on the full accepted dataset
- [ ] 10.3 Run 8k versus 16k versus 32k tokenizer ablation on the selected model size
- [ ] 10.4 Run identity-mixture ablation at 0%, 15%, and 25%
- [ ] 10.5 Run at least three seeds for the final competing configurations
- [ ] 10.6 Evaluate greedy, beam-2, and beam-4 decoding
- [ ] 10.7 Select the best checkpoint without viewing final human-gold labels
- [ ] 10.8 Run the frozen final test suite once for the selected candidates

## 11. Human Review

- [ ] 11.1 Randomize and blind model identifiers for review
- [ ] 11.2 Collect semantic, grammar, naturalness, preservation, and severity labels
- [ ] 11.3 Measure reviewer agreement
- [ ] 11.4 Adjudicate major and critical disagreements
- [ ] 11.5 Add corrected examples to a future-training set, not the frozen current test
- [ ] 11.6 Publish human-review results with limitations

## 12. CTranslate2 Export and Benchmark

- [ ] 12.1 Export the selected checkpoint to CTranslate2 FP32
- [ ] 12.2 Export or load the selected checkpoint with INT8 CPU compute
- [ ] 12.3 Verify token IDs, decoding, and sentinel restoration after export
- [ ] 12.4 Run the complete frozen quality suite on PyTorch and CTranslate2
- [ ] 12.5 Report aggregate and category-level quantization deltas
- [ ] 12.6 Build short, medium, long, multi-span, protected, and identity benchmarks
- [ ] 12.7 Benchmark batch sizes 1, 4, and 16
- [ ] 12.8 Record p50/p95/p99, throughput, RSS, startup time, model size, and threads
- [ ] 12.9 Repeat benchmark runs and report variance
- [ ] 12.10 Compare results with the declared performance gate

## 13. Decision and Documentation

- [ ] 13.1 Compare Marian models with identity, deterministic, and rule-only baselines
- [ ] 13.2 Create a category-by-category error table
- [ ] 13.3 Document representative major and critical failures
- [x] 13.4 State whether failures are model, data, tokenizer, decoding, or policy issues
- [x] 13.5 Produce `proceed`, `hybrid only`, or `reject` recommendation
- [x] 13.6 List remaining unknowns and required follow-up experiments
- [ ] 13.7 Write model card and dataset-processing card
- [ ] 13.8 Add reproducible commands for training, evaluation, export, and benchmark
- [x] 13.9 Run OpenSpec validation
- [ ] 13.10 Verify implementation against every scenario in the delta spec

## 14. Structured Spoken-Domain Adaptation

- [x] 14.1 Define the spoken structured-domain policy separately from protected spans
- [x] 14.2 Add user coordinate and `XX розділ` regression fixtures
- [x] 14.3 Generate new train and validation seeds from the verified 30-family oracle
- [x] 14.4 Build canonical splits and verify zero exact-source leakage
- [x] 14.5 Record the current Marian baseline on the frozen acceptance and regressions
- [x] 14.6 Fine-tune Marian from E1c with a reset optimizer
- [x] 14.7 Select a checkpoint using validation only
- [x] 14.8 Evaluate the selected checkpoint once on the frozen 15k acceptance set
- [x] 14.9 Require 100% exact match on the explicit user regression set
- [x] 14.10 Restart Gradio with the selected structured-domain checkpoint
- [x] 14.11 Document residual errors and retain deterministic fallback recommendation

## 15. External 15k Quality Parity

- [x] 15.1 Identify and smoke-test the website's default `/normalize` endpoint
- [x] 15.2 Implement ordered, append-only, resumable reference capture
- [ ] 15.3 Capture external outputs for every frozen 15k row
- [x] 15.4 Implement three-way target and output-parity metrics
- [ ] 15.5 Classify every substantive local and external disagreement by category
- [x] 15.6 Generate corrective training data with zero frozen-source overlap
- [x] 15.7 Train and select an improved checkpoint using non-frozen validation only
- [x] 15.8 Verify 100% exact match on the complete frozen 15k target suite
- [ ] 15.9 Verify Marian is no worse than the external comparator on target exact,
      CER, WER, semantic flags, and critical failures over all 15k rows
- [x] 15.10 Document policy differences and reproducible capture/evaluation commands
- [x] 15.11 Serve the verified selected checkpoint in Gradio with selectable GPU/CPU inference

## 16. Main 95k News-Domain Parity

- [x] 16.1 Audit `main_train_95k` and its disjoint official `golden_5k` test
- [x] 16.2 Quarantine ambiguous duplicate sources and build a deterministic validation split
- [x] 16.3 Record the E5 baseline on all 5,000 official test rows
- [x] 16.4 Train news-only E6 and quantify structured-domain forgetting
- [x] 16.5 Train E7 with equal news/structured replay and zero final-test overlap
- [x] 16.6 Repair the selected E8 checkpoint without reading golden targets or remote outputs
- [x] 16.7 Verify 15,000/15,000 structured exact and 4/4 user regressions
- [x] 16.8 Capture Gemma 3 270M outputs for all 5,000 official test rows
- [x] 16.9 Pass exact/CER/WER, semantic-risk, and critical-failure no-worse gates
- [x] 16.10 Add a machine-readable completion audit and background finalizer
- [x] 16.11 Serve E12 in Gradio with selectable GPU/CPU BF16 inference

## 17. Learned Roman-Numeral Adaptation

- [x] 17.1 Remove inference-time Roman conversion and pass Roman glyphs to Marian
- [x] 17.2 Generate 150k train, 1.4k validation, and 2.5k test rows with disjoint values/templates
- [x] 17.3 Train E9–E12 with news, structured, signed-range, and Roman replay
- [x] 17.4 Reach 97.60% exact on the unseen Roman holdout and 4/4 explicit regressions
- [x] 17.5 Verify E12 on structured 15k and news golden 5k with production BF16 greedy decoding
- [x] 17.6 Pass the complete public-reference no-worse gate and completion audit

## 18. Decades, Large Integers, and Financial Identifiers

- [x] 18.1 Add Ukrainian plural-ordinal normalization for decade forms ending in `-х`
- [x] 18.2 Verbalize standalone integers of five or more digits before model inference
- [x] 18.3 Treat grouped payment-card and `UA` IBAN digits as exact sequences
- [x] 18.4 Extend post-generation fidelity repair to card and IBAN sequences
- [x] 18.5 Verify focused regressions, the full test suite, and live GPU/CPU Gradio calls

## 19. Learned Latin-Letter Disambiguation

- [x] 19.1 Reproduce `вітамін D` being interpreted as Roman 500
- [x] 19.2 Remove all context-bound `C`/`D` inference rewrites
- [x] 19.3 Add diverse model-training contexts for letters `C` and `D`
- [x] 19.4 Verify 1,760/1,760 held-out letter contexts by model inference

## 20. Restrict Roman Training to I, V, and X

- [x] 20.1 Cap active Roman train/validation/test data at 39 and permit only `I`, `V`, `X`
- [x] 20.2 Make the generator and trainer reject values above 39 and add corpus-policy tests
- [x] 20.3 Reject E14/E15 because their 1–99 corpus contains `L`
- [x] 20.4 Train clean-lineage E16–E19 without any prior large-Roman checkpoint
- [x] 20.5 Select E19 step 8000 by letter, user-regression, and IVX holdout gates
