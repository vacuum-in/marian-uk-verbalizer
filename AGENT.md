# Verbalization policy

Never add, retain, or invoke hardcoded verbalization rules.

Do not convert, normalize, expand, inflect, group, mask, restore, correct,
validate, reject, or replace verbalized text with rule-based code, dictionaries,
candidate inventories, or post-processing.

The only permitted runtime logic is routing an unchanged input to a selected
model, or passing through a sentence that needs no verbalization. Improve
quality only by training or selecting models and by improving the training
dataset.

## Span routing (from E55, approved 2026-10-14)

Routing may also work below the sentence: `src/verbalizer/spans.py` marks the spans that may need verbalization
(digits, Latin, symbols, capitals, short abbreviations, footnote marks), from the shape of the characters only, with
no word lists. The model sees the sentence with one span marked and writes only that span's spoken form; the code
puts each model output at its span and copies everything else from the input unchanged. The code never chooses,
corrects or rejects a reading: what a span becomes is the model's output, including nothing (a footnote mark).
