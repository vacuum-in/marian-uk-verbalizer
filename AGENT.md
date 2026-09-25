# Verbalization policy

Never add, retain, or invoke hardcoded verbalization rules.

Do not convert, normalize, expand, inflect, group, mask, restore, correct,
validate, reject, or replace verbalized text with rule-based code, dictionaries,
candidate inventories, or post-processing.

The only permitted runtime logic is routing an unchanged input to a selected
model, or passing through a sentence that needs no verbalization. Improve
quality only by training or selecting models and by improving the training
dataset.
