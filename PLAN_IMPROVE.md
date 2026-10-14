# Improving the verbalizer: analysis and plan

Written 2026-10-14. The starting point is E54 (this repository), the verbalizer the live stack still serves, and
`skypro1111/gemma-3-270m-uk-verbalizer`, which `accentuk` 0.2.0 made its default on 2026-10-13. The evaluation kit for
both lives on the laptop in `~/verb-gemma` (`verb_eval.py`, the dataset splits, `errs_*.json` per system).

## 1. Where we are

Held-out splits of `skypro1111/uk-text-normalization` (3,619 rows; references plus listed alternatives), one sentence
at a time, CPU unless noted:

| system | exact | ignoring case and punctuation | time / sentence | size |
|---|---:|---:|---:|---:|
| E54, CT2 INT8 | 86.41% | 89.50% | 10 ms | 52 MB |
| Gemma 270M, CT2 INT8 (with its helpers) | 88.04% | 90.19% | 72 ms | 126 MB |
| Gemma 270M, CT2 float32 | 89.09% | 91.19% | 250 ms | 245 MB |
| Gemma 270M, transformers bf16, GPU | 88.97% | 90.99% | 167 ms | 250 MB |
| oracle: E54 or Gemma INT8 right | 93.42% | | | |

Per split (exact): E54 / Gemma INT8 — domains 88.53 / 89.59, hard mixed sentences 86.00 / 90.13, transliteration
72.89 / 72.63, numeral agreement 88.89 / 100, reported bugs 90.91 / 95.45.

What these numbers say:

- **For TTS the two models are closer than the exact-match figure suggests.** 112 of E54's 492 errors and 78 of
  Gemma's 433 differ from the reference only in case or punctuation ("де те пе" vs "Де те пе", a lowercase first
  word), which the listener never hears. Ignoring those, the gap is 0.7 points.
- **They fail on different rows.** Only 238 rows defeat both; either model alone gets 93.4%. A model that learns
  from both (section 4) has about four points of headroom before any new idea is needed.
- **INT8 costs Gemma about a point** (69 rows it gets wrong that the original gets right).

## 2. What goes wrong

Error classes (heuristic, `analyze.py` next to the error files), E54 / Gemma INT8:

| class | E54 | Gemma | typical case |
|---|---:|---:|---|
| Latin words and transliteration | 183 | 213 | `remont.ua` → "рімонт", `hotline` → "хотлайн"; many are spelling variants, not audible errors |
| numeral value or digit grouping | 47 | 64 | cards and accounts read in pairs instead of digit by digit, `№2600` as a quantity, repeated zeros miscounted |
| numeral form: case, gender, ordinal | 31 | 14 | **"12 дверей" → "дванадцяте дверей"** (reported), "XVI ст." → "шістнадцятого", "у 12 тис. приладів" → nominative |
| preposition and case of the governing phrase | in "other" | in "other" | "у 2019 році" → "з … року" |
| abbreviation expansion | in "other" | in "other" | "ср." → "сі" instead of "середа" |
| repeated or dropped words | 17 | 15 | "лімітед лімітед", an extra "нуль" in a run of zeros |
| case / punctuation only | 112 | 78 | inaudible |

The reported bug ("На 2 поверсі є 3 кімнати та 12 дверей" → "дванадцяте дверей") is E54's: Gemma reads it right.
The other report, a quick "він-він" at the start of "Він відчинив замок ключем", is not the verbalizer's: that
sentence has no digits, the router passes it through unchanged, and the text the TTS receives contains one "Він". The
repetition is produced by the TTS model; see section 6.

## 3. A constraint to settle first

`AGENT.md` forbids rule-based conversion, grouping and correction at runtime. Gemma as published needs two such steps,
`digit_router.py` (spells a bare digit string digit by digit without the model) and `bignum_spacing.py` (regroups long
numbers in the input), and `accentuk` 0.2.0 runs them. Its card measures what they buy: big numbers 13/20 → 20/20
with spacing, and the router exists because the model cannot count repeated digits.

Decision needed: either (a) the package keeps them as a documented exception for the third-party model, or (b) we
train a model that does not need them (tracks B and C below), and drop them when it is measured equal. Recommended:
(b), keeping (a) only until a policy-compliant model matches it.

## 4. Strategy

Quality comes from data, as this project has always held; the new levers are better labels and a better measure.

### Phase 0 — measure what the listener hears (1-2 days)

1. Add a **TTS-relevant score** next to exact match: case- and punctuation-insensitive, hyphens as spaces, and
   transliteration compared after a phonetic fold (ґ/г, е/є, doubled letters) so a spelling variant is not an error.
   Report both; decide on the new one.
2. Build a **real-text eval set**: 1,000 sentences with digits, Latin, abbreviations or symbols, drawn from our own
   text (audiobook texts in `audiotostress/artifacts/books`, news, YouTube transcripts), not from template
   generators. Reference = E54 and Gemma agree; disagreements labelled by Qwen3.8-27B with reasoning and then
   checked by hand. This set, not the synthetic one, decides releases.
3. Keep the per-class breakdown of section 2 in every report, so a gain in one class cannot hide a loss in another.

### Phase 1 — labels from narrators (pilot 2 days, then scale)

The audiobooks already aligned for stress mining are a source no text dataset has: **how a human actually read each
number, date and abbreviation in context.** For a book sentence with a digit, the miner's word timings bound the
audio span where the number was spoken; ASR on that span gives the spoken form ("9 квітня 1919 р." →
"дев'ятого квітня тисяча дев'ятсот дев'ятнадцятого року").

- Pilot: 300 book sentences with digits → ASR on the spans → compare with E54 / Gemma / references; measure how often
  the narrator's reading is usable (clean ASR, words not digits) and how often the models disagree with it.
- If the pilot holds (>= 70% usable), run it over all books and the new audio (the 3,756 h mined on 10-12): tens of
  thousands of real, context-inflected pairs, exactly the class (case and agreement after prepositions) where both
  models are weakest.
- These pairs derive from commercial audio and text: they stay on our machines, and only the trained weights leave.

### Phase 2 — consensus distillation on a large real corpus (1 week)

1. Collect ~500k sentences that need verbalization from our corpora (books, news, YouTube transcripts).
2. Label each with E54 and Gemma (float32, the more accurate run). Agreement (expected ~85-90% of rows) becomes a
   training pair. Disagreement goes to Qwen3.8-27B with reasoning, choosing between the two outputs or writing its
   own; a third of those spot-checked.
3. Every numeric target is cross-checked by parsing the numeral words back to a value, as the existing data
   generation does (offline, so allowed).
4. Mix with the existing E54 training data and the narrator pairs of phase 1; keep the replay share that E48-E54
   tuned so no old domain regresses.

### Phase 3 — train, two tracks, one gate

- **Track A, E55 (Marian 52M):** retrain on the phase 2 mix. Stays at 10 ms and 52 MB, fully policy-compliant.
- **Track B, Gemma 270M fine-tune:** continue training the published checkpoint on the same mix, with targeted data
  for bare digit strings, repeated zeros and long numbers (written as they are, not regrouped), so neither helper is
  needed. 72 ms INT8; worth it only if it beats track A by more than the INT8 loss.
- **Gate:** the phase 0 real-text set, TTS-relevant score, plus the per-class table and the 3,619 synthetic rows as a
  regression check. The winner replaces the default in `accentuk` and the live stack; the other is archived.

Targeted data in both tracks, by class:

| class | data |
|---|---|
| numeral + noun agreement (the reported bug) | counted nouns with plural-only and genitive-plural forms ("двері", "сани", "ножиці"), numbers 2-4 / 5-20 / 21-24 / compound, every case after prepositions |
| digit-by-digit reading | cards, IBAN, accounts, codes, ТТН, plates, phones, with long zero runs; label by the reading convention of the cue word |
| ordinals | centuries, floors, chapters, "N-тє", Roman numerals with all cases |
| abbreviations | weekday and month abbreviations, "ст.", "обл.", "р.", units; acronyms read letter by letter vs as a word |
| Latin | domains, brands, product names from the real corpus, references from transliteration of the spoken form (narrators and YouTube speakers say them) |

### Phase 4 — runtime (small, policy-compliant)

- Serve one sentence per call (batching changed INT8 Gemma outputs; E54 is unaffected but keep it uniform).
- On GPU, CT2 `int8` or `bfloat16` only: `float16` gave 0% and `int8_float16` 68% for Gemma.
- Route the live stack (`/v1/prepare` on :8000, still E54) to the winner of phase 3; until then, switch it to Gemma
  INT8 if the reported agreement bug matters more than the 7x latency.

## 5. Order and size

| step | time | output |
|---|---|---|
| phase 0 metric + real-text set | 1-2 days | `reports/real-text-v1.json`, score in every report |
| phase 1 pilot | 2 days | go / no-go on narrator labels |
| phase 1 full | 2-3 days GPU | narrator pairs |
| phase 2 consensus labels | ~1 week GPU (Qwen on disagreements) | `data/e55_consensus/` |
| phase 3 E55 + Gemma fine-tune | 2-4 days GPU | gated release |

The GPU is shared with the live stack and the OOV classification run (until about 10-16); each long run keeps the live
API guard used for the OOV runs.

## 6. Out of scope here: the repeated word in synthesis

"Він відчинив замок ключем" reaches the TTS as "Він відчини́в замо́к ключе́м." with one "Він"; the stressed text
differs from plain text only by U+0301 marks. A doubled short first word is a known failure of autoregressive TTS.
To test on our Pocket TTS: synthesize plain vs stressed text for ~200 sentences, run Whisper, count repeated words
in each. If the marks trigger it, the fix belongs in the TTS front end (how marks are tokenized) or in its training
data, not in the verbalizer.

## 7. Span pilot results (2026-10-14)

Span-to-span Marian (E54 warm-started; `src/verbalizer/spans.py`, `scripts/build_span_data.py`,
`scripts/evaluate_spans.py`; reports `reports/e5[5-9]-spans*.json`). Spoken match (case and punctuation not heard);
real = 1,102 audiobook sentences with references written by Qwen3.8-27B (a manual check of 40 found 2 wrong, 1 debatable);
altered prose = rows where a word outside every detected span is missing from the output.

| system | synthetic 3,619 | real 1,102 | altered prose (real) |
|---|---:|---:|---:|
| E54 whole sentence | 89.53 | 62.61 | 4.99% |
| Gemma whole sentence | ~90.2 | 58.89 | 8.08% |
| E55 span pilot (1 epoch, lr 1e-4) | 81.74 | 61.80 | |
| E56 (+ footnotes, 2 epochs at 5e-5) | 84.00 | 68.06 | 1.27% |
| E57 (+ wider detector, no E52 doubling) | 86.16 | 68.97 | 1.36% |
| E58 (+ ruler names, detector v2 at eval) | 87.40 | 70.42 | 1.45% |
| E59 (+ consistent footnotes, animate counts x3) | 87.73 | 69.51 | 1.45% |

What moved it: footnote marks read as nothing (real "bare number" 11% -> 79%), spans that keep a unit, currency or
abbreviation with its number, and not doubling E52 (it had pushed Roman numerals after names to the dative). The
remaining synthetic gap (-1.8) is the model's own span reading (409 errors vs E54's 379 in total); detector misses are
down to 35 rows. Still weaker than E54: Roman numerals in real text (33 vs 37), symbols.

Laptop iterations (RTX 5090, ~13 min per run):

| system | synthetic 3,619 | real 1,102 | altered prose (real) | time / sentence |
|---|---:|---:|---:|---:|
| E60 (detector: overlaps after a tail, ":" in paths, spacing of glued spans) | 89.09 | 69.78 | 0.45% | 1.1 ms GPU |
| E61 (E60 + 1 epoch at 2e-5) | 88.84 | 70.05 | 0.45% | |
| **E60, CTranslate2 INT8, CPU** | **88.86** | **70.05** | 0.45% | 10.2 ms |

E60 is the candidate: level with E54 on the synthetic splits (-0.4, within noise), +7.4 on real book text, and prose
outside the spans is never rewritten. Export: `scripts/convert_marian_ct2.py` from `.venv-gemma` (the training venv's
transformers 4.30 is too old for the CT2 converter) -> `outputs/marian-uk-verbalizer-e60-spans-ct2-int8` (54 MB).
CT2 must be called with `min_decoding_length=0`: its default of 1 forbids the empty reading of a footnote mark, and
the model then invents one ("[557]" -> "на"), which cost 5.6 points on real text.
