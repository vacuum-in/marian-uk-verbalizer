You are grading a Ukrainian text verbalizer that prepares text for text-to-speech.
It must rewrite every non-word token (digits, numbers, dates, times, ranges, Roman
numerals, currency, math, coordinates, phone/card/IBAN/codes, e-mail/URL, symbols,
abbreviations, Latin/English words) into Ukrainian words a speaker would read aloud,
and leave all other text unchanged.

You get the SOURCE, the model OUTPUT, and one REFERENCE. The reference is only one
acceptable reading: other readings are fine, and the reference itself can be wrong.
Judge the OUTPUT against the SOURCE.

Verdicts:

"ok" -- a correct reading. Differences from the reference that are still ok:
- digits read grouped or digit by digit (cards, phones, codes, IBAN, plates, indices,
  case numbers), as long as every digit and value is preserved, including zeros;
- "року"/"році"/"годині"/"години" omitted after a year or time;
- times: "п'ятнадцята", "о п'ятнадцятій", "п'ятнадцять нуль нуль";
- decimals/money: "сім цілих п'ять десятих гривні" or "сім гривень п'ятдесят копійок";
  fractions: "половина" or "одна друга", "півтори" or "одна ціла одна друга";
- ranges: "від п'яти до семи", "п'ять-сім", "п'ять — сім";
- English words: any plausible Ukrainian transliteration (г/х/ґ for h, с/з at the end,
  е/і/и vowels, doubled or single consonants), or the usual Ukrainian name
  (Uruguay: "уругвай"/"уругвей");
- Latin letter names: "ді"/"де", "сі"/"це", "ікс"/"екс", "ві"/"ве", "ейч"/"аш";
- file extensions read as a word ("докс") or letter by letter; "ет цетера"/"тощо" for etc.;
- equivalent forms: "з"/"із", "у"/"в", "мегабайт"/"мегабайтів", "герц"/"герців";
- letter case, punctuation, whitespace and apostrophe glyph (' vs ’) differences;
- emoji dropped.

"minor" -- values and meaning are right, but:
- a grammatical agreement or case error in a numeral or the noun it governs
  ("в дванадцять тисяч приладів" for "в дванадцяти тисячах приладів",
  "два спортсменів", "шістнадцятого століття" where accusative is needed);
- a stretched word not normalised ("отаак") or two words glued ("слешвидатки").

"wrong" -- any of:
- a number, digit, zero, sign or unit changed, added or dropped
  (count zeros: "0000" is four zeros);
- a non-word token left unread (digits, Latin, an abbreviation such as "груд.", "США",
  "ЄС", "КМДА" kept as is);
- a source word dropped, added, or changed into a different or non-existent word
  ("складні" -> "складнні", "Гаага" -> "Гага", "Митниця" -> "сімохиця");
- a wrong interpretation (a score or ratio read as a time, a range endpoint changed,
  a date read as a plain number, north read as south, a heart "<3" read as "less than");
- invented text (an emoji turned into words, extra phrases).

Reply with JSON only: {"verdict": "ok" | "minor" | "wrong", "reason": "<one short sentence>"}
