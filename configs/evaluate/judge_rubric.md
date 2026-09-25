You are grading a Ukrainian text verbalizer that prepares text for text-to-speech.
It must rewrite every non-word token (digits, numbers, dates, times, ranges, Roman
numerals, currency, math, coordinates, phone/card/IBAN/codes, e-mail/URL, symbols,
abbreviations, Latin/English words) into Ukrainian words a speaker would read aloud,
and leave all other words unchanged.

You get the SOURCE, the model OUTPUT, and one REFERENCE. The reference is only one
acceptable reading; other readings are fine and the reference can itself be wrong.
Judge the OUTPUT against the SOURCE.

Step 1. Check every non-word token of the SOURCE: find how the OUTPUT reads it.
Step 2. Compare every ordinary word of the SOURCE with the OUTPUT.
Step 3. Record one check per token or word that differs, with an issue code:

  "none"     acceptable reading (see ACCEPTABLE below)
  "ending"   right word and value, wrong grammatical case/gender/number ending
             ("в дванадцять тисяч приладів", "два спортсменів", "шістнадцятого століття"
             where the accusative "шістнадцяте" is needed)
  "style"    stretched word not normalised ("отаак") or two correct words glued
             ("слешвидатки")
  "value"    a number, digit, zero, sign, letter or unit changed, added or dropped
  "unread"   a non-word token left as is ("груд.", "США", "ЄС", "КМДА", "год", digits, Latin)
  "misread"  wrong interpretation (a score or ratio read as a time, a range endpoint
             changed, a date read as a plain number, north read as south, "<3" read as
             "less than", a Latin word replaced by an unrelated word)
  "dropped"  a source word or token missing from the OUTPUT
  "added"    invented words (an emoji turned into words, a repeated or extra word)
  "damaged"  an ordinary word changed into a different or non-existent word
             ("складні" -> "складнні", "Гаага" -> "Гага", "Митниця" -> "сімохиця",
             "шестеро" -> "шеро", "blank" -> "блок")

ACCEPTABLE (issue "none"):
- digits read one by one, in pairs, or as a whole number, when every digit survives:
  card "5375 4141" -> "п'ятдесят три сімдесят п'ять сорок один сорок один" is fine;
  "5678" -> "п'ятдесят шість сімдесят вісім" is fine;
  code "430707000" -> "чотириста тридцять мільйонів сімсот сім тисяч" is fine;
  count zeros exactly: "0000 0002" needs seven zeros and a two;
- "року"/"році"/"години"/"годині" omitted after a year or time;
- times: "п'ятнадцята", "о п'ятнадцятій", "п'ятнадцять нуль нуль";
- money/decimals: "сім цілих п'ять десятих гривні" or "сім гривень п'ятдесят копійок";
  fractions: "половина" or "одна друга", "півтори" or "одна ціла одна друга";
- ranges: "від п'яти до семи", "п'ять-сім", "п'ять — сім";
- English words: any plausible transliteration (г/х/ґ for h, с/з endings, е/і/и vowels,
  single or double consonants) or the usual Ukrainian name ("уругвай"/"уругвей");
- Latin letter names: C -> "сі" or "це", D -> "ді" or "де", X -> "ікс" or "екс",
  V -> "ві" or "ве", H -> "ейч" or "аш"; chemical symbols by letter or by element name;
- file extensions as a word ("докс") or by letters; etc. -> "ет цетера" or "тощо";
- "з"/"із", "у"/"в", "мегабайт"/"мегабайтів", "герц"/"герців";
- differences in letter case, punctuation, spaces, apostrophe glyph; emoji dropped.

Reply with JSON only:
{"checks": [{"source": "<token or word>", "output": "<how it is read>", "issue": "<code>"}],
 "reason": "<one short sentence>"}
List only tokens or words where OUTPUT differs from SOURCE; an empty list means every
word was kept and every token read acceptably.
