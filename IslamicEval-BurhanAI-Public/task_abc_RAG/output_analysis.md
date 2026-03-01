# Why do results look bad?

Below is a detailed analysis with examples and remediation steps.

### Executive summary
- 1A span detection is strong overall, but Hadith precision is weak.
- 1B correctness is dragged down because the model overwhelmingly labels as Wrong even when GT is Correct (especially for Ayah).
- 1C correction is often a placeholder (e.g., “خطأ”) instead of the exact canonical replacement, causing many misses.
- Errors are not due to span boundary IoU (matches have IoU near 1.0); the issue is label semantics and source validation.

### Metrics and distributions
- From evaluating ABC_MEDIUM:
  - 1A Macro F1: 0.905442
    - Per-class precision/recall/F1:
      - Neither: P=0.957, R=0.956, F1=0.957
      - Ayah: P=0.967, R=0.911, F1=0.938
      - Hadith: P=0.732, R=0.869, F1=0.795
  - 1B Accuracy: 0.451697 (346/766)
  - 1C Accuracy: 0.528150 (197/373)
  - IoU for matched spans: mean ≈ 0.988; IoU for 1B mismatches: mean ≈ 0.983. So 1B errors are not boundary-related.

- Predicted label distribution (medium submission):
  - WrongAyah: 560
  - WrongHadith: 255
  - CorrectAyah: 25
  - CorrectHadith: 0
- GT label distribution:
  - CorrectAyah: 309
  - WrongAyah: 228
  - WrongHadith: 154
  - CorrectHadith: 91
  - NoAnnotation: 16

- 1B correctness confusion (same-type matches only):
  - Ayah: Correct→Correct: 15, Correct→Wrong: 242, Wrong→Correct: 1, Wrong→Wrong: 203
  - Hadith: Correct→Correct: 0, Correct→Wrong: 84, Wrong→Correct: 0, Wrong→Wrong: 127
  - Type mismatched matched pairs: 2

Interpretation:
- The model almost never predicts Correct (only 25 CorrectAyah, zero CorrectHadith) even when GT says Correct. That’s the main reason 1B is low.
- Hadith precision is the weakest: lots of hadith-labeled spans that aren’t backed by GT hadith.

- 1C correction content quality for matched Wrong GTs:
  - Total: 382
  - Exact matches: 3
  - Placeholder (“خطأ”/“wrong”): 297
  - Partial/mismatch: 32
  - The model usually emits a placeholder instead of the requested canonical correction text.

### Concrete examples

- 1B mismatch for Ayah: GT CorrectAyah vs Pred WrongAyah (multiple in Q002)
  The text includes canonical Qur’an verses from Surah Luqman; GT says CorrectAyah but the model labeled WrongAyah.

```
Q002
... "وَإِذْ قَالَ لُقْمَانُ لِابْنِهِ ... لَا تُشْرِكْ بِاللَّهِ إِنَّ الشِّرْكَ لَظُلْمٌ عَظِيمٌ" ...
... "وَلَقَدْ آتَيْنَا لُقْمَانَ الْحِكْمَةَ ..." ...
... "وَوَصَّيْنَا الْإِنسَانَ بِوَالِدَيْهِ ..." ...
```

- 1B mismatch for Hadith: GT CorrectHadith vs Pred WrongHadith
  The model is overly skeptical and marks well-known hadiths as Wrong.

```
Q003
... "نصرت بالصبا وأُهلِكَت عاد بالدبور" (صحيح البخاري) ...
```

```
Q004
... "لعن الله من عمل عمل قوم لوط" ...
```

- 1C correction mismatch: GT provides the exact required correction; the model outputs placeholders or truncations.

```
Q003
GT correction: وَقِيلَ يَا أَرْضُ ابْلَعِي مَاءَكِ ... وَقِيلَ بُعْدًا لِّلْقَوْمِ الظَّالِمِينَ
Pred correction: خطأ
```

```
Q004
GT correction: إِنَّا خَلَقْنَا الْإِنسَانَ ... إِمَّا شَاكِرًا وَإِمَّا كَفُورًا
Pred correction: إِنَّا خَلَقْنَا الْإِنسَانَ ... سَمِيعًا بَصِيرًا   [missing the continuation]
```

- Hadith false positives: predicted hadith where GT has no hadith. Often fabricated attributions or paraphrases with “رواه مسلم/البخاري” tags but no real source, e.g., Q005 and Q058.

```
Q005
... "أَنَّ النبي صلى الله عليه وسلم نهى عن الزواج من الشُرَّعات" (رواه البخاري) ...
... "لا تَزَوَّجَنَّ أُختَكَ" (رواه مسلم) ...
```

```
Q058
... "قال لقمان: ... (رواية مسلم)" ...  [Luqman sayings falsely labeled as Hadith with “رواية مسلم”]
```

### Why this is happening
- 1B bias toward Wrong: The model is conservative and tends to label spans as Wrong, likely due to uncertainty, small deviations, diacritics, or an internal policy to avoid false “Correct” claims. This yields a huge Correct→Wrong error rate for Ayah and Hadith.
- 1C failure: The instruction requires providing an exact canonical correction, but the model often outputs “خطأ” or a partial snippet. This is a generation behavior issue; it needs stricter schema/guardrails and post-processing.
- Hadith precision: The model treats any “قال النبي/رواه ...” phrasing as hadith, including fabricated or paraphrased lines. There’s no verification against a canonical hadith store, so FP spikes.

### What to change to improve scores significantly

- Strengthen correctness decision (1B):
  - Add a retrieval/verification step:
    - For an Ayah-labeled span, normalize the span (strip diacritics/punctuation, collapse whitespace) and search `datasets/quranic_verses.json`. If a high-confidence match is found, set CorrectAyah; else mark as WrongAyah and record the canonical verse for 1C.
    - For a Hadith-labeled span, validate against `datasets/six_hadith_books.json` or a curated subset. If no credible match, either:
      - downgrade to Neither when the text is narrative, OR
      - keep as WrongHadith but ensure 1C correction is a verified hadith text when applicable. For your task, I would prefer downgrading to Neither to boost 1A precision and avoid hadith FPs.
  - Tighten rules in the prompt: explicitly say “Only label Correct if it exactly matches the canonical source; otherwise label Wrong and provide the exact canonical text as Correction.”

- Fix 1C correction generation:
  - Hard requirement in the schema: Correction must be the exact canonical source text for Wrong labels. Reject “خطأ” or empty; post-process to supply canonical text from the verified match.
  - Add normalizers (strip diacritics, unify punctuation) and a max-length cap with “must include full verse/hadith; do not truncate.”
  - For hadith, always include book and number in addition to text; you can add a separate metadata field to enforce provenance, then derive Correction from that.

- Reduce Hadith false positives (1A precision for Hadith):
  - Heuristics:
    - Require a verified source (book + number) from the hadith dataset to assign Hadith type. Without verification, prefer Ayah or Neither.
    - Penalize generic patterns (“قال النبي”, “رواه مسلم”) if not accompanied by a concrete reference that exists in the dataset.
    - Down-weight hadith detection in responses that are didactic/general advice without isnad references.
  - Prompt changes: include counter-examples of fabricated attributions and instruct the model to avoid labeling them as Hadith.

- Optional: Dynamic effort strategy
  - Since ABC_MEDIUM is explicit, the builder bucketed everything into medium. If you want a more natural distribution (and separate analyses of behavior under different reasoning depths), stop setting `reasoning={"effort":"medium"}` and let token thresholds place examples into low/medium/high; then compare how performance changes with effort.

### Additional diagnostics (already computed)
- Questions with many Hadith FPs:
  - Q005: 9
  - Q058: 7
  - Q083: 4
  - Q077: 3
  - Q094: 3
  - Others: Q008, Q090, Q127, Q131
- Average citations per question in medium TSV: ≈ 6.18
- Medium submission coverage: 840 rows across 136 questions. The builder reports 147 medium-classified questions, but only 136 have non-empty extracted citations in the TSV.

### Concrete implementation plan
- In the pipeline after model generation:
  - Normalize extracted spans.
  - Verify Ayah via `quranic_verses.json`:
    - If match above threshold, flip to CorrectAyah if not already, and set Correction to empty.
    - If no match, keep/flip to WrongAyah and set Correction to the best canonical verse string.
  - Verify Hadith via `six_hadith_books.json`:
    - If verified, flip to CorrectHadith; else either drop to Neither or keep WrongHadith and provide exact Correction from the closest verified hadith (include source).
  - Reject or fix Correction values of “خطأ”/empty; fill with canonical text.
  - Consider adding a min-confidence threshold to even emit a Hadith span.

If you want, I can wire a verification module into `task_abc_pipeline.py` that:
- Loads the Quran and Hadith datasets once.
- Adds normalization and fuzzy matching.
- Post-edits the model’s labels and corrections according to the above rules.
- Rebuilds and re-evaluates high/medium/low to quantify improvements.