# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Run Subtask C (Span Correction) — primary focus
```bash
cd subtask_C
python span_corrector.py
```

### Run any subtask via unified entry point
```bash
python main.py --task C --input datasets/Test_Subtask_1C_USER.tsv --output corrected.tsv
python main.py --task A --mode dev
python main.py --task B --tsv dev_data.tsv --xml questions.xml
```

### Rebuild search indexes (only needed if source data changes)
```bash
cd subtask_C
python quran_inverse_index.py    # reads datasets/quranic_verses.json → quran_index/
python hadith_inverse_index.py   # reads datasets/six_hadith_books.json → hadith_index/
```

### Install dependencies
```bash
pip install -r requirements.txt
```

## Architecture

### Task Pipeline (Subtask C)
The correction task takes incorrectly cited Quran/Hadith spans from LLM responses and finds the canonical correct text, or returns `خطأ` if the span is hallucinated/unfixable.

**Entry point:** `subtask_C/span_corrector.py` → `SpanCorrector.process_test_tsv_file()`
For each row in the input TSV, it extracts the span text (preferring XML character-position extraction over the truncated `Original_Span` column), determines source type (`Ayah→quran`, `Hadith→hadith`), and calls the matcher.

**Core matching (`subtask_C/matcher.py`, `QuranHadithSpanMatcher`):**
1. Normalize Arabic text (remove tashkeel/diacritics, punctuation, numbers)
2. Try exact n-gram substring match (fastest path)
3. Word-overlap candidate retrieval (top 100) → `difflib.SequenceMatcher` similarity scoring
4. For Quran: try combining 2–5 consecutive ayahs (cross-ayah matching)
5. Fallback: composite score of n-gram overlap + phrase matching + substring containment
6. Re-rank top 30 candidates with `BAAI/bge-reranker-v2-m3`
7. Apply threshold: Quran ≥ 0.80, Hadith ≥ 0.70 → return canonical text or `خطأ`

**Re-ranker (`subtask_C/reranker.py`, `ReRanker`):**
Wraps `BAAI/bge-reranker-v2-m3`. Final score = `0.6 * original_similarity + 0.4 * reranker_score`. If model fails to load, falls back to rule-based re-ranking (but `fallback_rerank_candidates` is called on `ReRanker` which doesn't define it — latent bug).

### Pre-built Indexes
Located at `subtask_C/quran_index/` and `subtask_C/hadith_index/`. These are JSON files loaded at startup — no DB required.
- Quran: 6,236 ayahs, key format `"ayah_{surah_id}_{ayah_id}"`
- Hadith: 34,994 hadiths from 6 books (Bukhari, Muslim, Abu Dawud, Tirmidhi, Ibn Majah, Nasai). Two text fields per hadith: `Matn` (body only) and `hadithTxt` (full with isnad).

### Data Files
- `datasets/dev_SubtaskC.tsv` — ground truth for development (180 rows; 111 are `خطأ`, 69 have valid corrections)
- `datasets/Test_Subtask_1C_USER.tsv` — test input (308 rows, no labels)
- `datasets/Test_Subtask_1C.xml` — full LLM response text used for span extraction by character position
- `subtask_C/submission_C/reranker_q8_h7.tsv` — best submission (Quran=0.8, Hadith=0.7 thresholds)

### Known Bugs in Current Code
1. **BGE re-ranker input format is wrong** (`reranker.py:84`): concatenates query and passage as `f"{query} [SEP] {passage}"` in a single string, but the tokenizer should receive them as two separate inputs for cross-encoder scoring.
2. **`fallback_rerank_candidates` is missing** from `ReRanker` class but called in `matcher.py:497` — will raise `AttributeError` when HF re-ranker is loaded but falls to the else branch.
3. **Thresholds over-conservative**: dev set expects ~38% corrections; the best submission only corrects ~19% of test spans.
4. **Arabic normalization incomplete**: alef variants (أ/إ/آ→ا), hamza, and ta-marbuta (ة→ه) are not normalized, which degrades fuzzy matching.
