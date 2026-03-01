# Repo Structure Notes

## Overview
HUMAIN's IslamicEval 2025 solution. 3 subtasks, we're improving Subtask C.

## Full File Tree
```
/
├── datasets/
│   ├── quranic_verses.json         — Full Quran (raw source for index)
│   ├── six_hadith_books.json       — 34,994 hadiths from 6 books (raw source)
│   ├── dev_SubtaskC.tsv            — Dev ground truth for 1C (180 rows)
│   ├── dev_SubtaskC.xml            — LLM responses for dev set
│   ├── Test_Subtask_1C_USER.tsv    — Test input (308 rows, no corrections)
│   └── Test_Subtask_1C.xml         — LLM responses for test set
├── subtask_C/
│   ├── matcher.py                  — Core matching engine (QuranHadithSpanMatcher)
│   ├── reranker.py                 — BGE-based re-ranker (ReRanker)
│   ├── span_corrector.py           — File I/O + pipeline (SpanCorrector)
│   ├── quran_inverse_index.py      — Script to build quran_index/
│   ├── hadith_inverse_index.py     — Script to build hadith_index/
│   ├── quran_index/                — Pre-built Quran index
│   │   ├── quran_word_index.json   — word → [ayah_ids]
│   │   ├── quran_ayah_lookup.json  — ayah_id → {surah_id, ayah_id, ayah_text, normalized_text}
│   │   └── index_stats.json
│   ├── hadith_index/
│   │   ├── hadith_word_index.json  — word → [hadith_ids]
│   │   ├── hadith_lookup.json      — hadith_id → {hadithID, BookID, title, hadithTxt, Matn}
│   │   └── index_stats.json
│   └── submission_C/
│       └── reranker_q8_h7.tsv      — Best submission (Quran=0.8, Hadith=0.7)
└── main.py                         — Unified entry point
```

## Subtask C Data Flow
```
Test_Subtask_1C_USER.tsv + Test_Subtask_1C.xml
        ↓ SpanCorrector.process_test_tsv_file()
For each row:
  - Extract span text (from XML using span positions, or fallback to Original_Span)
  - Determine source type (Ayah→quran, Hadith→hadith)
  - QuranHadithSpanMatcher.match_span_with_verse_splitting()
      → split on verse separators (Arabic numbers, asterisks, etc.)
      → if single verse: match_single_verse()
      → if multiple verses: match_multiple_verses()
          → find_matches_by_sequence()
              [1] find_single_ayah_matches()
                  → find_exact_substring_matches() (fastest, n-gram index)
                  → word overlap candidates → SequenceMatcher similarity
              [2] find_cross_ayah_matches() (Quran only, spans 2-5 ayahs)
              [3] fallback_matching() (more lenient thresholds)
          → rerank_candidates() (BGE re-ranker or rule-based fallback)
      → apply threshold (Quran=0.80, Hadith=0.70)
      → return canonical text or "خطأ"
        ↓
submission.tsv [Sequence_ID, Correction]
```

## Key Classes

### QuranHadithSpanMatcher (matcher.py)
- `normalize_text(text)` — removes tashkeel/diacritics, punctuation, numbers; normalizes whitespace
- `find_exact_substring_matches(normalized_query, source_type)` — n-gram index lookup then substring check
- `find_single_ayah_matches(query, source_type)` — word overlap → SequenceMatcher; early exit at 0.95
- `find_cross_ayah_matches(query)` — combines 2-5 consecutive ayahs, checks similarity ≥ 0.85
- `fallback_matching(query, source_type)` — n-gram + phrase + substring scores composite
- `find_matches_by_sequence(query, source_type)` — orchestrates all strategies + deduplication + reranking
- `match_span_with_verse_splitting(span, source_type)` — entry point; splits multi-verse spans

### ReRanker (reranker.py)
- Model: `BAAI/bge-reranker-v2-m3`
- `prepare_pairs()` — formats as single string: `f"{query} [SEP] {passage}"` ← BUG (should be list pairs)
- `rerank_candidates(query, candidates, top_k=10, alpha=0.6)` — combined_score = 0.6*original + 0.4*reranker
- `is_available()` — checks model loaded
- **Note:** `fallback_rerank_candidates()` is called in matcher but doesn't exist in ReRanker class

### SpanCorrector (span_corrector.py)
- `extract_full_span_from_xml()` — extracts text using byte positions from XML
- `process_test_tsv_file()` — main loop, generates submission TSV + detailed JSON
- `determine_search_source(span_type)` — "Ayah" → "quran", "Hadith" → "hadith"

## Scoring
- Similarity function: `difflib.SequenceMatcher(None, norm1, norm2).ratio()` (char-level)
- Multi-metric fallback: n-gram overlap + phrase matching + substring containment (0.4/0.35/0.25)
- Final ranking: uses re-ranker score if available (alpha=0.6 original, 0.4 HF)
- Thresholds: Quran=0.80, Hadith=0.70 (applied after reranking in match_single_verse)

## Hadith Data Structure
```json
{
  "hadithID": "bukhari_1",
  "BookID": "bukhari",
  "title": "...",
  "hadithTxt": "full text with isnad",
  "Matn": "just the matn (body text)"
}
```
Hadith lookup key is an integer string index (e.g., "0", "1", ..., "34993").
Both `Matn` and `hadithTxt` are indexed separately in the n-gram index (suffixed with `_matn` or `_hadith`).

## Quran Data Structure
```json
{
  "surah_id": 1,
  "surah_name": "الفاتحة",
  "ayah_id": 1,
  "ayah_text": "بِسْمِ اللَّهِ...",
  "normalized_text": "بسم الله..."
}
```
Ayah lookup key: `"ayah_{surah_id}_{ayah_id}"` (e.g., `"ayah_1_1"`)
