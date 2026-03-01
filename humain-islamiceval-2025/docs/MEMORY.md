# HUMAIN IslamicEval 2025 - Quick Reference

## Goal
Beat HUMAIN's best solution on IslamicEval 2025 Shared Task, focusing on **Subtask 1C (Span Correction)**.

## See detailed notes:
- `repo_structure.md` — full repo/code walkthrough
- `data_analysis.md` — dataset analysis and error patterns
- `improvement_ideas.md` — ranked list of ideas to improve 1C

## Quick Reference

### Key file paths (all relative to repo root)
- Input test: `datasets/Test_Subtask_1C_USER.tsv` (308 spans)
- Input XML: `datasets/Test_Subtask_1C.xml`
- Dev ground truth: `datasets/dev_SubtaskC.tsv` (180 spans)
- Dev XML: `datasets/dev_SubtaskC.xml`
- Reference: `datasets/quranic_verses.json`, `datasets/six_hadith_books.json`
- Quran index: `subtask_C/quran_index/` (6,236 ayahs, 32,480 unique words)
- Hadith index: `subtask_C/hadith_index/` (34,994 hadiths, 170,518 unique words)
- Core matcher: `subtask_C/matcher.py` (QuranHadithSpanMatcher)
- Re-ranker: `subtask_C/reranker.py` (BAAI/bge-reranker-v2-m3)
- Processor: `subtask_C/span_corrector.py` (SpanCorrector)
- Best submission: `subtask_C/submission_C/reranker_q8_h7.tsv`

### Task 1C Summary
- Input: incorrectly cited Quran/Hadith spans from LLM responses
- Output: corrected canonical text OR "خطأ" (if no valid match found)
- Label types: `WrongAyah`, `WrongHadith`
- Span_Type in test TSV: `Ayah`, `Hadith`

### Official Evaluation Results
- Metric: **Accuracy** (exact match prediction vs. gold label)
- **HUMAIN dev accuracy: 72.74%** — best in the challenge
- **HUMAIN test accuracy: 68.18%** — best in the challenge (1st place)
- BurhanAI test accuracy: 66.56% (2nd place)
- Majority baseline (always خطأ): ~62%
- Our goal: beat 68.18% on test set

### Submission Stats
- Thresholds: Quran=0.80, Hadith=0.70
- Submission: 59/308 corrected (19.2%), 249/308 خطأ (80.8%)
- Dev GT: 69/180 corrected (38.3%), 111/180 خطأ (61.7%)
- The model over-predicts خطأ on test set vs. dev set distribution

### Known Bugs / Issues
1. Re-ranker input format WRONG: uses `f"{query} [SEP] {passage}"` as single string;
   BGE cross-encoder needs `[[query, passage]]` pairs
2. Thresholds too aggressive (0.80/0.70) — misses many valid corrections
3. Only top 100 candidates retrieved before similarity scoring
4. Re-ranker only applied to top 30 candidates
5. `fallback_rerank_candidates` method referenced but not defined in ReRanker class
   (only `rerank_candidates` and `calculate_similarity_scores` exist)
