# Improvement Ideas for Subtask 1C

## Bug Fixes (do these first — guaranteed improvements)

### 1. Fix BGE re-ranker input format (HIGH PRIORITY)
**Current (WRONG):**
```python
input_text = f"{query} [SEP] {passage}"
batch_inputs.append(input_text)
# Then: self.tokenizer(batch_inputs, ...)
```
**Correct for cross-encoder:**
```python
# BGE reranker needs pairs: [[query, passage], ...]
inputs = self.tokenizer(
    [query] * len(batch_pairs),
    [passage for _, passage in batch_pairs],
    padding=True, truncation=True, max_length=512, return_tensors="pt"
)
```
This fix alone could significantly improve re-ranking quality.

### 2. Fix missing `fallback_rerank_candidates` method
In `matcher.py:497`, when HF reranker isn't available:
```python
return self.reranker.fallback_rerank_candidates(query, candidates, top_k)
```
But `ReRanker` class has no `fallback_rerank_candidates` method. This causes AttributeError.
Fix: remove the call or implement the method.

## Threshold Tuning (easy wins)

### 3. Lower thresholds for testing
Current: Quran=0.80, Hadith=0.70
Try: Quran=0.72-0.75, Hadith=0.60-0.65
Dev set shows 38% corrections expected; current system gives only 19% on test.

### 4. Separate threshold for multi-ayah vs single ayah matches

## Algorithmic Improvements

### 5. Better Arabic text normalization
- Handle alef variations: أ، إ، آ، ا → ا
- Handle hamza variations: ء، ئ، ؤ → base forms
- Handle ta marbuta: ة → ه
- Handle waw: و → و (usually fine but some edge cases)
This is crucial for Arabic NLP.

### 6. Use BM25 instead of (or in addition to) word overlap for candidate retrieval
Word overlap with min_overlap = len(words)//3 is very aggressive.
BM25 scoring would be more principled.

### 7. Increase candidate pool
Currently capped at 100. Try 200-500 for better recall.

### 8. Better scoring for partial Quran spans
The current system uses SequenceMatcher ratio which favors length-matched text.
For partial spans (LLM only cited part of an ayah), try:
- Score = overlap_words / len(query_words) [containment score]
- Match if query is mostly CONTAINED in ayah (not just similar)

### 9. Arabic-aware similarity
SequenceMatcher on Arabic chars can be tricky. Consider:
- Word-level edit distance (Levenshtein on words)
- Jaccard similarity on word sets
- TF-IDF cosine similarity

## Model Improvements

### 10. Use a better Arabic NLP model for re-ranking
- `CAMeL-Lab/bert-base-arabic-camelbert-msa` for Arabic BERT embeddings
- `aubmindlab/bert-large-arabertv2` for Arabic
- `UBC-NLP/MARBERT` for Arabic
These would give better semantic similarity for Arabic text than BGE.

### 11. Try sentence-transformers with Arabic support
`sentence-transformers/paraphrase-multilingual-mpnet-base-v2` supports Arabic
and might be better than BGE for this specific task.

### 12. Use ALLaM (Arabic LLM) for direct correction
If API access available, prompt an Arabic LLM directly to identify the correct
ayah/hadith from the span. This is the "nuclear option".

## Pipeline Improvements

### 13. Use XML context more aggressively
Currently: extracts span by character positions from XML Body.
Improvement: also use surrounding context from the LLM response to disambiguate
which ayah/hadith is most likely being cited.

### 14. Two-stage pipeline: first classify if خطأ, then correct
Train a binary classifier: is this span a real Quran/Hadith text or hallucinated?
If classified as real → do correction with lower threshold.
If classified as hallucinated → return خطأ directly.

## Quick Wins to Try First (in order)
1. Fix BGE re-ranker input format
2. Lower thresholds to Quran=0.75, Hadith=0.65
3. Add Arabic letter normalization (alef, hamza, ta marbuta)
4. Increase candidate pool to 200
5. Test on dev set to measure improvement
