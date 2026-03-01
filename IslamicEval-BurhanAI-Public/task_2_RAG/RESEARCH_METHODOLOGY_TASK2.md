## Subtask 2: Retrieval-Augmented Generation (RAG) – Research Methodology and Design Rationale

### Problem framing
- **Task intent**: Given an MSA question, return up to 20 answer-bearing sources drawn strictly from two corpora: the Thematic Qur'an Passage Collection (QPC) and Sahih Al‑Bukhari. “Answer‑bearing” means a citation that encloses the information needed to answer the question directly or with minimal bridging inference.
- **Unit of judgement**: The evaluator scores at the passage level for Qur'an (QPC passage‑ids such as `2:233-233`) and hadith level for Bukhari (`hadith_id`). Our system therefore must (a) ground search in the two corpora, (b) identify verse or hadith references precisely, and (c) deterministically map them to the organizers’ identifiers.

### Datasets
- **Questions (AyaTEC v1.3)**: `train`, `dev`, `test` TSVs under `datasets/task_2_data/` with `id\tquestion` per line.
- **QPC**: `datasets/task_2_data/Thematic_QPC/QH-QA-25_Subtask2_QPC_v1.1.tsv` providing the canonical passage segmentation for Qur'anic evaluation.
- **Sahih Al‑Bukhari (curated)**: `datasets/task_2_data/Sahih-Bukhari/QH-QA-25_Subtask2_Sahih-Bukhari_v1.0.jsonl` with `hadith_id` and `source_hadith_id`.
- **QRels (train/dev only)**: `qrels_train.gold`, `qrels_dev.gold`. No public hadith gold in train/dev; no public gold for test.

### System architecture
- **Grounded retrieval substrate**: OpenAI `file_search` over a vector store populated with the above corpora. The model is explicitly instructed to search only these files and not hallucinate beyond them. We upload: `q_full.json` (Qur'an text), QPC TSV, and Bukhari JSONL. This keeps search space closed and auditable.
- **Agent policy and schema**:
  - Developer instruction asks for up to 20 citations that “potentially enclose the answer(s),” ranked by directness/completeness.
  - Strict JSON Schema per citation: `type ∈ {quran,hadeeth}`, with `(sura_number, aya_number)` for Qur'an items and `hadith_id` for hadith items. The compact schema reduces failure modes and enables lossless mapping.
- **Reasoning and verbosity**: Default `reasoning.effort` is medium with `text.verbosity=medium` for stability; final reported runs used `--effort high` to encourage deeper search chains while keeping the same schema.
- **Tools**: `file_search` (primary) plus `code_interpreter` for light provenance. If the vector store is unavailable, the pipeline degrades gracefully by disabling tools, but Task 2 results rely on file_search being enabled.

### Retrieval philosophy: “answer‑enclosing” vs. topical relevance
- The prompt biases the model toward citations that directly contain the answer, not merely topical mentions. For example, verses explicitly naming a required entity or stating a ruling are ranked above general thematic verses. This framing narrows the semantic gap inherent in Arabic QA and reduces false‑positive passages that require extensive external reasoning.
- For Qur'an, returning individual ayat lets us aggregate to QPC units deterministically, aligning the model’s fine‑grained reasoning with the evaluator’s coarser unit of judgement.

### Deterministic post‑processing and mapping
- **Qur'an mapping**: Map each `(sura_number, aya_number)` to a QPC `passage-id` using the QPC TSV. Multiple verses often collapse to the same passage; we emit a single docid and keep the earliest rank to preserve model intent while avoiding redundancy.
- **Hadith mapping**: Resolve either `hadith_id` or `source_hadith_id` to the canonical `hadith_id` using the curated JSONL. We de‑duplicate by `hadith_id` while preserving first‑seen rank.
- **Ranking and scoring**: Preserve the model’s order to reflect its judgement of directness; assign monotonically decreasing scores to satisfy TREC run format. No heuristic re‑ranking is applied post‑hoc to avoid misaligning with the prompt’s optimization target.

### Orchestration, resilience, and observability
- **Concurrency and retries**: Async execution with configurable concurrency and exponential backoff+random jitter. This supports large splits with robust completion.
- **Resume‑safety**: Per‑question JSON artifacts allow interrupted runs to resume without recomputation.
- **Provenance**: Each question’s raw text and model JSON are persisted under `outputs_task_2/.../`, with a `run_meta.json` capturing dataset split, model pool, effort, durations, environment, and vector store id.

### Evaluation protocol and nuances
- We use the organizers’ code unmodified. The evaluator expects three qrels files: overall (`combined_qrels.tsv`), Qur'an (`quran_sample.qrels`), and Hadith (`hadith_sample.qrels`). Because train/dev lack hadith gold, our combined qrels capture Qur'an supervision only; `hadith_sample.qrels` remains empty, hence MAP_H@5 is 0 by construction.
- For test, there is no public gold; we run only the submission checker to validate format.

### Results (reference runs)
- **ALL (train+dev)**
  - Artifacts: `outputs_task_2/burhanai_task2_RAG_gpt_5_high_20250809_170234/`
  - Status: 250/250 completed; checker passed
  - Evaluation (merged train+dev qrels): MAP@10=0.6199, MAP_Q@5=0.5761, MAP_H@5=0.0000 (expected given missing hadith gold)
- **TEST**
  - Artifacts: `outputs_task_2/burhanai_task2_RAG_gpt_5_high_20250809_170435/`
  - Status: 71/71 completed; checker passed; evaluation N/A

### Error analysis and limitations
- **Hadith supervision gap**: Lack of train/dev hadith qrels prevents offline tuning for hadith retrieval; MAP_H is uninformative on these splits.
- **Passage aggregation assumptions**: QPC mapping assumes verses returned by the model fall cleanly within single passages. Edge cases (verses spanning adjacent passages) are rare but possible; we favor the exact QPC mapping provided.
- **Arabic linguistic variability**: No explicit morphological or diacritic normalization is applied in post‑processing; robustness relies on the vector store and the model’s search iterations. Queries with rare morphology or dialect could under‑retrieve.
- **Ranking fidelity**: We preserve the model’s ordering; if the model occasionally ranks supportive context above the most answer‑direct citation, we do not override it to avoid introducing new biases.

### Design choices that mattered
- **Closed‑world grounding**: Limiting search to two auditable corpora plus a strict schema eliminates source drift and “almost‑right” citations from outside the scoring universe.
- **Schema minimalism**: Requiring only `sura/aya` or `hadith_id` made the mapping deterministic and resilient across SDK output variations.
- **Thematic aggregation**: Emitting QPC passage‑ids for Qur'an aligns model granularity to the evaluator’s unit, reducing fragmentation and improving MAP.

### Future work
- Explore vector store ingestion strategies (chunk sizing, overlap) and Arabic‑specialized embedding models to improve recall on paraphrastic questions.
- Add optional query‑expansion prompts (synonyms, tafsir‑guided paraphrases) while retaining closed‑world constraints.
- Consider shallow re‑ranking informed by lightweight heuristics (entity match, directive/answer verbs) only if it demonstrably preserves “answer‑enclosing” priority.
- Evaluate adding auxiliary corpora (e.g., tafsir) as side channels for query reformulation without polluting the scoring universe.

### Reproducibility
```bash
# Environment
pip install -r requirements.txt
pip install -r original_docs/task_2_docs_and_organizers_code/Evaluation/requirements.txt
export OPENAI_API_KEY=...

# ALL (train+dev) with evaluation (uses merged qrels as described)
python -m task_2_RAG.pipeline_task2_rag \
  --split all \
  --models gpt-5 \
  --effort high \
  --concurrency 1000 \
  --evaluate

# TEST (no public gold)
python -m task_2_RAG.pipeline_task2_rag \
  --split test \
  --models gpt-5 \
  --effort high \
  --concurrency 1000 \
  --evaluate
```

### Files of interest
- `task_2_RAG/pipeline_task2_rag.py` — end‑to‑end agent pipeline, schema, mapping, evaluator integration.
- `datasets/task_2_data/...` — AyaTEC questions, QPC TSV, Sahih Al‑Bukhari JSONL, qrels.
- `original_docs/task_2_docs_and_organizers_code/Evaluation/` — submission checker and evaluator (used as‑is).
