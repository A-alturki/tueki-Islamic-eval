Subtask 2 RAG pipeline

Overview
- Generates a TREC-formatted run file (TSV) for Subtask 2 using LLM-assisted retrieval over a vector store populated with Qur'an and Sahih Al-Bukhari.
- Maps individual Quran ayah citations to thematic QPC passage-ids (e.g., 2:233-233) per organizers' collection.
- Resolves hadith citations strictly by hadith_id from the official JSONL.
- Produces outputs under `outputs_task_2/burhanai_task2_RAG_<models>_<effort>_<timestamp>` and can run the organizers' checker/evaluator without modifying their code.

Prerequisites
- Python 3.10+
- Install dependencies: `pip install -r requirements.txt` and `pip install -r original_docs/task_2_docs_and_organizers_code/Evaluation/requirements.txt`
- Set `OPENAI_API_KEY` in your environment.

Datasets expected
- Questions: `datasets/task_2_data/QH-QA-25_Subtask2_ayatec_v1.3_{train,dev}.tsv`
- QPC: `datasets/task_2_data/Thematic_QPC/QH-QA-25_Subtask2_QPC_v1.1.tsv`
- Sahih Al-Bukhari: `datasets/task_2_data/Sahih-Bukhari/QH-QA-25_Subtask2_Sahih-Bukhari_v1.0.jsonl`

Usage
```
python -m task_2_RAG.pipeline_task2_rag \
  --split dev \
  --models gpt-5 \
  --effort medium \
  --concurrency 4 \
  --evaluate
```

Notes
- Resume-safe: reuses existing JSON outputs in the selected output directory.
- Retries with exponential backoff on API errors.
- The `--evaluate` flag copies the produced run file to the organizers' evaluator input, runs the submission checker, and then computes MAP scores on the sample qrels.


