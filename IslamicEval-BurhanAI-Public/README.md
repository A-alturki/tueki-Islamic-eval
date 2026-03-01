## BurhanAI – IslamicEval 2025 Systems

End-to-end pipelines and resources accompanying our paper (see “Paper” below) for the IslamicEval 2025 Shared Task: capturing hallucination in Islamic content. This repo provides complete, runnable code for all subtasks:

- **Subtask 1A**: Identify spans of intended Qur’anic ayahs and Hadiths in LLM responses
- **Subtask 1B**: Validate correctness of each extracted citation
- **Subtask 1C**: Correct erroneous citations to their canonical text
- **Subtask 2**: Islamic QA retrieval over Qur’an passages (QPC) and Sahih Al‑Bukhari


### Repository map

- `abubakr/`
  - Finetuning for 1A phrase detection, dataset generation, search indices, and 1A/1B/1C submission builders
- `task_a_prompt_engineering/`
  - `pipeline_task_a.py`: Agentic 1A span extractor (tools) → Codabench‑ready TSV
- `task_c_prompt_engineering/`
  - `pipeline_task_c.py`: Agentic 1C correction pipeline (dev/test), produces submission ZIP and local scores
  - `scoring_from_organizers/`: Mirrored organizer scorer and metadata for 1C
- `task_2_RAG/`
  - `pipeline_task2_rag.py`: Retrieval‑augmented pipeline for Subtask 2 (TREC run + evaluator integration)
- `datasets/`
  - Shared data, inputs/GTs, canonical Qur’an/Hadith JSON, task‑2 corpora
- `original_docs/`
  - Task descriptions, organizers’ evaluation scripts, and references
 - `README_OLD.md`
   - Documentation for an earlier exploratory pipeline that combines all tasks (A,B, and C) for subtask 1 (kept for reference only)


### Prerequisites

- Python 3.10+
- `OPENAI_API_KEY` set in your environment
- Install dependencies:

```bash
pip install -r requirements.txt
# For task 2 evaluator integration as packaged:
pip install -r original_docs/task_2_docs_and_organizers_code/Evaluation/requirements.txt
```

Optional:

- `VECTOR_STORE_ID`: If provided, pipelines reuse an existing vector store; otherwise they create one and cache it to `.vector_store_id` files


### Datasets expected

Place the following under `datasets/` (already included in this repo snapshot where allowed by the task rules):

- Task 1 inputs/GTs: `TaskA/B/C_Input.xml`, `TaskA/B/C_GT.tsv`
- Canonical sources: `quranic_verses.json` (Othmani script) and `six_hadith_books.json`
- Subtask 2:
  - Questions: `task_2_data/QH-QA-25_Subtask2_ayatec_v1.3_{train,dev,test}.tsv`
  - Thematic QPC: `task_2_data/Thematic_QPC/QH-QA-25_Subtask2_QPC_v1.1.tsv`
  - Hadith: `task_2_data/Sahih-Bukhari/QH-QA-25_Subtask2_Sahih-Bukhari_v1.0.jsonl`

## Subtask 1A — Span detection (two approaches)

### Approach A: Fine‑tuned span detector

- Code: `abubakr/taskA/finetuning/*`, submission builder `abubakr/taskA/03-build-subtaskA-submission.py`
- Data creation: `abubakr/taskA/finetuning/01-phrase_detection_dataset_llm_generation.py` and `02-phrase_detection_dataset_creation.py`; training: `03-phrase_detection_finetuning_task.py`
- Run (example):

```bash
python abubakr/taskA/03-build-subtaskA-submission.py ft:<your_model_id> \
  --test_file datasets/taskA-testdata/Test_Subtask_1A/test_Subtask_1A.jsonl \
  --output abubakr/taskA/tasks_output/submission_task_A.tsv
```

### Approach B: Agentic span extractor (tools)

- Code: `task_a_prompt_engineering/pipeline_task_a.py`
- Span hygiene (trim/normalize/deduplicate) and plausibility checks against canonical corpora
- Run:

```bash
python task_a_prompt_engineering/pipeline_task_a.py --limit 5
python task_a_prompt_engineering/pipeline_task_a.py
# Output TSV at outputs_task_a/TaskA_submission.tsv
```


## Subtask 1B — Validation (hierarchical search)

- Code: `abubakr/taskA/04-build-subtaskB-submission.py`
- Uses a hierarchical search engine over Qur’an/Hadith with exact→normalized→fuzzy→semantic checks; optional LLM verification
- Prereq (indices): `abubakr/taskA/01-index-religion-dataset-for-search.py` (Whoosh/Qdrant optional) and data under `abubakr/datasets/`
- Run:

```bash
python abubakr/taskA/04-build-subtaskB-submission.py \
  --phrases-detected-tsv datasets/taskA-testdata/Test_Subtask_1B/Test_Subtask_1B_USER.tsv \
  --test-data-file datasets/taskA-testdata/Test_Subtask_1B/Test_Subtask_1B.jsonl \
  --output_file_name abubakr/taskA/tasks_output/submission_task_B.tsv
```


## Subtask 1C — Correction (two approaches)

### Approach A: Hierarchical correction

- Code: `abubakr/taskA/05-build-subtaskC-submission.py`
- Reuses the hierarchical search engine to propose canonical corrections; falls back to “خطأ” when no confident match exists
- Run:

```bash
python abubakr/taskA/05-build-subtaskC-submission.py \
  --phrases-detected-tsv datasets/taskA-testdata/Test_Subtask_1C/Test_Subtask_1C_USER.tsv \
  --test-data-file datasets/taskA-testdata/Test_Subtask_1C/Test_Subtask_1C.jsonl \
  --output_file_name abubakr/taskA/tasks_output/submission_task_C.tsv
```

### Approach B: Agentic correction (dev/test)

Produces a 1C submission TSV/ZIP and optional local dev accuracy mirroring organizer rules.

```bash
# Dev mode (scores on TaskC_GT.tsv)
python task_c_prompt_engineering/pipeline_task_c.py --mode dev --outdir outputs_task_c --effort high

# Test mode (builds the two‑column TSV and a zip package)
python task_c_prompt_engineering/pipeline_task_c.py \
  --mode test \
  --test_user_tsv datasets/1_c_test_data/Test_Subtask_1C_USER.tsv \
  --test_xml datasets/1_c_test_data/Test_Subtask_1C.xml
```

Key design decisions:

- Uses only the provided canonical DBs via code_interpreter (no external sources)
- Enforces exact diacritics; returns only “خطأ” when no confident canonical match exists
- Saves per‑case JSON, builds `Subtask1C_submission.tsv`, and packages a ZIP for submission


## Subtask 2 – Islamic QA RAG pipeline

LLM‑assisted retrieval and run generation with end‑to‑end evaluator integration.

```bash
python -m task_2_RAG.pipeline_task2_rag \
  --split dev \
  --models gpt-5 \
  --effort medium \
  --concurrency 4 \
  --evaluate
```

Features:

- Vector store populated with Qur’an and Sahih Al‑Bukhari
- Quran ayah → QPC passage mapping (e.g., `2:233-233`) per organizers’ collection
- Hadith citations resolved strictly by official `hadith_id`
- Writes outputs under `outputs_task_2/burhanai_task2_RAG_<models>_<effort>_<timestamp>/`
- `--evaluate` copies the run to the organizers’ checker and computes MAP on sample qrels



## Reproducing our runs

- Output directories under `outputs_task_a/`, `outputs_task_c/`, and `outputs_task_2/` include multiple model/effort variants used in ablations


## Key results

- Task 1A (agentic tools): Macro‑F1 90.06%
- Task 1A (fine‑tuned): Macro‑F1 87.10% (87.78% for a variant in ablations)
- Task 1B (hierarchical validation): Accuracy 88.60%
- Task 1C (hierarchical correction / agentic): Accuracy 66.56% / 61.04%
- Task 2 (RAG): MAP@10 0.6199; MAP_Q@5 0.5761; MAP_H@5 0.0000 (no hadith gold in sampled qrels)


## Paper

The full system description, methodology, and results are in our submission PDF:

- “BurhanAI_submission___final.pdf” (at the repo root)


## Citation

If you use any part of this repository, please cite our system description paper:

```bibtex
@inproceedings{adel2025burhan,
  title={BurhanAI at IslamicEval 2025 Shared Task: Combating Hallucinations in LLMs for Islamic Content; Evaluation, Correction, and Retrieval-Based Solution},
  author={Al Adel, Arij and Soliman, Abu Bakr and Sawan, Mohamed Sakher and Al-Najjar, Rahaf and Amin, Sameh},
  booktitle={IslamicEval @ ArabicNLP 2025},
  year={2025},
  url={https://openreview.net/forum?id=r00SAkJo7o}
}
```


## Troubleshooting

- If vector store errors occur, delete `.vector_store_id` (and `.vector_store_id_task2`) to force recreation and re‑upload of dataset files
- For partial submissions, ensure the JSON outputs exist in the directory your split builder reads (default `outputs/`)
- Ensure `OPENAI_API_KEY` is set and that your Python environment matches the required versions


