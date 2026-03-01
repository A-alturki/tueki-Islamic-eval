## Research Methodology: Approach 2 — AI Agent with Specialized Tools and Post‑processing (Task 1C)

### Objective
- **Goal**: For each erroneous or approximate Arabic citation span labeled as Ayah or Hadith within model responses, produce the exact canonical text from the provided databases, or return "خطأ" if no authentic candidate can be confidently identified.

### Data and Inputs
- **Development inputs**:
  - `datasets/TaskC_Input.xml` (responses by `Question_ID`)
  - `datasets/TaskC_GT.tsv` (gold corrections for accuracy computation)
- **Test inputs**:
  - `datasets/1_c_test_data/Test_Subtask_1C_USER.tsv` (rows with `Sequence_ID, Question_ID, Span_Type, Span_Start, Span_End, Original_Span`)
  - Optional context files: `datasets/1_c_test_data/Test_Subtask_1C.xml` or `datasets/1_c_test_data/Test_Subtask_1C.jsonl`
- **Canonical sources (attached to the agent tool runtime)**:
  - `task_c_prompt_engineering/scoring_from_organizers/quranic_verses.json`
  - `task_c_prompt_engineering/scoring_from_organizers/six_hadith_books.json`

### Agent design and tools
- **Model**: `gpt-5` (configurable; `--model` flag), via OpenAI Responses API.
- **Reasoning**: `reasoning.effort ∈ {low, medium, high}` (default: high; `--effort`).
- **Tools**: `code_interpreter` container only. The two canonical databases are uploaded once per run and reused across all calls. Vector store and retrieval are explicitly disabled.
- **Concurrency and reliability**:
  - Asynchronous execution with a high-concurrency semaphore.
  - Exponential backoff and retry on transient API errors.
  - Resume support: existing per‑case JSON outputs are detected and skipped.

### Prompting and output schema
- **Developer instruction (abridged)**:
  - Use only the attached Quran and Hadith JSON files; no external sources.
  - Enforce exactness: copy the selected canonical text verbatim with diacritics; no quotations, references, or isnads.
  - Matching strategy: normalize (strip diacritics/tatweel/punctuation; unify hamza; collapse spaces); search via substring, token overlap, n‑grams, and approximate ratios across both DBs; test both the full span computed from indices and the user‑provided snippet.
  - Decision rule: accept a candidate when robust similarity to the full span ≥ 0.85; otherwise output "خطأ". When resolvable, ayah numbers may be appended in parentheses; if not certain, return pure text.
  - Final copy‑out: programmatically slice from the DB string to avoid character drift.
- **Schema‑constrained output**: strict JSON object with a single required field `correction` (string). The agent returns only this field; explanations are disallowed.

### Pipeline flow (Task 1C)
- **Case construction**:
  - Development: iterate gold rows to create cases with `Question_ID`, span type, start/end indices, and the organizer‑provided snippet; map `Question_ID → Response` from `TaskC_Input.xml`.
  - Test: read cases from `Test_Subtask_1C_USER.tsv`; and augment response context from XML/JSONL.
- **Per‑case call**:
  - Compute `computed_span` by slicing the full response text with the provided indices; persist a compact provenance record per case.
  - Invoke the Responses API with: developer message, per‑case user prompt (span type, full and snippet spans, indices, and full response context), strict JSON schema, reasoning settings, and the `code_interpreter` attached to the Quran/Hadith files.
  - Save the full model JSON output; extract `correction` from the message content. Invalid or missing JSON yields the fallback output "خطأ".
- **Submission building**: produce `outputs_task_c/Subtask1C_submission.tsv` with two columns (no header): `Sequence_ID`, `Correction`. A ZIP is packaged per competition rules and a `run_meta.json` captures model, effort, environment, durations, and file paths.

### Post‑processing and scoring
- **Normalization for local validation**:
  - Remove default diacritics and apply small orthographic normalizations to both prediction and reference.
  - A prediction is counted correct if it equals the reference after normalization, or if the reference is a substring of the prediction and the prediction (normalized) exists verbatim in either the Quran or Hadith DBs.
- **No external retrieval**: all matching and verification occur using only the two provided databases within the code interpreter container.

### Reproducibility
```bash
# 1) Environment
pip install -r requirements.txt
export OPENAI_API_KEY=...

# 2) Development evaluation (computes local accuracy and writes submission + ZIP)
python task_c_prompt_engineering/pipeline_task_c.py \
  --mode dev \
  --outdir outputs_task_c_gpt5_high_codex_final \
  --effort high \
  --model gpt-5

# 3) Test submission (reads organizer USER TSV, optional context)
python task_c_prompt_engineering/pipeline_task_c.py \
  --mode test \
  --test_user_tsv datasets/1_c_test_data/Test_Subtask_1C_USER.tsv \
  --test_xml datasets/1_c_test_data/Test_Subtask_1C.xml \
  --outdir outputs_task_c_gpt5_high_codex_final
```

### Design choices and limitations
- **Faithfulness vs. coverage**: The agent favors exact canonical reproduction with diacritics. When spans are paraphrased or truncated, approximate search and token‑level similarity improve coverage but may still yield "خطأ" for highly abstracted paraphrases.
- **Database completeness**: Corrections are bounded by the provided Quran and Hadith JSONs; authentic texts absent from these files cannot be recovered.
- **Tooling boundary**: Retrieval and reasoning are confined to the code interpreter with pre‑uploaded files; no web search or external corpora are used. This ensures auditability and competition compliance.
- **Subjictivity In Golden Ground-Truth**: We believe there is a bit of subjictivity to this task in terms of deciding when to return 'خطأ' and when not to - some times it is possible to find a relevant verse but the wording is quite different, or there are multiple verses that could be related to this 'claimed' verse, which makes it difficult to decide on the line.

### Files of interest
- `task_c_prompt_engineering/pipeline_task_c.py` — end‑to‑end agent pipeline, prompt/spec, JSON schema, orchestration, and submission building.
- `task_c_prompt_engineering/scoring_from_organizers/scoring.py` — organizer‑mirrored normalization and accuracy logic.
- `datasets/TaskC_Input.xml`, `datasets/TaskC_GT.tsv`, `datasets/1_c_test_data/Test_Subtask_1C_USER.tsv` — inputs for development and test (from the repo and codebench files section).


