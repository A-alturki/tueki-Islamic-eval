## Research Methodology: AI Agent with Specialized Tools (Task 1A)

### Objective
- **Goal**: Identify every intended or claimed Quranic ayah or Prophetic hadith span inside Arabic responses and output precise character spans with labels, without verifying against external sources.

### Data and inputs
- **Development input**: `datasets/TaskA_Input.xml` (repeated `<Question>` blocks; no outer XML root; parsed via regex blocks then XML).
- **Ground truth**: `datasets/TaskA_GT.tsv`.
- **Optional test subset**: `datasets/Test_Subtask_1A.xml`.

### Agent design and tools
- **Models**: `gpt-5` by default; configurable via `--models` (comma‑separated). Round‑robin assignment across the pool.
- **API**: OpenAI Responses API with structured text output constrained by a JSON Schema.
- **Reasoning**: `reasoning.effort ∈ {low, medium, high}` (default: high; `--effort`).
- **Tools**: `code_interpreter` with a per‑question uploaded file `text_to_analyse.txt` that contains the exact raw response text. If a chosen model does not support tools (e.g., some o‑series), the pipeline falls back to a no‑tools prompt variant that computes indices directly from the provided string.
- **Retrieval**: Vector store (`file_search`) intentionally disabled for 1A to avoid canonicalization effects; we extract what the writer intended, even if inaccurate.
- **Concurrency and reliability**:
  - High‑concurrency async execution with exponential backoff on transient errors.
  - Resume support: previously generated per‑question JSON outputs are detected and skipped.

### Prompting and output schema
- **Developer instruction (abridged)**:
  - Extract what the writer claims or intends to be ayah/hadith, even if fabricated or paraphrased.
  - Select minimal contiguous MATN; exclude quotes, brackets, punctuation, tatweel, ellipses, decorative marks, sura names/numbers, hadith tags, and isnad.
  - Compute indices against the exact raw text; indices are [start, end) exclusive‑end.
  - Local cues within ±64 chars guide label choice (e.g., قال الله تعالى / قال رسول الله ﷺ). Do not over‑include narration.
  - Coverage: extract every distinct occurrence by position; do not collapse duplicates.
  - Avoid overly short/generic fragments; prefer ≥10 Arabic letters unless explicitly quoted with strong cues.
- **Schema‑constrained output**: array `citations` of objects with required fields: `span_start`, `span_end`, `span_text`, `citation_type ∈ {"Aya","Hadeeth"}`.

### Pipeline flow
- **Case construction**: Parse `TaskA_Input.xml` into `(Question_ID, Response)` pairs. For each question, save the raw response to `outputs_task_a/<Question_ID>.txt` and upload it to the tool container if tools are supported.
- **Per‑question call**: Invoke the Responses API with the developer message, the raw response (as user input), strict JSON schema, reasoning settings, and (when supported) the `code_interpreter` attached to `text_to_analyse.txt`. Persist the full JSON output per question.
- **Extraction and building rows**:
  - Parse the model’s JSON and collect candidate spans.
  - Apply deterministic post‑processing (below) and build `TaskA_submission.tsv` with columns: `Question_ID, Annotation_ID, Label, Span_Start, Span_End, Original_Span`.
- **Required competition format**: Build `Subtask1A_submission.tsv` (no header) using inclusive end indices; if no real spans, emit `A-Qxxx\t0\t0\tNo_Spans`.

### Post‑processing (deterministic)
- **Alignment and trimming**:
  - Snap `(start,end)` to the nearest exact occurrence of `span_text` in a ±64‑char window; global search fallback if needed.
  - Trim edge quotes/punctuation/RTL marks/tatweel; keep indices within bounds of the original string.
- **Label adjustments (optional, off by default)**: Local cue‑based correction can flip labels when evidence is strong and unambiguous.
- **Deduplication**:
  - Remove nested duplicates and identical spans.
  - For identical text/label within close proximity (≤120 chars), keep the one with higher cue score; tie‑break by earliest start.
- **Conflict pruning (optional, off by default)**: For heavily overlapping cross‑label spans, keep the one with higher cue score, else longer, else earlier.

### Evaluation protocol
- **Metric**: Character‑level macro‑averaged F1 over {Neither, Ayah, Hadith} using `evaluation.py`.
- **Notes**: Matching is character‑wise; Neither is treated as a class. Reproduce by passing explicit paths to inputs and submission.

### Reproducibility
```bash
# 1) Environment
pip install -r requirements.txt
export OPENAI_API_KEY=...

# 2) Run Task 1A (example: test subset)
python task_a_prompt_engineering/pipeline_task_a.py \
  --input datasets/Test_Subtask_1A.xml \
  --outdir outputs_task_a_test_gpt-5_high_0.81 \
  --models gpt-5 \
  --effort high

# 3) Evaluate (macro F1 over {Neither, Ayah, Hadith})
python evaluation.py \
  --task A \
  --submission outputs_task_a_test_gpt-5_high_0.81/TaskA_submission.tsv \
  --gt datasets/TaskA_GT.tsv \
  --input datasets/Test_Subtask_1A.xml
```

### Design choices and limitations
- **Claimed‑span extraction**: We intentionally avoid canonical verification; the task rewards identifying what the writer intended, which may increase false positives on spurious attributions.
- **Tooling variability**: For models without hosted tools, we fall back to a no‑tools prompt that computes indices from the raw string; behavior is aligned but may be less robust without filesystem‑based alignment.
- **Vector search disabled**: Prevents unintended normalization and keeps spans faithful to the original response.
- **Span hygiene**: Minimum‑length heuristic and deduplication mitigate overly short and redundant fragments but do not target semantic redundancy over distant positions.

### Files of interest
- `task_a_prompt_engineering/pipeline_task_a.py` — agent pipeline, prompts, schema, orchestration, submission building.
- `datasets/TaskA_Input.xml`, `datasets/TaskA_GT.tsv` — inputs and labels.
- `evaluation.py` — character‑level macro F1 computation for 1A.
