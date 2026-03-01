## Task 1A Pipeline (Standalone)

This directory contains a focused, robust pipeline for Subtask 1A: extracting spans of intended Quranic ayahs and Hadith from model responses and producing a Codabench-ready TSV.

### Why a dedicated pipeline?
- Tailored developer prompt for 1A only (no validation/correction steps that can distract the model)
- Strong span hygiene: quote/punctuation trimming, RTL/tatweel cleanup, deduplication of nested shards
- Vector-store assisted plausibility checks using `datasets/quranic_verses.json` and `datasets/six_hadith_books.json`

### Input/Output
- Input: `datasets/TaskA_Input.xml`
- Output TSV: `outputs_task_a/TaskA_submission.tsv`
- Also saves per-question artifacts (`A-Qxx.txt` and `.json`) under `outputs_task_a/`

The TSV schema matches `datasets/TaskA_GT.tsv`:

```
Question_ID	Annotation_ID	Label	Span_Start	Span_End	Original_Span
```

Where `Label` ∈ {`Ayah`, `Hadith`, `NoAnnotation`}.

### Usage

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=...

python task_a_prompt_engineering/pipeline_task_a.py --limit 5  # quick dry-run
python task_a_prompt_engineering/pipeline_task_a.py            # full run
```

Artifacts are written to `outputs_task_a/`.

### Notes and learnings applied from ABC pipeline
- Indices computed on the exact raw response file fed to tools to prevent off-by-one errors
- Aggressive trimming of quotes, brackets, punctuation, RTL controls, and tatweel
- Deduplicate contained spans, prefer longer unified spans
- Use vector store to avoid spurious short spans that don't plausibly match Quran/Hadith


