import argparse
import asyncio
import json
import math
import os
import random
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import platform
import sys
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, OpenAI, RateLimitError
from tqdm import tqdm
try:
    import subprocess
except Exception:  # pragma: no cover - fallback when subprocess unavailable
    subprocess = None


# -----------------------------
# Configuration
# -----------------------------

MODEL_POOL = [
    "gpt-5"
]

# Default reasoning effort; can be overridden via CLI
REASONING_EFFORT = "high"

OUTPUT_DIR = "outputs_task_a"
VECTOR_STORE_ID = os.getenv("VECTOR_STORE_ID")
VECTOR_ID_FILE = ".vector_store_id"


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class Question:
    id: str
    response: str


# -----------------------------
# Dataset utilities
# -----------------------------

def parse_xml_task_a(path: str) -> List[Question]:
    """Parse TaskA_Input.xml which contains repeated <Question> blocks.

    The file has no outer root, so we regex-extract blocks first, then parse.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    blocks = re.findall(r"<Question>.*?</Question>", raw, flags=re.DOTALL)
    questions: List[Question] = []
    for block in blocks:
        try:
            q = ET.fromstring(block)
        except ET.ParseError:
            continue
        qid = (q.findtext("ID") or "").strip()
        # Preserve raw response text exactly as in the XML (no strip)
        resp = q.findtext("Response")
        response = resp if resp is not None else ""
        if qid:
            questions.append(Question(id=qid, response=response))
    return questions


# -----------------------------
# Vector store bootstrap
# -----------------------------

def ensure_vector_store() -> str:
    """Return a usable vector store ID, creating one with dataset files if needed."""
    client = OpenAI()
    global VECTOR_STORE_ID

    if VECTOR_STORE_ID is None and os.path.exists(VECTOR_ID_FILE):
        with open(VECTOR_ID_FILE, "r", encoding="utf-8") as f:
            VECTOR_STORE_ID = f.read().strip() or None

    if VECTOR_STORE_ID:
        try:
            client.vector_stores.retrieve(VECTOR_STORE_ID)
            return VECTOR_STORE_ID
        except Exception:
            print("Stored vector_store_id invalid – creating a new one…")
            VECTOR_STORE_ID = None

    # For Task 1A we do NOT use file_search. Return a dummy placeholder to satisfy code paths.
    VECTOR_STORE_ID = "disabled-for-task-1a"
    return VECTOR_STORE_ID


# -----------------------------
# Model prompt (Task 1A focused)
# -----------------------------

DEVELOPER_MESSAGE = '''
Task 1A: Identify every INTENDED or CLAIMED Quranic ayah or Prophetic hadith span inside the given Arabic response text and return spans ONLY.

Scoring: character-level F1 over classes {Neither, Ayah, Hadith}. Your spans MUST match exact substring indices in the provided raw text. Do not normalize or rewrite the text. Compute indices against the raw file bytes as loaded by Python.

Important policy: extract what the writer CLAIMS or INTENDS to be an ayah or hadith, EVEN IF IT IS FABRICATED, MISQUOTED, OR PARAPHRASED. Do not verify against sources and do not skip doubtful spans. If the text implies a citation (e.g., قال الله تعالى / يقول الله / كما قال تعالى / قال رسول الله / في الحديث / روي عن النبي), select the span exactly as it appears.

Extraction rules (apply strictly):
1. Span content
   - Select only the minimal contiguous MATN text of the ayah/hadith present in the response.
   - Exclude: surrounding quotes, brackets, punctuation, emojis, tatweel (ـ), ellipses, decorative marks, sura names, verse numbers, hadith collection tags, and narrator chains (عن فلان قال…).
   - If paraphrased but clearly intended, select the paraphrase exactly as written.
2. Boundaries and indices
   - Use Python to open the file "text_to_analyse.txt" and work with its raw string.
   - Treat indices as [start, end) exclusive-end. Compute start by locating the first character of the chosen span in the raw string and compute end = start + len(span_text).
   - Trim ONLY edge characters if present at the span edges before finalizing indices: whitespace, newlines, tabs, RTL controls, quotes «»"' and brackets ()[]{} <> and punctuation ، ؛ : . , ! ? … and tatweel ـ.
   - After computing, assert that raw[start:end] == span_text. If not, re-check by sliding the window by at most 2 characters inward to remove stray quotes/newlines, then re-assert. If still failing, recompute start by searching for the exact span_text in the raw string and use that index.
3. Ayah vs Hadith decision (based on local cues)
   - Ayah cues within ±64 chars: قال الله تعالى، يقول الله، الآية، كما قال تعالى، سورة، (4:3) etc.
   - Hadith cues within ±64 chars: قال رسول الله، النبي ﷺ، في الحديث، روي عن، أخرجه البخاري/مسلم/الترمذي…
   - If both cues appear, classify by the MATN itself (what the span text is), not the surrounding narration.
   - Narration words like "عن"، "حدثنا"، "قال" تخص الراوي ليست جزءاً من المتن، ولا تحدد النوع وحدها.
4. Coverage and duplicates (very important)
   - Extract EVERY distinct occurrence (by position) of each ayah/hadith phrase in the response. Do NOT collapse identical phrases; each occurrence must be a separate span if it appears separately.
   - Scan the entire text: when you identify a phrase span, search the raw text for other exact occurrences of the same phrase and include them too, provided local cues still indicate a citation.
5. Avoid overly short or generic fragments
   - Reject single words or very short tokens unless they are clearly iconic and explicitly cited with strong cues and quotes. As a rule of thumb, require ≥ 10 Arabic letters for a span unless enclosed within quotes with explicit cue.
   - Do not extract formulaic phrases such as "بسم الله الرحمن الرحيم" unless the text explicitly presents them as a citation.
6. Selecting the correct boundaries in long narratives
   - Prefer the smallest exact quoted statement that constitutes the ayah/hadith, not entire narrative paragraphs or stories. If a long paragraph includes one or more quoted statements, extract each quoted statement separately.
   - If the span contains an embedded ayah within a hadith narration, extract only the ayah text as an "Aya" span; do not label the whole narration as "Aya".
7. Overlaps
   - Do not produce nested duplicates or overlapping shards of the same citation. Separate spans are allowed only when they represent distinct occurrences.
'''

DEVELOPER_MESSAGE_NO_TOOLS = '''
Task 1A: Identify every INTENDED or CLAIMED Quranic ayah or Prophetic hadith span inside the given Arabic response text and return spans ONLY.

You will receive the raw response text directly in the user message. Compute indices against that exact string. Do not normalize or rewrite the text.

Extraction rules (apply strictly):
1. Span content
   - Select only the minimal contiguous MATN text of the ayah/hadith present in the response.
   - Exclude: surrounding quotes, brackets, punctuation, emojis, tatweel (ـ), ellipses, decorative marks, sura names, verse numbers, hadith collection tags, and narrator chains (عن فلان قال…).
   - If paraphrased but clearly intended, select the paraphrase exactly as written.
2. Boundaries and indices
   - Treat indices as [start, end) exclusive-end. Compute start by locating the first character of the chosen span in the raw text and end = start + len(span_text).
   - Trim ONLY edge characters if present at the span edges before finalizing indices: whitespace, newlines, tabs, RTL controls, quotes «»"' and brackets ()[]{} <> and punctuation ، ؛ : . , ! ? … and tatweel ـ.
   - After computing, assert that raw_text[start:end] == span_text. If not, re-check by sliding inward by up to 2 chars to remove stray quotes/newlines, then re-assert. If still failing, recompute start by searching for the exact span_text in the raw string and use that index.
3. Ayah vs Hadith decision (based on local cues)
   - Ayah cues within ±64 chars: قال الله تعالى، يقول الله، الآية، كما قال تعالى، سورة، (4:3) etc.
   - Hadith cues within ±64 chars: قال رسول الله، النبي ﷺ، في الحديث، روي عن، أخرجه البخاري/مسلم/الترمذي…
   - If both cues appear, classify by the MATN itself (what the span text is), not the surrounding narration.
4. Coverage and duplicates (very important)
   - Extract EVERY distinct occurrence (by position) of each ayah/hadith phrase in the response. Do NOT collapse identical phrases; each occurrence must be a separate span if it appears separately.
5. Avoid overly short or generic fragments
   - Reject single words or very short tokens unless clearly iconic and explicitly cited. Prefer ≥ 10 Arabic letters unless quoted with an explicit cue.
6. Overlaps
   - Do not produce nested duplicates or overlapping shards of the same citation. Separate spans are allowed only when they represent distinct occurrences.
'''

SCHEMA = {
    "type": "object",
    "properties": {
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "span_start": {"type": "integer"},
                    "span_end": {"type": "integer"},
                    "span_text": {"type": "string"},
                    "citation_type": {"type": "string", "enum": ["Aya", "Hadeeth"]},
                },
                "required": ["span_start", "span_end", "span_text", "citation_type"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["citations"],
    "additionalProperties": False,
}


# -----------------------------
# OpenAI calling helpers
# -----------------------------

async def _call_model_once(
    client: AsyncOpenAI,
    q: Question,
    semaphore: asyncio.Semaphore,
    *,
    model_name: str,
    reasoning_effort: str,
) -> Optional[str]:
    async with semaphore:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        text_path = os.path.join(OUTPUT_DIR, f"{q.id}.txt")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(q.response)

        file_id: Optional[str] = None
        try:
            # Resolve model-specific capabilities
            model_lower = (model_name or "").lower()
            supports_tools = True
            text_verbosity = "high"
            if "o3-mini" in model_lower:
                supports_tools = False  # code_interpreter unsupported per API errors
                text_verbosity = "medium"  # safe default
            elif model_lower.startswith("o1") or " o1" in model_lower or model_lower == "o1":
                # o1 does not support hosted tools per API errors
                supports_tools = False
                text_verbosity = "medium"
            elif "o3" in model_lower or "o4-mini" in model_lower:
                # These models require text verbosity = medium
                text_verbosity = "medium"

            # Upload file only if we will use tools
            if supports_tools:
                with open(text_path, "rb") as fb:
                    buf = BytesIO(fb.read())
                    buf.name = "text_to_analyse.txt"
                    uploaded = await client.files.create(file=buf, purpose="assistants")
                file_id = uploaded.id

            # Choose developer message depending on tool support
            developer_msg = DEVELOPER_MESSAGE if supports_tools else DEVELOPER_MESSAGE_NO_TOOLS

            response = await client.responses.create(
                model=model_name,
                input=[
                    {
                        "role": "developer",
                        "content": [{"type": "input_text", "text": developer_msg}],
                    },
                    {"role": "user", "content": [{"type": "input_text", "text": q.response}]},
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "task_1a_spans",
                        "strict": True,
                        "schema": SCHEMA,
                    },
                    "verbosity": text_verbosity,
                },
                reasoning={"effort": reasoning_effort, "summary": None},
                tools=(
                    [
                        {
                            "type": "code_interpreter",
                            "container": {"type": "auto", "file_ids": [file_id]},
                        },
                    ]
                    if supports_tools and file_id
                    else []
                ),
                store=False,
            )

            json_path = os.path.join(OUTPUT_DIR, f"{q.id}.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(response.model_dump(), f, ensure_ascii=False, indent=2)
            return json_path
        finally:
            if file_id:
                try:
                    await client.files.delete(file_id)
                except Exception:
                    pass


async def call_model_with_retry(
    client: AsyncOpenAI,
    q: Question,
    semaphore: asyncio.Semaphore,
    *,
    model_name: str,
    reasoning_effort: str,
    retries: int = 5,
) -> Optional[str]:
    for attempt in range(retries):
        try:
            return await _call_model_once(
                client, q, semaphore, model_name=model_name, reasoning_effort=reasoning_effort
            )
        except (RateLimitError, APITimeoutError, APIConnectionError) as e:
            wait = (2 ** attempt) * 5 + random.uniform(0, 2)
            print(f"⚠️  {e.__class__.__name__} for {q.id} – retry {attempt + 1}/{retries} in {wait:.1f}s")
            await asyncio.sleep(wait)
        except Exception as e:
            print(f"❌ Unexpected error on {q.id}: {e}")
            return None
    print(f"❌ Failed after {retries} retries for {q.id}")
    return None


# -----------------------------
# Post-processing for spans
# -----------------------------

_TRIM_CHARS = "\u200f\u200e\u202a\u202b\u202c\ufeff\u061c\u0640"  # RTL marks, tatweel
_QUOTE_CHARS = "'\"“”‘’«»()[]{}<>،؛:.,!?…\n\t "

# Local cue lexicons for label correction/scoring
ENABLE_LABEL_CUES = False
ENABLE_DEDUP_IDENTICAL = True
ENABLE_PRUNE_CONFLICTS = False
_AYAH_CUES = [
    "قال الله", "يقول الله", "كما قال تعالى", "الآية", "سورة", "آية", "﴿", "﴾",
]
_HADITH_CUES = [
    "قال رسول الله", "النبي", "ﷺ", "في الحديث", "روي عن", "أخرجه", "حديث",
]


def _trim_span(text: str, start: int, end: int) -> Tuple[int, int]:
    start = max(0, start)
    end = min(len(text), end)
    while start < end and (text[start] in _QUOTE_CHARS or text[start] in _TRIM_CHARS):
        start += 1
    while end > start and (text[end - 1] in _QUOTE_CHARS or text[end - 1] in _TRIM_CHARS):
        end -= 1
    return start, end


def _cue_score(text: str, start: int, label: str) -> int:
    window_start = max(0, start - 64)
    window_end = min(len(text), start + 64)
    ctx = text[window_start:window_end]
    cues = _AYAH_CUES if label == "Ayah" else _HADITH_CUES
    score = 0
    for c in cues:
        if c in ctx:
            score += 1
    return score


def _dedupe_spans(spans: List[Tuple[int, int, str, str]], original: str) -> List[Tuple[int, int, str, str]]:
    """Remove nested duplicates and repeated identical texts; prefer stronger cues then earliest.

    spans: list of (start, end, label, text)
    """
    spans_sorted = sorted(spans, key=lambda s: (s[0], -(s[1] - s[0])))
    result: List[Tuple[int, int, str, str]] = []
    for s in spans_sorted:
        s_start, s_end, s_label, s_text = s
        keep = True
        # Drop contained shards of same type
        for r in result:
            r_start, r_end, r_label, r_text = r
            if s_label == r_label and s_start >= r_start and s_end <= r_end:
                keep = False
                break
            if s_start == r_start and s_end == r_end and s_label == r_label:
                keep = False
                break
        if not keep:
            continue

        if ENABLE_DEDUP_IDENTICAL:
            # For identical text with same label in close proximity, keep the one with higher cue score; tie -> earliest
            duplicates = [
                (idx, r)
                for idx, r in enumerate(result)
                if r[2] == s_label and r[3] == s_text and abs(r[0] - s_start) <= 120
            ]
            if duplicates:
                s_score = _cue_score(original, s_start, s_label)
                # Compare against the best existing
                best_idx = duplicates[0][0]
                best = result[best_idx]
                best_score = _cue_score(original, best[0], best[2])
                if s_score > best_score or (s_score == best_score and s_start < best[0]):
                    # Replace existing with stronger/earlier occurrence
                    result[best_idx] = s
                # Else drop this one
                continue

        result.append(s)
    return result


def _maybe_correct_label(original: str, start: int, end: int, label: str) -> str:
    """Adjust label based on local cue phrases if strong evidence contradicts model label."""
    window_start = max(0, start - 64)
    window_end = min(len(original), end + 16)
    ctx = original[window_start:window_end]
    has_ayah = any(c in ctx for c in _AYAH_CUES)
    has_hadith = any(c in ctx for c in _HADITH_CUES)
    if ENABLE_LABEL_CUES and has_ayah and not has_hadith:
        return "Ayah"
    if ENABLE_LABEL_CUES and has_hadith and not has_ayah:
        return "Hadith"
    # if both or none, keep original label
    return label


def _snap_to_text(original: str, start: int, end: int, span_text: Optional[str]) -> Tuple[int, int]:
    """Align (start, end) to the nearest exact occurrence of span_text in original.

    - If original[start:end] already equals span_text, return as-is after trimming.
    - Else search in a ±64-char window around start for the first exact match; choose the nearest occurrence.
    - If not found in window, search the whole string and choose the nearest occurrence to start.
    - If still not found or span_text invalid, return trimmed (start, end).
    """
    if not span_text or not isinstance(span_text, str):
        return _trim_span(original, start, end)

    # Fast path: exact match
    if 0 <= start <= end <= len(original) and original[start:end] == span_text:
        return _trim_span(original, start, end)

    # Windowed search
    window_start = max(0, start - 64)
    window_end = min(len(original), start + 64)
    window = original[window_start:window_end]
    best_abs = None
    best_idx = None
    idx = window.find(span_text)
    while idx != -1:
        abs_idx = window_start + idx
        delta = abs(abs_idx - start)
        if best_abs is None or delta < best_abs:
            best_abs = delta
            best_idx = abs_idx
        idx = window.find(span_text, idx + 1)
    if best_idx is None:
        # Global fallback
        idx = original.find(span_text)
        while idx != -1:
            delta = abs(idx - start)
            if best_abs is None or delta < best_abs:
                best_abs = delta
                best_idx = idx
            idx = original.find(span_text, idx + 1)
    if best_idx is not None:
        snapped = (best_idx, best_idx + len(span_text))
        return _trim_span(original, *snapped)
    return _trim_span(original, start, end)


def _prune_conflicts(spans: List[Tuple[int, int, str, str]], original: str) -> List[Tuple[int, int, str, str]]:
    """Drop overlapping spans of different labels when they largely overlap.

    Keep the one with higher cue score; on tie keep the longer; on tie keep earliest.
    """
    if not spans or not ENABLE_PRUNE_CONFLICTS:
        return spans
    spans_sorted = sorted(spans, key=lambda s: (s[0], s[1]))
    kept: List[Tuple[int, int, str, str]] = []
    for s in spans_sorted:
        s_start, s_end, s_label, s_text = s
        drop = False
        for i, r in enumerate(list(kept)):
            r_start, r_end, r_label, r_text = r
            # IoU
            inter = max(0, min(s_end, r_end) - max(s_start, r_start))
            if inter <= 0:
                continue
            len_s = max(1, s_end - s_start)
            len_r = max(1, r_end - r_start)
            iou = inter / (len_s + len_r - inter)
            if iou >= 0.8 and s_label != r_label:
                s_score = _cue_score(original, s_start, s_label)
                r_score = _cue_score(original, r_start, r_label)
                if s_score > r_score or (s_score == r_score and (len_s > len_r or (len_s == len_r and s_start < r_start))):
                    kept[i] = s
                    drop = True  # drop s from further comparisons by replacing r
                else:
                    drop = True
                break
        if not drop:
            kept.append(s)
    return kept


def _extract_message_text(full: dict) -> str:
    try:
        outputs = (full or {}).get("output", [])
        for item in outputs:
            if item.get("type") == "message":
                for part in item.get("content", []):
                    if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                        return part["text"]
    except Exception:
        pass
    return "{}"


def build_rows_for_task_a(question: Question, json_path: str) -> List[Dict[str, str]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    text = _extract_message_text(data)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = {}

    citations = payload.get("citations", []) if isinstance(payload, dict) else []

    # Load the original response text for trimming/validation
    response_text_path = os.path.join(OUTPUT_DIR, f"{question.id}.txt")
    with open(response_text_path, "r", encoding="utf-8") as f:
        original = f.read()

    spans: List[Tuple[int, int, str, str]] = []  # (start, end, label, text)
    for cit in citations:
        try:
            st = int(cit.get("span_start"))
            en = int(cit.get("span_end"))
            ctype = cit.get("citation_type")
            label = "Ayah" if ctype == "Aya" else "Hadith"
        except Exception:
            continue
        # First, attempt to align using the provided span_text if any to avoid off-by-one
        span_text_value = cit.get("span_text") if isinstance(cit, dict) else None
        st_aligned, en_aligned = _snap_to_text(original, st, en, span_text_value)
        st2, en2 = _trim_span(original, st_aligned, en_aligned)
        if 0 <= st2 < en2 <= len(original):
            # Adjust label using local cues
            label2 = _maybe_correct_label(original, st2, en2, label)
            spans.append((st2, en2, label2, original[st2:en2]))

    spans = _dedupe_spans(spans, original)
    spans = _prune_conflicts(spans, original)

    rows: List[Dict[str, str]] = []
    if not spans:
        # NoAnnotation row
        rows.append(
            {
                "Question_ID": question.id,
                "Annotation_ID": 0,
                "Label": "NoAnnotation",
                "Span_Start": 0,
                "Span_End": 0,
                "Original_Span": "",
            }
        )
        return rows

    for idx, (st, en, label, _text) in enumerate(spans, start=1):
        rows.append(
            {
                "Question_ID": question.id,
                "Annotation_ID": idx,
                "Label": label,
                "Span_Start": st,
                "Span_End": en,
                "Original_Span": original[st:en],
            }
        )
    return rows


# -----------------------------
# Main orchestration
# -----------------------------

async def main(
    limit: Optional[int] = None,
    input_path: Optional[str] = None,
    outdir: Optional[str] = None,
    models: Optional[List[str]] = None,
    effort: Optional[str] = None,
):
    # Ensure vector store is ready
    store_id = ensure_vector_store()
    globals()["VECTOR_STORE_ID"] = store_id

    # Resolve output dir if provided
    global OUTPUT_DIR
    if outdir:
        OUTPUT_DIR = outdir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Resolve model pool and reasoning effort
    global MODEL_POOL, REASONING_EFFORT
    if models and isinstance(models, list) and len(models) > 0:
        MODEL_POOL = models
    if effort:
        REASONING_EFFORT = effort

    # Load Task A dataset (allow custom input path)
    dataset_path = input_path or os.path.join("datasets", "TaskA_Input.xml")
    questions = parse_xml_task_a(dataset_path)
    if limit is not None:
        questions = questions[:limit]

    # Resume: detect existing JSONs
    json_paths_by_id: Dict[str, str] = {}
    pending: List[Question] = []
    for q in questions:
        out_path = os.path.join(OUTPUT_DIR, f"{q.id}.json")
        if os.path.exists(out_path):
            json_paths_by_id[q.id] = out_path
        else:
            pending.append(q)

    # Prepare containers to capture per-run details for metadata
    question_model_assignment: Dict[str, str] = {}
    question_duration_sec: Dict[str, float] = {}
    failed_questions: List[str] = []

    if pending:
        client = AsyncOpenAI()
        max_concurrency = 1000
        semaphore = asyncio.Semaphore(max_concurrency)

        async def process(q: Question, model_name: str):
            start = time.perf_counter()
            path = await call_model_with_retry(
                client, q, semaphore, model_name=model_name, reasoning_effort=REASONING_EFFORT
            )
            duration = time.perf_counter() - start
            return q.id, path, duration, model_name

        pool = MODEL_POOL or ["gpt-5"]
        tasks = [process(q, pool[i % len(pool)]) for i, q in enumerate(pending)]

        completed = 0
        total_duration = 0.0
        assumed_avg = 120.0
        pbar = tqdm(total=len(tasks), desc="Task 1A: processing", unit="q")
        for coro in asyncio.as_completed(tasks):
            qid, path, duration, model_used = await coro
            if path:
                json_paths_by_id[qid] = path
                question_model_assignment[qid] = model_used
                question_duration_sec[qid] = float(duration)
            completed += 1
            total_duration += duration

            avg_time = (total_duration / completed) if completed >= 3 else assumed_avg
            remaining = len(tasks) - completed
            eta_batches = math.ceil(remaining / max_concurrency) if max_concurrency else 0
            eta_seconds = eta_batches * avg_time

            def fmt(sec: float) -> str:
                m, s = divmod(int(sec), 60)
                h, m = divmod(m, 60)
                return f"{h:d}:{m:02d}:{s:02d}"

            pbar.update(1)
            pbar.set_postfix(avg=f"{avg_time/60:.1f}m", eta=fmt(eta_seconds), last=qid)
        pbar.close()
    else:
        print("✔ All Task 1A questions already processed – nothing to do.")

    # Build TSV
    all_rows: List[Dict[str, str]] = []
    for q in questions:
        jpath = json_paths_by_id.get(q.id)
        if jpath:
            all_rows.extend(build_rows_for_task_a(q, jpath))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tsv_path = os.path.join(OUTPUT_DIR, "TaskA_submission.tsv")
    fieldnames = [
        "Question_ID",
        "Annotation_ID",
        "Label",
        "Span_Start",
        "Span_End",
        "Original_Span",
    ]

    import csv

    with open(tsv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Wrote Task 1A submission to {tsv_path}")

    # Build required 4-column submission for Subtask 1A with inclusive end indices, no header
    from collections import defaultdict
    import re as _re

    def _fmt_qid(qid: str) -> str:
        # Keep as-is if already formatted; normalize A-Q# to zero-padded 3 digits
        m = _re.fullmatch(r"A-Q(\d+)", qid)
        if m:
            return f"A-Q{int(m.group(1)):03d}"
        return qid

    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in all_rows:
        grouped[r["Question_ID"]].append(r)

    sub1a_tsv = os.path.join(OUTPUT_DIR, "Subtask1A_submission.tsv")
    with open(sub1a_tsv, "w", encoding="utf-8") as f:
        for q in questions:
            qid = q.id
            rows = grouped.get(qid, [])
            real = [r for r in rows if r.get("Label") in ("Ayah", "Hadith")]
            if not real:
                f.write(f"{_fmt_qid(qid)}\t0\t0\tNo_Spans\n")
                continue
            for r in real:
                s = int(r["Span_Start"]) if r.get("Span_Start") is not None else 0
                e_excl = int(r["Span_End"]) if r.get("Span_End") is not None else 0
                e_incl = (e_excl - 1) if e_excl > 0 else 0
                f.write(f"{_fmt_qid(qid)}\t{s}\t{e_incl}\t{r['Label']}\n")

    # Package zip with only the TSV (metadata saved alongside, not inside zip)
    from datetime import datetime
    import zipfile
    # Build comprehensive run metadata for provenance/reproducibility
    # Try to fetch code version information (non-fatal if unavailable)
    git_commit = None
    git_branch = None
    try:
        if subprocess is not None:
            git_commit = (
                subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
                .decode("utf-8")
                .strip()
            )
            git_branch = (
                subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL)
                .decode("utf-8")
                .strip()
            )
    except Exception:
        pass

    # Identify which questions were processed this run vs. pre-existing
    processed_this_run = sorted(list(question_model_assignment.keys()))
    skipped_existing = sorted([q.id for q in questions if q.id not in question_model_assignment])

    meta = {
        "task": "Subtask 1A",
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "output_dir": os.path.abspath(OUTPUT_DIR),
        "output_file": os.path.basename(sub1a_tsv),
        "num_questions_total": len(questions),
        "input_path": dataset_path,
        "model_pool": MODEL_POOL,
        "vector_store": {
            "enabled": VECTOR_STORE_ID is not None and VECTOR_STORE_ID != "disabled-for-task-1a",
            "vector_store_id": VECTOR_STORE_ID,
        },
        "developer_message": DEVELOPER_MESSAGE,
        "json_schema": SCHEMA,
        "api_call": {
            "endpoint": "responses.create",
            "request_template": {
                "model": "<assigned per question>",
                "input": [
                    {"role": "developer", "content": [{"type": "input_text", "text": "<DEVELOPER_MESSAGE>"}]},
                    {"role": "user", "content": [{"type": "input_text", "text": "<raw response text>"}]}
                ],
                "text": {
                    "format": {"type": "json_schema", "name": "task_1a_spans", "strict": True, "schema": "<SCHEMA>"},
                    "verbosity": "high"
                },
                "reasoning": {"effort": "high", "summary": None},
                "tools": [{"type": "code_interpreter", "container": {"type": "auto", "file_ids": ["<uploaded file id>"]}}],
                "store": False
            }
        },
        "postprocessing": {
            "trim_chars": _TRIM_CHARS,
            "quote_chars": _QUOTE_CHARS,
            "label_cues_enabled": ENABLE_LABEL_CUES,
            "dedupe_identical_enabled": ENABLE_DEDUP_IDENTICAL,
            "prune_conflicts_enabled": ENABLE_PRUNE_CONFLICTS,
            "snap_window_chars": 64,
            "dedupe_proximity_chars": 120
        },
        "batching": {
            "max_concurrency": 1000,
            "retries": 5,
            "backoff_formula": "wait = (2**attempt)*5 + U[0,2] seconds"
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "openai_package_version": getattr(OpenAI, "__version__", None),
            "openai_api_key_present": bool(os.getenv("OPENAI_API_KEY")),
            "git_commit": git_commit,
            "git_branch": git_branch,
        },
        "questions": {
            "processed_this_run": processed_this_run,
            "skipped_already_present": skipped_existing,
            "model_assignments_this_run": question_model_assignment,
            "durations_sec_this_run": question_duration_sec,
        },
    }
    meta_path = os.path.join(OUTPUT_DIR, "run_meta.json")
    with open(meta_path, "w", encoding="utf-8") as mf:
        json.dump(meta, mf, ensure_ascii=False, indent=2)

    # Build a descriptive ZIP filename for future runs (task, model, effort, timestamp)
    def _sanitize_for_filename(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()

    used_models = sorted(set(question_model_assignment.values())) or sorted(set(MODEL_POOL))
    models_part = _sanitize_for_filename("+".join(used_models)) if used_models else "gpt_5"
    effort_part = _sanitize_for_filename(REASONING_EFFORT or "high")
    timestamp_part = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    zip_filename = f"burhanai_1a_prompt_{models_part}_{effort_part}_{timestamp_part}.zip"
    zip_path = os.path.join(OUTPUT_DIR, zip_filename)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(sub1a_tsv, arcname=os.path.basename(sub1a_tsv))
    print(f"Wrote Subtask 1A TSV to {sub1a_tsv} and packaged {zip_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 1A: span identification pipeline")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions (for quick runs)")
    parser.add_argument("--input", type=str, default=None, help="Path to Input XML (e.g., datasets/Test_Subtask_1A.xml)")
    parser.add_argument("--outdir", type=str, default=None, help="Output directory (default: outputs_task_a)")
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of models to use (e.g., gpt-5,gpt-5-mini). If omitted, defaults to gpt-5.",
    )
    parser.add_argument(
        "--effort",
        type=str,
        choices=["low", "medium", "high"],
        default=None,
        help="Reasoning effort to use for the run (low, medium, high). If omitted, defaults to high.",
    )
    args = parser.parse_args()
    model_list: Optional[List[str]] = None
    if args.models:
        model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    asyncio.run(
        main(
            limit=args.limit,
            input_path=args.input,
            outdir=args.outdir,
            models=model_list,
            effort=args.effort,
        )
    )


