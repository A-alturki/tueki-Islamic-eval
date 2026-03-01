import argparse
import asyncio
import csv
import json
import math
import os
import platform
import random
import re
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from io import BytesIO
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import subprocess
except Exception:  # pragma: no cover
    subprocess = None

from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, OpenAI, RateLimitError


# -----------------------------
# Configuration
# -----------------------------

MODEL_NAME = "gpt-5"  # default; can be overridden via --model
REASONING_EFFORT_DEFAULT = "high"  # low, medium, high

OUTPUT_DIR = "outputs_task_c"

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ORG_SCORING_DIR = os.path.join(THIS_DIR, "scoring_from_organizers")
QURAN_DB_PATH = os.path.join(ORG_SCORING_DIR, "quranic_verses.json")
HADITH_DB_PATH = os.path.join(ORG_SCORING_DIR, "six_hadith_books.json")

# Vector store disabled by request – code interpreter search only
VECTOR_STORE_ID: Optional[str] = None
VECTOR_ID_FILE = None


# -----------------------------
# Data structures
# -----------------------------


@dataclass
class Case:
    sequence_id: str
    question_id: str
    span_type: str  # "Ayah" or "Hadith"
    span_start: Optional[int]
    span_end: Optional[int]
    original_span: str


# -----------------------------
# Dataset utilities (Task C)
# -----------------------------


def parse_taskc_input_xml(path: str) -> Dict[str, str]:
    """Return mapping Question_ID -> Response text for Task C XML.

    The XML contains repeated <Question> elements with <ID> and <Response>.
    The file is a valid XML document with a root, or contains top-level repeated blocks.
    """
    # Try normal parse first
    try:
        tree = ET.parse(path)
        root = tree.getroot()
        mapping: Dict[str, str] = {}
        for q in root.findall(".//Question"):
            qid = (q.findtext("ID") or "").strip()
            resp = q.findtext("Response")
            if qid:
                mapping[qid] = resp if resp is not None else ""
        if mapping:
            return mapping
    except ET.ParseError:
        pass

    # Fallback: similar to Task A regex extraction of Question blocks
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    blocks = re.findall(r"<Question>.*?</Question>", raw, flags=re.DOTALL)
    mapping: Dict[str, str] = {}
    for block in blocks:
        try:
            q = ET.fromstring(block)
        except ET.ParseError:
            continue
        qid = (q.findtext("ID") or "").strip()
        resp = q.findtext("Response")
        if qid:
            mapping[qid] = resp if resp is not None else ""
    return mapping


def parse_taskc_gt_tsv(path: str) -> List[Tuple[str, str, int, int, str, str]]:
    """Parse development GT TSV for Task C.

    Columns: Question_ID, Label, Span_Start, Span_End, Original_Span, Correction
    Returns list of tuples (question_id, span_type, start, end, original_span, correction)
    where span_type is "Ayah" or "Hadith".
    """
    rows: List[Tuple[str, str, int, int, str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            qid = (r.get("Question_ID") or "").strip()
            label = (r.get("Label") or "").strip()
            st = int(r.get("Span_Start") or 0)
            en = int(r.get("Span_End") or 0)
            orig = r.get("Original_Span") or ""
            corr = r.get("Correction") or ""
            span_type = "Ayah" if "Ayah" in label else "Hadith"
            rows.append((qid, span_type, st, en, orig, corr))
    return rows


def parse_taskc_test_user_tsv(path: str) -> List[Case]:
    """Parse Test_Subtask_1C_USER.tsv provided by organizers for test submissions.

    Columns: Sequence_ID, Question_ID, Span_Type, Span_Start, Span_End, Original_Span
    """
    cases: List[Case] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            cases.append(
                Case(
                    sequence_id=str(r.get("Sequence_ID") or "").strip(),
                    question_id=str(r.get("Question_ID") or "").strip(),
                    span_type=str(r.get("Span_Type") or "").strip(),
                    span_start=int(r.get("Span_Start") or 0),
                    span_end=int(r.get("Span_End") or 0),
                    original_span=r.get("Original_Span") or "",
                )
            )
    return cases


def parse_test_context(path_xml: Optional[str], path_jsonl: Optional[str]) -> Dict[str, str]:
    """Read test context (responses) from .xml or .jsonl; returns Question_ID -> Response.

    jsonl lines should have at least keys: id, Response or similar. We attempt common fields.
    """
    mapping: Dict[str, str] = {}
    if path_xml and os.path.exists(path_xml):
        mapping.update(parse_taskc_input_xml(path_xml))
    if path_jsonl and os.path.exists(path_jsonl):
        with open(path_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    qid = (obj.get("ID") or obj.get("id") or "").strip()
                    resp = obj.get("Response") or obj.get("response") or ""
                    if qid:
                        mapping[qid] = resp
                except Exception:
                    continue
    return mapping


# -----------------------------
# Prompt & schema (Task 1C)
# -----------------------------


DEVELOPER_MESSAGE = """
Subtask 1C: Correction of Erroneous Content.

Goal: Given an Arabic span labeled as Ayah or Hadith that appears in an LLM response, return the CORRECT intended canonical text from the provided databases or return "خطأ" only if there is no similar authentic Ayah/Hadith that can be confidently identified.

Important rules:
1) Sources: Use ONLY the attached files via code interpreter – explicitly open and search these files:
   - quranic_verses.json (field: ayah_text)
   - six_hadith_books.json (fields: hadithTxt or Matn)
   Do not use external sources.
2) Exactness (critical): The output MUST be copied verbatim from the source DB with EXACT diacritics (tashkeel) and orthography. Any mismatch is unacceptable. Do NOT add quotes, surah names, verse numbers, isnads, or references. Return only the pure canonical text.
3) Matching strategy:
   - First, make 2–3 initial hypotheses (sura/ayah or hadith MATN ideas) from the span/context.
   - Normalize for search: strip diacritics, tatweel, punctuation; unify hamza; lowercase; collapse spaces. Search both DBs using multiple strategies: substring, token-overlap, n-grams, longest common substring, approximate ratio.
   - Always test BOTH inputs: (a) FULL substring extracted from the response using (start,end), and (b) the user-provided snippet.
   - If the span is paraphrased/truncated, broaden search with synonyms or partial phrases, then verify candidates by comparing against the FULL substring context.
   - If multiple ayat compose the intended quote, return the combined ayat in the correct canonical form exactly as stored in the DB.
   - For Hadiths, return the MATN (hadithTxt or Matn) without isnad.
4) Hallucination: Use "خطأ" ONLY when no reasonably similar authentic candidate exists in the DBs after thorough search. Prefer selecting the closest correct canonical text when a clear intended target is identifiable.
5) Output: Return a single field "correction" only. No explanations.
6) Acceptance threshold and numbering:
   - Compute a robust similarity between a candidate and the FULL substring (e.g., normalized LCS or token F1). If < 0.85, return "خطأ"; if ≥ 0.85, accept.
   - If you can reliably resolve ayah numbers (e.g., candidate includes single or multiple ayat with known indices), you may append ayah numbers in parentheses immediately after each ayah (e.g., (18) (19)). Do NOT invent numbers; only add if you can determine them from the DB metadata. If unsure, return the pure text.
7) Final copy-out: After selecting the candidate, copy the text EXACTLY as stored in the DB (with tashkeel). Do not retype; programmatically slice/copy the DB string to avoid character drift.
""".strip()


SCHEMA = {
    "type": "object",
    "properties": {
        "correction": {"type": "string"},
    },
    "required": ["correction"],
    "additionalProperties": False,
}


# -----------------------------
# OpenAI helpers
# -----------------------------


async def _upload_file_return_id(client: AsyncOpenAI, abs_path: str) -> Optional[str]:
    if not abs_path or not os.path.exists(abs_path):
        return None
    with open(abs_path, "rb") as fb:
        buf = BytesIO(fb.read())
        buf.name = os.path.basename(abs_path)
        uploaded = await client.files.create(file=buf, purpose="assistants")
        return uploaded.id


async def _call_model_once(
    client: AsyncOpenAI,
    case: Case,
    semaphore: asyncio.Semaphore,
    *,
    reasoning_effort: str,
    response_context: Optional[str],
    db_file_ids: Optional[List[str]],
) -> Optional[str]:
    async with semaphore:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        # Persist a compact case description for provenance
        text_path = os.path.join(OUTPUT_DIR, f"{case.sequence_id}_{case.question_id}.txt")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "sequence_id": case.sequence_id,
                        "question_id": case.question_id,
                        "span_type": case.span_type,
                        "span_start": case.span_start,
                        "span_end": case.span_end,
                        # Store the FULL original span from indices when available
                        "original_span": None,  # filled below after computing
                        "provided_span_snippet": case.original_span,
                        "response_context": response_context or "",
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )

        # Reuse pre-uploaded DB files; do not re-upload per case
        file_ids: List[str] = list(db_file_ids or [])
        try:
            # Prepare model input with provided snippet and computed FULL substring
            computed_span = None
            if response_context and isinstance(case.span_start, int) and isinstance(case.span_end, int):
                try:
                    computed_span = response_context[max(0, case.span_start) : max(0, case.span_end)]
                except Exception:
                    computed_span = None

            # Overwrite original_span in the saved JSON with the FULL computed substring when available
            try:
                with open(text_path, "r", encoding="utf-8") as rf:
                    meta_obj = json.load(rf)
                meta_obj["original_span"] = computed_span if computed_span is not None else case.original_span
                with open(text_path, "w", encoding="utf-8") as wf:
                    json.dump(meta_obj, wf, ensure_ascii=False, indent=2)
            except Exception:
                pass

            response = await client.responses.create(
                model=MODEL_NAME,
                input=[
                    {"role": "developer", "content": [{"type": "input_text", "text": DEVELOPER_MESSAGE}]},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": (
                                    f"Task: Correct a {case.span_type}.\n"
                                    f"Original_Span (FULL from indices): {computed_span or ''}\n"
                                    f"Provided_Span_Snippet: {case.original_span}\n"
                                    f"Question_ID: {case.question_id}\n"
                                    f"Span_Start: {case.span_start}, Span_End: {case.span_end}\n"
                                    f"Response_Context (full):\n{response_context or ''}"
                                ),
                            }
                        ],
                    },
                ],
                text={
                    "format": {"type": "json_schema", "name": "task_1c_correction", "strict": True, "schema": SCHEMA},
                    "verbosity": "high",
                },
                reasoning={"effort": reasoning_effort, "summary": None},
                tools=[
                    {"type": "code_interpreter", "container": {"type": "auto", "file_ids": file_ids}}
                ],
                store=False,
            )

            json_path = os.path.join(OUTPUT_DIR, f"{case.sequence_id}.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(response.model_dump(), f, ensure_ascii=False, indent=2)
            return json_path
        finally:
            # We keep DB file_ids for reuse; do not delete them here
            pass


async def call_model_with_retry(
    client: AsyncOpenAI,
    case: Case,
    semaphore: asyncio.Semaphore,
    *,
    reasoning_effort: str,
    response_context: Optional[str],
    db_file_ids: Optional[List[str]],
    retries: int = 5,
) -> Optional[str]:
    for attempt in range(retries):
        try:
            return await _call_model_once(
                client,
                case,
                semaphore,
                reasoning_effort=reasoning_effort,
                response_context=response_context,
                db_file_ids=db_file_ids,
            )
        except (RateLimitError, APITimeoutError, APIConnectionError) as e:
            wait = (2 ** attempt) * 5 + random.uniform(0, 2)
            print(f"⚠️  {e.__class__.__name__} for seq {case.sequence_id} – retry {attempt + 1}/{retries} in {wait:.1f}s")
            await asyncio.sleep(wait)
        except Exception as e:
            print(f"❌ Unexpected error on seq {case.sequence_id}: {e}")
            return None
    print(f"❌ Failed after {retries} retries for seq {case.sequence_id}")
    return None


# -----------------------------
# Response parsing helpers
# -----------------------------


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


def extract_correction(json_path: str) -> str:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    text = _extract_message_text(data)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = {}
    corr = payload.get("correction") if isinstance(payload, dict) else None
    if not isinstance(corr, str):
        return "خطأ"
    corr = corr.strip()
    return corr if corr else "خطأ"


# -----------------------------
# Local scoring for dev (mirror organizer logic)
# -----------------------------


def _remove_default_diac(s: str) -> str:
    out = s
    out = out.replace("َا", "ا")
    out = out.replace("ِي", "ي")
    out = out.replace("ُو", "و")
    out = out.replace("الْ", "ال")
    out = out.replace("ْ", "")
    out = out.replace("َّ", "َّ")
    out = out.replace("ِّ", "ِّ")
    out = out.replace("ُّ", "ُّ")
    out = out.replace("ًّ", "ًّ")
    out = out.replace("ٍّ", "ٍّ")
    out = out.replace("ٌّ", "ٌّ")
    out = out.replace("اَ", "ا")
    out = out.replace("اِ", "ا")
    out = out.replace("لِا", "لا")
    out = out.replace("اً", "ًا")
    return out


def _load_dbs() -> Tuple[List[str], List[str]]:
    with open(HADITH_DB_PATH, "r", encoding="utf-8") as f:
        hadith_books = json.load(f)
    with open(QURAN_DB_PATH, "r", encoding="utf-8") as f:
        quranic_verses = json.load(f)
    hadith_db = [
        _remove_default_diac(item.get("hadithTxt", ""))
        for item in hadith_books
        if isinstance(item, dict) and item.get("hadithTxt")
    ] + [
        _remove_default_diac(item.get("Matn", ""))
        for item in hadith_books
        if isinstance(item, dict) and item.get("Matn")
    ]
    quran_db = [
        _remove_default_diac(item.get("ayah_text", ""))
        for item in quranic_verses
        if isinstance(item, dict) and item.get("ayah_text")
    ]
    return hadith_db, quran_db


def score_dev_predictions(gt_rows: List[Tuple[str, str, int, int, str, str]], predictions: List[Tuple[int, str]]) -> Dict[str, float]:
    """Compute accuracy mirroring the organizer script logic.

    gt_rows: list of tuples (..., correction)
    predictions: list of (Sequence_ID (1-based), Correction)
    """
    # Build maps by Sequence_ID (assumed 1..N in the same order as gt_rows)
    pred_map: Dict[int, str] = {int(seq): pred for seq, pred in predictions}
    hadith_db, quran_db = _load_dbs()

    num_correct = 0
    total = len(gt_rows)
    for idx, (_qid, _stype, _st, _en, _orig, ref_corr) in enumerate(gt_rows, start=1):
        if idx not in pred_map:
            continue
        pred_corr = pred_map[idx]
        pred_norm = _remove_default_diac(pred_corr)
        ref_norm = _remove_default_diac(ref_corr)
        ok = False
        if pred_norm == ref_norm:
            ok = True
        else:
            # If ref is substring of pred and pred exists in one of DBs
            if ref_norm and ref_norm in pred_norm and (
                pred_norm in hadith_db or pred_norm in quran_db
            ):
                ok = True
        if ok:
            num_correct += 1
    acc = (num_correct / total) if total else 0.0
    return {"accuracy": acc}


# Vector store helpers removed (search disabled)


# -----------------------------
# Main orchestration
# -----------------------------


async def main(
    mode: str,
    *,
    outdir: Optional[str],
    effort: Optional[str],
    dev_input_xml: Optional[str],
    dev_gt_tsv: Optional[str],
    test_user_tsv: Optional[str],
    test_xml: Optional[str],
    test_jsonl: Optional[str],
    max_cases: Optional[int] = None,
):
    # Resolve output dir
    global OUTPUT_DIR
    if outdir:
        OUTPUT_DIR = outdir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Resolve effort
    reasoning_effort = effort or REASONING_EFFORT_DEFAULT
    if reasoning_effort not in ("low", "medium", "high"):
        reasoning_effort = REASONING_EFFORT_DEFAULT

    # Build cases
    cases: List[Case] = []
    response_by_qid: Dict[str, str] = {}
    groundtruth_rows: List[Tuple[str, str, int, int, str, str]] = []

    if mode == "dev":
        dev_input_xml = dev_input_xml or os.path.join("datasets", "TaskC_Input.xml")
        dev_gt_tsv = dev_gt_tsv or os.path.join("datasets", "TaskC_GT.tsv")
        response_by_qid = parse_taskc_input_xml(dev_input_xml)
        gt_rows = parse_taskc_gt_tsv(dev_gt_tsv)
        groundtruth_rows = gt_rows
        # Assign sequential Sequence_IDs matching GT row order (1-based)
        iterable: Iterable[Tuple[str, str, int, int, str, str]] = gt_rows
        if isinstance(max_cases, int) and max_cases > 0:
            iterable = gt_rows[:max_cases]
        for idx, (qid, span_type, st, en, orig, _corr) in enumerate(iterable, start=1):
            cases.append(
                Case(
                    sequence_id=str(idx),
                    question_id=qid,
                    span_type=span_type,
                    span_start=st,
                    span_end=en,
                    original_span=orig,
                )
            )
    elif mode == "test":
        if not test_user_tsv:
            raise ValueError("--test_user_tsv is required in test mode")
        cases = parse_taskc_test_user_tsv(test_user_tsv)
        response_by_qid = parse_test_context(test_xml, test_jsonl)
    else:
        raise ValueError("mode must be one of: dev, test")

    # Resume detection
    json_paths_by_seq: Dict[str, str] = {}
    pending: List[Case] = []
    for c in cases:
        out_path = os.path.join(OUTPUT_DIR, f"{c.sequence_id}.json")
        if os.path.exists(out_path):
            json_paths_by_seq[c.sequence_id] = out_path
        else:
            pending.append(c)

    # Process with GPT-5
    question_model_assignment: Dict[str, str] = {}
    case_duration_sec: Dict[str, float] = {}

    if pending:
        client = AsyncOpenAI()
        max_concurrency = 1000
        semaphore = asyncio.Semaphore(max_concurrency)

        # Upload DB files once for reuse across all calls
        db_file_ids: List[str] = []
        try:
            for p in (QURAN_DB_PATH, HADITH_DB_PATH):
                fid = await _upload_file_return_id(client, p)
                if fid:
                    db_file_ids.append(fid)
        except Exception:
            db_file_ids = []

        async def process_one(c: Case) -> Tuple[str, Optional[str], float, str]:
            ctx = response_by_qid.get(c.question_id, "")
            start = time.perf_counter()
            path = await call_model_with_retry(
                client,
                c,
                semaphore,
                reasoning_effort=reasoning_effort,
                response_context=ctx,
                db_file_ids=db_file_ids,
            )
            duration = time.perf_counter() - start
            return c.sequence_id, path, duration, MODEL_NAME

        tasks = [process_one(c) for c in pending]
        pbar_total = len(tasks)
        completed = 0
        total_duration = 0.0
        assumed_avg = 90.0
        print(f"Task 1C: processing {pbar_total} cases…", flush=True)
        for coro in asyncio.as_completed(tasks):
            seq, path, duration, model_used = await coro
            if path:
                json_paths_by_seq[seq] = path
                question_model_assignment[seq] = model_used
                case_duration_sec[seq] = float(duration)
            completed += 1
            total_duration += duration
            avg_time = (total_duration / completed) if completed >= 3 else assumed_avg
            remaining = pbar_total - completed
            eta_batches = math.ceil(remaining / max_concurrency) if max_concurrency else 0
            eta_seconds = eta_batches * avg_time

            def fmt(sec: float) -> str:
                m, s = divmod(int(sec), 60)
                h, m = divmod(m, 60)
                return f"{h:d}:{m:02d}:{s:02d}"

            print(
                f" progress {completed}/{pbar_total} • avg {avg_time/60:.1f}m • eta {fmt(eta_seconds)} • last {seq}",
                flush=True,
            )
    else:
        print("✔ All Task 1C cases already processed – nothing to do.", flush=True)

    # Build two-column TSV: Sequence_ID, Correction
    predictions: List[Tuple[int, str]] = []
    for c in cases:
        jpath = json_paths_by_seq.get(c.sequence_id)
        if not jpath:
            continue
        corr = extract_correction(jpath)
        predictions.append((int(c.sequence_id), corr))

    predictions.sort(key=lambda x: x[0])
    sub1c_tsv = os.path.join(OUTPUT_DIR, "Subtask1C_submission.tsv")
    with open(sub1c_tsv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        for sid, corr in predictions:
            w.writerow([sid, corr])
    print(f"Wrote Subtask 1C TSV to {sub1c_tsv}", flush=True)

    # Optional: local scoring in dev mode
    local_scores: Optional[Dict[str, float]] = None
    if mode == "dev" and groundtruth_rows:
        local_scores = score_dev_predictions(groundtruth_rows, predictions)
        print(f"Local dev scores: {json.dumps(local_scores, ensure_ascii=False)}", flush=True)

    # Package a zip as required (TSV only)
    from datetime import datetime
    import zipfile

    def _sanitize_for_filename(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()

    effort_part = _sanitize_for_filename(reasoning_effort or "high")
    timestamp_part = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"burhanai_1c_prompt_{MODEL_NAME.replace('-', '_')}_{effort_part}_{timestamp_part}.zip"
    zip_path = os.path.join(OUTPUT_DIR, zip_filename)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(sub1c_tsv, arcname=os.path.basename(sub1c_tsv))
    print(f"Packaged {zip_path}", flush=True)

    # Write run metadata
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

    meta = {
        "task": "Subtask 1C",
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "output_dir": os.path.abspath(OUTPUT_DIR),
        "output_file": os.path.basename(sub1c_tsv),
        "model": MODEL_NAME,
        "effort": reasoning_effort,
        "developer_message": DEVELOPER_MESSAGE,
        "json_schema": SCHEMA,
        "api_call": {
            "endpoint": "responses.create",
            "request_template": {
                "model": MODEL_NAME,
                "tools": [
                    {
                        "type": "code_interpreter",
                        "container": {"type": "auto", "file_ids": ["<quran.json>", "<hadith.json>"]},
                    }
                ],
                "reasoning": {"effort": reasoning_effort, "summary": None},
            },
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "openai_package_version": getattr(OpenAI, "__version__", None),
            "openai_api_key_present": bool(os.getenv("OPENAI_API_KEY")),
            "git_commit": git_commit,
            "git_branch": git_branch,
        },
        "run": {
            "num_cases": len(cases),
            "processed_this_run": sorted(list(question_model_assignment.keys())),
            "durations_sec_this_run": case_duration_sec,
            "zip_path": os.path.abspath(zip_path),
        },
        "local_scores": local_scores,
    }
    meta_path = os.path.join(OUTPUT_DIR, "run_meta.json")
    with open(meta_path, "w", encoding="utf-8") as mf:
        json.dump(meta, mf, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 1C: correction pipeline (GPT-5 only)")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["dev", "test"],
        default="dev",
        help="Run on dev (TaskC_GT.tsv) or build test submission from Test_Subtask_1C_USER.tsv",
    )
    parser.add_argument("--outdir", type=str, default=None, help="Output directory (default: outputs_task_c)")
    parser.add_argument(
        "--effort",
        type=str,
        choices=["low", "medium", "high"],
        default=None,
        help="Reasoning effort (low/medium/high). Default: high",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name override (e.g., gpt-5-nano)",
    )
    # Dev inputs
    parser.add_argument("--dev_input_xml", type=str, default=None, help="Path to datasets/TaskC_Input.xml")
    parser.add_argument("--dev_gt_tsv", type=str, default=None, help="Path to datasets/TaskC_GT.tsv")
    parser.add_argument("--max_cases", type=int, default=None, help="Limit number of dev rows to process")
    # Test inputs
    parser.add_argument("--test_user_tsv", type=str, default=None, help="Path to Test_Subtask_1C_USER.tsv")
    parser.add_argument("--test_xml", type=str, default=None, help="Optional path to Test_Subtask_1C.xml")
    parser.add_argument("--test_jsonl", type=str, default=None, help="Optional path to Test_Subtask_1C.jsonl")
    args = parser.parse_args()

    # Apply model override if provided
    if args.model:
        globals()["MODEL_NAME"] = args.model

    asyncio.run(
        main(
            mode=args.mode,
            outdir=args.outdir,
            effort=args.effort,
            dev_input_xml=args.dev_input_xml,
            dev_gt_tsv=args.dev_gt_tsv,
            test_user_tsv=args.test_user_tsv,
            test_xml=args.test_xml,
            test_jsonl=args.test_jsonl,
            max_cases=args.max_cases,
        )
    )


