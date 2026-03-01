import asyncio
import json
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Dict, Optional
from io import BytesIO

from openai import AsyncOpenAI, OpenAI, RateLimitError, APITimeoutError, APIConnectionError
import argparse
from tqdm import tqdm
import time
import math
import random

# Constants
MODEL_NAME = "gpt-5"
# Round-robin pool to spread requests across models to reduce rate limiting
MODEL_POOL = [
    "gpt-5",
    "gpt-5-25",
    "gpt-5",
    "gpt-5",
    "gpt-5"
]
VECTOR_STORE_ID = os.getenv("VECTOR_STORE_ID")  # May be overridden by id file
VECTOR_ID_FILE = ".vector_store_id"
OUTPUT_DIR = "outputs"


def ensure_vector_store() -> str:
    """Return a usable vector store ID, creating one with dataset files if necessary."""
    # Use synchronous client for simplicity
    client = OpenAI()
    global VECTOR_STORE_ID

    # 1) Check id file if env var missing
    if VECTOR_STORE_ID is None and os.path.exists(VECTOR_ID_FILE):
        with open(VECTOR_ID_FILE, "r", encoding="utf-8") as f:
            VECTOR_STORE_ID = f.read().strip() or None

    # 2) Try to retrieve existing store if we have an id
    if VECTOR_STORE_ID:
        try:
            client.vector_stores.retrieve(VECTOR_STORE_ID)
            return VECTOR_STORE_ID
        except Exception:
            print("Stored vector_store_id invalid – will create new one…")
            VECTOR_STORE_ID = None  # reset

    print("Vector store not found – creating a new one and uploading files…")
    vs = client.vector_stores.create(name="IslamicEvalStore")
    file_ids = []
    for path in [
        os.path.join("datasets", "six_hadith_books.json"),
        os.path.join("datasets", "quranic_verses.json"),
    ]:
        if os.path.exists(path):
            f = client.files.create(file=open(path, "rb"), purpose="assistants")
            file_ids.append(f.id)
        else:
            print(f"Warning: {path} not found – skipping upload")

    if file_ids:
        client.vector_stores.file_batches.create(
            vector_store_id=vs.id,
            file_ids=file_ids,
        )
    VECTOR_STORE_ID = vs.id
    # persist to file for future runs
    with open(VECTOR_ID_FILE, "w", encoding="utf-8") as f:
        f.write(vs.id)
    return vs.id

DEVELOPER_MESSAGE = (
    "Task: Extract every Quranic ayah or Prophetic hadith quoted (or paraphrased) in the *user response text* and fill the JSON schema.  IMPORTANT: quality will be judged with exact-string matching, so follow these rules precisely.\n\n"
    "1 Spans – Use code-interpreter to compute **exact** character offsets [start, end) inside the response.  Exclude surrounding quote marks, narrator chains, sura labels, verse numbers, etc.\n"
    "2 Validation – Set is_valid = true only when the *span text* **exactly** equals the canonical source including diacritics.  A single diacritic error means is_valid = false.\n"
    "3 Corrections – For each invalid citation:\n"
    "   • If you can identify the intended canonical text with high confidence, copy **the full canonical Arabic text** (identical spelling & diacritics) into corrected_text.\n"
    "   • Otherwise return the single word 'خطأ' and set completely_invalid = true.  Never invent a partial or approximate correction.\n"
    "4 Source lookup – Use ONLY the provided files via file_search:  'quranic_verses.json' (Othmani script) and 'six_hadith_books.json'.  Do **not** quote external websites.\n"
    "5 References – When citation_type == Aya fill aya_reference (sura_name, sura_number, aya_number).  When citation_type == Hadeeth fill hadeeth_reference (collection_name in English, book_number, hadith_number).  Leave the unused reference object empty.\n"
    "6 Self-check – Before final answer use code-interpreter to compare every corrected_text with the canonical text you fetched.  If mismatch keep searching or fall back to 'خطأ'.\n"
    "7 Output – Return **only** the top-level JSON object that matches the schema.  No markdown, no commentary.\n\n"
    "The full response text is provided to the code interpreter as the file 'text_to_analyse.txt'.  You have file_search for Quran & Hadith and web_search for secondary confirmation if needed."
)

# Schema copied from sample_code/test_code.py
SCHEMA = {
    "type": "object",
    "properties": {
        "citations": {
            "type": "array",
            "description": "List of citations with details, validation, references, and per-citation corrections.",
            "items": {
                "type": "object",
                "properties": {
                    "span_start": {
                        "type": "integer",
                        "description": "Inclusive start character index of the citation within the text (exclude the qutation marks).",
                    },
                    "span_end": {
                        "type": "integer",
                        "description": "Exclusive end character index of the citation within the text (exclude the qutation marks).",
                    },
                    "span_text": {
                        "type": "string",
                        "description": "The text content of the citation span.",
                    },
                    "citation_type": {
                        "type": "string",
                        "description": "Type of the citation: either Aya (Quran) or Hadeeth (Prophetic tradition).",
                        "enum": ["Aya", "Hadeeth"],
                    },
                    "is_valid": {
                        "type": "boolean",
                        "description": "True if the cited Aya or Hadeeth is recognized and valid, false otherwise.",
                    },
                    "aya_reference": {
                        "type": "object",
                        "description": "Quranic reference details (required if citation_type is Aya; empty object if not applicable).",
                        "properties": {
                            "sura_name": {"type": "string", "description": "Name of the sura (chapter) in the Quran."},
                            "sura_number": {"type": "integer", "description": "Number of the sura (chapter) in the Quran."},
                            "aya_number": {"type": "integer", "description": "Number of the aya (verse) in the Quran."},
                        },
                        "required": ["sura_name", "sura_number", "aya_number"],
                        "additionalProperties": False,
                    },
                    "hadeeth_reference": {
                        "type": "object",
                        "description": "Hadeeth reference details (required if citation_type is Hadeeth; empty object if not applicable).",
                        "properties": {
                            "collection_name": {"type": "string", "description": "Name of the Hadeeth collection in english (e.g. Sahih Bukhari, Muslim)."},
                            "book_number": {"type": "integer", "description": "Book number within the collection."},
                            "hadith_number": {"type": "integer", "description": "Hadith number within the specified book."},
                        },
                        "required": ["collection_name", "book_number", "hadith_number"],
                        "additionalProperties": False,
                    },
                    "corrected_text": {
                        "type": "string",
                        "description": "The corrected version of the citation text (or the text itself if it was correct in the first place) (e.g. if there are certain words that were invalid this version must have this corrected), or 'خطأ' for completely invalid citations (i.e. no close enough citation can be found to correct the text). If there are close texts in Quran or Hadith - but too far from the citation, this must be considered 'خطأ' and completely_invalid must be true.",
                    },
                    "completely_invalid": {
                        "type": "boolean",
                        "description": "Set to true if the citation is completely invalid i.e. there can't be a close enough text to correct it, (corrected_text must be 'خطأ').",
                    },
                },
                "required": [
                    "span_start",
                    "span_end",
                    "span_text",
                    "citation_type",
                    "is_valid",
                    "aya_reference",
                    "hadeeth_reference",
                    "corrected_text",
                    "completely_invalid",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["citations"],
    "additionalProperties": False,
}

@dataclass
class Question:
    id: str
    response: str


def parse_xml(path: str) -> List[Question]:
    """Parse dataset XML with repeated <Question> segments and no outer root."""
    import re

    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    blocks = re.findall(r"<Question>.*?</Question>", raw, flags=re.DOTALL)
    questions: List[Question] = []
    for block in blocks:
        try:
            q = ET.fromstring(block)
        except ET.ParseError:
            # Skip malformed blocks
            continue
        qid = q.findtext("ID")
        resp = q.findtext("Response") or ""
        questions.append(Question(id=qid, response=resp))
    return questions


async def _call_openai_once(client: AsyncOpenAI, q: Question, semaphore: asyncio.Semaphore, *, model_name: str) -> str:
    """Single attempt; may raise OpenAI errors."""
    async with semaphore:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        text_path = os.path.join(OUTPUT_DIR, f"{q.id}.txt")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(q.response)
        file_id: Optional[str] = None
        try:
            # Ensure the local file handle is closed promptly
            # Create an in-memory file-like object with an overridable name for upload
            with open(text_path, "rb") as fb:
                buf = BytesIO(fb.read())
                buf.name = "text_to_analyse.txt"
                uploaded = await client.files.create(file=buf, purpose="assistants")
            file_id = uploaded.id

            response = await client.responses.create(
                model=model_name,
                input=[
                    {
                        "role": "developer",
                        "content": [{"type": "input_text", "text": DEVELOPER_MESSAGE}],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": q.response}]
                    },
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "citations_validation",
                        "strict": True,
                        "schema": SCHEMA,
                    },
                     "verbosity": "high",
                },
                reasoning={"effort": "medium", "summary": None},
                tools=[
                    {"type": "file_search", "vector_store_ids": [VECTOR_STORE_ID]},
                    #{
                    #    "type": "web_search_preview",
                    #    "user_location": {"type": "approximate"},
                    #    "search_context_size": "medium",
                    #},
                    {
                        "type": "code_interpreter",
                        "container": {"type": "auto", "file_ids": [file_id]},
                    },
                ],
                store=False,
            )
            json_path = os.path.join(OUTPUT_DIR, f"{q.id}.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(response.model_dump(), f, ensure_ascii=False, indent=2)
            return json_path
        finally:
            # Always attempt to delete the uploaded OpenAI file object
            if file_id:
                try:
                    await client.files.delete(file_id)
                except Exception as e:
                    print(f"Warning: failed to delete OpenAI file {file_id}: {e}")


async def call_openai_with_retry(client: AsyncOpenAI, q: Question, semaphore: asyncio.Semaphore,
                                 retries: int = 5, *, model_name: str) -> Optional[str]:
    """Wrapper that retries on rate-limit, timeout, and connection errors."""
    for attempt in range(retries):
        try:
            return await _call_openai_once(client, q, semaphore, model_name=model_name)
        except (RateLimitError, APITimeoutError, APIConnectionError) as e:
            wait = (2 ** attempt) * 5 + random.uniform(0, 2)
            print(f"⚠️  {e.__class__.__name__} for Q{q.id} – retry {attempt + 1}/{retries} in {wait:.1f}s")
            await asyncio.sleep(wait)
        except Exception as e:
            print(f"❌ Unexpected error on Q{q.id}: {e}")
            return None
    print(f"❌ Failed after {retries} retries for Q{q.id}")
    return None



def build_rows(question_id: str, json_path: str) -> List[Dict[str, str]]:
    def _find_message_text(data: dict) -> str:
        try:
            outputs = (data or {}).get("output", [])
            for item in outputs:
                if item.get("type") == "message":
                    for part in item.get("content", []):
                        if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                            return part["text"]
        except Exception:
            pass
        return "{}"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    text = _find_message_text(data)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = {}
    citations = payload.get("citations", []) if isinstance(payload, dict) else []
    rows: List[Dict[str, str]] = []
    for idx, cit in enumerate(citations, start=1):
        ctype = cit.get("citation_type")
        is_valid = bool(cit.get("is_valid"))
        if ctype == "Aya":
            label = "CorrectAyah" if is_valid else "WrongAyah"
        else:
            label = "CorrectHadith" if is_valid else "WrongHadith"
        rows.append(
            {
                "Question_ID": question_id,
                "Annotation_ID": idx,
                "Label": label,
                "Span_Start": cit.get("span_start", ""),
                "Span_End": cit.get("span_end", ""),
                "Original_Span": cit.get("span_text", ""),
                "Correction": cit.get("corrected_text", ""),
            }
        )
    return rows


async def main(limit: Optional[int] = None):
    """Process TaskABC questions with resume support and detailed progress reporting.

    Args:
        limit: If provided, process only the first `limit` questions – handy for dry-runs.
    """
    # Ensure vector-store exists and is accessible.
    store_id = ensure_vector_store()
    globals()["VECTOR_STORE_ID"] = store_id  # keep global in sync

    # Load dataset
    questions = parse_xml(os.path.join("datasets", "TaskABC_Input.xml"))
    if limit is not None:
        questions = questions[:limit]

    # Detect already-processed questions (resume capability)
    json_paths_by_id: Dict[str, str] = {}
    pending_questions: List[Question] = []
    for q in questions:
        out_path = os.path.join(OUTPUT_DIR, f"{q.id}.json")
        if os.path.exists(out_path):
            json_paths_by_id[q.id] = out_path
        else:
            pending_questions.append(q)

    if pending_questions:
        client = AsyncOpenAI()
        max_concurrency = 10  # reduced to minimise rate limits
        semaphore = asyncio.Semaphore(max_concurrency)

        async def process_question(q: Question, model_name: str):
            start = time.perf_counter()
            path = await call_openai_with_retry(client, q, semaphore, model_name=model_name)
            duration = time.perf_counter() - start
            return q.id, path, duration

        # Assign models in round-robin across the pool
        pool = MODEL_POOL or [MODEL_NAME]
        tasks = [
            process_question(q, pool[i % len(pool)])
            for i, q in enumerate(pending_questions)
        ]
        completed = 0
        total_duration = 0.0
        assumed_avg = 180.0  # seconds – used until we have 3 samples

        pbar = tqdm(total=len(tasks), desc="Processing questions", unit="q")
        for coro in asyncio.as_completed(tasks):
            qid, path, duration = await coro
            if path:
                json_paths_by_id[qid] = path
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
        print("✔ All questions already processed – nothing to do.")

    # Build TSV combining previous and newly generated outputs
    all_rows: List[Dict[str, str]] = []
    for q in questions:
        jpath = json_paths_by_id.get(q.id)
        if jpath:
            all_rows.extend(build_rows(q.id, jpath))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tsv_path = os.path.join(OUTPUT_DIR, "TaskABC_submission.tsv")
    fieldnames = [
        "Question_ID",
        "Annotation_ID",
        "Label",
        "Span_Start",
        "Span_End",
        "Original_Span",
        "Correction",
    ]
    with open(tsv_path, "w", encoding="utf-8", newline="") as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Wrote submission file to {tsv_path}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process TaskABC dataset")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions to process (useful for testing)")
    args = parser.parse_args()
    asyncio.run(main(limit=args.limit))
