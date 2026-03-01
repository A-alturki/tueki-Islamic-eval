import asyncio
import json
import os
import platform
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Any

from openai import AsyncOpenAI, OpenAI


# -----------------------------
# Constants and globals
# -----------------------------

OUTPUT_ROOT = "outputs_task_2"
VECTOR_STORE_ID = os.getenv("VECTOR_STORE_ID")
VECTOR_ID_FILE = os.path.join("task_2_RAG", ".vector_store_id_task2")


# -----------------------------
# Data models
# -----------------------------

@dataclass
class Question:
    id: str
    text: str


@dataclass
class QpcPassage:
    passage_id: str  # e.g., "2:233-233"
    sura: int
    start_ayah: int
    end_ayah: int
    text: str


# -----------------------------
# I/O helpers
# -----------------------------

def parse_ayatec_tsv(path: str) -> List[Question]:
    questions: List[Question] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            qid, qtext = parts[0], parts[1]
            questions.append(Question(id=str(qid), text=qtext))
    return questions


def load_qpc_tsv(path: str) -> Tuple[List[QpcPassage], Dict[Tuple[int, int], str]]:
    passages: List[QpcPassage] = []
    verse_to_passage: Dict[Tuple[int, int], str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            tab = line.find("\t")
            if tab == -1:
                continue
            pid = line[:tab]
            ptext = line[tab + 1 :]
            # pid format: Chapter#:StartVerse#-EndVerse# (e.g., 2:233-233)
            try:
                chapter_part, verses_part = pid.split(":", 1)
                sura = int(chapter_part)
                start_str, end_str = verses_part.split("-", 1)
                start_ayah = int(start_str)
                end_ayah = int(end_str)
            except Exception:
                # skip malformed lines
                continue
            passages.append(
                QpcPassage(
                    passage_id=pid,
                    sura=sura,
                    start_ayah=start_ayah,
                    end_ayah=end_ayah,
                    text=ptext,
                )
            )
            for aya in range(start_ayah, end_ayah + 1):
                verse_to_passage[(sura, aya)] = pid
    return passages, verse_to_passage


def load_hadith_ids(jsonl_path: str) -> Dict[str, int]:
    """Return maps for hadith id resolution.

    Returns a mapping from stringified hadith_id and source_hadith_id to hadith_id.
    """
    id_map: Dict[str, int] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                # Some files might be single-quoted python dicts; try eval as fallback
                try:
                    obj = eval(line)
                except Exception:
                    continue
            hadith_id = int(obj.get("hadith_id"))
            id_map[str(hadith_id)] = hadith_id
            source_id = obj.get("source_hadith_id")
            if source_id is not None:
                id_map[str(source_id)] = hadith_id
    return id_map


# -----------------------------
# Vector store bootstrap (RAG)
# -----------------------------

def ensure_vector_store_for_task2() -> str:
    client = OpenAI()
    global VECTOR_STORE_ID

    # If user provided an explicit vector store id via env, honor it and never create a new one
    env_vs = os.getenv("VECTOR_STORE_ID")
    if env_vs:
        VECTOR_STORE_ID = env_vs
        print(f"Using provided VECTOR_STORE_ID from env: {env_vs}")
        return VECTOR_STORE_ID

    # Load cached id from file if env not set
    if VECTOR_STORE_ID is None and os.path.exists(VECTOR_ID_FILE):
        try:
            with open(VECTOR_ID_FILE, "r", encoding="utf-8") as f:
                VECTOR_STORE_ID = f.read().strip() or None
        except Exception:
            VECTOR_STORE_ID = None

    if VECTOR_STORE_ID:
        try:
            client.vector_stores.retrieve(VECTOR_STORE_ID)
            return VECTOR_STORE_ID
        except Exception:
            print("Stored vector_store_id invalid – will create a new one…")
            VECTOR_STORE_ID = None

    try:
        print("Creating a new vector store and uploading Task 2 corpora…")
        vs = client.vector_stores.create(name="IslamicEvalTask2Store")
    except Exception as e:
        print(f"Warning: unable to create vector store (will disable RAG tools): {e}")
        # Fall back to a disabled sentinel so callers can proceed without tools
        VECTOR_STORE_ID = "disabled-for-task-2"
        return VECTOR_STORE_ID

    file_ids: List[str] = []
    candidate_paths = [
        os.path.join("datasets", "q_full.json"),
        os.path.join("datasets", "task_2_data", "Thematic_QPC", "QH-QA-25_Subtask2_QPC_v1.1.tsv"),
        os.path.join("datasets", "task_2_data", "Sahih-Bukhari", "QH-QA-25_Subtask2_Sahih-Bukhari_v1.0.jsonl"),
    ]
    for cpath in candidate_paths:
        if os.path.exists(cpath):
            try:
                fobj = client.files.create(file=open(cpath, "rb"), purpose="assistants")
                file_ids.append(fobj.id)
                print(f"Uploaded {cpath}")
            except Exception as e:
                print(f"Warning: failed to upload {cpath}: {e}")
        else:
            print(f"Warning: {cpath} not found – skipping upload")

    if file_ids:
        client.vector_stores.file_batches.create(vector_store_id=vs.id, file_ids=file_ids)

    VECTOR_STORE_ID = vs.id
    try:
        os.makedirs(os.path.dirname(VECTOR_ID_FILE), exist_ok=True)
        with open(VECTOR_ID_FILE, "w", encoding="utf-8") as f:
            f.write(vs.id)
    except Exception:
        pass
    return vs.id


# -----------------------------
# Model prompt and schema
# -----------------------------

DEVELOPER_MESSAGE = (
    "Give me up to 20 (possibly fewer, possibly 0) citations from the Holy Qur'an or Sahih Al-Bukhari only that POTENTIALLY enclose the answer(s) to the user's Arabic question.\n"
    "STRICT rules:\n"
    "- Use ONLY the attached files (Qur'an ayat and the Sahih Al-Bukhari JSONL).\n"
    "- Search as many times as you need using file search; do not hallucinate.\n"
    "- Rank citations by how directly and completely they answer the question. Best first.\n"
    "- For Qur'an items, MUST provide integer sura_number and aya_number.\n"
    "- For Hadith items, MUST provide the integer hadith_id EXACTLY as in the field 'hadith_id' in the attached Sahih Al-Bukhari JSONL. Do NOT invent numbers.\n"
    "- If nothing is answer-bearing, return an empty list."
)

SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "citations": {
            "type": "array",
            "description": "Ordered list (<=20) of citations from Quran or Sahih Bukhari.",
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["quran", "hadeeth"],
                    },
                    "sura_number": {
                        "type": "integer",
                        "description": "1..114 for Quran items; -1 otherwise",
                    },
                    "aya_number": {
                        "type": "integer",
                        "description": "Ayah number for Quran items; -1 otherwise",
                    },
                    "hadith_id": {
                        "type": "integer",
                        "description": "Exact hadith_id from the Sahih Al-Bukhari JSONL; -1 for Quran items",
                    },
                },
                "required": ["type", "sura_number", "aya_number", "hadith_id"],
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
        text_path = os.path.join(OUTPUT_DIR, f"{q.id}.txt")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(q.text)

        # Determine tool support based on vector store availability (and scopes)
        enable_tools = VECTOR_STORE_ID not in (None, "disabled-for-task-2")

        # Upload the question text for provenance (optional for code_interpreter)
        file_id: Optional[str] = None
        try:
            if enable_tools:
                with open(text_path, "rb") as fb:
                    buf = BytesIO(fb.read())
                    buf.name = "question.txt"
                    uploaded = await client.files.create(file=buf, purpose="assistants")
                file_id = uploaded.id

            response = await client.responses.create(
                model=model_name,
                input=[
                    {"role": "developer", "content": [{"type": "input_text", "text": DEVELOPER_MESSAGE}]},
                    {"role": "user", "content": [{"type": "input_text", "text": q.text}]},
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "task_2_citations",
                        "strict": True,
                        "schema": SCHEMA,
                    },
                    "verbosity": "medium",
                },
                reasoning={"effort": reasoning_effort, "summary": None},
                tools=(
                    [
                        {"type": "file_search", "vector_store_ids": [VECTOR_STORE_ID]},
                        {"type": "code_interpreter", "container": {"type": "auto", "file_ids": [file_id]}},
                    ]
                    if (enable_tools and file_id)
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
    for attempt in range(retries + 1):
        try:
            return await _call_model_once(
                client, q, semaphore, model_name=model_name, reasoning_effort=reasoning_effort
            )
        except Exception as e:
            if attempt >= retries:
                print(f"[FATAL] qid={q.id} failed after {retries} retries: {e}")
                return None
            sleep_s = (2 ** attempt) * 2 + random.uniform(0, 2)
            print(f"[WARN] qid={q.id} attempt {attempt+1}/{retries} failed: {e}; retrying in {sleep_s:.1f}s")
            await asyncio.sleep(sleep_s)


# -----------------------------
# Post-processing: build run TSV
# -----------------------------

def _extract_citations_from_json(path: str) -> List[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return []

    # Responses API payload layout
    # Expect: obj["output"][0]["content"][0]["text"] parsed as JSON per schema
    # But safer: attempt to find a parsed form in obj["response"]["output"] → varies by sdk
    # Fall back to parsing top-level text if present
    # Here we try common shapes
    # 1) SDK model_dump() includes 'output' root list with message content
    data = None
    try:
        outputs = obj.get("output") or obj.get("outputs")
        if outputs and isinstance(outputs, list):
            for out in outputs:
                content = out.get("content") if isinstance(out, dict) else None
                if content and isinstance(content, list):
                    for c in content:
                        if c.get("type") == "output_text":
                            cand = c.get("text")
                            if isinstance(cand, str):
                                data = json.loads(cand)
                                raise StopIteration
    except StopIteration:
        pass
    except Exception:
        data = None

    if data is None:
        # 2) Direct 'response' with 'output_text'
        try:
            resp = obj.get("response") or {}
            content = resp.get("output") or []
            for c in content:
                if c.get("type") == "output_text":
                    data = json.loads(c.get("text", "{}"))
                    break
        except Exception:
            data = None

    if data is None:
        # 3) Sometimes SDK mirrors at obj["response"]["output"][0]["content"][0]["text"]
        try:
            txt = (
                obj["response"]["output"][0]["content"][0]["text"]
            )
            data = json.loads(txt)
        except Exception:
            data = None

    if isinstance(data, dict):
        items = data.get("citations") or []
        if isinstance(items, list):
            return [x for x in items if isinstance(x, dict)]
    return []


def map_quran_to_passages(
    citations: List[Dict[str, Any]], verse_to_passage: Dict[Tuple[int, int], str]
) -> List[str]:
    passage_ids: List[str] = []
    seen: set = set()
    for c in citations:
        if c.get("type") != "quran":
            continue
        sura = int(c.get("sura_number", -1))
        aya = int(c.get("aya_number", -1))
        if sura <= 0 or aya <= 0:
            continue
        pid = verse_to_passage.get((sura, aya))
        if pid and pid not in seen:
            seen.add(pid)
            passage_ids.append(pid)
    return passage_ids


def map_hadith_ids(
    citations: List[Dict[str, Any]], hadith_id_map: Dict[str, int]
) -> List[str]:
    ids: List[str] = []
    seen: set = set()
    for c in citations:
        if c.get("type") != "hadeeth":
            continue
        hid = c.get("hadith_id")
        if hid is None:
            continue
        # Allow either exact hadith_id or source_hadith_id to resolve
        resolved = hadith_id_map.get(str(hid))
        if resolved is None:
            # also allow ints in map directly
            resolved = hadith_id_map.get(str(int(hid))) if isinstance(hid, (int, float)) else None
        if resolved is None:
            continue
        sid = str(resolved)
        if sid not in seen:
            seen.add(sid)
            ids.append(sid)
    return ids


def build_trec_rows_for_question(
    qid: str,
    citations: List[Dict[str, Any]],
    verse_to_passage: Dict[Tuple[int, int], str],
    hadith_id_map: Dict[str, int],
    *,
    max_items: int = 20,
) -> List[Tuple[str, str, int, float, str]]:
    # Preserve original order for ranking
    qpc_ids = map_quran_to_passages(citations, verse_to_passage)
    h_ids = map_hadith_ids(citations, hadith_id_map)
    combined: List[str] = qpc_ids + h_ids
    combined = combined[:max_items]
    rows: List[Tuple[str, str, int, float, str]] = []
    for rank, docid in enumerate(combined):
        # Use a stable decreasing score; larger is better
        score = 1_000_000.0 - rank * 1000.0
        rows.append((qid, "Q0", docid, rank, score, "burhanai_task2_rag"))
    return rows


def write_run_tsv(rows: List[Tuple[str, str, str, int, float, str]], out_path: str) -> None:
    # Ensure columns: query, q0, docid, rank, score, system (tab-separated)
    with open(out_path, "w", encoding="utf-8") as f:
        for q, q0, d, r, s, sysname in rows:
            f.write(f"{q}\t{q0}\t{d}\t{r}\t{s}\t{sysname}\n")


# -----------------------------
# Orchestrator
# -----------------------------

async def main(
    *,
    dataset_split: str,
    outdir: Optional[str],
    models: Optional[List[str]],
    effort: Optional[str],
    limit: Optional[int],
    concurrency: int,
    evaluate: bool,
) -> None:
    # Resolve vector store
    store_id = ensure_vector_store_for_task2()
    globals()["VECTOR_STORE_ID"] = store_id

    # Prepare output directory
    global OUTPUT_DIR
    if outdir:
        OUTPUT_DIR = outdir
    else:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        models_part = re.sub(r"[^A-Za-z0-9]+", "_", "+".join(models or ["gpt_5"]))
        effort_part = re.sub(r"[^A-Za-z0-9]+", "_", (effort or "high"))
        OUTPUT_DIR = os.path.join(
            OUTPUT_ROOT, f"burhanai_task2_RAG_{models_part}_{effort_part}_{ts}"
        )
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load datasets
    base = os.path.join("datasets", "task_2_data")
    dev_path = os.path.join(base, "QH-QA-25_Subtask2_ayatec_v1.3_dev.tsv")
    train_path = os.path.join(base, "QH-QA-25_Subtask2_ayatec_v1.3_train.tsv")
    test_path = os.path.join(base, "QH-QA-25_Subtask2_ayatec_v1.3_test.tsv")
    if dataset_split == "dev":
        questions = parse_ayatec_tsv(dev_path)
    elif dataset_split == "train":
        questions = parse_ayatec_tsv(train_path)
    elif dataset_split == "test":
        questions = parse_ayatec_tsv(test_path)
    else:
        # all = train + dev
        questions = parse_ayatec_tsv(train_path) + parse_ayatec_tsv(dev_path)
    if limit is not None:
        questions = questions[:limit]

    # Load corpora mappings
    qpc_path = os.path.join(base, "Thematic_QPC", "QH-QA-25_Subtask2_QPC_v1.1.tsv")
    _, verse_to_passage = load_qpc_tsv(qpc_path)
    hadith_map = load_hadith_ids(
        os.path.join(base, "Sahih-Bukhari", "QH-QA-25_Subtask2_Sahih-Bukhari_v1.0.jsonl")
    )

    # Resume detection
    json_paths_by_id: Dict[str, str] = {}
    pending: List[Question] = []
    for q in questions:
        out_path = os.path.join(OUTPUT_DIR, f"{q.id}.json")
        if os.path.exists(out_path):
            json_paths_by_id[q.id] = out_path
        else:
            pending.append(q)

    # OpenAI client and execution pool
    client = AsyncOpenAI()
    sem = asyncio.Semaphore(concurrency)
    model_pool = models or ["gpt-5"]
    reasoning_effort = effort or "medium"

    # Assign models round-robin for diversity
    async def _process(q: Question, idx: int) -> Tuple[str, Optional[str], float, str]:
        model_name = model_pool[idx % len(model_pool)]
        t0 = time.time()
        path = await call_model_with_retry(
            client,
            q,
            sem,
            model_name=model_name,
            reasoning_effort=reasoning_effort,
        )
        dt = time.time() - t0
        return q.id, path, dt, model_name

    tasks: List[asyncio.Task] = []
    for i, q in enumerate(pending):
        tasks.append(asyncio.create_task(_process(q, i)))

    processed_this_run: Dict[str, str] = {}
    durations_this_run: Dict[str, float] = {}
    failures: List[str] = []
    total_to_process = len(pending)
    started_at = time.time()
    last_report = 0.0
    if tasks:
        completed = 0
        for coro in asyncio.as_completed(tasks):
            qid, path, dt, model_name = await coro
            if path is None:
                failures.append(qid)
            else:
                json_paths_by_id[qid] = path
                processed_this_run[qid] = model_name
                durations_this_run[qid] = dt
            completed += 1
            # Progress reporting with ETA
            now = time.time()
            # Rate-limit progress logs to ~2/sec
            if now - last_report > 0.5:
                done = completed
                remaining = total_to_process - done
                avg_dt = (sum(durations_this_run.values()) / max(1, len(durations_this_run))) if durations_this_run else 0.0
                eta_sec = int(remaining * avg_dt) if avg_dt > 0 else 0
                m, s = divmod(max(0, eta_sec), 60)
                h, m = divmod(m, 60)
                elapsed = int(now - started_at)
                em, es = divmod(elapsed, 60)
                eh, em = divmod(em, 60)
                print(
                    f"Progress [{dataset_split}]: {done}/{total_to_process} done, {len(failures)} failed, "
                    f"remaining {remaining}, elapsed {eh:02d}:{em:02d}:{es:02d}, ETA {h:02d}:{m:02d}:{s:02d}")
                last_report = now

    # Build run rows
    all_rows: List[Tuple[str, str, str, int, float, str]] = []
    for q in questions:
        jpath = json_paths_by_id.get(q.id)
        citations = _extract_citations_from_json(jpath) if jpath else []
        rows = build_trec_rows_for_question(
            q.id, citations, verse_to_passage, hadith_map, max_items=20
        )
        all_rows.extend(rows)

    # Write run file
    run_tsv = os.path.join(OUTPUT_DIR, "run.tsv")
    write_run_tsv(all_rows, run_tsv)
    print(f"Wrote run file: {run_tsv}")

    # Persist run metadata
    meta = {
        "task": "subtask_2_rag",
        "time": datetime.utcnow().isoformat() + "Z",
        "output_dir": os.path.abspath(OUTPUT_DIR),
        "dataset_split": dataset_split,
        "num_questions": len(questions),
        "processed_this_run": list(processed_this_run.keys()),
        "skipped_existing": [qid for qid in json_paths_by_id.keys() if qid not in processed_this_run],
        "failures": failures,
        "model_pool": model_pool,
        "reasoning_effort": reasoning_effort,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "openai_package_version": getattr(OpenAI, "__version__", None),
            "openai_api_key_present": bool(os.getenv("OPENAI_API_KEY")),
            "vector_store_id": VECTOR_STORE_ID,
        },
    }
    with open(os.path.join(OUTPUT_DIR, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Copy to evaluator input location and run checker/evaluator if requested
    if evaluate:
        eval_dir = os.path.join(
            "original_docs", "task_2_docs_and_organizers_code", "Evaluation"
        )
        eval_res_dir = os.path.join(eval_dir, "input", "res")
        os.makedirs(eval_res_dir, exist_ok=True)
        eval_run_path = os.path.join(eval_res_dir, "run_sample.tsv")
        try:
            # Write a copy for evaluator
            with open(run_tsv, "r", encoding="utf-8") as src, open(
                eval_run_path, "w", encoding="utf-8"
            ) as dst:
                dst.write(src.read())
            print("Copied run to evaluator input path.")
        except Exception as e:
            print(f"Warning: failed to prepare evaluator run file: {e}")

        # 1) Run format checker
        try:
            import subprocess

            cmd = [
                sys.executable,
                os.path.join(eval_dir, "checker.py"),
                "-m",
                eval_run_path,
            ]
            print("Running submission checker…")
            result = subprocess.run(cmd, capture_output=True, text=True)
            print(result.stdout)
            if result.returncode != 0:
                print(result.stderr)
        except Exception as e:
            print(f"Warning: checker failed to execute: {e}")

        # 2) Run evaluator script (uses its internal fixed paths)
        try:
            import subprocess

            cmd = [sys.executable, os.path.join(eval_dir, "evaluate.py")]
            print("Running evaluator…")
            result = subprocess.run(cmd, capture_output=True, text=True)
            print(result.stdout)
            if result.returncode != 0:
                print(result.stderr)
        except Exception as e:
            print(f"Warning: evaluator failed to execute: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="IslamicEval 2025 Subtask 2 – RAG pipeline to generate TREC run files"
    )
    parser.add_argument(
        "--split", choices=["train", "dev", "test", "all"], default="all", help="Dataset split"
    )
    parser.add_argument(
        "--outdir", type=str, default=None, help="Output directory (auto-named if omitted)"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["gpt-5"],
        help="One or more model names to use (round-robin)",
    )
    parser.add_argument(
        "--effort",
        choices=["low", "medium", "high"],
        default="medium",
        help="Reasoning effort",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions")
    parser.add_argument("--concurrency", type=int, default=4, help="Max concurrent calls")
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run checker and evaluator on the produced run file",
    )

    args = parser.parse_args()

    # Make sure root output exists
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    asyncio.run(
        main(
            dataset_split=args.split,
            outdir=args.outdir,
            models=args.models,
            effort=args.effort,
            limit=args.limit,
            concurrency=args.concurrency,
            evaluate=args.evaluate,
        )
    )


