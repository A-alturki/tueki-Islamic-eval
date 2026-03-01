#!/usr/bin/env python3
import csv
import json
import os
from typing import Dict, List, Optional, Tuple

OUTPUT_DIR = "outputs"
HIGH_TSV = os.path.join(OUTPUT_DIR, "TaskABC_partial_submission_high_reason.tsv")
LOW_TSV = os.path.join(OUTPUT_DIR, "TaskABC_partial_submission_low_reason.tsv")
MEDIUM_TSV = os.path.join(OUTPUT_DIR, "TaskABC_partial_submission_medium_reason.tsv")

FIELDNAMES = [
    "Question_ID",
    "Annotation_ID",
    "Label",
    "Span_Start",
    "Span_End",
    "Original_Span",
    "Correction",
]


def list_output_jsons() -> List[str]:
    if not os.path.exists(OUTPUT_DIR):
        return []
    return sorted(
        os.path.join(OUTPUT_DIR, f)
        for f in os.listdir(OUTPUT_DIR)
        if f.startswith("Q") and f.endswith(".json")
    )


def read_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def get_effort(data: dict) -> Optional[str]:
    # Prefer explicit reasoning effort if present
    eff = (data or {}).get("reasoning", {}).get("effort")
    if eff in {"low", "medium", "high"}:
        return eff
    # Fallback to reasoning token heuristic if explicit effort is absent
    # Thresholds can be adjusted via env vars if desired
    high_min = int(os.environ.get("REASONING_TOKENS_HIGH_MIN", "12000"))
    medium_min = int(os.environ.get("REASONING_TOKENS_MEDIUM_MIN", "6000"))
    rtoks = (data or {}).get("usage", {}).get("output_tokens_details", {}).get("reasoning_tokens")
    if isinstance(rtoks, int):
        if rtoks >= high_min:
            return "high"
        if rtoks >= medium_min:
            return "medium"
        return "low"
    return None


def find_message_text(data: dict) -> str:
    """Locate assistant message text that contains the citations JSON string."""
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


def infer_question_id_from_path(path: str) -> str:
    base = os.path.basename(path)
    return os.path.splitext(base)[0]


def extract_rows_for_question(qid: str, json_path: str) -> List[Dict[str, str]]:
    data = read_json(json_path)
    if not data:
        return []
    text = find_message_text(data)
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
                "Question_ID": qid,
                "Annotation_ID": idx,
                "Label": label,
                "Span_Start": cit.get("span_start", ""),
                "Span_End": cit.get("span_end", ""),
                "Original_Span": cit.get("span_text", ""),
                "Correction": cit.get("corrected_text", ""),
            }
        )
    return rows


def write_tsv(path: str, all_rows: List[Dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, delimiter="\t")
        writer.writeheader()
        writer.writerows(all_rows)


def main():
    json_files = list_output_jsons()
    if not json_files:
        print("No output JSONs found under 'outputs/'. Nothing to do.")
        return

    high_qids: List[Tuple[str, str]] = []
    medium_qids: List[Tuple[str, str]] = []
    low_qids: List[Tuple[str, str]] = []

    for jpath in json_files:
        data = read_json(jpath)
        if not data:
            continue
        eff = get_effort(data)
        qid = infer_question_id_from_path(jpath)
        if eff == "high":
            high_qids.append((qid, jpath))
        elif eff == "medium":
            medium_qids.append((qid, jpath))
        elif eff == "low":
            low_qids.append((qid, jpath))
        else:
            print(f"Warning: No reasoning.effort for {jpath}; defaulting to 'low'.")
            low_qids.append((qid, jpath))

    high_rows: List[Dict[str, str]] = []
    for qid, jpath in high_qids:
        high_rows.extend(extract_rows_for_question(qid, jpath))

    medium_rows: List[Dict[str, str]] = []
    for qid, jpath in medium_qids:
        medium_rows.extend(extract_rows_for_question(qid, jpath))

    low_rows: List[Dict[str, str]] = []
    for qid, jpath in low_qids:
        low_rows.extend(extract_rows_for_question(qid, jpath))

    write_tsv(HIGH_TSV, high_rows)
    write_tsv(MEDIUM_TSV, medium_rows)
    write_tsv(LOW_TSV, low_rows)

    print("Wrote partial submissions:")
    print(f" - {HIGH_TSV}    ({len(high_rows)} rows from {len(high_qids)} questions)")
    print(f" - {MEDIUM_TSV}  ({len(medium_rows)} rows from {len(medium_qids)} questions)")
    print(f" - {LOW_TSV}     ({len(low_rows)} rows from {len(low_qids)} questions)")


if __name__ == "__main__":
    main()
