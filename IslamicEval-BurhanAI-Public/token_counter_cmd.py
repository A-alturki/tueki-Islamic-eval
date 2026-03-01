#!/usr/bin/env python3
"""
Token counter CLI for JSON/JSONL/TXT files using tiktoken (o200k_base by default).

Examples:
  - Count tokens for big JSON files in datasets:
      python token_counter_cmd.py datasets --extensions .json --encoding o200k_base

  - Only count specific JSON keys:
      python token_counter_cmd.py datasets/quranic_verses.json --json-keys text arabic

  - Save results to JSON:
      python token_counter_cmd.py datasets --extensions .json --save results.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm import tqdm

try:
    import tiktoken
except Exception as exc:  # pragma: no cover
    print(
        "Missing dependency 'tiktoken'. Please run: python -m pip install tiktoken",
        file=sys.stderr,
    )
    raise


# -----------------------------
# Tokenizer helpers
# -----------------------------


def resolve_encoder(encoding: Optional[str], model: Optional[str]):
    """Return a tiktoken encoder. Defaults to o200k_base if resolution fails.

    New OpenAI vision/reasoning models (e.g., GPT-4o family) use o200k_base.
    We assume GPT-5-compatible tokenization is the same unless specified.
    """
    # Prefer model mapping when provided
    if model:
        try:
            return tiktoken.encoding_for_model(model)
        except Exception:
            # Fall back to explicit encoding
            pass
    # Then explicit encoding
    if encoding:
        try:
            return tiktoken.get_encoding(encoding)
        except Exception:
            pass
    # Default fallback
    return tiktoken.get_encoding("o200k_base")


def count_tokens_text(encoder, text: str) -> int:
    if not text:
        return 0
    # Ensure string
    if not isinstance(text, str):
        text = str(text)
    return len(encoder.encode(text))


def iter_strings_from_obj(obj: Any, only_keys: Optional[Sequence[str]] = None) -> Iterable[str]:
    """Yield all string values from a JSON-like object.

    - If only_keys is provided and obj is a dict, only yields strings under those keys (recursively).
    - Otherwise, traverses lists/dicts recursively and yields all strings.
    """
    if isinstance(obj, str):
        yield obj
        return

    if isinstance(obj, dict):
        if only_keys:
            for key in only_keys:
                if key in obj:
                    yield from iter_strings_from_obj(obj[key], only_keys=None)
        else:
            for value in obj.values():
                yield from iter_strings_from_obj(value, only_keys=None)
        return

    if isinstance(obj, list):
        for item in obj:
            yield from iter_strings_from_obj(item, only_keys=only_keys)
        return

    # Primitive types other than string: skip unless keys were targeted and cast above
    return


# -----------------------------
# Data structures
# -----------------------------


@dataclass
class FileTokenStats:
    path: str
    file_type: str
    num_records: int
    total_tokens: int
    avg_tokens: float
    p50: float
    p90: float
    p99: float


def compute_quantiles(values: List[int], qs: Sequence[float]) -> List[float]:
    if not values:
        return [0.0 for _ in qs]
    sorted_vals = sorted(values)
    out: List[float] = []
    n = len(sorted_vals)
    for q in qs:
        if n == 1:
            out.append(float(sorted_vals[0]))
            continue
        idx = q * (n - 1)
        lo = math.floor(idx)
        hi = math.ceil(idx)
        if lo == hi:
            out.append(float(sorted_vals[lo]))
        else:
            frac = idx - lo
            out.append(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac)
    return out


# -----------------------------
# File readers
# -----------------------------


def detect_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".jsonl", ".ndjson"}:
        return "jsonl"
    if ext == ".json":
        return "json"
    if ext in {".txt", ".tsv", ".csv", ".md"}:
        return "text"
    return "unknown"


def gather_files(inputs: Sequence[str], allowed_extensions: Optional[Sequence[str]]) -> List[Path]:
    files: List[Path] = []
    allow = None
    if allowed_extensions:
        allow = {ext.lower() for ext in allowed_extensions}
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            for sub in p.rglob("*"):
                if sub.is_file():
                    if allow is None or sub.suffix.lower() in allow:
                        files.append(sub)
        elif p.is_file():
            if allow is None or p.suffix.lower() in allow:
                files.append(p)
    # Deduplicate, stable
    seen = set()
    unique: List[Path] = []
    for f in files:
        if f not in seen:
            unique.append(f)
            seen.add(f)
    return unique


def count_tokens_in_file(
    path: Path,
    encoder,
    json_keys: Optional[Sequence[str]] = None,
    max_records: Optional[int] = None,
) -> Tuple[FileTokenStats, List[int]]:
    ftype = detect_type(path)
    per_record: List[int] = []

    if ftype == "jsonl":
        total = 0
        n = 0
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    # Treat as plain text line
                    tokens = count_tokens_text(encoder, line)
                    per_record.append(tokens)
                    total += tokens
                    n += 1
                    if max_records and n >= max_records:
                        break
                    continue
                tokens = 0
                for s in iter_strings_from_obj(obj, only_keys=json_keys):
                    tokens += count_tokens_text(encoder, s)
                per_record.append(tokens)
                total += tokens
                n += 1
                if max_records and n >= max_records:
                    break

    elif ftype == "json":
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        # If list: per item record; else: single record
        if isinstance(obj, list):
            for idx, item in enumerate(obj):
                tokens = 0
                for s in iter_strings_from_obj(item, only_keys=json_keys):
                    tokens += count_tokens_text(encoder, s)
                per_record.append(tokens)
                if max_records and (idx + 1) >= max_records:
                    break
        else:
            tokens = 0
            for s in iter_strings_from_obj(obj, only_keys=json_keys):
                tokens += count_tokens_text(encoder, s)
            per_record.append(tokens)

    elif ftype == "text":
        total = 0
        n = 0
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                tokens = count_tokens_text(encoder, line)
                per_record.append(tokens)
                total += tokens
                n += 1
                if max_records and n >= max_records:
                    break
    else:
        # Unknown: treat whole file as one string
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            text = ""
        per_record = [count_tokens_text(encoder, text)]

    total_tokens = int(sum(per_record))
    num_records = len(per_record)
    avg_tokens = float(total_tokens) / num_records if num_records else 0.0
    p50, p90, p99 = compute_quantiles(per_record, [0.5, 0.9, 0.99])

    stats = FileTokenStats(
        path=str(path),
        file_type=ftype,
        num_records=num_records,
        total_tokens=total_tokens,
        avg_tokens=avg_tokens,
        p50=p50,
        p90=p90,
        p99=p99,
    )
    return stats, per_record


def format_table(rows: List[FileTokenStats]) -> str:
    if not rows:
        return "No files matched."
    # Column widths
    headers = [
        "path",
        "type",
        "records",
        "total_tokens",
        "avg",
        "p50",
        "p90",
        "p99",
    ]
    data = []
    for r in rows:
        data.append([
            r.path,
            r.file_type,
            str(r.num_records),
            str(r.total_tokens),
            f"{r.avg_tokens:.2f}",
            f"{r.p50:.2f}",
            f"{r.p90:.2f}",
            f"{r.p99:.2f}",
        ])
    col_widths = [len(h) for h in headers]
    for row in data:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    def fmt_row(row: Sequence[str]) -> str:
        return "  ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row))

    lines = [fmt_row(headers), fmt_row(["-" * w for w in col_widths])]
    for row in data:
        lines.append(fmt_row(row))
    # Totals line
    total_tokens = sum(r.total_tokens for r in rows)
    total_records = sum(r.num_records for r in rows)
    avg = (total_tokens / total_records) if total_records else 0.0
    lines.append("")
    lines.append(
        f"TOTALS  records={total_records}  total_tokens={total_tokens}  avg_per_record={avg:.2f}"
    )
    return "\n".join(lines)


def save_results(rows: List[FileTokenStats], save_path: Path) -> None:
    if save_path.suffix.lower() == ".json":
        with save_path.open("w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in rows], f, ensure_ascii=False, indent=2)
        return
    if save_path.suffix.lower() in {".csv", ".tsv"}:
        delimiter = "," if save_path.suffix.lower() == ".csv" else "\t"
        with save_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter=delimiter)
            writer.writerow(["path", "file_type", "num_records", "total_tokens", "avg_tokens", "p50", "p90", "p99"])
            for r in rows:
                writer.writerow([
                    r.path,
                    r.file_type,
                    r.num_records,
                    r.total_tokens,
                    f"{r.avg_tokens:.6f}",
                    f"{r.p50:.6f}",
                    f"{r.p90:.6f}",
                    f"{r.p99:.6f}",
                ])
        return
    raise ValueError("Unsupported save format. Use .json, .csv, or .tsv")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count tokens in files using tiktoken")
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input files or directories",
    )
    parser.add_argument(
        "--encoding",
        default="o200k_base",
        help="Tiktoken encoding name (default: o200k_base)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name to infer encoding (overrides --encoding when resolvable)",
    )
    parser.add_argument(
        "--json-keys",
        nargs="*",
        default=None,
        help="If provided, only count these keys in JSON objects",
    )
    parser.add_argument(
        "--extensions",
        nargs="*",
        default=None,
        help="Only include files with these extensions (e.g., .json .jsonl .txt)",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional cap on records processed per file (for sampling)",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Optional path to save results (.json, .csv, or .tsv)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    encoder = resolve_encoder(args.encoding, args.model)

    files = gather_files(args.inputs, args.extensions)
    if not files:
        print("No matching files.")
        return 2

    rows: List[FileTokenStats] = []
    for fpath in tqdm(files, desc="Counting tokens", unit="file"):
        try:
            stats, _ = count_tokens_in_file(
                fpath, encoder, json_keys=args.json_keys, max_records=args.max_records
            )
            rows.append(stats)
        except Exception as exc:  # pragma: no cover
            print(f"Error processing {fpath}: {exc}", file=sys.stderr)

    print(format_table(rows))

    if args.save:
        save_results(rows, Path(args.save))
        print(f"\nSaved results to {args.save}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


