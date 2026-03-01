#!/usr/bin/env python3
"""
Compute token statistics for all hadiths across the six books.

Per-book and overall stats: count, total, avg, min, max, p50, p90, p99.

Usage:
  python compute_hadith_token_stats.py \
    --dir datasets/six_hadith_books_split \
    --encoding o200k_base

Notes:
  - Counts tokens across ALL string fields in each hadith object.
  - Uses tiktoken (o200k_base by default) which matches GPT-4o/5 tokenization.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import tiktoken


def resolve_encoder(encoding: Optional[str], model: Optional[str]):
    if model:
        try:
            return tiktoken.encoding_for_model(model)
        except Exception:
            pass
    if encoding:
        try:
            return tiktoken.get_encoding(encoding)
        except Exception:
            pass
    return tiktoken.get_encoding("o200k_base")


def iter_strings_from_obj(obj: Any) -> Iterable[str]:
    if isinstance(obj, str):
        yield obj
        return
    if isinstance(obj, dict):
        for v in obj.values():
            yield from iter_strings_from_obj(v)
        return
    if isinstance(obj, list):
        for item in obj:
            yield from iter_strings_from_obj(item)
        return
    return


def count_tokens_text(encoder, text: Any) -> int:
    if text is None:
        return 0
    if not isinstance(text, str):
        text = str(text)
    return len(encoder.encode(text))


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hadith token statistics across books")
    p.add_argument("--dir", default="datasets/six_hadith_books_split", help="Directory with book_*.json files")
    p.add_argument("--encoding", default="o200k_base", help="tiktoken encoding (default: o200k_base)")
    p.add_argument("--model", default=None, help="Optional model name to infer encoding")
    return p.parse_args()


def load_book_files(directory: Path) -> List[Path]:
    return sorted(directory.glob("book_*.json"))


def compute_book_stats(path: Path, encoder) -> Tuple[str, Dict[str, float]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}")

    per_hadith_tokens: List[int] = []
    for rec in data:
        tokens = 0
        for s in iter_strings_from_obj(rec):
            tokens += count_tokens_text(encoder, s)
        per_hadith_tokens.append(tokens)

    total = int(sum(per_hadith_tokens))
    count = len(per_hadith_tokens)
    avg = (total / count) if count else 0.0
    min_v = min(per_hadith_tokens) if count else 0
    max_v = max(per_hadith_tokens) if count else 0
    p50, p90, p99 = compute_quantiles(per_hadith_tokens, [0.5, 0.9, 0.99])

    return (
        path.name,
        {
            "count": float(count),
            "total": float(total),
            "avg": float(avg),
            "min": float(min_v),
            "max": float(max_v),
            "p50": float(p50),
            "p90": float(p90),
            "p99": float(p99),
        },
    )


def main() -> int:
    args = parse_args()
    directory = Path(args.dir)
    encoder = resolve_encoder(args.encoding, args.model)

    files = load_book_files(directory)
    if not files:
        print(f"No book_*.json files found in {directory}")
        return 2

    results: List[Tuple[str, Dict[str, float]]] = []
    overall_tokens: List[int] = []

    for fp in files:
        name, stats = compute_book_stats(fp, encoder)
        results.append((name, stats))

    # Compute overall from per-hadith values by re-reading to avoid storing huge lists per book
    for fp in files:
        with fp.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for rec in data:
            t = 0
            for s in iter_strings_from_obj(rec):
                t += count_tokens_text(encoder, s)
            overall_tokens.append(t)

    overall_total = int(sum(overall_tokens))
    overall_count = len(overall_tokens)
    overall_avg = (overall_total / overall_count) if overall_count else 0.0
    overall_min = min(overall_tokens) if overall_count else 0
    overall_max = max(overall_tokens) if overall_count else 0
    overall_p50, overall_p90, overall_p99 = compute_quantiles(overall_tokens, [0.5, 0.9, 0.99])

    # Print table
    headers = ["book", "count", "total", "avg", "min", "max", "p50", "p90", "p99"]
    col_widths = [len(h) for h in headers]
    rows: List[List[str]] = []
    for name, st in results:
        row = [
            name,
            f"{int(st['count'])}",
            f"{int(st['total'])}",
            f"{st['avg']:.2f}",
            f"{int(st['min'])}",
            f"{int(st['max'])}",
            f"{st['p50']:.2f}",
            f"{st['p90']:.2f}",
            f"{st['p99']:.2f}",
        ]
        rows.append(row)
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    # Overall row
    overall_row = [
        "ALL",
        f"{overall_count}",
        f"{overall_total}",
        f"{overall_avg:.2f}",
        f"{overall_min}",
        f"{overall_max}",
        f"{overall_p50:.2f}",
        f"{overall_p90:.2f}",
        f"{overall_p99:.2f}",
    ]
    for i, cell in enumerate(overall_row):
        col_widths[i] = max(col_widths[i], len(cell))

    def fmt(row: Sequence[str]) -> str:
        return "  ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row))

    print(fmt(headers))
    print(fmt(["-" * w for w in col_widths]))
    for row in rows:
        print(fmt(row))
    print("" )
    print(fmt(overall_row))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


