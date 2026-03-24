#!/usr/bin/env python3
"""
ensemble_voting.py
==================
Post-processing script for zero-shot LLM baseline results.

Reads all result JSON files from the results/ directory and:
  1. Prints and saves a comparison table (Model × Metric).
  2. Computes a majority-vote ensemble for Subtask 1B.
  3. Runs an annotation-error analysis for Subtask 1C:
     finds spans where ALL models agree on a correction that
     differs from the gold label (strong evidence of annotation errors).

Usage:
  python ensemble_voting.py --results-dir results/
  python ensemble_voting.py --results-dir results/ --output-dir analysis/
"""

import os
import re
import sys
import json
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter, defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# RESULT LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_results(results_dir: Path) -> Dict[str, Dict]:
    """Load all zero-shot result JSON files from results_dir.

    Expects files named: {model_safe_name}_{subtask}_zeroshot.json
    where subtask is 1A, 1B, or 1C.

    Returns:
        Nested dict: results[subtask][model_name] = result_dict
    """
    results: Dict[str, Dict] = {"1A": {}, "1B": {}, "1C": {}}

    json_files = sorted(results_dir.glob("*_zeroshot.json"))
    if not json_files:
        print(f"[WARN] No *_zeroshot.json files found in {results_dir}")
        return results

    for filepath in json_files:
        with open(filepath, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        subtask = data.get("subtask", "")  # "1A", "1B", or "1C"
        model   = data.get("model", filepath.stem)

        if subtask not in results:
            print(f"  [SKIP] Unknown subtask '{subtask}' in {filepath.name}")
            continue

        results[subtask][model] = data
        print(f"  Loaded: {filepath.name}  (model={model}, subtask={subtask})")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

def build_comparison_table(results: Dict[str, Dict]) -> List[Dict]:
    """Build one row per model containing all available metrics.

    Columns:
      Model | 1A_MacroF1 | 1B_Acc | 1B_Acc_Ayah | 1B_Acc_Hadith |
            | 1C_Acc | 1C_Acc_Ayah | 1C_Acc_Hadith

    Missing values are represented as "" (empty string).
    """
    # Collect all model names across subtasks
    all_models = set()
    for subtask_results in results.values():
        all_models.update(subtask_results.keys())

    rows = []
    for model in sorted(all_models):
        row: Dict = {"Model": model}

        # ── Subtask 1A ───────────────────────────────────────────────────────
        if model in results["1A"]:
            agg = results["1A"][model].get("aggregate_metrics", {})
            row["1A_MacroF1"] = f"{agg.get('macro_f1', 0.0):.4f}"
            per_class = agg.get("per_class", {})
            row["1A_F1_Ayah"]    = f"{per_class.get('ayah',   0.0):.4f}"
            row["1A_F1_Hadith"]  = f"{per_class.get('hadith', 0.0):.4f}"
            row["1A_F1_Neither"] = f"{per_class.get('neither',0.0):.4f}"
        else:
            row["1A_MacroF1"] = row["1A_F1_Ayah"] = row["1A_F1_Hadith"] = row["1A_F1_Neither"] = ""

        # ── Subtask 1B ───────────────────────────────────────────────────────
        if model in results["1B"]:
            agg = results["1B"][model].get("aggregate_metrics", {})
            row["1B_Acc"]       = f"{agg.get('accuracy',        0.0):.4f}"
            row["1B_Acc_Ayah"]  = f"{agg.get('accuracy_ayah',  0.0):.4f}"
            row["1B_Acc_Hadith"]= f"{agg.get('accuracy_hadith', 0.0):.4f}"
            row["1B_N"]         = str(agg.get("n_total", ""))
        else:
            row["1B_Acc"] = row["1B_Acc_Ayah"] = row["1B_Acc_Hadith"] = row["1B_N"] = ""

        # ── Subtask 1C ───────────────────────────────────────────────────────
        if model in results["1C"]:
            agg = results["1C"][model].get("aggregate_metrics", {})
            row["1C_Acc"]       = f"{agg.get('accuracy',        0.0):.4f}"
            row["1C_Acc_Ayah"]  = f"{agg.get('accuracy_ayah',  0.0):.4f}"
            row["1C_Acc_Hadith"]= f"{agg.get('accuracy_hadith', 0.0):.4f}"
            row["1C_N"]         = str(agg.get("n_total", ""))
        else:
            row["1C_Acc"] = row["1C_Acc_Ayah"] = row["1C_Acc_Hadith"] = row["1C_N"] = ""

        rows.append(row)

    return rows


def print_comparison_table(rows: List[Dict]) -> None:
    """Pretty-print the comparison table to stdout."""
    if not rows:
        print("No results to display.")
        return

    # Define column order and display widths
    columns = [
        ("Model",         30),
        ("1A_MacroF1",    12),
        ("1A_F1_Ayah",    12),
        ("1A_F1_Hadith",  12),
        ("1A_F1_Neither", 12),
        ("1B_Acc",        10),
        ("1B_Acc_Ayah",   12),
        ("1B_Acc_Hadith", 13),
        ("1C_Acc",        10),
        ("1C_Acc_Ayah",   12),
        ("1C_Acc_Hadith", 13),
    ]

    # Header
    header = "  ".join(col.ljust(w) for col, w in columns)
    print("\n" + "=" * len(header))
    print("ZERO-SHOT BASELINE COMPARISON TABLE")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for row in rows:
        line = "  ".join(str(row.get(col, "")).ljust(w) for col, w in columns)
        print(line)

    print("=" * len(header))


def save_comparison_csv(rows: List[Dict], outfile: Path) -> None:
    """Save the comparison table to a CSV file."""
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(outfile, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Comparison table saved -> {outfile}")


# ─────────────────────────────────────────────────────────────────────────────
# SUBTASK 1A — SPAN ENSEMBLE (MAJORITY & WEIGHTED)
# ─────────────────────────────────────────────────────────────────────────────

def _char_f1(text_len: int, gold_spans: List[Dict], pred_spans: List[Dict]) -> Dict:
    """Character-level macro F1 for one sample (mirrors zero_shot_baselines.py)."""
    from sklearn.metrics import f1_score

    def spans_to_labels(spans):
        labels = ["neither"] * text_len
        for sp in spans:
            stype = "ayah" if sp.get("type") == "q" else "hadith"
            for i in range(max(0, sp["start"]), min(text_len, sp["end"])):
                labels[i] = stype
        return labels

    gold_labels = spans_to_labels(gold_spans)
    pred_labels = spans_to_labels(pred_spans)

    if gold_labels == pred_labels:
        classes = set(gold_labels)
        per_class = {c: 1.0 for c in ("ayah", "hadith", "neither")}
        macro = sum(per_class[c] for c in classes) / len(classes)
        return {"macro_f1": macro, "f1_ayah": per_class["ayah"],
                "f1_hadith": per_class["hadith"], "f1_neither": per_class["neither"]}

    macro = f1_score(gold_labels, pred_labels, average="macro", zero_division=0)
    per = f1_score(gold_labels, pred_labels,
                   labels=["ayah", "hadith", "neither"],
                   average=None, zero_division=0)
    return {"macro_f1": float(macro),
            "f1_ayah": float(per[0]), "f1_hadith": float(per[1]), "f1_neither": float(per[2])}


def ensemble_1a(results: Dict[str, Dict], weighted: bool = False) -> Dict:
    """Ensemble Subtask 1A predictions via character-level voting.

    For each character position in the response text, every model casts a vote:
      "q" (Quran), "h" (Hadith), or implicitly "neither".
    The winning label at each position is determined by:
      - Majority vote  (weighted=False): strict majority of models
      - Weighted vote  (weighted=True):  each model's vote is scaled by its
                                         overall macro F1 score

    Contiguous runs of the same non-neither label are then converted back to spans
    and evaluated against gold using character-level macro F1.
    """
    a_results = results.get("1A", {})
    if len(a_results) < 2:
        print("[INFO] Need >=2 models for 1A ensemble.")
        return {}

    models = list(a_results.keys())
    method = "weighted" if weighted else "majority"
    print(f"\nEnsemble 1A ({method}): combining {len(models)} models")

    # Model weights: overall macro F1 (or 1.0 for pure majority)
    model_weights = {}
    for model, data in a_results.items():
        f1 = data.get("aggregate_metrics", {}).get("macro_f1", 0.5)
        model_weights[model] = float(f1) ** 20 if weighted else 1.0

    # Index per-sample predictions: {qid: {model: [spans]}}
    per_qid_preds: Dict[str, Dict] = defaultdict(dict)
    per_qid_gold:  Dict[str, List] = {}

    for model, data in a_results.items():
        for sample in data.get("per_sample", []):
            qid = sample.get("sample_id", "")
            if not qid:
                continue
            spans = (sample.get("prediction") or {}).get("spans", []) or []
            per_qid_preds[qid][model] = spans
            if qid not in per_qid_gold and sample.get("gold_spans") is not None:
                per_qid_gold[qid] = sample["gold_spans"]

    all_f1, per_sample_out = [], []

    for qid in sorted(per_qid_preds.keys()):
        model_spans = per_qid_preds[qid]
        gold_spans  = per_qid_gold.get(qid, [])

        # Determine text length from max span end across all predictions + gold
        text_len = max(
            (sp["end"] for spans in model_spans.values() for sp in spans if "end" in sp),
            default=0
        )
        text_len = max(
            text_len,
            max((sp["end"] for sp in gold_spans if "end" in sp), default=0)
        ) + 1

        # Accumulate weighted votes at each character position
        votes_q = [0.0] * text_len
        votes_h = [0.0] * text_len

        for model, spans in model_spans.items():
            w = model_weights[model]
            for sp in spans:
                start = max(0, sp.get("start", 0))
                end   = min(text_len, sp.get("end", 0))
                if end <= start:
                    continue
                if sp.get("type") == "q":
                    for i in range(start, end):
                        votes_q[i] += w
                else:
                    for i in range(start, end):
                        votes_h[i] += w

        # Threshold: >50% of total weight (or >50% of models for majority)
        total_w = sum(model_weights[m] for m in model_spans)
        threshold = total_w / 2.0

        # Assign per-character label
        labels = []
        for i in range(text_len):
            if votes_q[i] > threshold and votes_q[i] >= votes_h[i]:
                labels.append("q")
            elif votes_h[i] > threshold and votes_h[i] > votes_q[i]:
                labels.append("h")
            else:
                labels.append("n")

        # Convert runs of same non-n label to spans
        ensemble_spans = []
        i = 0
        while i < text_len:
            if labels[i] != "n":
                stype = labels[i]
                start = i
                while i < text_len and labels[i] == stype:
                    i += 1
                ensemble_spans.append({"type": stype, "start": start, "end": i})
            else:
                i += 1

        metrics = _char_f1(text_len, gold_spans, ensemble_spans)
        all_f1.append(metrics["macro_f1"])

        per_sample_out.append({
            "sample_id":      qid,
            "ensemble_spans": ensemble_spans,
            "gold_spans":     gold_spans,
            "metrics":        metrics,
            "n_models":       len(model_spans),
        })

    avg_f1 = sum(all_f1) / len(all_f1) if all_f1 else 0.0
    per_class = {}
    for cls, key in [("ayah", "f1_ayah"), ("hadith", "f1_hadith"), ("neither", "f1_neither")]:
        vals = [s["metrics"][key] for s in per_sample_out if s.get("metrics")]
        per_class[cls] = sum(vals) / len(vals) if vals else 0.0

    print(f"  Ensemble 1A ({method}) macro F1: {avg_f1:.4f}  "
          f"(Ayah={per_class['ayah']:.4f}, Hadith={per_class['hadith']:.4f})  "
          f"[{len(per_sample_out)} questions, {len(models)} models]")
    if weighted:
        print("  Model weights (F1):")
        for m, w in sorted(model_weights.items(), key=lambda x: -x[1]):
            print(f"    {m}: {w:.4f}")

    return {
        "method":            method,
        "models":            models,
        "model_weights":     model_weights,
        "aggregate_macro_f1": avg_f1,
        "per_class":         per_class,
        "per_sample":        per_sample_out,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SUBTASK 1B — MAJORITY VOTE ENSEMBLE
# ─────────────────────────────────────────────────────────────────────────────

def ensemble_1b(results: Dict[str, Dict]) -> Dict:
    """Compute majority-vote ensemble accuracy for Subtask 1B.

    For each (question_id, annotation_id) span, the ensemble prediction is
    "correct" if more than half of the models predict "correct", else "incorrect".
    Ties are broken in favour of "incorrect" (the safer label).

    Returns:
        dict with ensemble_accuracy, ensemble_accuracy_ayah, ensemble_accuracy_hadith,
        and per_span details.
    """
    b_results = results.get("1B", {})
    if not b_results:
        print("[INFO] No 1B results available for ensemble.")
        return {}

    models = list(b_results.keys())
    print(f"\nEnsemble 1B: combining {len(models)} models: {models}")

    # Index per-span predictions from each model.
    # Key: (question_id, annotation_id) or (question_id, span_index)
    # Value: {model_name: pred_correct (bool), gold_correct (bool), span_type (str)}
    span_data: Dict[Tuple, Dict] = defaultdict(dict)
    span_gold:  Dict[Tuple, bool] = {}
    span_type_map: Dict[Tuple, str] = {}

    for model, data in b_results.items():
        # Flatten all per-sample span predictions
        for sample in data.get("per_sample", []):
            qid = sample.get("sample_id", "")
            gold_spans = sample.get("gold_spans", [])
            # The prediction list is stored in per_sample for each model
            # We need the flat per-span predictions stored in all_span_predictions
        # Use all_span_predictions if available (flat list of span dicts)
        flat_preds = data.get("all_span_predictions", [])
        for idx, pred in enumerate(flat_preds):
            qid       = pred.get("question_id", "")
            ann_id    = pred.get("annotation_id", idx)
            key       = (qid, str(ann_id))
            span_data[key][model]  = pred.get("pred_correct", False)
            span_gold[key]         = pred.get("gold_correct", False)
            span_type_map[key]     = pred.get("span_type", "Unknown")

    if not span_data:
        print("[WARN] Could not extract per-span predictions for ensemble.")
        return {}

    # Compute ensemble prediction and accuracy
    total = correct = 0
    ayah_total = ayah_correct = 0
    hadith_total = hadith_correct = 0
    per_span_details = []

    for key, model_preds in span_data.items():
        qid, ann_id = key
        gold = span_gold.get(key, False)
        stype = span_type_map.get(key, "Unknown")

        # Majority vote: count how many models say "correct"
        n_correct = sum(1 for v in model_preds.values() if v)
        n_total_models = len(model_preds)
        # Ensemble: "correct" only if strictly more than half say correct
        ensemble_pred = (n_correct > n_total_models / 2)

        match = (ensemble_pred == gold)

        total += 1
        if match:
            correct += 1
        if stype == "Ayah":
            ayah_total += 1
            if match:
                ayah_correct += 1
        elif stype == "Hadith":
            hadith_total += 1
            if match:
                hadith_correct += 1

        per_span_details.append({
            "question_id":     qid,
            "annotation_id":   ann_id,
            "span_type":       stype,
            "gold_correct":    gold,
            "ensemble_pred":   ensemble_pred,
            "match":           match,
            "model_votes":     {m: bool(v) for m, v in model_preds.items()},
            "n_correct_votes": n_correct,
            "n_total_votes":   n_total_models,
        })

    acc       = correct / total if total else 0.0
    acc_ayah  = ayah_correct / ayah_total if ayah_total else 0.0
    acc_hadith= hadith_correct / hadith_total if hadith_total else 0.0

    print(f"  Ensemble 1B accuracy: {acc:.4f}  "
          f"(Ayah={acc_ayah:.4f}, Hadith={acc_hadith:.4f})  "
          f"[{total} spans from {len(models)} models]")

    return {
        "models":              models,
        "ensemble_accuracy":        acc,
        "ensemble_accuracy_ayah":   acc_ayah,
        "ensemble_accuracy_hadith": acc_hadith,
        "n_total":   total,
        "n_ayah":    ayah_total,
        "n_hadith":  hadith_total,
        "per_span":  per_span_details,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SUBTASK 1C — ANNOTATION ERROR ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def normalize_arabic(text: str) -> str:
    """Light normalisation used to compare corrections across models.

    Mirrors the normalisation from zero_shot_baselines.py / subtask_C/matcher.py.
    """
    import re
    if not text:
        return ""
    # Remove diacritics
    text = re.sub(r"[\u064B-\u0652\u0670\u0640]", "", text)
    # Replace punctuation with space
    text = re.sub(r"[ۚ۩﴾﴿؛،.,:;!?()\[\]\"'«»\-–—]", " ", text)
    # Remove digits
    text = re.sub(r"[\u06F0-\u06F9\u0660-\u0669\d]", "", text)
    # Normalize whitespace
    return " ".join(text.split())


def annotation_error_analysis(results: Dict[str, Dict]) -> List[Dict]:
    """Find 1C spans where ALL models agree on a correction differing from gold.

    These are strong candidates for annotation errors in the dev set.

    Returns:
        Sorted list of candidate annotation errors (highest model consensus first).
        Each entry:
          span_text, gold_correction, model_corrections (dict),
          n_models_agree, consensus_correction, all_agree_with_each_other
    """
    c_results = results.get("1C", {})
    if not c_results:
        print("[INFO] No 1C results available for annotation error analysis.")
        return []

    models = list(c_results.keys())
    print(f"\nAnnotation Error Analysis 1C: {len(models)} models")

    # Index per-span predictions from each model.
    # Key: (question_id, span_start)  — unique identifier for each incorrect span
    span_data: Dict[Tuple, Dict] = defaultdict(dict)
    span_gold: Dict[Tuple, str] = {}
    span_text_map: Dict[Tuple, str] = {}
    span_type_map: Dict[Tuple, str] = {}

    for model, data in c_results.items():
        for sample in data.get("per_sample", []):
            qid       = sample.get("sample_id", "")
            span_text = sample.get("span_text", "")
            pred      = (sample.get("prediction") or {}).get("correction", "")
            gold      = sample.get("gold", "")
            stype     = sample.get("span_type", "Unknown")

            # Use (qid, normalized_span_text) as the key to match across models
            key = (qid, normalize_arabic(span_text)[:60])  # truncated for stability
            span_data[key][model]  = pred
            span_gold[key]         = gold
            span_text_map[key]     = span_text
            span_type_map[key]     = stype

    candidates = []

    for key, model_preds in span_data.items():
        qid = key[0]
        gold = span_gold.get(key, "")
        span_text = span_text_map.get(key, "")
        stype = span_type_map.get(key, "Unknown")

        # Normalize gold and model predictions for comparison
        gold_norm = normalize_arabic(gold)
        if gold.strip() == "خطأ":
            gold_norm = "خطأ"

        # Collect model predictions (normalized)
        norm_preds = {}
        for m, pred in model_preds.items():
            p_norm = normalize_arabic(pred)
            if pred.strip() == "خطأ":
                p_norm = "خطأ"
            norm_preds[m] = p_norm

        # Count how many models' predictions DIFFER from gold
        models_differ = [m for m, pn in norm_preds.items() if pn != gold_norm]
        n_differ = len(models_differ)

        if n_differ == 0:
            continue  # All models agree with gold — not an annotation issue

        # Check if models agree with each other (ignoring gold)
        unique_model_preds = set(norm_preds.values())
        all_agree_with_each_other = (len(unique_model_preds) == 1)

        # Consensus correction = most common model prediction
        counter = Counter(norm_preds.values())
        consensus_correction, n_consensus = counter.most_common(1)[0]

        candidates.append({
            "question_id":             qid,
            "span_type":               stype,
            "span_text":               span_text,
            "gold_correction":         gold,
            "consensus_correction":    consensus_correction,
            "n_models_total":          len(model_preds),
            "n_models_differ_from_gold": n_differ,
            "all_agree_with_each_other": all_agree_with_each_other,
            "model_corrections":       {m: model_preds[m] for m in model_preds},
            "model_corrections_normalized": norm_preds,
        })

    # Sort: models that agree with each other AND all differ from gold come first,
    # then by proportion of models differing from gold (descending)
    candidates.sort(
        key=lambda x: (
            -int(x["all_agree_with_each_other"]),
            -x["n_models_differ_from_gold"] / max(x["n_models_total"], 1),
        )
    )

    print(f"  Found {len(candidates)} potential annotation error candidates "
          f"(out of {len(span_data)} total spans).")

    # Print summary of the top candidates
    top_n = min(10, len(candidates))
    if top_n > 0:
        print(f"\n  Top {top_n} annotation error candidates:")
        for i, c in enumerate(candidates[:top_n], 1):
            agree_str = "ALL MODELS AGREE" if c["all_agree_with_each_other"] else "MIXED"
            print(f"  {i}. [{c['span_type']}] {c['question_id']} — "
                  f"{c['n_models_differ_from_gold']}/{c['n_models_total']} models differ "
                  f"({agree_str})")
            print(f"     Span:  {c['span_text'][:80]}…")
            print(f"     Gold:  {c['gold_correction'][:80]}")
            print(f"     Models say: {c['consensus_correction'][:80]}")

    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ensemble voting and analysis over zero-shot baseline results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ensemble_voting.py --results-dir results/
  python ensemble_voting.py --results-dir results/ --output-dir analysis/
        """,
    )
    parser.add_argument(
        "--results-dir", default="results",
        help="Directory containing *_zeroshot.json result files (default: results/)",
    )
    parser.add_argument(
        "--output-dir", default="results",
        help="Directory to save CSV and JSON outputs (default: same as --results-dir)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_dir.exists():
        print(f"ERROR: results directory not found: {results_dir}")
        sys.exit(1)

    # ── 1. Load all results ──────────────────────────────────────────────────
    print(f"Loading results from {results_dir} …")
    results = load_results(results_dir)

    n_loaded = sum(len(v) for v in results.values())
    if n_loaded == 0:
        print("No results loaded. Exiting.")
        sys.exit(1)
    print(f"Loaded {n_loaded} result file(s).")

    # ── 2. Comparison table ──────────────────────────────────────────────────
    print("\nBuilding comparison table…")
    table_rows = build_comparison_table(results)
    print_comparison_table(table_rows)
    save_comparison_csv(table_rows, output_dir / "comparison_table.csv")

    # ── 3. 1A ensemble voting ────────────────────────────────────────────────
    for weighted in (False, True):
        ens_1a = ensemble_1a(results, weighted=weighted)
        if ens_1a:
            method = ens_1a["method"]
            outfile = output_dir / f"ensemble_1a_{method}.json"
            with open(outfile, "w", encoding="utf-8") as fh:
                json.dump(ens_1a, fh, ensure_ascii=False, indent=2)
            print(f"Ensemble 1A ({method}) saved -> {outfile}")

    # ── 4. 1B ensemble voting ────────────────────────────────────────────────
    ensemble_result = ensemble_1b(results)
    if ensemble_result:
        ens_outfile = output_dir / "ensemble_1b.json"
        with open(ens_outfile, "w", encoding="utf-8") as fh:
            json.dump(ensemble_result, fh, ensure_ascii=False, indent=2)
        print(f"Ensemble 1B details saved -> {ens_outfile}")

    # ── 5. 1C annotation error analysis ─────────────────────────────────────
    annotation_errors = annotation_error_analysis(results)
    if annotation_errors:
        err_outfile = output_dir / "annotation_error_candidates_1c.json"
        with open(err_outfile, "w", encoding="utf-8") as fh:
            json.dump(annotation_errors, fh, ensure_ascii=False, indent=2)
        print(f"Annotation error candidates saved -> {err_outfile}")

    print("\nDone.")


if __name__ == "__main__":
    main()
