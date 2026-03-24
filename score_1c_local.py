"""
Local scorer for Subtask 1C — mirrors scoring_C.py but uses local paths.

Usage:
    python score_1c_local.py <model_name>
    python score_1c_local.py gemini-3-flash-preview
    python score_1c_local.py --all          # score every *_1C_zeroshot.json in results/

The script:
  1. Reads results/<model>_1C_zeroshot.json produced by zero_shot_baselines.py
  2. Builds an in-memory pred TSV  (Sequence_ID, Correction)
  3. Builds an in-memory ref  TSV  from datasets/dev_SubtaskC.tsv
  4. Applies the same scoring logic as scoring_C.py (remove_default_diac + corpus lookup)
  5. Prints per-model accuracy (overall, Ayah, Hadith)
"""

import json
import os
import sys
import re
import glob

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR  = os.path.join(BASE_DIR, "results")
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
SCORING_DIR  = os.path.join(BASE_DIR, "scoring")

QURAN_FILE  = os.path.join(DATASETS_DIR, "quranic_verses.json")
HADITH_FILE = os.path.join(DATASETS_DIR, "six_hadith_books.json")
REF_TSV     = os.path.join(DATASETS_DIR, "dev_SubtaskC.tsv")


# ── Normalization (copied verbatim from scoring_C.py) ─────────────────────────
def remove_default_diac(s):
    import unicodedata, re as _re
    out = unicodedata.normalize("NFC", s)
    out = _re.sub("\u0640(?=\u0670)", "", out)
    out = _re.sub(r"(?<= )([\u0600-\u06FF][\u064B-\u0650]?)\u0651", r"\1", out)
    out = _re.sub(r"^([\u0600-\u06FF][\u064B-\u0650]?)\u0651", r"\1", out)
    out = out.replace("َا", "ا")
    out = out.replace("ِي", "ي")
    out = out.replace("ُو", "و")
    out = out.replace("الْ", "ال")

    out = out.replace("ْ", "")

    out = out.replace("َّ", "َّ")
    out = out.replace("ِّ", "ِّ")
    out = out.replace("ُّ", "ُّ")
    out = out.replace("ًّ", "ًّ")
    out = out.replace("ٍّ", "ٍّ")
    out = out.replace("ٌّ", "ٌّ")

    out = out.replace("اَ", "ا")
    out = out.replace("اِ", "ا")
    out = out.replace("لِا", "لا")
    out = out.replace("اً", "ًا")

    return out


# ── Load corpora ──────────────────────────────────────────────────────────────
def load_corpora():
    with open(QURAN_FILE, encoding="utf-8") as f:
        quranic_verses = json.load(f)
    with open(HADITH_FILE, encoding="utf-8") as f:
        hadith_books = json.load(f)

    hadith_db = set()
    for item in hadith_books:
        if item is None:
            continue
        if item.get("hadithTxt"):
            hadith_db.add(remove_default_diac(item["hadithTxt"]))
        if item.get("Matn"):
            hadith_db.add(remove_default_diac(item["Matn"]))

    quranic_db = set()
    for item in quranic_verses:
        if item is None:
            continue
        if item.get("ayah_text"):
            quranic_db.add(remove_default_diac(item["ayah_text"]))

    return quranic_db, hadith_db


# ── Scoring logic (mirrors scoring_C.py line 108) ────────────────────────────
def is_correct(pred_correction: str, ref_correction: str,
               quranic_db: set, hadith_db: set) -> bool:
    pred_n = remove_default_diac(str(pred_correction))
    ref_n  = remove_default_diac(str(ref_correction))

    # Exact match after normalisation
    if pred_n == ref_n:
        return True

    # Partial match: ref is contained in pred AND pred is in the canonical corpus
    if ref_n in pred_n and (pred_n in hadith_db or pred_n in quranic_db):
        return True

    return False


# ── Score one model ───────────────────────────────────────────────────────────
def score_model(model_name: str, ref_df: pd.DataFrame,
                quranic_db: set, hadith_db: set) -> dict:
    result_file = os.path.join(RESULTS_DIR, f"{model_name}_1C_zeroshot.json")
    if not os.path.exists(result_file):
        print(f"  [SKIP] {result_file} not found")
        return {}

    with open(result_file, encoding="utf-8") as f:
        data = json.load(f)

    per_sample = data.get("per_sample", [])
    if not per_sample:
        print(f"  [SKIP] {model_name} — no per_sample data")
        return {}

    # Build lookup: span_id (= sample_id = "C-QXX@start-end") -> predicted correction
    pred_lookup = {}
    for s in per_sample:
        span_id = s.get("sample_id") or s.get("sequence_id")
        pred    = s.get("prediction", {})
        if isinstance(pred, dict):
            correction = pred.get("correction", "")
        else:
            correction = str(pred)
        pred_lookup[span_id] = correction

    total = ayah_total = hadith_total = 0
    correct = ayah_correct = hadith_correct = 0

    for _, row in ref_df.iterrows():
        # Build the same unique span_id used by run_1c
        qid      = row["Question_ID"]
        start    = int(row["Span_Start"])
        end      = int(row["Span_End"])
        span_id  = f"{qid}@{start}-{end}"
        ref_corr = str(row["Correction"])
        stype    = str(row["Label"])   # WrongAyah / WrongHadith

        if span_id not in pred_lookup:
            print(f"  [WARN] {span_id} missing from predictions")
            total += 1
            continue

        pred_corr = pred_lookup[span_id]
        ok = is_correct(pred_corr, ref_corr, quranic_db, hadith_db)

        total   += 1
        correct += int(ok)

        if "Ayah" in stype:
            ayah_total   += 1
            ayah_correct += int(ok)
        elif "Hadith" in stype:
            hadith_total   += 1
            hadith_correct += int(ok)

    acc        = correct / total if total else 0.0
    acc_ayah   = ayah_correct   / ayah_total   if ayah_total   else 0.0
    acc_hadith = hadith_correct / hadith_total if hadith_total else 0.0

    return {
        "model":       model_name,
        "accuracy":    round(acc,        4),
        "acc_ayah":    round(acc_ayah,   4),
        "acc_hadith":  round(acc_hadith, 4),
        "n_total":     total,
        "n_correct":   correct,
        "n_ayah":      ayah_total,
        "n_hadith":    hadith_total,
        "status":      data.get("status", "?"),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ref_df      = pd.read_csv(REF_TSV, sep="\t")
    quranic_db, hadith_db = load_corpora()
    print(f"Loaded {len(quranic_db):,} Quran ayahs, {len(hadith_db):,} hadith entries\n")

    # Determine which models to score
    if len(sys.argv) >= 2 and sys.argv[1] == "--all":
        files = glob.glob(os.path.join(RESULTS_DIR, "*_1C_zeroshot.json"))
        models = [os.path.basename(f).replace("_1C_zeroshot.json", "") for f in files]
    elif len(sys.argv) >= 2:
        models = [sys.argv[1]]
    else:
        print("Usage: python score_1c_local.py <model_name>  |  --all")
        sys.exit(1)

    rows = []
    for model in sorted(models):
        result = score_model(model, ref_df, quranic_db, hadith_db)
        if result:
            rows.append(result)
            print(f"{model:30s}  acc={result['accuracy']:.4f}  "
                  f"ayah={result['acc_ayah']:.4f}  hadith={result['acc_hadith']:.4f}  "
                  f"({result['n_correct']}/{result['n_total']})  [{result['status']}]")

    if len(rows) > 1:
        print()
        print(f"{'Model':30s}  {'Acc':>7}  {'Ayah':>7}  {'Hadith':>7}  {'N':>5}")
        print("-" * 65)
        for r in sorted(rows, key=lambda x: -x["accuracy"]):
            print(f"{r['model']:30s}  {r['accuracy']:7.4f}  {r['acc_ayah']:7.4f}  "
                  f"{r['acc_hadith']:7.4f}  {r['n_total']:>5}")


if __name__ == "__main__":
    main()
