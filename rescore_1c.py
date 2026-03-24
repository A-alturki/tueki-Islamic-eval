"""
rescore_1c.py
-------------
Re-score existing 1C result JSONs using:
  - Updated remove_default_diac (tatweel + assimilation-shadda fixes)
  - Updated snapping: Quran ON, returns خطأ if no canonical match found

Reads from:  results/*_1C_zeroshot.json
Writes to:   results_rescored/*_1C_zeroshot.json
"""

import json, re, sys, os
from pathlib import Path
from difflib import SequenceMatcher
from typing import Optional, List
import unicodedata

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR   = Path("results")
OUT_DIR       = Path("results_rescored")
QURAN_FILE    = Path("datasets/quranic_verses.json")
HADITH_FILE   = Path("datasets/six_hadith_books.json")

OUT_DIR.mkdir(exist_ok=True)

# ── Normalisation (updated) ────────────────────────────────────────────────────
def normalize_hadith(s: str) -> str:
    """For Hadith comparison: strip ALL diacritics and punctuation (lenient match)."""
    # Strip all Arabic diacritics and annotation marks
    out = re.sub(
        r"[\u064B-\u0652\u0670\u0640\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]",
        "", s)
    # Strip punctuation: Arabic comma/semicolon/period + Western equivalents
    out = re.sub(r"[،؛؟,;.!?\"'()[\]{}]", "", out)
    # Collapse whitespace
    out = re.sub(r"\s+", " ", out).strip()
    return out


def remove_default_diac(s: str) -> str:
    out = unicodedata.normalize("NFC", s)
    # Remove tatweel only before superscript alif (أُولَـٰئِكَ corpus encoding)
    out = re.sub("\u0640(?=\u0670)", "", out)
    # Remove assimilation (إدغام) shadda — first consonant of a word only
    out = re.sub(r"(?<= )([\u0600-\u06FF][\u064B-\u0650]?)\u0651", r"\1", out)
    out = re.sub(r"^([\u0600-\u06FF][\u064B-\u0650]?)\u0651", r"\1", out)
    # Original scoring_C.py replacements
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
    # Normalise spacing around ayah-number markers "(N)"
    out = re.sub(r" \((\d+)\)", r"(\1)", out)
    return out


# ── Corpus loading ─────────────────────────────────────────────────────────────
def _strip_all_diac(text: str) -> str:
    return re.sub(
        r"[\u064B-\u0652\u0670\u0640\u06D6-\u06DC\u06DF-\u06E4\u06E5\u06E6\u06E7\u06E8\u06EA-\u06ED]",
        "", text)

_QURAN_CORPUS: Optional[List] = None
_HADITH_CORPUS: Optional[List] = None
_QURAN_CANDIDATES: Optional[List] = None

def _load_quran_corpus() -> List:
    global _QURAN_CORPUS
    if _QURAN_CORPUS is not None:
        return _QURAN_CORPUS
    with open(QURAN_FILE, encoding="utf-8") as f:
        data = json.load(f)
    data = sorted([x for x in data if x and x.get("ayah_text")],
                  key=lambda x: (x["surah_id"], x["ayah_id"]))
    _QURAN_CORPUS = [
        {"orig": x["ayah_text"], "stripped": _strip_all_diac(x["ayah_text"]),
         "surah": x["surah_id"], "ayah": x["ayah_id"]}
        for x in data
    ]
    return _QURAN_CORPUS

def _load_hadith_corpus() -> List:
    global _HADITH_CORPUS
    if _HADITH_CORPUS is not None:
        return _HADITH_CORPUS
    with open(HADITH_FILE, encoding="utf-8") as f:
        data = json.load(f)
    seen: set = set()
    entries = []
    for item in data:
        if item is None:
            continue
        text = item.get("Matn") or item.get("hadithTxt")
        if text and text not in seen:
            seen.add(text)
            entries.append({"orig": text, "stripped": _strip_all_diac(text)})
    _HADITH_CORPUS = entries
    return _HADITH_CORPUS


# ── Snapping ───────────────────────────────────────────────────────────────────
def retrieve_canonical(pred_text: str, span_type: str,
                       min_similarity: float = 0.55) -> Optional[str]:
    if not pred_text or pred_text.strip() == "خطأ":
        return None
    stripped_pred = _strip_all_diac(pred_text.strip())
    if not stripped_pred:
        return None
    pred_words = set(stripped_pred.split())
    best_ratio = 0.0
    best_text: Optional[str] = None

    if span_type == "Ayah":
        global _QURAN_CANDIDATES, _QURAN_INDEX
        if _QURAN_CANDIDATES is None:
            corpus = _load_quran_corpus()
            MAX_AYAH_SPAN = 10
            _QURAN_CANDIDATES = []
            _QURAN_INDEX = {}
            for idx, entry in enumerate(corpus):
                _QURAN_CANDIDATES.append((entry["orig"], entry["stripped"]))
                orig_parts     = [entry["orig"]]
                stripped_parts = [entry["stripped"]]
                for k in range(1, MAX_AYAH_SPAN):
                    j = idx + k
                    if j >= len(corpus) or corpus[j]["surah"] != entry["surah"]:
                        break
                    nxt = corpus[j]
                    orig_parts.append(f" ({corpus[j-1]['ayah']}) " + nxt["orig"])
                    stripped_parts.append(" " + nxt["stripped"])
                    combined_orig = "".join(orig_parts) + f" ({nxt['ayah']})"
                    combined_str  = "".join(stripped_parts)
                    _QURAN_CANDIDATES.append((combined_orig, combined_str))
            for i, (_, stripped) in enumerate(_QURAN_CANDIDATES):
                for w in stripped.split():
                    _QURAN_INDEX.setdefault(w, set()).add(i)

        from collections import Counter as _Counter
        hit_counts = _Counter()
        for w in pred_words:
            for i in _QURAN_INDEX.get(w, set()):
                hit_counts[i] += 1
        min_common = max(2, len(pred_words) // 4)
        shortlist_idx = {i for i, c in hit_counts.items() if c >= min_common}
        candidates = [_QURAN_CANDIDATES[i] for i in shortlist_idx]
    else:
        corpus = _load_hadith_corpus()
        candidates = [(e["orig"], e["stripped"]) for e in corpus]

    for orig, stripped in candidates:
        corp_words = set(stripped.split())
        if corp_words and len(pred_words & corp_words) / max(len(pred_words), 1) < 0.30:
            continue
        ratio = SequenceMatcher(None, stripped_pred, stripped).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_text = orig

    return best_text if best_ratio >= min_similarity else None


def _crop_canonical_to_model(pred_stripped: str, canon_orig: str) -> str:
    """Return the slice of canon_orig whose words best align with pred_stripped.

    Uses word-level SequenceMatcher to find which span of canonical words the
    model was targeting, then returns those words with their canonical diacritics.
    Falls back to the full canonical if alignment fails.
    """
    pred_words = pred_stripped.split()
    if not pred_words:
        return canon_orig

    # Pair original and stripped words together, skipping standalone waqf/
    # annotation marks that disappear after stripping (e.g. ۚ ۖ).
    pairs = [(o, _strip_all_diac(o))
             for o in canon_orig.split()
             if _strip_all_diac(o).strip()]
    if not pairs:
        return canon_orig

    canon_words_o = [p[0] for p in pairs]
    canon_words_s = [p[1] for p in pairs]

    matcher = SequenceMatcher(None, pred_words, canon_words_s, autojunk=False)
    blocks = [b for b in matcher.get_matching_blocks() if b.size > 0]
    if not blocks:
        return canon_orig

    start = blocks[0].b
    end   = blocks[-1].b + blocks[-1].size
    return " ".join(canon_words_o[start:end]) if end > start else canon_orig


def snap(model_raw: str, span_type: str) -> str:
    """Apply updated snapping: both Ayah and Hadith; return خطأ if no match."""
    if model_raw.strip() == "خطأ":
        return "خطأ"
    # Strip a trailing lone "(N)" — model labelled a single ayah with its number.
    # e.g. "فَأُمُّهُۥ هَاوِيَةٌ (9)" → "فَأُمُّهُۥ هَاوِيَةٌ"
    # A multi-ayah output has (N) in the middle too, so this only fires when
    # there is exactly one (N) marker and it is at the very end.
    cleaned = re.sub(r"\s*\(\d+\)\s*$", "", model_raw.strip())
    if re.search(r"\(\d+\)", cleaned):
        cleaned = model_raw.strip()

    canonical = retrieve_canonical(cleaned, span_type)
    if not canonical:
        return "خطأ"

    # Length-ratio check: if model output length is comparable to canonical,
    # snap fully. Otherwise crop the canonical to match what the model covered.
    stripped_cleaned   = _strip_all_diac(cleaned)
    stripped_canonical = _strip_all_diac(canonical)
    len_ratio = len(stripped_cleaned) / max(len(stripped_canonical), 1)

    if 0.7 <= len_ratio <= 1.3:
        return canonical  # normal full snap
    else:
        return _crop_canonical_to_model(stripped_cleaned, canonical)


# ── Main rescoring loop ────────────────────────────────────────────────────────
result_files = sorted(RESULTS_DIR.glob("*_1C_zeroshot.json"))
if not result_files:
    print("No 1C result files found in results/")
    sys.exit(1)

print(f"Rescoring {len(result_files)} files -> {OUT_DIR}/\n")

summary_rows = []

for path in result_files:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    model = data.get("model", path.stem)
    per_sample = data.get("per_sample", [])
    if not per_sample:
        print(f"  {path.name}: no per_sample — skipping")
        continue

    n_total = n_correct = n_ayah = n_ayah_correct = n_hadith = n_hadith_correct = 0

    new_per_sample = []
    for s in per_sample:
        pred_info  = s.get("prediction") or {}
        model_raw  = pred_info.get("model_raw", pred_info.get("correction", "")) or ""
        gold       = s.get("gold", "")
        span_type  = s.get("span_type", "Ayah")

        # Re-snap with updated logic
        snapped = snap(model_raw, span_type)
        if span_type == "Hadith":
            exact = normalize_hadith(gold) == normalize_hadith(snapped)
        else:
            exact = remove_default_diac(gold.strip()) == remove_default_diac(snapped.strip())

        n_total += 1
        if span_type == "Ayah":
            n_ayah += 1
            if exact: n_ayah_correct += 1
        else:
            n_hadith += 1
            if exact: n_hadith_correct += 1
        if exact:
            n_correct += 1

        new_s = dict(s)
        new_s["prediction"] = {"correction": snapped, "model_raw": model_raw}
        new_s["metrics"]    = {"exact_match": exact}
        new_per_sample.append(new_s)

    acc        = n_correct        / n_total   if n_total   else 0
    acc_ayah   = n_ayah_correct   / n_ayah    if n_ayah    else 0
    acc_hadith = n_hadith_correct / n_hadith  if n_hadith  else 0

    out_data = dict(data)
    out_data["per_sample"] = new_per_sample
    out_data["aggregate_metrics"] = {
        "accuracy":        round(acc,        4),
        "accuracy_ayah":   round(acc_ayah,   4),
        "accuracy_hadith": round(acc_hadith, 4),
        "n_total":         n_total,
        "n_correct":       n_correct,
        "n_ayah":          n_ayah,
        "n_ayah_correct":  n_ayah_correct,
        "n_hadith":        n_hadith,
        "n_hadith_correct":n_hadith_correct,
    }

    out_path = OUT_DIR / path.name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    summary_rows.append((model, n_total, n_correct, acc, acc_ayah, acc_hadith))
    print(f"  {model:<30} acc={acc:.3f}  ayah={acc_ayah:.3f}  hadith={acc_hadith:.3f}  ({n_correct}/{n_total})")

print(f"\nDone. Results written to {OUT_DIR}/")
