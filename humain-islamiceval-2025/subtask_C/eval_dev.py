"""
Dev set evaluation script.
Runs the matcher against dev_SubtaskC.tsv and compares predictions to ground truth.
Reports exact match accuracy, precision, recall, and F1.
"""
import csv
from matcher import QuranHadithSpanMatcher
from span_corrector import SpanCorrector

# ── Config ────────────────────────────────────────────────────────────────────
DEV_TSV  = "datasets/dev_SubtaskC.tsv"
DEV_XML  = "datasets/dev_SubtaskC.xml"

USE_RERANKER   = False   # set True to use BGE (slower but better)
QURAN_THRESH   = 0.70
HADITH_THRESH  = 0.65
# ─────────────────────────────────────────────────────────────────────────────

def normalize_for_compare(text: str) -> str:
    """Normalize for comparison: remove diacritics and letter variants."""
    import re
    text = text.strip()
    # Remove diacritics
    text = re.sub(r'[\u064B-\u0652\u0670\u0640]', '', text)
    # Normalize alef variants
    text = re.sub(r'[أإآٱ]', 'ا', text)
    # Normalize hamza
    text = re.sub(r'[ئؤء]', 'ء', text)
    # Ta marbuta → ha
    text = text.replace('ة', 'ه')
    # Alef maqsura → ya
    text = text.replace('ى', 'ي')
    # Remove kashida
    text = text.replace('ـ', '')
    # Normalize whitespace
    text = ' '.join(text.split())
    return text

def evaluate(pred: str, gold: str) -> str:
    """Return 'TP', 'TN', 'FP', or 'FN'."""
    # Check خطأ BEFORE normalizing (it's Arabic so normalization would mangle it)
    pred_is_error = (pred.strip() == "خطأ")
    gold_is_error = (gold.strip() == "خطأ")
    p = normalize_for_compare(pred)
    g = normalize_for_compare(gold)

    if not gold_is_error and not pred_is_error:
        return "TP" if p == g else "FP_WRONG"   # found something but wrong text
    if not gold_is_error and pred_is_error:
        return "FN"      # missed a real correction
    if gold_is_error and pred_is_error:
        return "TN"      # correctly said خطأ
    if gold_is_error and not pred_is_error:
        return "FP"      # falsely returned a correction


print("=" * 60)
print("DEV SET EVALUATION")
print(f"  Quran threshold : {QURAN_THRESH}")
print(f"  Hadith threshold: {HADITH_THRESH}")
print(f"  Re-ranker       : {'ON (BGE)' if USE_RERANKER else 'OFF (rule-based)'}")
print("=" * 60)

# ── Init matcher ──────────────────────────────────────────────────────────────
matcher = QuranHadithSpanMatcher(
    quran_index_dir="subtask_C/quran_index",
    hadith_index_dir="subtask_C/hadith_index",
    use_hf_reranker=USE_RERANKER,
    verbose=False
)
matcher.quran_similarity_threshold  = QURAN_THRESH
matcher.hadith_similarity_threshold = HADITH_THRESH
processor = SpanCorrector(matcher, verbose=False)

# ── Read dev TSV ──────────────────────────────────────────────────────────────
# Dev format: Question_ID | Label | Span_Start | Span_End | Original_Span | Correction
rows = []
with open(DEV_TSV, encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        rows.append(row)

print(f"\nLoaded {len(rows)} dev rows\n")

# ── Run predictions ───────────────────────────────────────────────────────────
results = []
for i, row in enumerate(rows, 1):
    label      = row["Label"]           # WrongAyah / WrongHadith
    span_type  = "quran" if "Ayah" in label else "hadith"
    q_id       = row["Question_ID"]
    span_start = int(row["Span_Start"])
    span_end   = int(row["Span_End"])
    orig_span  = row["Original_Span"].strip()
    gold       = row["Correction"].strip()

    # Try to extract full span from XML
    full_span = processor.extract_full_span_from_xml(DEV_XML, q_id, span_start, span_end)
    if not full_span:
        full_span = orig_span

    # Predict
    match_result = matcher.match_span_with_verse_splitting(full_span, span_type)
    if match_result == "خطأ" or not isinstance(match_result, dict):
        pred = "خطأ"
    else:
        pred = str(match_result.get("text", "خطأ")).strip()

    outcome = evaluate(pred, gold)
    results.append({
        "q_id": q_id, "label": label,
        "gold": gold, "pred": pred, "outcome": outcome
    })

    if i % 20 == 0:
        print(f"  Processed {i}/{len(rows)}...")

# ── Compute metrics ───────────────────────────────────────────────────────────
tp       = sum(1 for r in results if r["outcome"] == "TP")
tn       = sum(1 for r in results if r["outcome"] == "TN")
fp       = sum(1 for r in results if r["outcome"] == "FP")
fn       = sum(1 for r in results if r["outcome"] == "FN")
fp_wrong = sum(1 for r in results if r["outcome"] == "FP_WRONG")
total    = len(results)

precision = tp / (tp + fp + fp_wrong) if (tp + fp + fp_wrong) > 0 else 0
recall    = tp / (tp + fn)            if (tp + fn) > 0            else 0
f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
accuracy  = (tp + tn) / total

print(f"\n{'=' * 60}")
print("RESULTS")
print(f"{'=' * 60}")
print(f"Total spans       : {total}")
print(f"")
print(f"TP (correct fix)  : {tp}")
print(f"TN (correct خطأ) : {tn}")
print(f"FN (missed fix)   : {fn}")
print(f"FP (false fix)    : {fp}")
print(f"FP_WRONG (bad fix): {fp_wrong}")
print(f"")
print(f"Accuracy          : {accuracy:.3f}  ({(tp+tn)}/{total})")
print(f"Precision         : {precision:.3f}")
print(f"Recall            : {recall:.3f}")
print(f"F1                : {f1:.3f}")
print(f"{'=' * 60}")

# ── Show some failures ────────────────────────────────────────────────────────
print("\nSample FN (missed corrections):")
fns = [r for r in results if r["outcome"] == "FN"][:5]
for r in fns:
    print(f"  [{r['q_id']} {r['label']}]")
    print(f"    Gold : {r['gold'][:80]}...")
    print(f"    Pred : {r['pred'][:80]}")

print("\nSample FP_WRONG (wrong text returned):")
fps = [r for r in results if r["outcome"] == "FP_WRONG"][:5]
for r in fps:
    print(f"  [{r['q_id']} {r['label']}]")
    print(f"    Gold : {r['gold'][:80]}...")
    print(f"    Pred : {r['pred'][:80]}...")
