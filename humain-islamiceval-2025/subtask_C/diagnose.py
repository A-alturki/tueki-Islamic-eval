"""
Trace exactly why high-scoring matches are dropped for FN cases.
"""
from matcher import QuranHadithSpanMatcher
import csv, re

matcher = QuranHadithSpanMatcher(
    quran_index_dir="quran_index",
    hadith_index_dir="hadith_index",
    use_hf_reranker=False,
    verbose=False
)
matcher.quran_similarity_threshold  = 0.80
matcher.hadith_similarity_threshold = 0.70

DEV_TSV = "../datasets/dev_SubtaskC.tsv"
DEV_XML = "../datasets/dev_SubtaskC.xml"

# Read dev set, find FN cases
from span_corrector import SpanCorrector
processor = SpanCorrector(matcher, verbose=False)

rows = []
with open(DEV_TSV, encoding="utf-8") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        rows.append(row)

def norm(text):
    text = re.sub(r'[\u064B-\u0652\u0670\u0640]', '', text)
    text = re.sub(r'[أإآٱ]', 'ا', text)
    text = re.sub(r'[ئؤء]', 'ء', text)
    text = text.replace('ة','ه').replace('ى','ي').replace('ـ','')
    return ' '.join(text.split())

print("Tracing first 10 FN cases (should be correct but predicted خطأ)\n")
fn_count = 0
for row in rows:
    gold = row["Correction"].strip()
    if gold == "خطأ":
        continue  # skip TN cases

    label     = row["Label"]
    src_type  = "quran" if "Ayah" in label else "hadith"
    q_id      = row["Question_ID"]
    orig_span = row["Original_Span"].strip()
    span_start = int(row["Span_Start"])
    span_end   = int(row["Span_End"])

    full_span = processor.extract_full_span_from_xml(DEV_XML, q_id, span_start, span_end)
    if not full_span:
        full_span = orig_span

    # Step through the pipeline manually
    matches = matcher.find_matches_by_sequence(full_span, src_type)
    threshold = matcher.quran_similarity_threshold if src_type == "quran" else matcher.hadith_similarity_threshold

    top_sim   = matches[0]["similarity"] if matches else 0.0
    top_text  = matches[0].get("text","")[:70] if matches else ""
    top_type  = matches[0].get("match_type","") if matches else ""

    passes    = top_sim >= threshold
    gold_norm = norm(gold)
    pred_norm = norm(top_text) if matches else ""
    correct   = gold_norm in pred_norm or pred_norm in gold_norm or gold_norm == pred_norm

    # Final result from the full pipeline
    result = matcher.match_span_with_verse_splitting(full_span, src_type)
    final  = "خطأ" if result == "خطأ" else result.get("text","")[:60]

    print(f"[{q_id} {label}]")
    print(f"  Span      : {full_span[:70]}...")
    print(f"  Gold      : {gold[:70]}...")
    print(f"  Top match : [{top_sim:.3f}] {top_type} → {top_text}...")
    print(f"  Threshold : {threshold}  →  {'PASSES ✓' if passes else 'FAILS ✗'}")
    print(f"  Text match: {'YES ✓' if correct else 'NO ✗'}")
    print(f"  Final out : {final}")
    print()

    fn_count += 1
    if fn_count >= 10:
        break
