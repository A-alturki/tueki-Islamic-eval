"""
Quick run — no reranker download needed. Uses rule-based matching only.
Results will be slightly lower quality than the full system but runs instantly.
"""
from matcher import QuranHadithSpanMatcher
from span_corrector import SpanCorrector

matcher = QuranHadithSpanMatcher(
    quran_index_dir="quran_index",
    hadith_index_dir="hadith_index",
    use_hf_reranker=False,   # skip the 1GB model download
    verbose=True
)

matcher.quran_similarity_threshold = 0.80
matcher.hadith_similarity_threshold = 0.70

processor = SpanCorrector(matcher, verbose=True)

submission_data, spans_data = processor.process_test_tsv_file(
    tsv_file="../datasets/Test_Subtask_1C_USER.tsv",
    xml_file="../datasets/Test_Subtask_1C.xml",
    output_file="submission_C/fast_no_reranker.tsv",
)

processor.print_submission_stats(submission_data)
