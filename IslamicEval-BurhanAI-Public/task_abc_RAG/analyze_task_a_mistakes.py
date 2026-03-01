#!/usr/bin/env python3
import csv
import os
import sys
import re
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class Annotation:
    question_id: str
    annotation_id: int
    label: str
    span_start: int
    span_end: int
    original_span_field: str


def read_tsv(path: str) -> List[Annotation]:
    annotations: List[Annotation] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            qid = row.get("Question_ID", "").strip()
            ann_id_raw = row.get("Annotation_ID", "0").strip()
            label = row.get("Label", "").strip()
            try:
                ann_id = int(ann_id_raw)
            except ValueError:
                # Some files may have non-integer; skip
                continue
            try:
                span_start = int(row.get("Span_Start", "0").strip())
                span_end = int(row.get("Span_End", "0").strip())
            except ValueError:
                span_start, span_end = 0, 0
            original_span_field = row.get("Original_Span", "")
            annotations.append(
                Annotation(
                    question_id=qid,
                    annotation_id=ann_id,
                    label=label,
                    span_start=span_start,
                    span_end=span_end,
                    original_span_field=original_span_field,
                )
            )
    return annotations


def parse_task_a_input_xml(path: str) -> Dict[str, str]:
    """
    Tolerant extractor: returns mapping Question_ID -> raw inner text of <Response>.
    Does not rely on well-formed XML; uses regex to find <Question>...</Question>, then
    extracts <ID>...</ID> and <Response>...</Response> as raw strings.
    """
    with open(path, "r", encoding="utf-8") as f:
        xml_text = f.read()

    id_to_response: Dict[str, str] = {}

    # Find all Question blocks non-greedily
    for m in re.finditer(r"<Question>(.*?)</Question>", xml_text, flags=re.DOTALL | re.IGNORECASE):
        block = m.group(1)
        m_id = re.search(r"<ID>(.*?)</ID>", block, flags=re.DOTALL | re.IGNORECASE)
        m_resp = re.search(r"<Response>(.*?)</Response>", block, flags=re.DOTALL | re.IGNORECASE)
        if not m_id or not m_resp:
            continue
        qid = (m_id.group(1) or "").strip()
        response_text = m_resp.group(1) or ""
        id_to_response[qid] = response_text
    return id_to_response


def slice_safe(text: str, start: int, end: int) -> str:
    start = max(0, start)
    end = max(0, end)
    if end < start:
        start, end = end, start
    if start > len(text):
        return ""
    return text[start:end]


def build_index(annotations: List[Annotation]) -> Dict[str, Dict[int, Annotation]]:
    mapping: Dict[str, Dict[int, Annotation]] = {}
    for ann in annotations:
        if ann.label == "NoAnnotation":
            # Skip placeholders
            continue
        if ann.question_id not in mapping:
            mapping[ann.question_id] = {}
        mapping[ann.question_id][ann.annotation_id] = ann
    return mapping


def compare_and_report(
    gt_index: Dict[str, Dict[int, Annotation]],
    sub_index: Dict[str, Dict[int, Annotation]],
    responses: Dict[str, str],
) -> Tuple[str, List[str]]:
    """Compact, errors-only report.

    Per question, list only mismatches or missing/extra, with minimal lines:
    - GT: label [s,e] | text
    - SUB: label [s,e] | text
    - ERR: comma-separated categories
    """
    lines: List[str] = []
    issues: List[str] = []
    # Aggregates
    per_q_issue_counts: Dict[str, int] = {}
    label_pair_counts: Dict[Tuple[str, str], int] = {}
    span_offset_counts: Dict[Tuple[int, int], int] = {}
    quote_boundary_count: int = 0
    text_only_mismatch_count: int = 0
    total_gt_spans = 0
    total_sub_spans = 0

    all_qids = sorted(set(list(gt_index.keys()) + list(sub_index.keys())))
    for qid in all_qids:
        resp_text = responses.get(qid, "")
        gt_anns = gt_index.get(qid, {})
        sub_anns = sub_index.get(qid, {})
        total_gt_spans += len(gt_anns)
        total_sub_spans += len(sub_anns)

        all_ann_ids = sorted(set(list(gt_anns.keys()) + list(sub_anns.keys())))
        q_lines: List[str] = []

        for ann_id in all_ann_ids:
            gt = gt_anns.get(ann_id)
            sub = sub_anns.get(ann_id)
            errs: List[str] = []

            if gt and not sub:
                gt_text = slice_safe(resp_text, gt.span_start, gt.span_end)
                q_lines.append(f"- id={ann_id} | ERR=MissingInSubmission")
                q_lines.append(f"  GT: {gt.label} [{gt.span_start},{gt.span_end}] | {gt_text}")
                issues.append(f"{qid} ann {ann_id}: MissingInSubmission")
                per_q_issue_counts[qid] = per_q_issue_counts.get(qid, 0) + 1
                continue

                
            if sub and not gt:
                sub_text = slice_safe(resp_text, sub.span_start, sub.span_end)
                q_lines.append(f"- id={ann_id} | ERR=ExtraInSubmission")
                q_lines.append(f"  SUB: {sub.label} [{sub.span_start},{sub.span_end}] | {sub_text}")
                issues.append(f"{qid} ann {ann_id}: ExtraInSubmission")
                per_q_issue_counts[qid] = per_q_issue_counts.get(qid, 0) + 1
                continue

            # Both exist
            assert gt and sub
            gt_text = slice_safe(resp_text, gt.span_start, gt.span_end)
            sub_text = slice_safe(resp_text, sub.span_start, sub.span_end)

            if gt.label != sub.label:
                errs.append(f"LabelMismatch(GT={gt.label},SUB={sub.label})")
                label_pair_counts[(gt.label, sub.label)] = label_pair_counts.get((gt.label, sub.label), 0) + 1
            if gt.span_start != sub.span_start or gt.span_end != sub.span_end:
                errs.append(f"SpanMismatch")
                span_offset_counts[(sub.span_start - gt.span_start, sub.span_end - gt.span_end)] = (
                    span_offset_counts.get((sub.span_start - gt.span_start, sub.span_end - gt.span_end), 0) + 1
                )
            if gt_text != sub_text:
                errs.append("TextMismatch")
                gt_strip = gt_text.strip()
                sub_strip = sub_text.strip()
                if (gt_strip.startswith('"') and gt_strip[1:] == sub_strip) or (
                    sub_strip.startswith('"') and sub_strip[1:] == gt_strip
                ) or (gt_strip.endswith('"') and gt_strip[:-1] == sub_strip) or (
                    sub_strip.endswith('"') and sub_strip[:-1] == gt_strip
                ):
                    quote_boundary_count += 1
                else:
                    text_only_mismatch_count += 1

            if errs:
                q_lines.append(f"- id={ann_id} | ERR={','.join(errs)}")
                q_lines.append(f"  GT: {gt.label} [{gt.span_start},{gt.span_end}] | {gt_text}")
                q_lines.append(f"  SUB: {sub.label} [{sub.span_start},{sub.span_end}] | {sub_text}")
                issues.append(f"{qid} ann {ann_id}: {','.join(errs)}")
                per_q_issue_counts[qid] = per_q_issue_counts.get(qid, 0) + 1

        if q_lines:
            lines.append(f"\n=== {qid} ===")
            lines.extend(q_lines)

    # Analytics footer
    analytics_lines: List[str] = []
    analytics_lines.append("\n=== Analytics ===")
    analytics_lines.append(f"GT spans: {total_gt_spans} | SUB spans: {total_sub_spans}")
    if per_q_issue_counts:
        worst = sorted(per_q_issue_counts.items(), key=lambda x: -x[1])[:10]
        analytics_lines.append("Top questions by issue count:")
        analytics_lines.append(", ".join([f"{q}:{c}" for q, c in worst]))
    if label_pair_counts:
        analytics_lines.append("Label mismatch pairs:")
        for (g, s), c in sorted(label_pair_counts.items(), key=lambda x: -x[1]):
            analytics_lines.append(f"  {g}->{s}: {c}")
    if span_offset_counts:
        analytics_lines.append("Top span offset deltas (d_start,d_end): count")
        for (d_start, d_end), c in sorted(span_offset_counts.items(), key=lambda x: -x[1])[:10]:
            analytics_lines.append(f"  ({d_start},{d_end}): {c}")
    analytics_lines.append(f"Quote-boundary-only text mismatches: {quote_boundary_count}")
    analytics_lines.append(f"Other text mismatches: {text_only_mismatch_count}")

    report_text = "\n".join(lines + analytics_lines)
    return report_text, issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Compact errors-only analyzer for Task A")
    parser.add_argument("--gt", default=os.path.join("datasets", "TaskA_GT.tsv"))
    parser.add_argument("--submission", default=os.path.join("outputs_task_a", "TaskA_submission.tsv"))
    parser.add_argument("--input", default=os.path.join("datasets", "TaskA_Input.xml"))
    parser.add_argument("--out-report", default=os.path.join("outputs", "task_a_error_report_compact.txt"))
    parser.add_argument("--out-summary", default=os.path.join("outputs", "task_a_error_summary_compact.txt"))
    args = parser.parse_args()

    gt_path = args.gt
    sub_path = args.submission
    input_xml_path = args.input
    out_report_path = args.out_report
    out_summary_path = args.out_summary

    for p in [gt_path, sub_path, input_xml_path]:
        if not os.path.exists(p):
            print(f"Missing required file: {p}")
            sys.exit(1)

    os.makedirs(os.path.dirname(out_report_path) or ".", exist_ok=True)

    gt_anns = read_tsv(gt_path)
    sub_anns = read_tsv(sub_path)
    responses = parse_task_a_input_xml(input_xml_path)

    gt_index = build_index(gt_anns)
    sub_index = build_index(sub_anns)

    report_text, issues = compare_and_report(gt_index, sub_index, responses)

    with open(out_report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    # Summary
    cats = {"MissingInSubmission": 0, "ExtraInSubmission": 0, "LabelMismatch": 0, "SpanMismatch": 0, "TextMismatch": 0}
    for iss in issues:
        if "MissingInSubmission" in iss:
            cats["MissingInSubmission"] += 1
        if "ExtraInSubmission" in iss:
            cats["ExtraInSubmission"] += 1
        if "LabelMismatch" in iss:
            cats["LabelMismatch"] += 1
        if "SpanMismatch" in iss:
            cats["SpanMismatch"] += 1
        if "TextMismatch" in iss:
            cats["TextMismatch"] += 1
    total_qs = len(set(list(gt_index.keys()) + list(sub_index.keys())))
    summary_lines: List[str] = [
        "Task A Compact Summary",
        f"Total questions: {total_qs}",
        f"Total issue entries: {len(issues)}",
        *(f"{k}: {v}" for k, v in cats.items()),
        f"Report file: {out_report_path}",
    ]
    with open(out_summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print(f"Wrote compact report to: {out_report_path}")
    print(f"Wrote compact summary to: {out_summary_path}")


if __name__ == "__main__":
    main()


