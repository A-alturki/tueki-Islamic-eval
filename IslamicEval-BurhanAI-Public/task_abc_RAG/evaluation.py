import argparse
import csv
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class Annotation:
    question_id: str
    annotation_id: str
    label_raw: str
    span_start: int
    span_end: int
    original_span: str
    correction: str

    @property
    def length(self) -> int:
        return max(0, self.span_end - self.span_start)

    @property
    def type_(self) -> Optional[str]:
        s = self.label_raw.lower()
        if "ayah" in s:
            return "Ayah"
        if "hadith" in s or "hadeeth" in s:
            return "Hadith"
        return None

    @property
    def correctness(self) -> Optional[str]:
        s = self.label_raw.lower()
        if "correct" in s:
            return "Correct"
        if "wrong" in s:
            return "Wrong"
        return None


def parse_xml_questions(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    blocks = re.findall(r"<Question>.*?</Question>", raw, flags=re.DOTALL)
    out: Dict[str, str] = {}
    import xml.etree.ElementTree as ET
    for block in blocks:
        try:
            q = ET.fromstring(block)
        except ET.ParseError:
            continue
        qid = q.findtext("ID")
        resp = q.findtext("Response") or ""
        if qid:
            out[qid] = resp
    return out


def load_tsv(path: str) -> List[Annotation]:
    rows: List[Annotation] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            rows.append(
                Annotation(
                    question_id=r["Question_ID"],
                    annotation_id=str(r["Annotation_ID"]),
                    label_raw=r.get("Label", ""),
                    span_start=int(r.get("Span_Start", 0) or 0),
                    span_end=int(r.get("Span_End", 0) or 0),
                    original_span=r.get("Original_Span", "") or "",
                    correction=r.get("Correction", "") or "",
                )
            )
    return rows


def group_by_question(rows: List[Annotation]) -> Dict[str, List[Annotation]]:
    byq: Dict[str, List[Annotation]] = {}
    for a in rows:
        byq.setdefault(a.question_id, []).append(a)
    return byq


def guess_input_from_gt(gt_path: str) -> Optional[str]:
    base = os.path.basename(gt_path)
    if base == "TaskABC_GT.tsv":
        return os.path.join("datasets", "TaskABC_Input.xml")
    if base == "TaskA_GT.tsv":
        return os.path.join("datasets", "TaskA_Input.xml")
    if base == "TaskB_GT.tsv":
        return os.path.join("datasets", "TaskB_Input.xml")
    if base == "TaskC_GT.tsv":
        return os.path.join("datasets", "TaskC_Input.xml")
    return None


def build_char_labels(length: int, spans: List[Tuple[int, int, str]]) -> List[int]:
    # 0 Neither, 1 Ayah, 2 Hadith
    arr = [0] * max(0, length)
    for s, e, t in spans:
        if s is None or e is None:
            continue
        s = max(0, s)
        e = min(length, e)
        if e <= s:
            continue
        code = 1 if t == "Ayah" else 2 if t == "Hadith" else 0
        if code == 0:
            continue
        for i in range(s, e):
            arr[i] = code
    return arr


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    if tp == 0 and (fp > 0 or fn > 0):
        return 0.0
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return (2 * p * r / (p + r)) if (p + r) else 0.0


def compute_char_macro_f1(
    gt_by_q: Dict[str, List[Annotation]],
    pred_by_q: Dict[str, List[Annotation]],
    texts: Dict[str, str],
) -> Tuple[float, Dict[str, Dict[str, int]]]:
    # Aggregate counts across all questions for each class
    # classes: 0 Neither, 1 Ayah, 2 Hadith
    counts = {
        0: {"tp": 0, "fp": 0, "fn": 0},
        1: {"tp": 0, "fp": 0, "fn": 0},
        2: {"tp": 0, "fp": 0, "fn": 0},
    }
    for qid, text in texts.items():
        n = len(text or "")
        gt_spans = [
            (a.span_start, a.span_end, a.type_)
            for a in gt_by_q.get(qid, [])
            if a.type_ in ("Ayah", "Hadith")
        ]
        pr_spans = [
            (a.span_start, a.span_end, a.type_)
            for a in pred_by_q.get(qid, [])
            if a.type_ in ("Ayah", "Hadith")
        ]
        gt_arr = build_char_labels(n, gt_spans)
        pr_arr = build_char_labels(n, pr_spans)
        for i in range(n):
            g = gt_arr[i]
            p = pr_arr[i]
            for cls in (0, 1, 2):
                if p == cls and g == cls:
                    counts[cls]["tp"] += 1
                elif p == cls and g != cls:
                    counts[cls]["fp"] += 1
                elif p != cls and g == cls:
                    counts[cls]["fn"] += 1

    f1_neither = f1_from_counts(**counts[0])
    f1_ayah = f1_from_counts(**counts[1])
    f1_hadith = f1_from_counts(**counts[2])
    macro = (f1_neither + f1_ayah + f1_hadith) / 3.0
    pretty_counts = {
        "Neither": counts[0],
        "Ayah": counts[1],
        "Hadith": counts[2],
    }
    return macro, pretty_counts


def iou(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    s1, e1 = a
    s2, e2 = b
    inter = max(0, min(e1, e2) - max(s1, s2))
    if inter <= 0:
        return 0.0
    len1 = max(0, e1 - s1)
    len2 = max(0, e2 - s2)
    return inter / (len1 + len2 - inter) if (len1 + len2 - inter) > 0 else 0.0


def greedy_match(
    gt: List[Annotation],
    pr: List[Annotation],
    iou_threshold: float,
) -> Dict[str, Optional[Annotation]]:
    pairs: List[Tuple[float, Annotation, Annotation]] = []
    for g in gt:
        for p in pr:
            ov = iou((g.span_start, g.span_end), (p.span_start, p.span_end))
            if ov >= iou_threshold:
                pairs.append((ov, g, p))
    pairs.sort(key=lambda t: t[0], reverse=True)
    matched_gt: Dict[str, Optional[Annotation]] = {g.annotation_id: None for g in gt}
    used_pred: set = set()
    used_gt: set = set()
    for ov, g, p in pairs:
        if g.annotation_id in used_gt:
            continue
        if (g.question_id, p.annotation_id) in used_pred:
            continue
        matched_gt[g.annotation_id] = p
        used_gt.add(g.annotation_id)
        used_pred.add((g.question_id, p.annotation_id))
    return matched_gt


def compute_1b_accuracy(
    gt_by_q: Dict[str, List[Annotation]],
    pred_by_q: Dict[str, List[Annotation]],
    iou_threshold: float = 0.5,
) -> Tuple[float, int, int]:
    correct = 0
    total = 0
    for qid, gts in gt_by_q.items():
        preds = pred_by_q.get(qid, [])
        matched = greedy_match(gts, preds, iou_threshold)
        for g in gts:
            total += 1
            p = matched.get(g.annotation_id)
            if not p:
                continue
            gc = g.correctness
            pc = p.correctness
            if gc is None or pc is None:
                continue
            if gc == pc:
                correct += 1
    acc = correct / total if total else 0.0
    return acc, correct, total


def normalize_text_for_exact(s: str) -> str:
    return (s or "").strip()


def compute_1c_accuracy(
    gt_by_q: Dict[str, List[Annotation]],
    pred_by_q: Dict[str, List[Annotation]],
    iou_threshold: float = 0.5,
) -> Tuple[float, int, int]:
    correct = 0
    total = 0
    for qid, gts in gt_by_q.items():
        preds = pred_by_q.get(qid, [])
        matched = greedy_match(gts, preds, iou_threshold)
        for g in gts:
            if g.correctness != "Wrong":
                continue
            total += 1
            p = matched.get(g.annotation_id)
            if not p:
                continue
            if normalize_text_for_exact(p.correction) == normalize_text_for_exact(g.correction):
                correct += 1
    acc = correct / total if total else 0.0
    return acc, correct, total


def resolve_defaults(task: str) -> Tuple[str, str, str]:
    task = task.upper()
    if task == "ABC":
        return (
            os.path.join("outputs", "TaskABC_submission.tsv"),
            os.path.join("datasets", "TaskABC_GT.tsv"),
            os.path.join("datasets", "TaskABC_Input.xml"),
        )
    if task == "ABC_HIGH":
        return (
            os.path.join("outputs", "TaskABC_partial_submission_high_reason.tsv"),
            os.path.join("datasets", "TaskABC_GT.tsv"),
            os.path.join("datasets", "TaskABC_Input.xml"),
        )
    if task == "ABC_LOW":
        return (
            os.path.join("outputs", "TaskABC_partial_submission_low_reason.tsv"),
            os.path.join("datasets", "TaskABC_GT.tsv"),
            os.path.join("datasets", "TaskABC_Input.xml"),
        )
    if task == "ABC_MEDIUM":
        return (
            os.path.join("outputs", "TaskABC_partial_submission_medium_reason.tsv"),
            os.path.join("datasets", "TaskABC_GT.tsv"),
            os.path.join("datasets", "TaskABC_Input.xml"),
        )
    if task == "A":
        return (
            os.path.join("outputs", "TaskA_submission.tsv"),
            os.path.join("datasets", "TaskA_GT.tsv"),
            os.path.join("datasets", "TaskA_Input.xml"),
        )
    if task == "B":
        return (
            os.path.join("outputs", "TaskB_submission.tsv"),
            os.path.join("datasets", "TaskB_GT.tsv"),
            os.path.join("datasets", "TaskB_Input.xml"),
        )
    if task == "C":
        return (
            os.path.join("outputs", "TaskC_submission.tsv"),
            os.path.join("datasets", "TaskC_GT.tsv"),
            os.path.join("datasets", "TaskC_Input.xml"),
        )
    raise ValueError(f"Unknown task: {task}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate IslamicEval Subtasks 1A, 1B, 1C and combined 1ABC."
    )
    parser.add_argument(
        "--task", type=str, default="ABC", choices=["A", "B", "C", "ABC", "ABC_HIGH", "ABC_MEDIUM", "ABC_LOW"],
        help="Which dataset to evaluate; defaults to ABC combined.",
    )
    parser.add_argument("--submission", type=str, default=None, help="Path to submission TSV.")
    parser.add_argument("--gt", type=str, default=None, help="Path to GT TSV.")
    parser.add_argument(
        "--input", type=str, default=None, help="Path to Input XML for the same split."
    )
    parser.add_argument(
        "--iou-threshold", type=float, default=0.5, help="IoU threshold for span matching."
    )
    parser.add_argument(
        "--restrict-to-submission", action="store_true",
        help="Restrict evaluation to the set of Question_IDs present in the submission.",
    )
    args = parser.parse_args()

    default_sub, default_gt, default_input = resolve_defaults(args.task)
    sub_path = args.submission or default_sub
    gt_path = args.gt or default_gt
    input_path = args.input or guess_input_from_gt(gt_path) or default_input

    if not os.path.exists(sub_path):
        raise SystemExit(f"Submission file not found: {sub_path}")
    if not os.path.exists(gt_path):
        raise SystemExit(f"GT file not found: {gt_path}")
    if not os.path.exists(input_path):
        raise SystemExit(f"Input XML file not found: {input_path}")

    pred_rows = load_tsv(sub_path)
    gt_rows = load_tsv(gt_path)
    pred_by_q = group_by_question(pred_rows)
    gt_by_q = group_by_question(gt_rows)
    texts = parse_xml_questions(input_path)

    # Auto-restrict if evaluating partial submissions or explicitly requested
    auto_restrict = any(x in os.path.basename(sub_path).lower() for x in ["partial", "high_reason", "medium_reason", "low_reason"]) or args.task in {"ABC_HIGH", "ABC_MEDIUM", "ABC_LOW"}
    if args.restrict_to_submission or auto_restrict:
        keep_qids = set(pred_by_q.keys())
        gt_by_q = {k: v for k, v in gt_by_q.items() if k in keep_qids}
        pred_by_q = {k: v for k, v in pred_by_q.items() if k in keep_qids}
        texts = {k: v for k, v in texts.items() if k in keep_qids}

    macro_f1, counts = compute_char_macro_f1(gt_by_q, pred_by_q, texts)
    acc_1b, correct_1b, total_1b = compute_1b_accuracy(
        gt_by_q, pred_by_q, iou_threshold=args.iou_threshold
    )
    acc_1c, correct_1c, total_1c = compute_1c_accuracy(
        gt_by_q, pred_by_q, iou_threshold=args.iou_threshold
    )

    print(f"Evaluated task: {args.task}")
    print(f"1A (span) Macro F1 (char-level over {{Neither, Ayah, Hadith}}): {macro_f1:.6f}")
    print("  Counts by class (tp/fp/fn):")
    for cls_name, c in counts.items():
        print(f"    {cls_name}: tp={c['tp']} fp={c['fp']} fn={c['fn']}")
    print("1B (validation) Accuracy: {:.6f}  ({}/{} correct)".format(acc_1b, correct_1b, total_1b))
    print(
        "1C (correction) Accuracy: {:.6f}  ({}/{} exact matches)".format(
            acc_1c, correct_1c, total_1c
        )
    )
    print(f"Span match IoU threshold: {args.iou_threshold}")


if __name__ == "__main__":
    main()
