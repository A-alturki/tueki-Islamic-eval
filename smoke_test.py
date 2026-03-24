#!/usr/bin/env python3
"""
smoke_test.py
=============
Minimal connectivity and sanity test for the zero-shot evaluation pipeline.

Checks:
  1. Which API keys are present in the environment.
  2. Sends one short Arabic test message to each available API to verify
     that the connection works and gets a response.
  3. Runs ONE sample from the Subtask 1A dev set through gpt-4.1-mini
     (if OPENAI_API_KEY is set) and prints the raw response + parsed spans
     so you can visually verify correctness.

Usage:
  python smoke_test.py
"""

import os
import re
import json
import time
import sys
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — mirrors MODEL_REGISTRY in zero_shot_baselines.py
# ─────────────────────────────────────────────────────────────────────────────

# (provider, model_id, env_var_name)
API_CONFIGS = {
    "OpenAI":     ("openai",    "gpt-4.1-mini",             "OPENAI_API_KEY"),
    "Anthropic":  ("anthropic", "claude-sonnet-4-20250514",  "ANTHROPIC_API_KEY"),
    "Gemini":     ("gemini",    "gemini-2.5-flash",          "GEMINI_API_KEY"),
    "Together":   ("together",  "Qwen/Qwen3-32B",            "TOGETHER_API_KEY"),
}

# Short Arabic test message used for connectivity checks
ARABIC_TEST_MSG = "ما هو أول ما نزل من القرآن الكريم؟"  # "What was the first revelation?"

# Path helpers (same as zero_shot_baselines.py)
REPO_ROOT   = Path(__file__).parent
DATASETS_DIR = REPO_ROOT / "datasets"
DEV_XML_1A  = DATASETS_DIR / "dev_SubtaskA.xml"
DEV_TSV_1A  = DATASETS_DIR / "dev_SubtaskA.tsv"


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — CHECK API KEYS
# ─────────────────────────────────────────────────────────────────────────────

def check_api_keys() -> dict:
    """Print which API keys are set and return a dict provider → bool."""
    print("=" * 60)
    print("STEP 1: API key availability")
    print("=" * 60)

    availability = {}
    for name, (provider, model_id, env_var) in API_CONFIGS.items():
        is_set = bool(os.environ.get(env_var))
        status = "✓ SET" if is_set else "✗ NOT SET"
        print(f"  {name:12s} ({env_var:25s})  →  {status}")
        availability[provider] = is_set

    print()
    return availability


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — CONNECTIVITY TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_openai_connectivity(model_id: str = "gpt-4.1-mini") -> bool:
    """Send one message to OpenAI and print the response."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user",   "content": ARABIC_TEST_MSG},
            ],
            max_tokens=200,
            temperature=0.1,
        )
        text = response.choices[0].message.content
        print(f"  [OpenAI/{model_id}] OK — response excerpt:\n    {text[:120]}")
        return True
    except Exception as exc:
        print(f"  [OpenAI] FAILED: {exc}")
        return False


def test_anthropic_connectivity(model_id: str = "claude-sonnet-4-20250514") -> bool:
    """Send one message to Anthropic and print the response."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        response = client.messages.create(
            model=model_id,
            max_tokens=200,
            system="You are a helpful assistant.",
            messages=[{"role": "user", "content": ARABIC_TEST_MSG}],
            temperature=0.1,
        )
        text = response.content[0].text
        print(f"  [Anthropic/{model_id}] OK — response excerpt:\n    {text[:120]}")
        return True
    except Exception as exc:
        print(f"  [Anthropic] FAILED: {exc}")
        return False


def test_gemini_connectivity(model_id: str = "gemini-2.5-flash") -> bool:
    """Send one message to Gemini and print the response."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        model = genai.GenerativeModel(
            model_name=model_id,
            system_instruction="You are a helpful assistant.",
        )
        response = model.generate_content(
            ARABIC_TEST_MSG,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1, max_output_tokens=200
            ),
        )
        text = response.text
        print(f"  [Gemini/{model_id}] OK — response excerpt:\n    {text[:120]}")
        return True
    except Exception as exc:
        print(f"  [Gemini] FAILED: {exc}")
        return False


def test_together_connectivity(model_id: str = "Qwen/Qwen3-32B") -> bool:
    """Send one message to Together AI and print the response."""
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=os.environ["TOGETHER_API_KEY"],
            base_url="https://api.together.xyz/v1",
        )
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user",   "content": ARABIC_TEST_MSG},
            ],
            max_tokens=200,
            temperature=0.1,
        )
        text = response.choices[0].message.content
        print(f"  [Together/{model_id}] OK — response excerpt:\n    {text[:120]}")
        return True
    except Exception as exc:
        print(f"  [Together] FAILED: {exc}")
        return False


def test_connectivity(availability: dict) -> None:
    """Run connectivity tests for all providers with available API keys."""
    print("=" * 60)
    print("STEP 2: API connectivity tests")
    print("=" * 60)

    dispatch = {
        "openai":    test_openai_connectivity,
        "anthropic": test_anthropic_connectivity,
        "gemini":    test_gemini_connectivity,
        "together":  test_together_connectivity,
    }

    for provider, is_available in availability.items():
        if not is_available:
            print(f"  [{provider}] SKIPPED (no API key)")
            continue
        print(f"\n  Testing {provider}…")
        success = dispatch[provider]()
        # Short sleep to respect rate limits even during smoke testing
        time.sleep(2)

    print()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — ONE SAMPLE THROUGH SUBTASK 1A (gpt-4.1-mini)
# ─────────────────────────────────────────────────────────────────────────────

def load_first_sample_1a():
    """Load the first question from dev_SubtaskA.xml and its gold spans.

    Returns:
        (question_id, response_text, gold_spans)
        or (None, None, None) if files are missing.
    """
    if not DEV_XML_1A.exists() or not DEV_TSV_1A.exists():
        print(f"  [WARN] Dev files not found:\n    {DEV_XML_1A}\n    {DEV_TSV_1A}")
        return None, None, None

    # Parse XML (regex approach, same as span_checker.py — no proper root element)
    with open(DEV_XML_1A, "r", encoding="utf-8") as fh:
        content = fh.read()

    questions = {}
    for block in re.findall(r"<Question>(.*?)</Question>", content, re.DOTALL):
        id_m  = re.search(r"<ID>(.*?)</ID>",           block, re.DOTALL)
        res_m = re.search(r"<Response>(.*?)</Response>", block, re.DOTALL)
        if id_m and res_m:
            questions[id_m.group(1).strip()] = res_m.group(1).strip()

    if not questions:
        print("  [WARN] No questions found in XML.")
        return None, None, None

    # Take the first question
    first_qid  = next(iter(questions))
    first_resp = questions[first_qid]

    # Load gold spans for this question from the TSV
    import pandas as pd
    gold_df = pd.read_csv(DEV_TSV_1A, sep="\t")
    qid_gold = gold_df[gold_df["Question_ID"] == first_qid]

    gold_spans = []
    for _, row in qid_gold.iterrows():
        gold_spans.append({
            "type":  "q" if str(row["Label"]).lower() == "ayah" else "h",
            "start": int(row["Span_Start"]),
            "end":   int(row["Span_End"]),  # exclusive
            "text":  first_resp[int(row["Span_Start"]):int(row["Span_End"])],
        })

    return first_qid, first_resp, gold_spans


def fix_span_indices_smoke(spans, response_text):
    """Simplified version of fix_span_indices for the smoke test."""
    fixed = []
    for span in spans:
        text  = span.get("text", "")
        start = span.get("start", 0)
        end   = span.get("end", 0)
        stype = span.get("type", "q")

        if not text:
            continue

        # Check if model's inclusive-end indexing is correct
        if response_text[start : end + 1] == text:
            fixed.append({"type": stype, "text": text,
                          "start": start, "end": end + 1})
            continue

        # Search for the text and pick the nearest occurrence
        pos, occurrences = 0, []
        while True:
            idx = response_text.find(text, pos)
            if idx == -1:
                break
            occurrences.append(idx)
            pos = idx + 1

        if not occurrences:
            print(f"    [WARN] Could not locate span text: '{text[:40]}'")
            continue
        best = min(occurrences, key=lambda i: abs(i - start))
        fixed.append({"type": stype, "text": text,
                      "start": best, "end": best + len(text)})

    return fixed


def test_1a_single_sample() -> None:
    """Run the first dev sample through Subtask 1A with gpt-4.1-mini.

    Prints the raw model response and the parsed spans so you can visually
    verify that span detection is working.
    """
    print("=" * 60)
    print("STEP 3: Subtask 1A single-sample test (gpt-4.1-mini)")
    print("=" * 60)

    if not os.environ.get("OPENAI_API_KEY"):
        print("  SKIPPED — OPENAI_API_KEY not set.")
        return

    qid, response_text, gold_spans = load_first_sample_1a()
    if qid is None:
        return

    print(f"\n  Question ID:  {qid}")
    print(f"  Response text ({len(response_text)} chars):\n")
    # Print first 500 chars of the response
    print("    " + response_text[:500].replace("\n", "\n    "))
    if len(response_text) > 500:
        print(f"    … [{len(response_text) - 500} more chars]")

    print(f"\n  Gold spans ({len(gold_spans)}):")
    for s in gold_spans:
        type_label = "Ayah" if s["type"] == "q" else "Hadith"
        print(f"    [{type_label}] chars {s['start']}–{s['end']}: \"{s['text'][:60]}\"")

    # Build the 1A prompt (same as in zero_shot_baselines.py)
    system_msg = (
        "You are an expert in Islamic texts. "
        "You can identify quotations from the Holy Quran and Prophetic Hadith "
        "(الأحاديث النبوية) within Arabic text."
    )
    user_prompt = f"""\
Analyze the following Arabic text and identify all spans that are quotations from the Quran (Ayah) or Hadith.

For each span, provide:
- "type": "q" for Quran or "h" for Hadith
- "text": the exact quoted text as it appears in the input (copy it exactly, character for character)
- "start": the character index where the span starts (0-indexed) in the original text
- "end": the character index where the span ends (0-indexed, inclusive)

Rules:
- Only include the actual quoted religious text, not introductory phrases like "قال الله تعالى" or "قال رسول الله صلى الله عليه وسلم"
- Include both correctly quoted and incorrectly quoted spans
- Be precise with start/end character indices
- If there are no quotations, return an empty spans array

Return ONLY valid JSON, no other text:
{{"spans": [{{"type": "q", "text": "...", "start": 0, "end": 10}}, ...]}}

Text:
<<<
{response_text}
>>>"""

    print("\n  Calling gpt-4.1-mini…")
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=4096,
        )
        raw = response.choices[0].message.content
    except Exception as exc:
        print(f"  [ERROR] API call failed: {exc}")
        return

    print("\n  ── RAW RESPONSE ──────────────────────────────────")
    print(raw[:2000])
    if len(raw) > 2000:
        print(f"  … [{len(raw) - 2000} more chars]")

    # Parse the JSON
    text_clean = re.sub(r"```(?:json)?\s*", "", raw).strip()
    text_clean = re.sub(r"```\s*$", "", text_clean).strip()
    parsed = None
    try:
        parsed = json.loads(text_clean)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text_clean, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
            except json.JSONDecodeError:
                pass

    if parsed is None:
        print("\n  [ERROR] Could not parse JSON from response.")
        return

    raw_spans = parsed.get("spans", [])
    print(f"\n  ── PARSED SPANS (before index fix): {len(raw_spans)} spans ──")
    for j, s in enumerate(raw_spans, 1):
        print(f"  {j}. type={s.get('type')} start={s.get('start')} "
              f"end={s.get('end')}  text=\"{str(s.get('text',''))[:60]}\"")

    # Fix character indices
    fixed_spans = fix_span_indices_smoke(raw_spans, response_text)
    print(f"\n  ── FIXED SPANS (after index fix): {len(fixed_spans)} spans ──")
    for j, s in enumerate(fixed_spans, 1):
        type_label = "Ayah" if s["type"] == "q" else "Hadith"
        verified = (response_text[s["start"]:s["end"]] == s["text"])
        print(f"  {j}. [{type_label}] chars {s['start']}–{s['end']}  "
              f"verified={verified}  \"{s['text'][:60]}\"")

    print("\n  ── COMPARISON WITH GOLD ──")
    print(f"  Gold:  {len(gold_spans)} spans")
    print(f"  Model: {len(fixed_spans)} spans")
    print(f"  (Visual inspection recommended)")

    print("\nSmoke test complete.")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("IslamicEval 2025 — Zero-Shot Pipeline Smoke Test")
    print("=" * 60 + "\n")

    # Step 1: check API keys
    availability = check_api_keys()

    # Step 2: test connectivity for each available provider
    test_connectivity(availability)

    # Step 3: run one 1A sample with gpt-4.1-mini
    test_1a_single_sample()


if __name__ == "__main__":
    main()
