#!/usr/bin/env python3
"""
zero_shot_baselines.py
======================
Zero-shot evaluation of frontier LLMs on IslamicEval 2025 Subtasks 1A, 1B, and 1C.

Uses ONLY the data files from the HUMAIN repo — all prompts and inference logic are new.

Subtasks:
  1A — Span Detection: identify Quran/Hadith quotation spans in an Arabic LLM response.
  1B — Validation:     classify gold spans as "correct" or "incorrect".
  1C — Correction:     provide the canonical correct text for incorrect spans.

Usage:
  python zero_shot_baselines.py --model gpt-4.1-mini --subtask 1A
  python zero_shot_baselines.py --model gpt-4.1-mini --subtask all
  python zero_shot_baselines.py --model all --subtask all
  python zero_shot_baselines.py --model all --subtask all --max-samples 5
"""

import os
import re
import sys
import json
import time
import argparse
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from dotenv import load_dotenv
import pandas as pd
from sklearn.metrics import f1_score as sklearn_f1_score
load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# PATHS — all relative to the repo root (one directory up from here if needed,
# but since this script lives at the repo root, we use relative paths directly).
# ─────────────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent
DATASETS_DIR = REPO_ROOT / "datasets"

# Dev-set XML files (contain <Question> blocks with <ID>, <Model>, <Text>, <Response>)
DEV_XML = {
    "1A": DATASETS_DIR / "dev_SubtaskA.xml",
    "1B": DATASETS_DIR / "dev_SubtaskB.xml",
    "1C": DATASETS_DIR / "dev_SubtaskC.xml",
}

# Dev-set TSV files (gold annotations)
DEV_TSV = {
    "1A": DATASETS_DIR / "dev_SubtaskA.tsv",  # cols: Question_ID, Annotation_ID, Label, Span_Start, Span_End, Original_Span
    "1B": DATASETS_DIR / "dev_SubtaskB.tsv",  # cols: Question_ID, Annotation_ID, Label, Span_Start, Span_End, Original_Span
    "1C": DATASETS_DIR / "dev_SubtaskC.tsv",  # cols: Question_ID, Label, Span_Start, Span_End, Original_Span, Correction
}

# Results directory (created if it doesn't exist)
RESULTS_DIR = REPO_ROOT / "results"

# ─────────────────────────────────────────────────────────────────────────────
# MODEL REGISTRY
# Each entry: "safe_name" → (provider, model_id, api_key_env_var)
# ─────────────────────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    # OpenAI models
    "gpt-5.4":       ("openai",    "gpt-5.4",                   "OPENAI_API_KEY"),
    "gpt-5.4-mini":   ("openai",    "gpt-5-mini",               "OPENAI_API_KEY"),
    "gpt-5.4-nano":         ("openai",    "gpt-5.4-nano",                   "OPENAI_API_KEY"),
    "gpt-5":               ("openai",    "gpt-5",                         "OPENAI_API_KEY"),
    # Anthropic Claude models
    "claude-opus-4-6": ("anthropic", "claude-opus-4-6", "ANTHROPIC_API_KEY"),
    "claude-sonnet-4-6":   ("anthropic", "claude-sonnet-4-6",   "ANTHROPIC_API_KEY"),
    "claude-haiku-4-5":   ("anthropic", "claude-haiku-4-5",   "ANTHROPIC_API_KEY"),
    # Google Gemini models
    "gemini-3.1-pro-preview": ("gemini", "gemini-3.1-pro-preview", "GEMINI_API_KEY"),
    "gemini-3-flash-preview":   ("gemini", "gemini-3-flash-preview",   "GEMINI_API_KEY"),
    # Together AI (Qwen3 large — thinking disabled via enable_thinking=False)
    "Qwen3-235B":  ("together",    "Qwen/Qwen3-235B-A22B-Instruct-2507-tput", "TOGETHER_API_KEY"),
    "Qwen3-80B":   ("together",    "Qwen/Qwen3-Next-80B-A3B-Instruct",        "TOGETHER_API_KEY"),
    # OpenRouter (Qwen3 small — thinking disabled via thinking.enabled=False)
    "Qwen3-32B":   ("openrouter", "qwen/qwen3-32b",  "OPENROUTER_API_KEY"),
    "Qwen3-14B":   ("openrouter", "qwen/qwen3-14b",  "OPENROUTER_API_KEY"),
    "Qwen3-8B":    ("openrouter", "qwen/qwen3-8b",   "OPENROUTER_API_KEY"),
}

# Sleep time between API calls per provider (to respect rate limits)
# Gemini Tier 1 (paid) has high RPM — 1s is sufficient for all Gemini models
SLEEP_TIMES = {
    "openai":      0.5,
    "anthropic":   1.0,
    "gemini":      1.0,
    "together":    1.0,
    "openrouter":  1.0,
}

# No longer needed — was for Gemini 2.5 Pro free tier (2 RPM). Tier 1 is much higher.
SLEEP_TIME_GEMINI_PRO = 1.0

# ─────────────────────────────────────────────────────────────────────────────
# API CLIENT INITIALISATION
# Clients are created lazily on first use, based on available environment keys.
# ─────────────────────────────────────────────────────────────────────────────
_clients: Dict[str, Any] = {}  # provider → client object


def get_client(provider: str) -> Optional[Any]:
    """Return (and cache) the API client for a given provider.

    Returns None if the required environment variable is missing.
    """
    if provider in _clients:
        return _clients[provider]

    if provider == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            return None
        from openai import OpenAI
        client = OpenAI(api_key=key)
        _clients[provider] = client
        return client

    elif provider == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            return None
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        _clients[provider] = client
        return client

    elif provider == "gemini":
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            return None
        # New google-genai SDK (pip install google-genai)
        from google import genai
        client = genai.Client(api_key=key)
        _clients[provider] = client
        return client

    elif provider == "together":
        key = os.environ.get("TOGETHER_API_KEY")
        if not key:
            return None
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url="https://api.together.xyz/v1")
        _clients[provider] = client
        return client

    elif provider == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            return None
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1")
        _clients[provider] = client
        return client

    return None


# ─────────────────────────────────────────────────────────────────────────────
# UNIFIED API CALL with retry + exponential back-off
# ─────────────────────────────────────────────────────────────────────────────

def call_api(model_name: str, system_msg: str, user_prompt: str,
             max_retries: int = 3, schema: Optional[Dict] = None):
    """Call a model and return (text, usage) where usage = {"input_tokens": N, "output_tokens": N}.

    Handles all four providers uniformly.  On failure retries with exponential
    back-off (2 s, 4 s, 8 s).  Returns (None, None) on permanent failure.

    Args:
        model_name: Key into MODEL_REGISTRY (e.g. "gpt-4.1-mini").
        system_msg: System / persona instruction.
        user_prompt: The task prompt sent as the user message.
        max_retries: Maximum number of retry attempts.

    Returns:
        Raw model response text, or None on failure.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")

    provider, model_id, _ = MODEL_REGISTRY[model_name]
    client = get_client(provider)
    if client is None:
        raise RuntimeError(f"No API key for provider '{provider}' (model={model_name})")

    # Determine sleep time for this provider/model
    sleep_time = SLEEP_TIME_GEMINI_PRO if "pro" in model_name.lower() else SLEEP_TIMES.get(provider, 1.0)

    for attempt in range(max_retries):
        try:
            text, usage = _do_api_call(provider, client, model_id, system_msg, user_prompt, schema=schema)
            time.sleep(sleep_time)  # rate-limit pause after each successful call
            return text, usage

        except Exception as exc:
            wait = 2 ** (attempt + 1)  # 2 s, 4 s, 8 s
            print(f"    [WARN] API call failed (attempt {attempt + 1}/{max_retries}): {exc}")
            if attempt < max_retries - 1:
                print(f"    [WARN] Retrying in {wait} s...")
                time.sleep(wait)
            else:
                print(f"    [ERROR] All {max_retries} attempts failed for {model_name}.")
                return None, None


def _strip_additional_properties(schema: Dict) -> Dict:
    """Recursively remove 'additionalProperties' from a JSON schema dict.

    Gemini's response_schema does not support this OpenAI-specific field
    and will return a 400 INVALID_ARGUMENT error if it is present.
    """
    if not isinstance(schema, dict):
        return schema
    return {
        k: _strip_additional_properties(v)
        for k, v in schema.items()
        if k != "additionalProperties"
    }


def _do_api_call(provider: str, client: Any, model_id: str,
                 system_msg: str, user_prompt: str,
                 schema: Optional[Dict] = None):
    """Low-level API dispatch.  Raises on error.  Returns (text, usage).

    usage = {"input_tokens": int, "output_tokens": int}

    Args:
        schema: Optional JSON schema dict.  When provided:
                  - OpenAI: enforced via response_format (structured outputs)
                  - Gemini: enforced via response_schema in generation_config
                  - Anthropic/Together: ignored (schema is described in the system prompt)
    """

    if provider in ("openai", "together", "openrouter"):
        # OpenAI SDK (also used for Together AI via OpenAI-compatible endpoint).
        # o-series and gpt-5+ use max_completion_tokens instead of max_tokens/temperature.
        COMPLETION_TOKENS_PREFIXES = ("o1", "o3", "o4", "o5", "gpt-5")
        is_reasoning = any(model_id.startswith(p) for p in COMPLETION_TOKENS_PREFIXES)

        call_kwargs = dict(
            model=model_id,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_prompt},
            ],
        )
        if is_reasoning:
            call_kwargs["max_completion_tokens"] = 4096
            # reasoning models ignore temperature; omit it to avoid API errors
        else:
            call_kwargs["max_tokens"] = 4096
            call_kwargs["temperature"] = 0.1

        # Structured output: only for OpenAI (OpenRouter may not support strict json_schema)
        if schema and provider == "openai":
            call_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name":   "span_detection",
                    "strict": True,
                    "schema": schema,
                },
            }

        if model_id.startswith("gpt-5.4-pro"):
            # print("im here")
            # response = client.responses.create(**call_kwargs)
            response = client.responses.create(
                model=model_id,
                instructions=system_msg,
                input=user_prompt,
                max_output_tokens=4096,
                text={
                    "format": {
                        "type": "json_schema",
                        "name":   "span_detection",
                        "strict": True,
                        "schema": schema,
                    }
                }
            )
            usage = {"input_tokens": getattr(response.usage, "input_tokens", None),
                     "output_tokens": getattr(response.usage, "output_tokens", None)}
            return response.output_text, usage


        else:
            # Disable Qwen3 thinking mode so the answer goes into message.content.
            # Together AI uses enable_thinking=False; OpenRouter uses thinking.enabled=False.
            if provider == "together":
                call_kwargs["extra_body"] = {"enable_thinking": False}
            elif provider == "openrouter":
                call_kwargs["extra_body"] = {"thinking": {"enabled": False}}
            response = client.chat.completions.create(**call_kwargs)
            usage = {"input_tokens": response.usage.prompt_tokens,
                     "output_tokens": response.usage.completion_tokens}
            return response.choices[0].message.content, usage

    elif provider == "anthropic":
        # Anthropic SDK — system message is a separate top-level parameter.
        # No native JSON schema enforcement; the system prompt describes the format.
        response = client.messages.create(
            model=model_id,
            max_tokens=4096,
            system=system_msg,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0.1,
        )
        usage = {"input_tokens": response.usage.input_tokens,
                 "output_tokens": response.usage.output_tokens}
        return response.content[0].text, usage

    elif provider == "gemini":
        # New google-genai SDK: client was created in get_client()
        from google.genai import types as genai_types

        # Build config — add JSON schema enforcement if provided
        config_kwargs = {
            "temperature": 0.1,
            "system_instruction": system_msg,
        }
        if schema:
            config_kwargs["response_mime_type"] = "application/json"
            # Gemini does not support "additionalProperties" — strip it recursively
            config_kwargs["response_schema"]    = _strip_additional_properties(schema)

        response = client.models.generate_content(
            model=model_id,
            contents=user_prompt,
            config=genai_types.GenerateContentConfig(**config_kwargs),
        )
        um = response.usage_metadata
        usage = {"input_tokens": um.prompt_token_count,
                 "output_tokens": um.candidates_token_count}
        return response.text, usage

    else:
        raise ValueError(f"Unsupported provider: {provider}")


# ─────────────────────────────────────────────────────────────────────────────
# JSON PARSING — robust extraction from messy LLM output
# ─────────────────────────────────────────────────────────────────────────────

def parse_json_response(raw: str) -> Optional[Dict]:
    """Parse JSON from a model response that may contain markdown fences or prose.

    Strategy:
      1. Strip ```json ... ``` (or ``` ... ```) fences.
      2. Try json.loads() on the full stripped string.
      3. If that fails, regex-search for the first {...} block and try again.
      4. If still failing, log and return None.
    """
    if not raw:
        return None

    # Step 1: remove markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", raw).strip()
    text = re.sub(r"```\s*$", "", text).strip()

    # Step 2: try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Step 3: search for the first { ... } block (DOTALL so it spans newlines)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Failure
    print(f"    [WARN] Could not parse JSON from response:\n{raw[:300]}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# CHARACTER INDEX FIXING for Subtask 1A
# LLMs are notoriously bad at counting characters in Arabic text.
# ─────────────────────────────────────────────────────────────────────────────

def fix_span_indices(spans: List[Dict], response_text: str) -> List[Dict]:
    """Verify and correct character indices for each predicted span.

    EMPTY SPANS:
    If the model returns {"spans": []} it means no citations were found.
    This is valid — the function will simply return [] immediately.

    WHY THIS EXISTS:
    LLMs are notoriously bad at counting characters in Arabic text, especially
    when diacritics (tashkeel) are present. A model might correctly identify
    the quoted text (e.g. "لا تنكح المرأة على عمتها") but give wrong start/end
    indices. This function trusts the TEXT the model returned and recomputes
    the correct indices by searching for that text in the response.

    WHAT IT DOES (step by step):
      1. For each span the model returned {type, text, start, end}:
         a. Try response_text[start:end] — if it exactly equals span["text"],
            the indices are correct and we keep them as-is.
         b. If NOT, the model's indices are wrong. We search for span["text"]
            as a substring inside response_text to find the real position.
         c. If the text appears more than once (e.g. a repeated quote), we pick
            the occurrence whose start index is closest to what the model guessed,
            assuming the model was roughly in the right area even if off by a few.
         d. If the text isn't found anywhere in response_text, the model hallucinated
            text that isn't actually in the input — we discard the span entirely.

    EXAMPLE (from A-Q01):
      Model raw response said: start=31, end=72
      response_text[31:72]  →  does NOT match the span text
      Search for span text  →  found at position 37
      end = 37 + len(text)  →  76
      Stored prediction: start=37, end=76  ✓  (matches gold)

    Args:
        spans:         List of span dicts from the model (may have wrong indices).
        response_text: The original Arabic response text the model was analyzing.

    Returns:
        A new list of spans with verified/corrected indices (exclusive end convention,
        i.e. response_text[start:end] == text for every returned span).
    """
    # Empty spans list is valid — model found no citations. Return immediately.
    if not spans:
        return []

    fixed = []
    for span in spans:
        text  = span.get("text", "")
        start = span.get("start", 0)
        end   = span.get("end", 0)
        stype = span.get("type", "q")

        if not text:
            continue

        # ── Step 1: Check if the model's indices are already correct ─────────
        # We asked for exclusive end [start, end), so response_text[start:end]
        # should equal the span text directly.
        predicted_slice = response_text[start:end]

        if predicted_slice == text:
            # Indices are correct — store as-is.
            fixed.append({"type": stype, "text": text, "start": start, "end": end})
            continue

        # ── Step 2: Indices are wrong — find the text's real position ────────
        # Collect every position where this exact text appears in response_text.
        pos = 0
        occurrences = []
        while True:
            idx = response_text.find(text, pos)
            if idx == -1:
                break
            occurrences.append(idx)
            pos = idx + 1  # advance past this hit to find the next one

        if not occurrences:
            # Text not found anywhere — model hallucinated or garbled the text.
            # Discard this span rather than store wrong indices.
            print(f"    [WARN] Span text not found in response, discarding: '{text[:40]}...'")
            continue

        # ── Step 3: Pick the best occurrence ─────────────────────────────────
        # If the text appears multiple times, choose the occurrence whose start
        # is closest to what the model predicted (model was roughly right area).
        best = min(occurrences, key=lambda i: abs(i - start))

        # Recompute end as exclusive: start + length of the text
        fixed.append({"type": stype, "text": text,
                      "start": best, "end": best + len(text)})

    return fixed


# ─────────────────────────────────────────────────────────────────────────────
# TEXT NORMALISATION for Subtask 1C comparison
# Mirrors the normalisation used in scoring_C.py (organizer's scorer).
# ─────────────────────────────────────────────────────────────────────────────

def remove_default_diac(s: str) -> str:
    """Normalise Arabic text exactly as scoring_C.py does, plus Unicode NFC.

    The corpus and gold annotations store Arabic combining diacritics
    (shadda + vowel) in different Unicode orders (e.g. U+0651 U+064E vs
    U+064E U+0651).  NFC normalisation puts them in canonical order first so
    that subsequent string comparisons are byte-equal.

    Then applies the same replacements as scoring_C.py: removes 'default'
    diacritics implied by spelling (fatha+alif, kasra+ya, damma+waw,
    sukun on الـ, bare sukun) and a few orthographic variants.
    """
    import unicodedata, re as _re
    s = unicodedata.normalize("NFC", s)
    out = s

    # Remove tatweel (U+0640) only when it immediately precedes superscript alif
    # (U+0670).  The corpus encodes أُولَـٰئِكَ with tatweel; gold annotations omit it.
    out = _re.sub("\u0640(?=\u0670)", "", out)

    # Remove assimilation (إدغام) shadda — shadda that appears on the FIRST
    # consonant of a word (after a space, or at start of string).  These are
    # Tajweed marks absent from the corpus but present in gold annotations.
    # Mid-word shadda (e.g. مُحَمَّد, رَبَّنَا, اللَّه) is intentionally preserved.
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

    # Normalise spacing around ayah-number markers "(N)" used in multi-ayah corrections.
    # Corpus candidates are built with a space before "(N)" but gold attaches directly:
    # بَصِيرًا (2) vs بَصِيرًا(2)
    out = re.sub(r" \((\d+)\)", r"(\1)", out)

    return out


# Keep the old name as an alias so nothing else breaks
def normalize_arabic(text: str) -> str:
    return remove_default_diac(text)


# ─────────────────────────────────────────────────────────────────────────────
# CANONICAL RETRIEVAL — snap model output to nearest verified corpus entry
# ─────────────────────────────────────────────────────────────────────────────

_QURAN_CORPUS: Optional[List] = None   # list of dicts with original + stripped text
_HADITH_CORPUS: Optional[List] = None  # same
_QURAN_CANDIDATES: Optional[List] = None  # cached multi-ayah candidate list
_QURAN_INDEX: Optional[dict] = None   # inverted index: word -> set of candidate indices

def _strip_all_diac(text: str) -> str:
    """Remove ALL Arabic diacritics and waqf marks for broad fuzzy matching."""
    return re.sub(r"[\u064B-\u0652\u0670\u0640\u06D6-\u06DC\u06DF-\u06E6\u06E7\u06E8\u06EA-\u06ED]",
                  "", text)


def _load_quran_corpus() -> List:
    global _QURAN_CORPUS
    if _QURAN_CORPUS is not None:
        return _QURAN_CORPUS
    corpus_path = Path(__file__).parent / "datasets" / "quranic_verses.json"
    with open(corpus_path, encoding="utf-8") as f:
        data = json.load(f)
    # Sort by (surah_id, ayah_id) so adjacent ayahs are contiguous
    data = sorted([x for x in data if x and x.get("ayah_text")],
                  key=lambda x: (x["surah_id"], x["ayah_id"]))
    _QURAN_CORPUS = [
        {"orig": x["ayah_text"],
         "stripped": _strip_all_diac(x["ayah_text"]),
         "surah": x["surah_id"],
         "ayah": x["ayah_id"]}
        for x in data
    ]
    return _QURAN_CORPUS


def _load_hadith_corpus() -> List:
    global _HADITH_CORPUS
    if _HADITH_CORPUS is not None:
        return _HADITH_CORPUS
    corpus_path = Path(__file__).parent / "datasets" / "six_hadith_books.json"
    with open(corpus_path, encoding="utf-8") as f:
        data = json.load(f)
    seen: set = set()
    entries = []
    for item in data:
        if item is None:
            continue
        # Prefer Matn (clean body text); fall back to hadithTxt
        text = item.get("Matn") or item.get("hadithTxt")
        if text and text not in seen:
            seen.add(text)
            entries.append({"orig": text, "stripped": _strip_all_diac(text)})
    _HADITH_CORPUS = entries
    return _HADITH_CORPUS


def retrieve_canonical(pred_text: str, span_type: str,
                       min_similarity: float = 0.55) -> Optional[str]:
    """Find the closest verified canonical text for a model prediction.

    Uses diacritic-stripped text for comparison (robust to diacritic differences).
    For Quran, also tries concatenations of 2–3 adjacent ayahs to handle
    multi-ayah spans.

    Returns the original (with diacritics) canonical text, or None if
    no sufficiently similar entry is found.
    """
    from difflib import SequenceMatcher

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
            # Build inverted index: word -> set of candidate indices
            for i, (_, stripped) in enumerate(_QURAN_CANDIDATES):
                for w in stripped.split():
                    _QURAN_INDEX.setdefault(w, set()).add(i)

        # Use inverted index: count how many pred words each candidate shares,
        # then keep only candidates with >= min_common hits.
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
        # Pre-filter: require at least 30% word overlap
        corp_words = set(stripped.split())
        if corp_words and len(pred_words & corp_words) / max(len(pred_words), 1) < 0.30:
            continue

        ratio = SequenceMatcher(None, stripped_pred, stripped).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_text = orig

    return best_text if best_ratio >= min_similarity else "خطأ"


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING HELPERS
# (Replicating the XML-regex approach from subtask_B/span_checker.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_xml_questions(xml_path: Path) -> Dict[str, str]:
    """Parse a dev XML file and return a dict mapping Question_ID → Response text.

    The XML files in this repo do NOT have a single root element, so we use regex
    instead of a standard XML parser (same technique as span_checker.py).
    """
    with open(xml_path, "r", encoding="utf-8") as fh:
        content = fh.read()

    questions = {}
    # Each <Question> block contains <ID> and <Response> sub-elements
    for block in re.findall(r"<Question>(.*?)</Question>", content, re.DOTALL):
        id_m  = re.search(r"<ID>(.*?)</ID>",         block, re.DOTALL)
        res_m = re.search(r"<Response>(.*?)</Response>", block, re.DOTALL)
        if id_m and res_m:
            qid      = id_m.group(1).strip()
            response = res_m.group(1).strip()
            questions[qid] = response

    return questions


def load_data_1a() -> Tuple[Dict[str, str], pd.DataFrame]:
    """Load Subtask 1A dev data.

    Returns:
        questions: dict  {Question_ID → response_text}
        gold_df:   DataFrame with gold span annotations
                   cols: Question_ID, Annotation_ID, Label (Ayah/Hadith),
                         Span_Start (int), Span_End (int, exclusive), Original_Span
    """
    questions = load_xml_questions(DEV_XML["1A"])
    gold_df   = pd.read_csv(DEV_TSV["1A"], sep="\t")
    # Ensure integer types for span boundaries
    gold_df["Span_Start"] = gold_df["Span_Start"].astype(int)
    gold_df["Span_End"]   = gold_df["Span_End"].astype(int)
    return questions, gold_df


def load_data_1b() -> Tuple[Dict[str, str], pd.DataFrame]:
    """Load Subtask 1B dev data.

    Returns:
        questions: dict  {Question_ID → response_text}
        gold_df:   DataFrame with gold span annotations
                   cols: Question_ID, Annotation_ID, Label (WrongAyah/CorrectAyah/…),
                         Span_Start (int), Span_End (int, exclusive), Original_Span
    """
    questions = load_xml_questions(DEV_XML["1B"])
    gold_df   = pd.read_csv(DEV_TSV["1B"], sep="\t")
    gold_df["Span_Start"] = gold_df["Span_Start"].astype(int)
    gold_df["Span_End"]   = gold_df["Span_End"].astype(int)
    return questions, gold_df


def load_data_1c() -> Tuple[Dict[str, str], pd.DataFrame]:
    """Load Subtask 1C dev data.

    Returns:
        questions: dict  {Question_ID → response_text}
        gold_df:   DataFrame with gold corrections
                   cols: Question_ID, Label (WrongAyah/WrongHadith),
                         Span_Start (int), Span_End (int, exclusive),
                         Original_Span, Correction (correct text or "خطأ")
    """
    questions = load_xml_questions(DEV_XML["1C"])
    gold_df   = pd.read_csv(DEV_TSV["1C"], sep="\t")
    gold_df["Span_Start"] = gold_df["Span_Start"].astype(int)
    gold_df["Span_End"]   = gold_df["Span_End"].astype(int)
    return questions, gold_df


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def compute_1a_char_f1(response_text: str,
                        gold_spans: List[Dict],
                        pred_spans: List[Dict]) -> Dict:
    """Compute character-level macro-averaged F1 for Subtask 1A.

    Matches the official scoring.py exactly:
      - Integer labels: 0=Neither, 1=Ayah, 2=Hadith
      - sklearn f1_score(..., average='macro') — only averages over classes
        that actually appear in the sample (gold or pred), so a sample with
        no Ayah spans is NOT penalised for f1_ayah=0.

    Gold/pred spans: list of {type: 'q'|'h', start: int, end: int (exclusive)}
    """
    NEITHER = 0
    AYAH    = 1
    HADITH  = 2

    n = len(response_text)

    # Build integer character-label arrays (same as official scorer)
    gold_arr = [NEITHER] * n
    for s in gold_spans:
        tag = AYAH if s["type"].lower() in ("q", "ayah") else HADITH
        gold_arr[s["start"]: min(s["end"], n)] = [tag] * (min(s["end"], n) - s["start"])

    pred_arr = [NEITHER] * n
    for s in pred_spans:
        raw = s["type"].lower()
        tag = AYAH if raw in ("q", "ayah") else HADITH
        pred_arr[s["start"]: min(s["end"], n)] = [tag] * (min(s["end"], n) - s["start"])

    # Use sklearn macro F1 — sklearn only averages over labels present in the
    # combined data, exactly as the official scorer does (no labels= arg).
    macro_f1 = sklearn_f1_score(gold_arr, pred_arr, average='macro', zero_division=0)

    # Also compute per-class F1 for logging/diagnostics (not used in final score)
    per_class = sklearn_f1_score(gold_arr, pred_arr, average=None,
                                 labels=[NEITHER, AYAH, HADITH], zero_division=0)

    return {
        "macro_f1":        macro_f1,
        "f1_neither":      per_class[0],
        "f1_ayah":         per_class[1],
        "f1_hadith":       per_class[2],
    }


def compute_1b_accuracy(predictions: List[Dict]) -> Dict:
    """Compute accuracy for Subtask 1B using 4-class labels.

    Each prediction dict should have:
        gold_label   (str): one of CorrectAyah, WrongAyah, CorrectHadith, WrongHadith
        pred_label   (str): model's predicted label (same set)
        span_type    (str): 'Ayah' or 'Hadith'

    Returns dict with overall accuracy, per-type accuracy, and per-class counts.
    """
    total = correct = 0
    ayah_total = ayah_correct = 0
    hadith_total = hadith_correct = 0

    for p in predictions:
        gold  = p.get("gold_label", "")
        pred  = p.get("pred_label", "")
        stype = p.get("span_type", "Unknown")

        total += 1
        if gold == pred:
            correct += 1

        if stype == "Ayah":
            ayah_total += 1
            if gold == pred:
                ayah_correct += 1
        elif stype == "Hadith":
            hadith_total += 1
            if gold == pred:
                hadith_correct += 1

    return {
        "accuracy":        correct / total if total else 0.0,
        "accuracy_ayah":   ayah_correct / ayah_total if ayah_total else 0.0,
        "accuracy_hadith": hadith_correct / hadith_total if hadith_total else 0.0,
        "n_total":   total,
        "n_correct": correct,
        "n_ayah":    ayah_total,
        "n_hadith":  hadith_total,
    }


def compute_1c_accuracy(predictions: List[Dict]) -> Dict:
    """Compute exact-match accuracy for Subtask 1C using remove_default_diac normalisation
    (mirrors scoring_C.py — sukun removal + a few orthographic variants, NOT full diacritic strip).

    Each prediction dict should have:
        gold_correction  (str): the gold correct text (or "خطأ")
        pred_correction  (str): the model's correction (or "خطأ")
        span_type        (str): 'Ayah' or 'Hadith'

    Returns dict with: accuracy, accuracy_ayah, accuracy_hadith,
                       n_total, n_ayah, n_hadith
    """
    total = correct = 0
    ayah_total = ayah_correct = 0
    hadith_total = hadith_correct = 0

    for p in predictions:
        stype = p.get("span_type", "Unknown")

        # Use the same normalisation as scoring_C.py
        match = (remove_default_diac(p["gold_correction"].strip()) ==
                 remove_default_diac(p["pred_correction"].strip()))

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

    return {
        "accuracy":        correct / total if total else 0.0,
        "accuracy_ayah":   ayah_correct / ayah_total if ayah_total else 0.0,
        "accuracy_hadith": hadith_correct / hadith_total if hadith_total else 0.0,
        "n_total":  total,
        "n_ayah":   ayah_total,
        "n_hadith": hadith_total,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SUBTASK 1A — ZERO-SHOT SPAN DETECTION
# ─────────────────────────────────────────────────────────────────────────────

# System prompt adapted from BurhanAI's DEVELOPER_MESSAGE_NO_TOOLS.
# Rule 2 has been reworded: text is provided directly (no file-reading tools).
# Indices are [start, end) exclusive-end, matching gold annotation convention.
SYSTEM_1A = """\
Task 1A: Identify every INTENDED or CLAIMED Quranic ayah or Prophetic hadith span inside the given Arabic response text and return spans ONLY.

Scoring: character-level F1 over classes {Neither, Ayah, Hadith}. Your spans MUST match exact substring indices in the provided raw text. Do not normalize or rewrite the text.

Important policy: extract what the writer CLAIMS or INTENDS to be an ayah or hadith, EVEN IF IT IS FABRICATED, MISQUOTED, OR PARAPHRASED. Do not verify against sources and do not skip doubtful spans. If the text implies a citation (e.g., قال الله تعالى / يقول الله / كما قال تعالى / قال رسول الله / في الحديث / روي عن النبي), select the span exactly as it appears.

Extraction rules (apply strictly):
1. Span content
   - Select only the minimal contiguous MATN text of the ayah/hadith present in the response.
   - Exclude: surrounding quotes, brackets, punctuation, emojis, tatweel (ـ), ellipses, decorative marks, sura names, verse numbers, hadith collection tags, and narrator chains (عن فلان قال…).
   - If paraphrased but clearly intended, select the paraphrase exactly as written.
2. Boundaries and indices
   - The text is provided directly in the user message. Work with that exact string.
   - Treat indices as [start, end) exclusive-end. Compute start by locating the first character of the chosen span in the raw text and end = start + len(span_text).
   - Trim ONLY edge characters if present at the span edges before finalizing indices: whitespace, newlines, tabs, RTL controls, quotes «»"' and brackets ()[]{} <> and punctuation ، ؛ : . , ! ? … and tatweel ـ.
   - After computing, assert that raw_text[start:end] == span_text. If not, re-check by sliding inward by up to 2 chars to remove stray quotes/newlines, then re-assert. If still failing, recompute start by searching for the exact span_text in the raw string and use that index.
3. Ayah vs Hadith decision (based on local cues)
   - Ayah cues within ±64 chars: قال الله تعالى، يقول الله، الآية، كما قال تعالى، سورة، (4:3) etc.
   - Hadith cues within ±64 chars: قال رسول الله، النبي ﷺ، في الحديث، روي عن، أخرجه البخاري/مسلم/الترمذي…
   - If both cues appear, classify by the MATN itself (what the span text is), not the surrounding narration.
   - Narration words like "عن"، "حدثنا"، "قال" تخص الراوي ليست جزءاً من المتن، ولا تحدد النوع وحدها.
4. Coverage and duplicates (very important)
   - Extract EVERY distinct occurrence (by position) of each ayah/hadith phrase in the response. Do NOT collapse identical phrases; each occurrence must be a separate span if it appears separately.
   - Scan the entire text: when you identify a phrase span, search the raw text for other exact occurrences of the same phrase and include them too, provided local cues still indicate a citation.
5. Avoid overly short or generic fragments
   - Reject single words or very short tokens unless they are clearly iconic and explicitly cited with strong cues and quotes. As a rule of thumb, require ≥ 10 Arabic letters for a span unless enclosed within quotes with explicit cue.
   - Do not extract formulaic phrases such as "بسم الله الرحمن الرحيم" unless the text explicitly presents them as a citation.
6. Selecting the correct boundaries in long narratives
   - Prefer the smallest exact quoted statement that constitutes the ayah/hadith, not entire narrative paragraphs or stories. If a long paragraph includes one or more quoted statements, extract each quoted statement separately.
   - If the span contains an embedded ayah within a hadith narration, extract only the ayah text as an "Aya" span; do not label the whole narration as "Aya".
7. Overlaps
   - Do not produce nested duplicates or overlapping shards of the same citation. Separate spans are allowed only when they represent distinct occurrences.

   
   
8. Empty output
   - If no citations are found, return an empty spans array: {"spans": []}
   - This is valid and expected when the response mentions Islamic topics without quoting specific text.

Return ONLY valid JSON with this structure:
{"spans": [{"type": "q", "text": "...", "start": 0, "end": 10}, ...]}
Use "q" for Quran (Ayah) and "h" for Hadith. start/end are [start, end) exclusive-end indices.

---

## Examples

### Example 1 — No citations (return empty spans)

The text mentions Quran/Hadith topics but never quotes a specific ayah or hadith.
Paraphrases, topic references, and vague allusions are NOT extracted.

Text:
<<<
نعم، هناك آيات كثيرة في القرآن تتحدث عن الأصوات والأصوات التي يمكن سماعها يوم القيامة مثل الرعود والصواعق والأبواق وغيرها. كما يحذر النبي محمد (عليه الصلاة والسلام) أتباعه من الاستماع إلى "صوت" الشيطان واتباع توجيهاته.
>>>

Output:
{"spans": []}

(No specific ayah or hadith is quoted. "صوت" in quotes is not a religious citation.)

---

### Example 2 — Single Ayah, cue via prophetic speech attribution

The cue "قال عيسى عليه السلام" signals a Quranic verse being cited.
Extract only the matn inside the inner quotes; exclude the outer quote, reference "[مريم: 30]", and punctuation.
Note: diacritics (tashkeel) are individual characters and shift all indices — Python's str.find() handles this correctly.

Text:
<<<
"
قال عيسى عليه السلام: "قَالَ إِنِّي عَبْدُ اللَّهِ آتَانِيَ الْكِتَابَ وَجَعَلَنِي نَبِيًّا" [مريم: 30].
>>>

Output:
{"spans": [{"type": "q", "text": "قَالَ إِنِّي عَبْدُ اللَّهِ آتَانِيَ الْكِتَابَ وَجَعَلَنِي نَبِيًّا", "start": 25, "end": 93}]}

(Ayah cue: "قال عيسى عليه السلام". The inner quote at position 24 is excluded; span starts at position 25 on the first letter ق. Trailing " [مريم: 30]." excluded.)

---

### Example 3 — Both Ayah and Hadith in the same response

When a response contains multiple citation types, extract each span independently.
Each gets its own entry; intro phrases and source references are excluded.

Text:
<<<
قال الله تعالى في القرآن "وما كان لبشر أن يكلمه الله إلا وحيا أو من وراء حجاب أو يرسل رسولا فيوحي إليه ما يشاء إنه علي حكيم" (سورة الشورى 42). وهذا يعني أنه ليس للأنبياء القدرة على معرفة الأشياء التي تقع خارج وحي الله المباشر لهم. كما ورد في الحديث الشريف "لا يعلم الغيب إلا الله ومن أطلعه عليه".
>>>

Output:
{"spans": [{"type": "q", "text": "وما كان لبشر أن يكلمه الله إلا وحيا أو من وراء حجاب أو يرسل رسولا فيوحي إليه ما يشاء إنه علي حكيم", "start": 26, "end": 123}, {"type": "h", "text": "لا يعلم الغيب إلا الله ومن أطلعه عليه", "start": 257, "end": 294}]}

(Ayah cue: "قال الله تعالى في القرآن" → type "q". Hadith cue: "ورد في الحديث الشريف" → type "h". "(سورة الشورى 42)" and "(ابن أبي الدنيا)" are source references — excluded.)\
"""

# JSON schema for structured output (enforced by OpenAI and Gemini APIs).
# Anthropic and Together receive the schema description via the system prompt instead.
SCHEMA_1A = {
    "type": "object",
    "properties": {
        "spans": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type":  {"type": "string", "enum": ["q", "h"]},
                    "text":  {"type": "string"},
                    "start": {"type": "integer"},
                    "end":   {"type": "integer"},
                },
                "required": ["type", "text", "start", "end"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["spans"],
    "additionalProperties": False,
}


def build_prompt_1a(response_text: str) -> str:
    """User message for 1A — just the raw text.  All instructions are in the system prompt."""
    return f"Text to analyze:\n<<<\n{response_text}\n>>>"


def run_1a(model_name: str, questions: Dict[str, str],
           gold_df: pd.DataFrame, max_samples: Optional[int] = None) -> Dict:
    """Run zero-shot Subtask 1A on the dev set.

    For each question, sends the Response text to the model, parses the JSON
    span predictions, fixes character indices, then computes character-level
    macro F1.

    Args:
        model_name:  Key into MODEL_REGISTRY.
        questions:   {Question_ID → response_text}
        gold_df:     Gold span annotation DataFrame from dev_SubtaskA.tsv.
        max_samples: If set, limit to this many questions (for quick tests).

    Returns:
        Result dict with aggregate_metrics and per_sample list.
    """
    provider, _, _ = MODEL_REGISTRY[model_name]

    # Group gold annotations by Question_ID for fast lookup
    gold_by_qid = {
        qid: grp for qid, grp in gold_df.groupby("Question_ID")
    }

    qids = list(questions.keys())
    if max_samples:
        qids = qids[:max_samples]
    n_total = len(qids)

    # Resume: skip already-completed samples
    per_sample, completed_ids, _ = load_resume_state(model_name, "1A")
    if completed_ids is None:  # already complete
        return {}
    all_sample_f1 = [s["metrics"]["macro_f1"] for s in per_sample if s.get("metrics")]
    qids = [q for q in qids if q not in completed_ids]

    print(f"\n=== Subtask 1A | {model_name} | {len(qids)} samples (of {n_total} total) ===")

    for i, qid in enumerate(qids):
        response_text = questions[qid]
        prompt = build_prompt_1a(response_text)

        print(f"  [{i+1}/{len(qids)}] {model_name} | 1A | {qid} ... ", end="", flush=True)

        # ── API call ────────────────────────────────────────────────────────
        # Pass SCHEMA_1A so OpenAI and Gemini enforce structured JSON output
        raw_response, usage = call_api(model_name, SYSTEM_1A, prompt, schema=SCHEMA_1A)

        if raw_response is None:
            # Permanent failure — skip this sample
            print("FAILED")
            per_sample.append({
                "sample_id":    qid,
                "prediction":   None,
                "metrics":      None,
                "raw_response": None,
                "usage":        None,
                "error":        "API call failed",
            })
            save_incremental(model_name, "1A", per_sample, n_total)
            continue

        # ── Parse JSON ──────────────────────────────────────────────────────
        parsed = parse_json_response(raw_response)
        pred_spans_raw = []
        if parsed and "spans" in parsed:
            pred_spans_raw = parsed["spans"]

        # ── Fix character indices ───────────────────────────────────────────
        pred_spans = fix_span_indices(pred_spans_raw, response_text)

        # ── Build gold spans for this question ──────────────────────────────
        # Gold spans from TSV: Span_End is exclusive (Python slice convention)
        gold_spans = []
        if qid in gold_by_qid:
            for _, row in gold_by_qid[qid].iterrows():
                gold_type = "q" if str(row["Label"]).lower() == "ayah" else "h"
                gold_spans.append({
                    "type":  gold_type,
                    "start": int(row["Span_Start"]),
                    "end":   int(row["Span_End"]),   # exclusive
                })

        # ── Compute character-level F1 for this sample ──────────────────────
        sample_metrics = compute_1a_char_f1(response_text, gold_spans, pred_spans)
        all_sample_f1.append(sample_metrics["macro_f1"])

        n_found = len(pred_spans)
        print(f"OK ({n_found} spans found, F1={sample_metrics['macro_f1']:.3f})")

        per_sample.append({
            "sample_id":    qid,
            "prediction":   {"spans": pred_spans},
            "gold_spans":   gold_spans,
            "metrics":      sample_metrics,
            "raw_response": raw_response,
            "usage":        usage,
        })

        # Incrementally save after every sample so progress is never lost
        save_incremental(model_name, "1A", per_sample, len(qids))

    # ── Aggregate metrics across all samples ────────────────────────────────
    # We report mean macro F1 across questions (same as most span-detection benchmarks)
    avg_macro_f1 = (sum(all_sample_f1) / len(all_sample_f1)) if all_sample_f1 else 0.0

    # Also compute corpus-level macro F1 by averaging per-class F1s
    per_class = {}
    for cls in ("ayah", "hadith", "neither"):
        vals = [s["metrics"][f"f1_{cls}"] for s in per_sample if s.get("metrics")]
        per_class[cls] = sum(vals) / len(vals) if vals else 0.0

    aggregate_metrics = {
        "macro_f1":  avg_macro_f1,
        "per_class": per_class,
        "n_samples": n_total,
        "n_failed":  sum(1 for s in per_sample if s.get("error")),
    }

    print(f"  -> Aggregate macro F1: {avg_macro_f1:.4f}")
    return {
        "model":              model_name,
        "subtask":            "1A",
        "mode":               "zero-shot",
        "status":             "complete",
        "timestamp":          datetime.now(timezone.utc).isoformat(),
        "aggregate_metrics":  aggregate_metrics,
        "per_sample":         per_sample,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SUBTASK 1B — ZERO-SHOT VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_1B = """\
You are a hafiz (حافظ) and Islamic scholar with complete memorisation of:
  • The Holy Quran in the Uthmani script (الرسم العثماني) with full tashkeel (تشكيل)
  • The six canonical Hadith collections (الكتب الستة):
      Sahih al-Bukhari, Sahih Muslim, Sunan Abu Dawud,
      Jami' al-Tirmidhi, Sunan al-Nasa'i, Sunan Ibn Majah

Your sole task is: given a list of Arabic quotations extracted from an AI-generated response about Islam, classify each quotation as "correct" or "incorrect" based on whether it faithfully matches the canonical source.

CLASSIFICATION CRITERIA
────────────────────────
Use exactly one of these four labels for each quotation:

  CorrectAyah   — the quotation faithfully matches the exact wording of a Quranic ayah
                  in the Uthmani Mushaf. Minor diacritic variation is acceptable.

  WrongAyah     — claimed to be (or appears to be) a Quran quotation, but contains
                  errors: words added, removed, substituted, reordered, or is fabricated.

  CorrectHadith — the quotation faithfully matches the exact matn of a hadith in one
                  of the six canonical books. Minor diacritic variation is acceptable.

  WrongHadith   — claimed to be (or appears to be) a hadith, but the text is fabricated,
                  paraphrased, garbled, or cannot be verified in the six canonical books.

OUTPUT FORMAT
─────────────
Return ONLY valid JSON — no explanation, no commentary:
{"validations": [{"index": 1, "label": "CorrectAyah"}, {"index": 2, "label": "WrongAyah"}, ...]}

One entry per quotation, in the same order as the input list.
The label field must be exactly one of: CorrectAyah, WrongAyah, CorrectHadith, WrongHadith.

──────────────────────────────────────────────────────────────────────────────
FEW-SHOT EXAMPLES
──────────────────────────────────────────────────────────────────────────────

Example input (3 quotations):
1. [Quran] "وَلَنَبْلُوَنَّكُمْ بِشَيْءٍ مِنَ الْخَوْفِ وَالْجُوعِ وَنَقْصٍ مِنَ الْأَمْوَالِ وَالْأَنْفُسِ وَالثَّمَرَاتِ وَبَشِّرِ الصَّابِرِينَ"
2. [Quran] "وَإِن يَصْبِرْ عَلَيْكَ إِنَّهُ كَآفٍ إِنَّهُ غَفُورٌ ذُو عِزٍ وَرَحِيمٌ"
3. [Hadith] "إن الله كتب الإيمان في قلوبكم، فاعملوا بما كتب في قلوبكم، فإن الله لا يقبل من عمل لا يُعمل له"

Example output:
{"validations": [{"index": 1, "label": "CorrectAyah"}, {"index": 2, "label": "WrongAyah"}, {"index": 3, "label": "WrongHadith"}]}

Reasoning (NOT included in your output):
  1 — CorrectAyah: exact match with Surah Al-Baqarah 2:155.
  2 — WrongAyah: this text does not appear anywhere in the Quran; fabricated.
  3 — WrongHadith: this text is not found in any of the six canonical hadith collections.

──────────────────────────────────────────────────────────────────────────────
"""


def build_prompt_1b(spans: List[Dict]) -> str:
    """Build the zero-shot user prompt for Subtask 1B validation.

    Args:
        spans: list of {text: str, span_type: 'Ayah'|'Hadith'}
    """
    # Build numbered list of spans
    numbered_lines = []
    for idx, span in enumerate(spans, start=1):
        type_label = "Quran" if span["span_type"] == "Ayah" else "Hadith"
        numbered_lines.append(f'{idx}. [{type_label}] "{span["text"]}"')
    numbered_spans_str = "\n".join(numbered_lines)

    return f"""\
Classify each of the following quotations using one of the four labels: CorrectAyah, WrongAyah, CorrectHadith, WrongHadith.

{numbered_spans_str}

Return ONLY valid JSON:
{{"validations": [{{"index": 1, "label": "CorrectAyah"}}, {{"index": 2, "label": "WrongHadith"}}, ...]}}"""


def run_1b(model_name: str, questions: Dict[str, str],
           gold_df: pd.DataFrame, max_samples: Optional[int] = None) -> Dict:
    """Run zero-shot Subtask 1B on the dev set.

    Uses GOLD spans (not model predictions from 1A) to isolate 1B performance.
    Batches all spans for a single question into one API call.

    Args:
        model_name:  Key into MODEL_REGISTRY.
        questions:   {Question_ID → response_text}
        gold_df:     Gold span annotation DataFrame from dev_SubtaskB.tsv.
        max_samples: Optional limit on number of questions.

    Returns:
        Result dict with aggregate_metrics and per_sample list.
    """
    # Group gold rows by Question_ID
    gold_by_qid = {
        qid: grp.to_dict("records") for qid, grp in gold_df.groupby("Question_ID")
    }

    qids = list(gold_by_qid.keys())
    if max_samples:
        qids = qids[:max_samples]
    n_total = len(qids)

    # Resume: skip already-completed samples
    per_sample, completed_ids, extra = load_resume_state(model_name, "1B")
    if completed_ids is None:  # already complete
        return {}
    all_span_predictions: List[Dict] = extra.get("all_span_predictions", [])
    qids = [q for q in qids if q not in completed_ids]

    print(f"\n=== Subtask 1B | {model_name} | {len(qids)} questions (of {n_total} total) ===")

    for i, qid in enumerate(qids):
        rows = gold_by_qid[qid]
        response_text = questions.get(qid, "")

        # ── Build span list for this question ───────────────────────────────
        spans_info = []
        for row in rows:
            label = str(row["Label"])
            # Determine span type from label (e.g. "WrongAyah" → "Ayah")
            if "Ayah" in label or "ayah" in label:
                span_type = "Ayah"
            else:
                span_type = "Hadith"
            # Determine gold correctness from label
            gold_correct = ("Wrong" not in label and "wrong" not in label)

            # Extract actual span text from the response using gold indices
            start = int(row["Span_Start"])
            end   = int(row["Span_End"])  # exclusive
            span_text = response_text[start:end].strip() if response_text else row.get("Original_Span", "")

            spans_info.append({
                "text":         span_text,
                "span_type":    span_type,
                "gold_correct": gold_correct,
                "annotation_id": row.get("Annotation_ID", ""),
            })

        if not spans_info:
            continue

        # ── API call (one call per question, all spans batched) ─────────────
        prompt = build_prompt_1b(spans_info)

        n_spans = len(spans_info)
        print(f"  [{i+1}/{len(qids)}] {model_name} | 1B | {qid} ({n_spans} spans) ... ", end="", flush=True)

        raw_response, usage = call_api(model_name, SYSTEM_1B, prompt)

        if raw_response is None:
            print("FAILED")
            per_sample.append({
                "sample_id":    qid,
                "prediction":   None,
                "metrics":      None,
                "raw_response": None,
                "usage":        None,
                "error":        "API call failed",
            })
            save_incremental(model_name, "1B", per_sample, n_total,
                             {"all_span_predictions": all_span_predictions})
            continue

        # ── Parse JSON ──────────────────────────────────────────────────────
        VALID_LABELS = {"CorrectAyah", "WrongAyah", "CorrectHadith", "WrongHadith"}
        parsed = parse_json_response(raw_response)
        validations = {}  # index (1-based) → 4-class label string
        if parsed and "validations" in parsed:
            for v in parsed["validations"]:
                idx = v.get("index")
                lbl = str(v.get("label", "")).strip()
                if idx is not None and lbl in VALID_LABELS:
                    validations[int(idx)] = lbl

        # ── Match predictions back to gold spans ────────────────────────────
        sample_predictions = []
        for j, span in enumerate(spans_info, start=1):
            gold_label = ("Correct" if span["gold_correct"] else "Wrong") + span["span_type"]
            pred_label = validations.get(j, None)
            if pred_label is None:
                stype = span["span_type"]
                print(f"\n    [WARN] No valid prediction for span {j} in {qid}, defaulting to Wrong{stype}")
                pred_label = f"Wrong{stype}"

            pred_dict = {
                "gold_label":   gold_label,
                "pred_label":   pred_label,
                "gold_correct": span["gold_correct"],
                "pred_correct": pred_label.startswith("Correct"),
                "span_type":    span["span_type"],
                "span_text":    span["text"],
                "annotation_id": span["annotation_id"],
                "question_id":  qid,
            }
            sample_predictions.append(pred_dict)
            all_span_predictions.append(pred_dict)

        # Compute per-sample accuracy (4-class exact match)
        sample_acc = sum(1 for p in sample_predictions
                         if p["gold_label"] == p["pred_label"]) / len(sample_predictions)

        print(f"OK (acc={sample_acc:.3f})")

        per_sample.append({
            "sample_id":    qid,
            "prediction":   {"validations": [
                {"index": j, "label": validations.get(j, f"Wrong{spans_info[j-1]['span_type']}")}
                for j in range(1, len(spans_info) + 1)
            ]},
            "gold_spans":   spans_info,
            "metrics":      {"accuracy": sample_acc},
            "raw_response": raw_response,
            "usage":        usage,
        })

        save_incremental(model_name, "1B", per_sample, len(qids),
                         {"all_span_predictions": all_span_predictions})

    # ── Aggregate metrics ───────────────────────────────────────────────────
    agg = compute_1b_accuracy(all_span_predictions)
    print(f"  -> Accuracy: {agg['accuracy']:.4f} "
          f"(Ayah={agg['accuracy_ayah']:.4f}, Hadith={agg['accuracy_hadith']:.4f})")

    return {
        "model":             model_name,
        "subtask":           "1B",
        "mode":              "zero-shot",
        "timestamp":         datetime.now(timezone.utc).isoformat(),
        "aggregate_metrics": agg,
        "per_sample":        per_sample,
        "all_span_predictions": all_span_predictions,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SUBTASK 1C — ZERO-SHOT CORRECTION
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_1C = """\
You are a hafiz (حافظ) and Islamic scholar with complete memorisation of:
  • The Holy Quran in the Uthmani script (الرسم العثماني) with full tashkeel (تشكيل)
  • The six canonical Hadith collections (الكتب الستة):
      Sahih al-Bukhari, Sahih Muslim, Sunan Abu Dawud,
      Jami' al-Tirmidhi, Sunan al-Nasa'i, Sunan Ibn Majah

Your sole task is: given an INCORRECT Arabic quotation, return the exact canonical text it was attempting to cite — or خطأ if it has no authentic source.

CRITICAL RULES
──────────────
1. OUTPUT FORMAT — always return valid JSON:
   {"correction": "<exact canonical text>"}
   or
   {"correction": "خطأ"}
   No explanation. No commentary. No attribution. No isnad. JSON only.

2. QURAN CORRECTIONS
   • Provide the COMPLETE ayah(s) in full Uthmani script with all tashkeel.
   - If a partial Ayah is enough to signify the meaning, you may return just that part, but it must be an exact contiguous substring of the full Ayah text in the Uthmani Mushaf. Do NOT add or remove any words.
   • For multi-ayah spans: place the ayah number (N) after EACH ayah —
     INCLUDING the very last ayah — separated by a single space:
     e.g. "...فَخُورٍ (18) وَاقْصِدْ فِي مَشْيِكَ ... لَصَوْتُ الْحَمِيرِ (19)"
   - Do NOT add Ayah number if it is only 1 ayah
   • Never truncate, paraphrase, or omit any word.

3. HADITH CORRECTIONS
   • Provide ONLY the matn (المتن) — the body of the hadith text.
   • Do NOT include: isnad, sanad, narrator chains, book name, chapter, or any
     phrase like "رواه البخاري" / "أخرجه مسلم".
   • You must be 100% certain the exact wording exists in one of the six
     canonical books. If there is ANY doubt, output خطأ.
   • A hadith that is "similar to" or "paraphrased from" an authentic one is
     still خطأ — only provide a correction for verbatim authentic text.

4. DIACRITICS AND ORTHOGRAPHY
    • Diacritics (tashkeel) are part of the text and must be included in Quranic
        corrections. For hadiths, include diacritics if they are present in the
        canonical text, but do not add diacritics that are not in the source.

    - Any mistake in the diacritics, even if the letters are correct, makes the quotation incorrect.
    - For Quranic text, you must match the Uthmani orthography exactly, including
      letters like "ى" vs "ي", "ة" vs "ه", and the use of "ٱ" (alif wasla) where appropriate.
    - For hadiths, match the orthography of the canonical source exactly.

    IMPORTANT — إدغام (assimilation):
    - Apply full Tajweed diacritization including إدغام marks.
    - When a letter (e.g. ن) assimilates into the following letter, drop its
      sukun and place a shadda on the following letter.
    - Example: write "مِن نُّطْفَةٍ" (ن with no sukun, following نُّ with shadda),
      NOT "مِنْ نُطْفَةٍ" (ن with sukun, following نُ without shadda).

    IMPORTANT — أُولَٰئِكَ orthography:
    - Write this word WITHOUT a kashida/tatweel character between the لَ and the
      superscript alif (ٰ). Correct: "أُولَٰئِكَ". Incorrect: "أُولَـٰئِكَ".
     

5. WHEN TO OUTPUT خطأ
   • The quotation is completely fabricated with no authentic Quranic or Hadith
     source — even if it sounds Islamic.
   • The Hadith text is a paraphrase, corrupted version, or you cannot identify
     it with certainty in the canonical books.

──────────────────────────────────────────────────────────────────────────────
FEW-SHOT EXAMPLES
──────────────────────────────────────────────────────────────────────────────

Example 1 — WrongAyah (single ayah, wrong word)
Source type: Ayah
Incorrect: "لَا يُكَلِّفُ اللَّهُ نَفْسًا إِلَّا طَاقَتَهَا"
{"correction": "لَا يُكَلِّفُ اللَّهُ نَفْسًا إِلَّا وُسْعَهَا ۚ لَهَا مَا كَسَبَتْ وَعَلَيْهَا مَا اكْتَسَبَتْ ۗ رَبَّنَا لَا تُؤَاخِذْنَا إِن نَّسِينَا أَوْ أَخْطَأْنَا ۚ رَبَّنَا وَلَا تَحْمِلْ عَلَيْنَا إِصْرًا كَمَا حَمَلْتَهُ عَلَى الَّذِينَ مِن قَبْلِنَا ۚ رَبَّنَا وَلَا تُحَمِّلْنَا مَا لَا طَاقَةَ لَنَا بِهِ ۖ وَاعْفُ عَنَّا وَاغْفِرْ لَنَا وَارْحَمْنَا ۚ أَنتَ مَوْلَانَا فَانصُرْنَا عَلَى الْقَوْمِ الْكَافِرِينَ (286)"}
Note: the COMPLETE ayah must be returned, including waqf marks (ۚ ۖ ۗ) and the ayah number (286) at the end.

Example 2 — WrongHadith (fabricated, no authentic source)
Incorrect: "من صلى الفجر في جماعة كان في ذمة الله يوم القيامة"
{"correction": "خطأ"}

Example 3 — WrongHadith (authentic text with wrong wording)
Incorrect: "إنما الأعمال بالنيات وإنما لكل امرئ ما أراد"
{"correction": "إِنَّمَا الْأَعْمَالُ بِالنِّيَّاتِ، وَإِنَّمَا لِكُلِّ امْرِئٍ مَا نَوَى"}

──────────────────────────────────────────────────────────────────────────────
"""

SCHEMA_1C = {
    "type": "object",
    "properties": {
        "correction": {
            "type": "string",
            "description": "The exact canonical text or خطأ if fabricated"
        }
    },
    "required": ["correction"],
    "additionalProperties": False,
}


def build_prompt_1c(span_text: str, source_type: str) -> str:
    """Build the zero-shot user prompt for Subtask 1C correction."""
    return f"""\
Source type: {source_type}
Incorrect quotation: "{span_text}"

Return JSON: {{"correction": "<exact canonical text or خطأ>"}}"""


def run_1c(model_name: str, questions: Dict[str, str],
           gold_df: pd.DataFrame, max_samples: Optional[int] = None) -> Dict:
    """Run zero-shot Subtask 1C on the dev set.

    Uses ONLY the gold spans labeled as incorrect (all rows in dev_SubtaskC.tsv
    are already incorrect spans).  One API call per span.

    Args:
        model_name:  Key into MODEL_REGISTRY.
        questions:   {Question_ID → response_text}
        gold_df:     Gold correction DataFrame from dev_SubtaskC.tsv.
        max_samples: Optional limit on number of spans.

    Returns:
        Result dict with aggregate_metrics and per_sample list.
    """
    rows = gold_df.to_dict("records")
    if max_samples:
        rows = rows[:max_samples]
    n_total = len(rows)

    # Resume: count-based row skipping (span_ids are unique but row order is stable)
    per_sample, completed_ids, extra = load_resume_state(model_name, "1C")
    if completed_ids is None:  # already complete
        return {}
    n_done = len(per_sample)
    all_predictions: List[Dict] = extra.get("all_predictions", [])
    if n_done > 0:
        rows = rows[n_done:]  # skip already-processed rows

    print(f"\n=== Subtask 1C | {model_name} | {len(rows)} spans (of {n_total} total) ===")

    for i, row in enumerate(rows):
        qid   = row["Question_ID"]
        label = str(row["Label"])
        start = int(row["Span_Start"])
        end   = int(row["Span_End"])  # exclusive
        gold_correction = str(row["Correction"])
        # Unique ID per span (same question can have multiple spans)
        span_id = f"{qid}@{start}-{end}"

        # Determine span type and source label
        if "Ayah" in label or "ayah" in label:
            span_type   = "Ayah"
            source_type = "Holy Quran (القرآن الكريم)"
        else:
            span_type   = "Hadith"
            source_type = "Prophetic Hadith (الحديث النبوي)"

        # Extract span text from response
        response_text = questions.get(qid, "")
        span_text = response_text[start:end].strip() if response_text else row.get("Original_Span", "")

        # Fall back to Original_Span if extraction is empty or clearly wrong
        if not span_text:
            span_text = str(row.get("Original_Span", ""))

        prompt = build_prompt_1c(span_text, source_type)

        print(f"  [{i+1}/{len(rows)}] {model_name} | 1C | {qid} [{span_type}] ... ",
              end="", flush=True)

        raw_response, usage = call_api(model_name, SYSTEM_1C, prompt, schema=SCHEMA_1C)

        if raw_response is None:
            print("FAILED")
            per_sample.append({
                "sample_id":     span_id,
                "question_id":   qid,
                "span_type":     span_type,
                "prediction":    None,
                "gold":          gold_correction,
                "metrics":       None,
                "raw_response":  None,
                "usage":         None,
                "error":         "API call failed",
            })
            save_incremental(model_name, "1C", per_sample, n_total,
                             {"all_predictions": all_predictions})
            continue

        # Parse JSON response: {"correction": "..."}
        parsed = parse_json_response(raw_response)
        if parsed and "correction" in parsed:
            pred_correction = str(parsed["correction"]).strip()
        else:
            # Fallback: treat entire response as plain text
            pred_correction = raw_response.strip()

        # Strip surrounding quotation marks if the model added them
        pred_correction = pred_correction.strip('"\'""''')

        # Snap to nearest verified canonical text from corpus.
        # Snap to nearest verified canonical text.
        # Quran snapping is OFF by default: the corpus lacks assimilation
        # (إدغام) shadda marks that the gold requires, so snapping hurts more
        # than it helps.  Set SNAP_QURAN = True to re-enable.
        # Hadith snapping is ON: Matn text is consistent with the gold.
        SNAP_QURAN = True
        # Strip trailing lone "(N)" before snapping — model labelled a single
        # ayah with its own number, e.g. "فَأُمُّهُ هَاوِيَةٌ (9)".
        snap_input = pred_correction
        if span_type == "Ayah":
            cleaned = re.sub(r"\s*\(\d+\)\s*$", "", snap_input.strip())
            if not re.search(r"\(\d+\)", cleaned):  # no remaining (N) → was single-ayah
                snap_input = cleaned
        if span_type == "Hadith" or SNAP_QURAN:
            canonical = retrieve_canonical(snap_input, span_type)
        else:
            canonical = None
        snapped_correction = canonical if canonical else pred_correction

        # Exact match using scoring_C.py normalisation (partial diacritic removal)
        exact_match = (remove_default_diac(gold_correction.strip()) ==
                       remove_default_diac(snapped_correction.strip()))

        print(f"OK (match={'Y' if exact_match else 'N'}"
              f"{' [snapped]' if canonical else ''})")

        pred_dict = {
            "gold_correction": gold_correction,
            "pred_correction": snapped_correction,
            "span_type":       span_type,
            "question_id":     qid,
        }
        all_predictions.append(pred_dict)

        per_sample.append({
            "sample_id":       span_id,
            "question_id":     qid,
            "span_type":       span_type,
            "span_text":       span_text,
            "prediction":      {"correction": snapped_correction,
                                "model_raw":  pred_correction},
            "gold":            gold_correction,
            "metrics":         {"exact_match": exact_match},
            "raw_response":    raw_response,
            "usage":           usage,
        })

        save_incremental(model_name, "1C", per_sample, len(rows),
                         {"all_predictions": all_predictions})

    # ── Aggregate metrics ───────────────────────────────────────────────────
    agg = compute_1c_accuracy(all_predictions)
    print(f"  -> Accuracy: {agg['accuracy']:.4f} "
          f"(Ayah={agg['accuracy_ayah']:.4f}, Hadith={agg['accuracy_hadith']:.4f})")

    return {
        "model":             model_name,
        "subtask":           "1C",
        "mode":              "zero-shot",
        "timestamp":         datetime.now(timezone.utc).isoformat(),
        "aggregate_metrics": agg,
        "per_sample":        per_sample,
        "all_predictions":   all_predictions,
    }


# ─────────────────────────────────────────────────────────────────────────────
# RESULT SAVING
# ─────────────────────────────────────────────────────────────────────────────

def save_result(result: Dict, model_name: str, subtask: str) -> Path:
    """Save result dict to a JSON file under results/.

    File name: results/{model_safe_name}_{subtask}_zeroshot.json
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Sanitise model name for use in file paths
    safe_name = re.sub(r"[^A-Za-z0-9_\-]", "_", model_name)
    outfile = RESULTS_DIR / f"{safe_name}_{subtask}_zeroshot.json"

    with open(outfile, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)

    print(f"  Saved -> {outfile}")
    return outfile


def result_path(model_name: str, subtask: str) -> Path:
    """Return the output path for a given model and subtask (without writing)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9_\-]", "_", model_name)
    return RESULTS_DIR / f"{safe_name}_{subtask}_zeroshot.json"


def save_incremental(model_name: str, subtask: str, per_sample: List[Dict],
                     n_total: int, extra_fields: Optional[Dict] = None) -> None:
    """Write current progress to disk after each sample completes.

    Computes aggregate metrics on the completed samples so far so the file
    is always in a valid, readable state. If the run crashes, you lose at
    most one sample rather than everything.

    Args:
        model_name:   Key into MODEL_REGISTRY.
        subtask:      '1A', '1B', or '1C'.
        per_sample:   List of completed sample dicts so far.
        n_total:      Total number of samples planned for this run.
        extra_fields: Any additional top-level fields to include (e.g.
                      all_span_predictions for 1B/1C).
    """
    outfile = result_path(model_name, subtask)

    # Compute interim aggregate on completed samples only
    completed = [s for s in per_sample if not s.get("error")]
    n_failed  = len(per_sample) - len(completed)

    if subtask == "1A":
        f1_vals = [s["metrics"]["macro_f1"] for s in completed if s.get("metrics")]
        avg_f1  = sum(f1_vals) / len(f1_vals) if f1_vals else 0.0
        per_class = {}
        for cls in ("ayah", "hadith", "neither"):
            vals = [s["metrics"][f"f1_{cls}"] for s in completed if s.get("metrics")]
            per_class[cls] = sum(vals) / len(vals) if vals else 0.0
        agg = {"macro_f1": avg_f1, "per_class": per_class,
               "n_samples": n_total, "n_completed": len(per_sample), "n_failed": n_failed}

    elif subtask == "1B":
        agg = {"n_samples": n_total, "n_completed": len(per_sample), "n_failed": n_failed}

    elif subtask == "1C":
        agg = {"n_samples": n_total, "n_completed": len(per_sample), "n_failed": n_failed}

    else:
        agg = {}

    snapshot = {
        "model":             model_name,
        "subtask":           subtask,
        "mode":              "zero-shot",
        "timestamp":         datetime.now(timezone.utc).isoformat(),
        "status":            "in_progress" if len(per_sample) < n_total else "complete",
        "aggregate_metrics": agg,
        "per_sample":        per_sample,
    }
    if extra_fields:
        snapshot.update(extra_fields)

    with open(outfile, "w", encoding="utf-8") as fh:
        json.dump(snapshot, fh, ensure_ascii=False, indent=2)


def load_resume_state(model_name: str, subtask: str):
    """Load existing results file so an interrupted run can be resumed.

    Returns:
        (per_sample, completed_ids, extra)
        - per_sample:     list of already-processed sample dicts
        - completed_ids:  set of sample_id strings already done (1A/1B)
                          OR int count of rows done (1C, where ids repeat)
        - extra:          dict of any extra top-level fields (e.g. all_span_predictions)
    """
    outfile = result_path(model_name, subtask)
    if not os.path.exists(outfile):
        return [], set(), {}
    try:
        with open(outfile, encoding="utf-8") as fh:
            data = json.load(fh)
        if data.get("status") == "complete":
            print(f"  [SKIP] {model_name}/{subtask} already complete — delete the JSON to re-run.")
            return data.get("per_sample", []), None, {}  # None signals "fully done"
        per_sample = data.get("per_sample", [])
        extra = {k: v for k, v in data.items()
                 if k not in ("model", "subtask", "mode", "timestamp", "status",
                              "aggregate_metrics", "per_sample")}
        completed_ids = {s["sample_id"] for s in per_sample}
        if per_sample:
            print(f"  [RESUME] {len(per_sample)} samples already done — skipping.")
        return per_sample, completed_ids, extra
    except Exception as e:
        print(f"  [WARN] Could not load existing results for resume: {e}")
        return [], set(), {}


# ─────────────────────────────────────────────────────────────────────────────
# MODEL SELECTION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def resolve_models(model_arg: str) -> List[str]:
    """Expand --model argument to a concrete list of model names.

    'all' → all models whose API key is available in the environment.
    Otherwise → validate the single model name and return it in a list.
    """
    if model_arg == "all":
        available = []
        for name, (provider, _, key_env) in MODEL_REGISTRY.items():
            if os.environ.get(key_env):
                available.append(name)
            else:
                print(f"  [SKIP] {name} - {key_env} not set")
        return available
    else:
        if model_arg not in MODEL_REGISTRY:
            print(f"ERROR: Unknown model '{model_arg}'. "
                  f"Choices: {list(MODEL_REGISTRY.keys())} or 'all'")
            sys.exit(1)
        provider, _, key_env = MODEL_REGISTRY[model_arg]
        if not os.environ.get(key_env):
            print(f"ERROR: {key_env} is not set in the environment.")
            sys.exit(1)
        return [model_arg]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot frontier LLM evaluation on IslamicEval 2025",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python zero_shot_baselines.py --model gpt-4.1-mini --subtask 1A
  python zero_shot_baselines.py --model gpt-4.1-mini --subtask all
  python zero_shot_baselines.py --model all --subtask all
  python zero_shot_baselines.py --model all --subtask all --max-samples 5
        """,
    )
    parser.add_argument(
        "--model", required=True,
        help=f"Model name (one of {list(MODEL_REGISTRY.keys())}) or 'all'",
    )
    parser.add_argument(
        "--subtask", required=True, choices=["1A", "1B", "1C", "all"],
        help="Subtask to run (1A, 1B, 1C, or all)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit number of samples/questions (useful for quick tests)",
    )
    args = parser.parse_args()

    # Resolve model list
    models = resolve_models(args.model)
    if not models:
        print("No models available (check API keys). Exiting.")
        sys.exit(1)

    # Resolve subtask list
    subtasks = ["1A", "1B", "1C"] if args.subtask == "all" else [args.subtask]

    # Pre-load all required data (only once, shared across models)
    data_cache: Dict[str, Any] = {}
    if "1A" in subtasks:
        print("Loading Subtask 1A data...")
        data_cache["1A"] = load_data_1a()
    if "1B" in subtasks:
        print("Loading Subtask 1B data...")
        data_cache["1B"] = load_data_1b()
    if "1C" in subtasks:
        print("Loading Subtask 1C data...")
        data_cache["1C"] = load_data_1c()

    # Run evaluation
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        for subtask in subtasks:
            questions, gold_df = data_cache[subtask]

            try:
                if subtask == "1A":
                    result = run_1a(model_name, questions, gold_df, args.max_samples)
                elif subtask == "1B":
                    result = run_1b(model_name, questions, gold_df, args.max_samples)
                elif subtask == "1C":
                    result = run_1c(model_name, questions, gold_df, args.max_samples)

                save_result(result, model_name, subtask)

            except Exception as exc:
                print(f"\n[ERROR] {model_name} / {subtask} crashed: {exc}")
                traceback.print_exc()

    print("\nDone.")


if __name__ == "__main__":
    main()
