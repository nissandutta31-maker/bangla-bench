#!/usr/bin/env python3
"""
debug_dump.py — pinpoint the BanglaBench scoring bug.

Symptom: below-random accuracy (16.7% on a 4-way MCQ; floor is 25%) with
100% parse rate. That combination has exactly two usual causes:

    (1) the passage never reaches the prompt  -> model answers blind
    (2) the gold letter is misaligned with the choices (off-by-one, etc.)
        -> a correct answer gets marked wrong -> accuracy collapses BELOW chance

This script makes both visible for the first N items. It computes the gold
letter INDEPENDENTLY (no dependency on the runner's internals, so it can't
inherit the same bug) and, if it can locate the runner's own prompt-builder,
prints the exact prompt the runner sends so you can confirm the passage is in it.

Usage:
    python3 debug_dump.py                          # 3 items from belebele_ben_sample.jsonl
    python3 debug_dump.py belebele_ben_full.jsonl 5
"""

import importlib
import json
import sys

ITEMS_PATH = sys.argv[1] if len(sys.argv) > 1 else "belebele_ben_sample.jsonl"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 3
LETTERS = ["A", "B", "C", "D"]


def first(item, *keys):
    """Return the first present, non-empty field among `keys`."""
    for k in keys:
        if k in item and item[k] not in (None, ""):
            return item[k]
    return None


def canonical_gold(item):
    """What the gold letter SHOULD be per the Belebele schema.
    correct_answer_num is 1-indexed ('1'..'4') -> A..D."""
    raw = first(item, "correct_answer_num", "answer", "correct_answer")
    if raw is None:
        return None, "MISSING gold field"
    s = str(raw).strip()
    if s.isdigit():
        n = int(s)
        if 1 <= n <= 4:
            return LETTERS[n - 1], f"num {n} -> {LETTERS[n - 1]} (1-indexed)"
        return None, f"out-of-range num {n}"
    if s.upper() in LETTERS:
        return s.upper(), f"letter {s.upper()}"
    return None, f"unrecognized gold value {s!r}"


def get_choices(item):
    c = first(item, "choices")
    if isinstance(c, list):
        return list(c)
    return [item.get(f"mc_answer{i}") for i in range(1, 5)]


# --- try to locate the runner's own prompt builder (optional) ---------------
runner = None
build_fn = None
try:
    runner = importlib.import_module("bangla_bench_runner")
    for name in (
        "build_prompt", "render_prompt", "format_prompt", "make_prompt",
        "item_to_prompt", "build_user_prompt", "render_item", "format_item",
        "build_question", "render_question",
    ):
        if hasattr(runner, name):
            build_fn = getattr(runner, name)
            print(f"[ok] using runner.{name}() to render the real prompt\n")
            break
except Exception as e:  # noqa: BLE001
    print(f"[note] couldn't import bangla_bench_runner ({e}); "
          f"showing independent analysis only.\n")

if runner is not None and build_fn is None:
    candidates = [
        n for n in dir(runner)
        if callable(getattr(runner, n))
        and any(w in n.lower() for w in ("prompt", "render", "format", "normal"))
    ]
    print("[note] no known prompt-builder found by name. Likely candidates in "
          f"the module: {candidates or '(none obvious)'}")
    print("       Set build_fn to the right one near the top of this script to "
          "also see the exact rendered prompt.\n")

# --- dump -------------------------------------------------------------------
with open(ITEMS_PATH, encoding="utf-8") as fh:
    items = [json.loads(line) for line in fh if line.strip()][:N]

for i, item in enumerate(items, 1):
    passage = first(item, "flores_passage", "passage", "context")
    question = first(item, "question", "question_text")
    choices = get_choices(item)
    gold, how = canonical_gold(item)

    print("=" * 78)
    print(f"ITEM {i}   id={first(item, 'item_id', 'id', 'task_id')}")
    print(f"  passage present in data? "
          f"{'YES (len=%d)' % len(str(passage)) if passage else 'NO  <-- RED FLAG'}")
    print(f"  question: {question!r}")
    for letter, text in zip(LETTERS, choices):
        mark = "   <-- GOLD (canonical)" if letter == gold else ""
        print(f"    {letter}. {text}{mark}")
    print(f"  canonical gold letter: {gold}   [{how}]")

    if build_fn is not None:
        try:
            rendered = build_fn(item)
        except TypeError:
            rendered = None
            print("  [runner builder didn't accept the raw item; it likely takes "
                  "a normalized item — wire it manually to confirm the passage]")
        if rendered is not None:
            included = bool(passage) and (str(passage)[:40] in str(rendered))
            print(f"  passage included in RENDERED prompt? "
                  f"{'YES' if included else 'NO  <-- RED FLAG'}")
            print("  ---------- exact prompt the runner builds ----------")
            print(rendered)
            print("  ----------------------------------------------------")
    print()

print("=" * 78)
print("HOW TO READ THIS")
print(" 1. 'passage present in data? NO'  -> your JSONL is missing flores_passage.")
print(" 2. passage in data but NOT in the rendered prompt -> the prompt builder is")
print("    dropping it. The model is answering blind. Fix the builder to include")
print("    the passage. (This alone can push a frontier model below chance.)")
print(" 3. You are a native Bengali speaker — read the choice marked GOLD. Is it")
print("    actually the correct answer? If the genuinely correct option sits on a")
print("    DIFFERENT letter, your gold mapping is off (0-indexed vs 1-indexed, or")
print("    choices reordered before scoring). That single off-by-one produces")
print("    below-random accuracy with a 100% parse rate.")
print()
print("CONFIRM IT END-TO-END (no guessing):")
print("   a) head -3 %s > dbg3.jsonl" % ITEMS_PATH)
print("   b) in config.yaml set:  logging.csv_include_response: true")
print("                           logging.prompt_preview_chars: 100000")
print("   c) python3 bangla_bench_runner.py eval dbg3.jsonl -o dbg_results.jsonl")
print("   d) open dbg_results.jsonl and compare the runner's recorded GOLD for")
print("      each item to the canonical gold printed above. If they differ, the")
print("      bug is in the runner's gold mapping. If they match but the answer is")
print("      marked wrong, check the full prompt in the CSV log for the passage.")
