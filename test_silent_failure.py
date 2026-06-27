#!/usr/bin/env python3
"""Offline unit tests for the silent-failure detector and haystack builder.

No external API calls and no inspect-ai dependency required.

Run with:  python3 test_silent_failure.py
Exits 0 if all checks pass, 1 otherwise.
"""
from src.validators.silent_failure import (
    EMPTY_RESPONSE,
    FALSE_NEGATIVE_CONTEXT_DROP,
    MOJIBAKE_DETECTED,
    REASONING_TRUNCATED,
    SCRIPT_DRIFT_ENGLISH,
    TOKENIZER_EXPLOSION,
    FailureContext,
    detect_silent_failures,
)
from tasks.haystack import DEFAULT_NEEDLE, build_haystack, build_prompt, load_filler_sentences


def check(cond, label):
    print(f"[{'PASS' if cond else 'FAIL'}] {label}")
    return cond


def flags(text, **kw):
    return detect_silent_failures(FailureContext(response_text=text, **kw))


def main() -> int:
    ok = True

    # --- clean responses produce no flags ---------------------------------- #
    ok &= check(
        flags("টমেটো", expects_bangla=True, needle_answer="টমেটো") == [],
        "clean Bangla answer -> no flags",
    )

    # --- empty response short-circuits ------------------------------------- #
    ok &= check(flags("   ") == [EMPTY_RESPONSE], "whitespace-only -> EMPTY_RESPONSE")
    ok &= check(flags("") == [EMPTY_RESPONSE], "empty string -> EMPTY_RESPONSE")

    # --- context drop: refusal markers ------------------------------------- #
    ok &= check(
        FALSE_NEGATIVE_CONTEXT_DROP in flags("আমি জানি না।", needle_answer="টমেটো"),
        "Bangla 'jani na' refusal -> context drop",
    )
    ok &= check(
        FALSE_NEGATIVE_CONTEXT_DROP in flags("I don't know", needle_answer="টমেটো"),
        "English 'don't know' -> context drop",
    )
    # --- context drop: needle simply absent -------------------------------- #
    ok &= check(
        FALSE_NEGATIVE_CONTEXT_DROP in flags("শিহাব আম খায়", needle_answer="টমেটো"),
        "wrong fact (needle absent) -> context drop",
    )
    ok &= check(
        FALSE_NEGATIVE_CONTEXT_DROP not in flags("সে টমেটো খায়", needle_answer="টমেটো"),
        "needle present in longer answer -> no context drop",
    )

    # --- truncation -------------------------------------------------------- #
    ok &= check(
        flags("partial answer", finish_reason="length") == [REASONING_TRUNCATED],
        "finish_reason=length -> truncation",
    )
    ok &= check(
        REASONING_TRUNCATED not in flags("done", finish_reason="stop"),
        "finish_reason=stop -> no truncation",
    )

    # --- mojibake ---------------------------------------------------------- #
    ok &= check(
        MOJIBAKE_DETECTED in flags("টম�েটো", expects_bangla=True),
        "replacement char -> mojibake",
    )
    ok &= check(
        MOJIBAKE_DETECTED in flags("bad\x07bell"),
        "control char -> mojibake",
    )

    # --- script drift ------------------------------------------------------ #
    ok &= check(
        SCRIPT_DRIFT_ENGLISH in flags("The answer is Dhaka", expects_bangla=True),
        "English answer on Bangla task -> script drift",
    )
    ok &= check(
        SCRIPT_DRIFT_ENGLISH not in flags("The answer is Dhaka", expects_bangla=False),
        "English answer on non-Bangla task -> no script drift",
    )
    ok &= check(
        SCRIPT_DRIFT_ENGLISH not in flags("ঢাকা শহর", expects_bangla=True),
        "Bangla answer on Bangla task -> no script drift",
    )

    # --- tokenizer explosion ----------------------------------------------- #
    ok &= check(
        TOKENIZER_EXPLOSION
        in flags("x" * 50, short_answer_task=True, prompt_tokens=10, completion_tokens=50),
        "runaway output on short task -> tokenizer explosion",
    )
    ok &= check(
        TOKENIZER_EXPLOSION
        not in flags("x", short_answer_task=False, prompt_tokens=10, completion_tokens=50),
        "long output on long-form task -> no tokenizer explosion",
    )

    # --- multiple independent flags can co-occur --------------------------- #
    multi = flags("I don't know", expects_bangla=True, needle_answer="টমেটো", finish_reason="length")
    ok &= check(
        FALSE_NEGATIVE_CONTEXT_DROP in multi
        and SCRIPT_DRIFT_ENGLISH in multi
        and REASONING_TRUNCATED in multi,
        "co-occurring failures all flagged",
    )

    # --- haystack builder -------------------------------------------------- #
    filler = load_filler_sentences()
    ok &= check(len(filler) > 0, "filler sentences loaded from Belebele")
    h = build_haystack(DEFAULT_NEEDLE, target_tokens=2000, depth=0.5, filler=filler)
    ok &= check(DEFAULT_NEEDLE.fact in h, "needle embedded in haystack")
    top = build_haystack(DEFAULT_NEEDLE, 2000, 0.0, filler).split("\n")
    ok &= check(top.index(DEFAULT_NEEDLE.fact) == 0, "depth=0.0 -> needle at top")
    bottom = build_haystack(DEFAULT_NEEDLE, 2000, 1.0, filler).split("\n")
    ok &= check(
        bottom.index(DEFAULT_NEEDLE.fact) == len(bottom) - 1,
        "depth=1.0 -> needle at bottom",
    )
    ok &= check(DEFAULT_NEEDLE.question in build_prompt(DEFAULT_NEEDLE, h), "prompt contains question")

    print("\n" + ("ALL PASSED" if ok else "SOME FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
