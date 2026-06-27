"""Silent-failure detection for Bangla LLM completions.

A "silent failure" is a response that the model emits *confidently* but that is
broken in a way exact-match scoring will not catch: it dropped the context, it
got truncated mid-thought, it corrupted the Bengali script, or it ran away
generating tokens. These are the failures worth a post-mortem, so we flag them
as structured metadata on every completion rather than collapsing the run to a
single pass/fail.

This module is deliberately framework-agnostic and dependency-free so it can be
unit-tested offline (see ``test_silent_failure.py``) and reused outside of
inspect-ai. The inspect-ai scorer wrapper lives in ``tasks/scorers.py``.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# Unicode block for Bengali script (includes Assamese): U+0980-U+09FF.
_BANGLA_RANGE = (0x0980, 0x09FF)

# C0/C1 control chars (excluding the normal whitespace \t \n \r) plus the
# Unicode replacement character (U+FFFD) -- both are reliable mojibake signatures.
_CONTROL_CHARS = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f�]")

# Phrases a model emits when it has effectively dropped the context and is
# refusing to answer, in both English and Bengali.
_DONT_KNOW_MARKERS = (
    "don't know",
    "do not know",
    "cannot find",
    "can't find",
    "not mentioned",
    "no information",
    "জানি না",   # "jani na" (I don't know)
    "জানিনা",     # "janina"
    "জানা নেই",  # "jana nei" (not known)
    "উল্লেখ নেই",  # "ullekh nei" (not mentioned)
    "তথ্য নেই",  # "tothyo nei" (no information)
    "পাওয়া যায়নি",  # "paoa jayni" (not found)
)

# Flag string constants -- referenced by tests and downstream aggregation SQL.
FALSE_NEGATIVE_CONTEXT_DROP = "FALSE_NEGATIVE_CONTEXT_DROP"
REASONING_TRUNCATED = "REASONING_TRUNCATED"
MOJIBAKE_DETECTED = "MOJIBAKE_DETECTED"
SCRIPT_DRIFT_ENGLISH = "SCRIPT_DRIFT_ENGLISH"
TOKENIZER_EXPLOSION = "TOKENIZER_EXPLOSION"
EMPTY_RESPONSE = "EMPTY_RESPONSE"


@dataclass
class FailureContext:
    """Everything the detector needs about one completion.

    Only ``response_text`` is required. The richer the context, the more
    failure modes can be detected -- e.g. ``finish_reason`` enables truncation
    detection, ``needle_answer`` enables context-drop detection.
    """

    response_text: str
    finish_reason: str | None = None  # e.g. "stop", "length", "content_filter"
    expects_bangla: bool = False      # is the task supposed to elicit Bengali?
    needle_answer: str | None = None  # ground-truth fact that MUST appear
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    # For tokenizer-explosion: only flag runaway output on short-answer tasks,
    # where a large completion is clearly a loop rather than a long-form answer.
    short_answer_task: bool = False
    metadata: dict = field(default_factory=dict)


def _bangla_ratio(text: str) -> float:
    """Fraction of *letter* characters that fall in the Bengali block.

    Digits, punctuation and whitespace are ignored so that an all-Bengali
    answer containing a date or a quoted English proper noun is not penalised
    by the denominator.
    """
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    lo, hi = _BANGLA_RANGE
    bangla = sum(1 for c in letters if lo <= ord(c) <= hi)
    return bangla / len(letters)


def detect_silent_failures(ctx: FailureContext) -> list[str]:
    """Return the list of silent-failure flags for one completion.

    The list is empty for a clean response. Flags are independent -- a single
    completion can trip several at once (e.g. truncated AND script-drifted).
    """
    flags: list[str] = []
    text = ctx.response_text or ""
    stripped = text.strip()

    # 0. Empty / whitespace-only response. Nothing else is meaningful after this.
    if not stripped:
        return [EMPTY_RESPONSE]

    lowered = stripped.lower()

    # 1. False-negative context drop: the needle answer was retrievable, yet the
    #    model either refused ("don't know") or never produced the fact.
    if ctx.needle_answer:
        said_dont_know = any(m in lowered or m in stripped for m in _DONT_KNOW_MARKERS)
        # Normalise whitespace before substring-checking the needle so that a
        # line-wrapped answer still counts as containing it.
        norm = unicodedata.normalize("NFC", " ".join(stripped.split()))
        needle = unicodedata.normalize("NFC", ctx.needle_answer.strip())
        missing_needle = bool(needle) and needle not in norm
        if said_dont_know or missing_needle:
            flags.append(FALSE_NEGATIVE_CONTEXT_DROP)

    # 2. Reasoning truncation: the provider cut us off at the token ceiling.
    if ctx.finish_reason == "length":
        flags.append(REASONING_TRUNCATED)

    # 3. Mojibake: control chars or the Unicode replacement char in the output.
    if _CONTROL_CHARS.search(text):
        flags.append(MOJIBAKE_DETECTED)

    # 4. Script drift: a Bengali task whose answer came back mostly non-Bengali.
    if ctx.expects_bangla and _bangla_ratio(stripped) < 0.30:
        flags.append(SCRIPT_DRIFT_ENGLISH)

    # 5. Tokenizer explosion: on a short-answer task the model emitted far more
    #    tokens than it consumed -- a strong signal of a generation loop, which
    #    bites poorly-tokenized scripts (Bengali numerals) on small local models.
    if (
        ctx.short_answer_task
        and ctx.prompt_tokens
        and ctx.completion_tokens
        and ctx.completion_tokens > 2 * ctx.prompt_tokens
    ):
        flags.append(TOKENIZER_EXPLOSION)

    return flags
