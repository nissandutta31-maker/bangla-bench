"""inspect-ai scorers that surface silent failures alongside correctness.

``retrieval_with_silent_failures`` scores a needle-retrieval sample two ways at
once: the binary "did the needle answer appear?" *and* the structured
silent-failure flags from ``src.validators.silent_failure``. Correctness is the
headline metric; the flags are recorded on each ``Score`` so a post-mortem can
join them (``WHERE flag = 'REASONING_TRUNCATED'``) without re-running anything.
"""
from __future__ import annotations

import sys
import unicodedata
from collections.abc import Sequence
from os.path import abspath, dirname

from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Metric,
    SampleScore,
    Score,
    Target,
    Value,
    accuracy,
    metric,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState

# Allow `from src.validators...` whether inspect is launched from the repo root
# or elsewhere -- the task package sits one level below the repo root.
_REPO_ROOT = dirname(dirname(abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.validators.silent_failure import FailureContext, detect_silent_failures  # noqa: E402

# inspect-ai reports truncation as one of these stop reasons; the validator
# speaks the OpenAI-style "length", so normalise on the way in.
_TRUNCATION_STOP_REASONS = {"max_tokens", "model_length"}


@metric
def silent_failure_rate() -> Metric:
    """Fraction of samples that tripped at least one silent-failure flag."""

    def compute(scores: Sequence[SampleScore]) -> Value:
        if not scores:
            return 0.0
        flagged = sum(
            1 for s in scores if (s.score.metadata or {}).get("silent_failure_flags")
        )
        return flagged / len(scores)

    return compute


@scorer(metrics=[accuracy(), stderr(), silent_failure_rate()])
def retrieval_with_silent_failures():
    """Score needle retrieval and attach silent-failure flags as metadata."""

    async def score(state: TaskState, target: Target) -> Score:
        completion = state.output.completion or ""

        # Recover the provider stop reason + token usage when available.
        stop_reason = None
        if state.output.choices:
            stop_reason = state.output.choices[0].stop_reason
        finish_reason = "length" if stop_reason in _TRUNCATION_STOP_REASONS else stop_reason

        usage = state.output.usage
        prompt_tokens = usage.input_tokens if usage else None
        completion_tokens = usage.output_tokens if usage else None

        meta = state.metadata or {}
        needle_answer = target.text or meta.get("needle_answer")

        ctx = FailureContext(
            response_text=completion,
            finish_reason=finish_reason,
            expects_bangla=bool(meta.get("expects_bangla", True)),
            needle_answer=needle_answer,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            short_answer_task=bool(meta.get("short_answer_task", False)),
        )
        flags = detect_silent_failures(ctx)

        # Headline correctness: the gold needle answer must appear in the output.
        # Normalise both sides to NFC first, so a model that emits decomposed
        # Bengali (e.g. ড় as ড + ◌়) still matches an NFC needle -- and so this
        # verdict never disagrees with the detector's context-drop flag, which
        # also NFC-normalises (src/validators/silent_failure.py).
        if needle_answer:
            norm_needle = unicodedata.normalize("NFC", needle_answer.strip())
            norm_completion = unicodedata.normalize("NFC", completion)
            correct = bool(norm_needle) and norm_needle in norm_completion
        else:
            correct = False

        return Score(
            value=CORRECT if correct else INCORRECT,
            answer=completion.strip()[:200],
            metadata={
                "silent_failure_flags": flags,
                "stop_reason": stop_reason,
                "depth": meta.get("depth"),
                "target_tokens": meta.get("target_tokens"),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        )

    return score
