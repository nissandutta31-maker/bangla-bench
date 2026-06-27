"""inspect-ai task: Bangla needle-in-a-haystack (context integrity).

Embeds a single Bengali fact inside a large body of fluent Bengali filler, then
asks the model to retrieve it. Sweeping the context size and the needle's depth
exposes *where* a model silently drops context -- the exact post-mortem signal
this harness is built to capture.

Run it with:

    inspect eval tasks/bangla_needle_haystack.py --model openai/gpt-5.5
    inspect eval tasks/bangla_needle_haystack.py --model ollama/gemma-2-9b \
        -T target_tokens=1000,8000 -T depths=0.0,0.5,1.0

The scorer (``retrieval_with_silent_failures``) reports accuracy AND a
silent-failure rate; per-sample flags are written into the inspect log.
"""
from __future__ import annotations

import sys
from os.path import abspath, dirname

# inspect-ai loads this file by path, so the `tasks` package is not on sys.path
# by default. Add the repo root (one level up) so the absolute imports below
# resolve whether run via `inspect eval` or imported directly.
_REPO_ROOT = dirname(dirname(abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.solver import generate

from tasks.haystack import (
    DEFAULT_NEEDLE,
    Needle,
    build_haystack,
    build_prompt,
    load_filler_sentences,
)
from tasks.scorers import retrieval_with_silent_failures

# Default sweep: small enough to run cheaply end-to-end, wide enough to show the
# drop-off. Override on the CLI with -T target_tokens=... -T depths=...
DEFAULT_TARGET_TOKENS = (1_000, 8_000, 32_000)
DEFAULT_DEPTHS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _as_int_tuple(value, default):
    if value is None:
        return tuple(default)
    if isinstance(value, (int, float)):
        return (int(value),)
    return tuple(int(v) for v in value)


def _as_float_tuple(value, default):
    if value is None:
        return tuple(default)
    if isinstance(value, (int, float)):
        return (float(value),)
    return tuple(float(v) for v in value)


def build_samples(
    target_tokens=DEFAULT_TARGET_TOKENS,
    depths=DEFAULT_DEPTHS,
    needle: Needle = DEFAULT_NEEDLE,
    filler_path: str | None = None,
) -> list[Sample]:
    """One Sample per (context size x needle depth) cell of the sweep."""
    filler = load_filler_sentences(filler_path)
    samples: list[Sample] = []
    for tokens in target_tokens:
        for depth in depths:
            haystack = build_haystack(needle, tokens, depth, filler)
            samples.append(
                Sample(
                    id=f"needle_t{tokens}_d{depth}",
                    input=build_prompt(needle, haystack),
                    target=needle.answer,
                    metadata={
                        "target_tokens": tokens,
                        "depth": depth,
                        "expects_bangla": True,
                        "needle_answer": needle.answer,
                        "needle_fact": needle.fact,
                    },
                )
            )
    return samples


@task
def bangla_needle_haystack(target_tokens=None, depths=None, filler_path=None) -> Task:
    """Bangla long-context retrieval sweep with silent-failure detection."""
    samples = build_samples(
        target_tokens=_as_int_tuple(target_tokens, DEFAULT_TARGET_TOKENS),
        depths=_as_float_tuple(depths, DEFAULT_DEPTHS),
        filler_path=filler_path,
    )
    return Task(
        dataset=MemoryDataset(samples=samples, name="bangla_needle_haystack"),
        solver=generate(),
        scorer=retrieval_with_silent_failures(),
    )
