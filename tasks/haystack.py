"""Build Bangla needle-in-a-haystack contexts from real Bengali text.

Pure, dependency-free helpers so the haystack construction can be unit-tested
offline. The inspect-ai ``Task`` that consumes these lives in
``bangla_needle_haystack.py``.

The filler ("haystack") is drawn from the project's own Belebele Bengali
passages so the distractor text is fluent, in-script Bengali rather than
machine noise -- a fairer stress test of long-context retrieval.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

# Bengali is token-inefficient on most tokenizers (frequently >1 token per
# character). This is only a rough budget knob for *constructing* contexts of a
# target size; the real token count is recorded from provider usage at run time.
APPROX_CHARS_PER_TOKEN = 2.0

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_FILLER = os.path.join(_REPO_ROOT, "belebele_ben_full.jsonl")


@dataclass(frozen=True)
class Needle:
    """A single retrievable fact and the question that retrieves it."""

    fact: str       # the sentence embedded in the haystack
    question: str   # what we ask the model afterwards
    answer: str     # the substring the answer MUST contain


# Default needle: a culturally-plausible Bengali fact unlikely to collide with
# the Belebele filler text. "Shihab eats tomatoes." -> "What does Shihab eat?"
DEFAULT_NEEDLE = Needle(
    fact="শিহাব প্রতিদিন সকালে টমেটো খায়।",
    question="শিহাব কী খায়?",
    answer="টমেটো",
)


def load_filler_sentences(path: str | None = None) -> list[str]:
    """Return Bengali passages from a Belebele-format JSONL file as filler.

    Accepts the same passage field-name variants the runner does
    (``flores_passage`` / ``passage`` / ``context``). Order is preserved so
    callers can build deterministic haystacks.
    """
    path = path or _DEFAULT_FILLER
    sentences: list[str] = []
    seen: set[str] = set()
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            passage = obj.get("flores_passage") or obj.get("passage") or obj.get("context")
            if passage and passage not in seen:
                seen.add(passage)
                sentences.append(passage.strip())
    if not sentences:
        raise ValueError(f"No Bengali passages found in filler source: {path}")
    return sentences


def build_haystack(
    needle: Needle,
    target_tokens: int,
    depth: float,
    filler: list[str],
    chars_per_token: float = APPROX_CHARS_PER_TOKEN,
) -> str:
    """Assemble a haystack of ~``target_tokens`` with the needle at ``depth``.

    ``depth`` is a fraction in [0, 1]: 0.0 places the needle at the very top of
    the context, 1.0 at the very bottom, 0.5 in the middle. Filler sentences are
    cycled (and prefixed with an index so repeats are not byte-identical) until
    the character budget is met, then the needle is spliced in at the requested
    depth between two sentence boundaries.
    """
    if not filler:
        raise ValueError("filler must be a non-empty list of Bengali passages")
    if not 0.0 <= depth <= 1.0:
        raise ValueError(f"depth must be in [0, 1], got {depth}")

    char_budget = int(target_tokens * chars_per_token)
    pieces: list[str] = []
    total = 0
    i = 0
    while total < char_budget:
        sent = filler[i % len(filler)]
        # Disambiguate cycled repeats so the context isn't trivially compressible.
        piece = f"[{i}] {sent}"
        pieces.append(piece)
        total += len(piece) + 1  # +1 for the joining space/newline
        i += 1

    # Splice the needle at the requested depth, on a sentence boundary.
    insert_at = round(depth * len(pieces))
    insert_at = max(0, min(insert_at, len(pieces)))
    pieces.insert(insert_at, needle.fact)
    return "\n".join(pieces)


def build_prompt(needle: Needle, haystack: str) -> str:
    """Wrap a haystack + question into a Bengali retrieval prompt."""
    return (
        "নিচে একটি দীর্ঘ অনুচ্ছেদ দেওয়া হলো। অনুচ্ছেদটি মনোযোগ দিয়ে পড়ুন এবং "
        "এর ভিত্তিতে প্রশ্নের উত্তর দিন।\n\n"
        f"--- অনুচ্ছেদ শুরু ---\n{haystack}\n--- অনুচ্ছেদ শেষ ---\n\n"
        f"প্রশ্ন: {needle.question}\n"
        "উত্তর বাংলায় সংক্ষেপে দিন।"
    )
