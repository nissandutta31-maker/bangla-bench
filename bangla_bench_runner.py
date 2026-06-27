#!/usr/bin/env python3
"""Bangla Bench LiteLLM runner.

Routes Bangla-language MCQ benchmark requests across multiple LLM providers
(NVIDIA hosted models, DeepSeek V4 Pro, Perplexity) using LiteLLM, with
per-provider exponential-backoff retries and cross-provider failover. Results
are written as JSONL and every attempt is appended to a CSV audit log.

No API keys are ever hardcoded; keys are read from environment variables whose
names are declared in the config file.
"""

from __future__ import annotations

import argparse
import atexit
import csv
import hashlib
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml

try:
    import litellm
    from litellm import completion
except ImportError:  # pragma: no cover - exercised only without the dep installed
    litellm = None
    completion = None

# Reasoning models reject params like temperature=0 / small max_tokens. Letting
# litellm silently drop unsupported params keeps a single config working across
# both classic and thinking models instead of erroring per provider.
if litellm is not None:
    litellm.drop_params = True


# --------------------------------------------------------------------------- #
# Config models
# --------------------------------------------------------------------------- #
@dataclass
class ProviderConfig:
    name: str
    model: str
    api_key_env: str
    api_base: Optional[str] = None
    temperature: Optional[float] = 0.0
    max_tokens: int = 16

    @property
    def api_key(self) -> Optional[str]:
        return os.environ.get(self.api_key_env)


@dataclass
class RetryConfig:
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    jitter: bool = True


@dataclass
class RunnerConfig:
    providers: list[ProviderConfig]
    retry: RetryConfig
    system_prompt: str
    csv_path: str
    prompt_preview_chars: int = 120
    csv_include_response: bool = False
    csv_batch_size: int = 100

    @classmethod
    def load(cls, path: str) -> "RunnerConfig":
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)

        providers = [ProviderConfig(**p) for p in raw.get("providers", [])]
        if not providers:
            raise ValueError("config must define at least one provider")

        retry = RetryConfig(**(raw.get("retry") or {}))
        logging_cfg = raw.get("logging") or {}
        return cls(
            providers=providers,
            retry=retry,
            system_prompt=raw.get("system_prompt", "").strip(),
            csv_path=logging_cfg.get("csv_path", "logs/bangla_bench_log.csv"),
            prompt_preview_chars=int(logging_cfg.get("prompt_preview_chars", 120)),
            csv_include_response=logging_cfg.get("csv_include_response", False),
            csv_batch_size=int(logging_cfg.get("csv_batch_size", 100)),
        )


# --------------------------------------------------------------------------- #
# Result / log record
# --------------------------------------------------------------------------- #
CSV_FIELDS = [
    "timestamp",
    "task_id",
    "provider",
    "model",
    "prompt_hash",
    "prompt_preview",
    "response_text",
    "parsed_answer",
    "tokens_used",
    "latency_seconds",
    "status",
    "error_type",
    "error_message",
    "retry_count",
    "failover_used",
]


@dataclass
class AttemptResult:
    timestamp: str
    task_id: str
    provider: str
    model: str
    prompt_hash: str
    prompt_preview: str
    response_text: str = ""
    parsed_answer: Optional[str] = None
    tokens_used: Optional[int] = None
    latency_seconds: Optional[float] = None
    status: str = "error"
    error_type: str = ""
    error_message: str = ""
    retry_count: int = 0
    failover_used: bool = False

    def as_row(self) -> dict[str, Any]:
        return {k: ("" if getattr(self, k) is None else getattr(self, k)) for k in CSV_FIELDS}


# --------------------------------------------------------------------------- #
# Logging (batched for performance)
# --------------------------------------------------------------------------- #
class BufferedCSVWriter:
    """Thread-safe buffered CSV writer that flushes in batches."""
    
    def __init__(self, csv_path: str, fieldnames: list[str], batch_size: int = 100):
        self.csv_path = csv_path
        self.fieldnames = fieldnames
        self.batch_size = batch_size
        self.buffer: list[dict[str, Any]] = []
        self.header_written = False
        self._lock = __import__("threading").Lock()
        
        # Check if file already exists with header
        path = Path(csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size > 0:
            self.header_written = True
    
    def write(self, row: dict[str, Any]) -> None:
        with self._lock:
            self.buffer.append(row)
            if len(self.buffer) >= self.batch_size:
                self._flush_locked()
    
    def flush(self) -> None:
        with self._lock:
            self._flush_locked()
    
    def _flush_locked(self) -> None:
        if not self.buffer:
            return
        Path(self.csv_path).parent.mkdir(parents=True, exist_ok=True)
        new_file = not self.header_written
        with open(self.csv_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.fieldnames)
            if new_file:
                writer.writeheader()
                self.header_written = True
            writer.writerows(self.buffer)
        self.buffer.clear()


# Global buffered writers per CSV path (thread-safe singleton pattern)
_csv_writers: dict[str, BufferedCSVWriter] = {}
_csv_writers_lock = __import__("threading").Lock()


def get_csv_writer(csv_path: str, fieldnames: list[str] = CSV_FIELDS, batch_size: int = 100) -> BufferedCSVWriter:
    """Get or create a buffered CSV writer for the given path."""
    with _csv_writers_lock:
        if csv_path not in _csv_writers:
            _csv_writers[csv_path] = BufferedCSVWriter(csv_path, fieldnames, batch_size)
        return _csv_writers[csv_path]


def log_attempt_csv_flush_all():
    """Flush all buffered CSV writers."""
    with _csv_writers_lock:
        for writer in _csv_writers.values():
            try:
                writer.flush()
            except OSError:
                # Path may already be gone (e.g. tempfile cleanup at interpreter exit).
                pass


# Safety net: on normal interpreter exit, flush any buffered CSV rows so the
# audit log is never silently lost (covers the single-prompt CLI path and any
# caller that forgets to flush explicitly).
atexit.register(log_attempt_csv_flush_all)


def log_attempt(csv_path: str, result: AttemptResult, include_response: bool = False, batch_size: int = 100) -> None:
    """Append a single attempt result to the CSV log via buffered writer.
    
    Args:
        csv_path: Path to CSV file
        result: AttemptResult to log
        include_response: If False (default), omits response_text to save I/O.
        batch_size: Buffer flush threshold.
    """
    row = result.as_row()
    if not include_response:
        row["response_text"] = ""
    writer = get_csv_writer(csv_path, batch_size=batch_size)
    writer.write(row)


# --------------------------------------------------------------------------- #
# Answer parsing & token extraction
# --------------------------------------------------------------------------- #
# Explicit "answer: X" style markers (English + Bengali). We take the LAST
# such match so a model that reasons aloud and then restates its final choice
# is scored on the restatement, not an earlier mention.
_ANSWER_MARKER_RE = re.compile(
    r"(?:answer|উত্তর|ans|final)\b[\s:：\-.)=]*(?:is\s+)?\b([ABCD])\b",
    re.IGNORECASE,
)
_STANDALONE_LETTER_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)
# Stricter fallback for reasoning / thinking models whose final letter is
# *fused* to Bengali script or punctuation (e.g. "সঠিক উত্তরঃC" or "উত্তর-C।").
# Python's \b sees Bengali code points as word chars, so \b([ABCD])\b never
# fires when a letter is glued to Bengali text -> the answer was being scored as
# unparsed (and therefore wrong). This pattern matches an UPPERCASE A-D that is
# only required to be free of adjacent ASCII alphanumerics, so it accepts a
# letter touching Bengali characters or a trailing danda while still ignoring
# things like "D-Day" -> "Day" (D is followed by an ASCII letter) and the
# English article "a"/"A" embedded in a word. Uppercase-only keeps it from
# grabbing the English article "a" or stray lowercase letters in prose.
_ISOLATED_LETTER_RE = re.compile(r"(?<![A-Za-z0-9])([ABCD])(?![A-Za-z0-9])")


def parse_answer(text: str) -> Optional[str]:
    """Extract a single A-D answer letter from model output, or None.

    Precedence (tuned for reasoning / CoT output so we never grab the first
    stray letter in a passage echo like "D-Day" or "A(H5N1)"):
      1. The LAST explicit answer marker ("answer: C", "উত্তর- B", ...).
      2. The whole reply being exactly one bare letter.
      3. The LAST standalone A-D letter on the LAST non-empty line only.
      4. (stricter fallback) The LAST UPPERCASE A-D on the last non-empty line
         that is merely free of adjacent ASCII alphanumerics -- recovers a
         letter fused to Bengali text/punctuation where step 3's \\b fails.
      5. (stricter fallback) The same isolated-letter scan walking the
         remaining lines bottom-up, for thinking models whose final letter
         lands a line or two above the literal last line.

    Steps 4-5 only ever run when steps 1-3 found nothing, so they can only turn
    a previously-unparsed (always-wrong) answer into a parsed one -- they can
    never change a letter that an earlier, higher-precedence step already
    resolved, and so can never lower a model's accuracy.
    """
    if not text:
        return None
    cleaned = text.strip()

    marker_matches = _ANSWER_MARKER_RE.findall(cleaned)
    if marker_matches:
        return marker_matches[-1].upper()

    if cleaned.upper() in _LETTERS:
        return cleaned.upper()

    nonempty_lines = [ln for ln in cleaned.splitlines() if ln.strip()]
    if nonempty_lines:
        last_line = nonempty_lines[-1]
        line_matches = _STANDALONE_LETTER_RE.findall(last_line)
        if line_matches:
            return line_matches[-1].upper()

        isolated = _ISOLATED_LETTER_RE.findall(last_line)
        if isolated:
            return isolated[-1].upper()

        for ln in reversed(nonempty_lines[:-1]):
            isolated = _ISOLATED_LETTER_RE.findall(ln)
            if isolated:
                return isolated[-1].upper()
    return None


def extract_tokens(response: Any) -> Optional[int]:
    """Pull total token usage out of a LiteLLM response object/dict."""
    usage = None
    if response is None:
        return None
    if isinstance(response, dict):
        usage = response.get("usage")
    else:
        usage = getattr(response, "usage", None)
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage.get("total_tokens")
    return getattr(usage, "total_tokens", None)


def extract_text(response: Any) -> str:
    """Pull the assistant message content out of a LiteLLM response."""
    try:
        if isinstance(response, dict):
            return response["choices"][0]["message"]["content"] or ""
        return response.choices[0].message.content or ""
    except (KeyError, IndexError, AttributeError, TypeError):
        return ""


# --------------------------------------------------------------------------- #
# Rate-limit / transient error detection
# --------------------------------------------------------------------------- #
_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}
_RETRYABLE_NAME_HINTS = (
    "ratelimit",
    "rate_limit",
    "timeout",
    "serviceunavailable",
    "service_unavailable",
    "apiconnection",
    "internalserver",
    "overloaded",
)


def is_retryable_error(exc: Exception) -> bool:
    """Best-effort classification of transient/rate-limit errors from LiteLLM."""
    # 1) LiteLLM exception classes (when available).
    if litellm is not None:
        retryable_classes = tuple(
            cls
            for cls in (
                getattr(litellm, "RateLimitError", None),
                getattr(litellm, "Timeout", None),
                getattr(litellm, "APIConnectionError", None),
                getattr(litellm, "ServiceUnavailableError", None),
                getattr(litellm, "InternalServerError", None),
                getattr(litellm, "APIError", None),
            )
            if cls is not None
        )
        if retryable_classes and isinstance(exc, retryable_classes):
            return True

    # 2) HTTP status code attached to the exception.
    status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
    if isinstance(status, int) and status in _RETRYABLE_STATUS:
        return True

    # 3) Fall back to fuzzy matching on the class name / message.
    name = type(exc).__name__.lower()
    if any(hint in name for hint in _RETRYABLE_NAME_HINTS):
        return True
    msg = str(exc).lower()
    if "rate limit" in msg or "too many requests" in msg or "429" in msg:
        return True
    return False


def backoff_delay(attempt: int, retry: RetryConfig) -> float:
    """Exponential backoff with optional full jitter. attempt is 0-indexed."""
    delay = min(retry.base_delay * (2 ** attempt), retry.max_delay)
    if retry.jitter:
        delay = random.uniform(0, delay)
    return delay


# --------------------------------------------------------------------------- #
# Core call + failover
# --------------------------------------------------------------------------- #
def _build_messages(system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return messages


def call_with_failover(
    config: RunnerConfig,
    user_prompt: str,
    task_id: str = "single",
    sleep_fn=time.sleep,
) -> AttemptResult:
    """Try each provider in priority order; retry transient errors with backoff.

    Returns the AttemptResult of the first success, or the last failed attempt
    if every provider is exhausted. Every individual attempt is logged to CSV.
    """
    if completion is None:
        raise RuntimeError(
            "litellm is not installed. Run `pip install -r requirements.txt`."
        )

    prompt_hash = hashlib.sha256(user_prompt.encode("utf-8")).hexdigest()[:16]
    preview = user_prompt.replace("\n", " ")[: config.prompt_preview_chars]
    messages = _build_messages(config.system_prompt, user_prompt)

    last_result: Optional[AttemptResult] = None

    for provider_idx, provider in enumerate(config.providers):
        failover_used = provider_idx > 0

        if not provider.api_key:
            result = AttemptResult(
                timestamp=_now(),
                task_id=task_id,
                provider=provider.name,
                model=provider.model,
                prompt_hash=prompt_hash,
                prompt_preview=preview,
                status="error",
                error_type="MissingAPIKey",
                error_message=f"env var {provider.api_key_env} is not set",
                failover_used=failover_used,
            )
            log_attempt(config.csv_path, result, include_response=config.csv_include_response, batch_size=config.csv_batch_size)
            last_result = result
            continue

        # max_retries total attempts against this provider.
        for attempt in range(config.retry.max_retries):
            start = time.monotonic()
            try:
                kwargs: dict[str, Any] = dict(
                    model=provider.model,
                    messages=messages,
                    api_key=provider.api_key,
                    max_tokens=provider.max_tokens,
                )
                if provider.temperature is not None:
                    kwargs["temperature"] = provider.temperature
                if provider.api_base:
                    kwargs["api_base"] = provider.api_base

                response = completion(**kwargs)
                latency = time.monotonic() - start
                text = extract_text(response)
                result = AttemptResult(
                    timestamp=_now(),
                    task_id=task_id,
                    provider=provider.name,
                    model=provider.model,
                    prompt_hash=prompt_hash,
                    prompt_preview=preview,
                    response_text=text,
                    parsed_answer=parse_answer(text),
                    tokens_used=extract_tokens(response),
                    latency_seconds=round(latency, 3),
                    status="success",
                    retry_count=attempt,
                    failover_used=failover_used,
                )
                log_attempt(config.csv_path, result, include_response=config.csv_include_response, batch_size=config.csv_batch_size)
                return result

            except Exception as exc:  # noqa: BLE001 - we classify below
                latency = time.monotonic() - start
                retryable = is_retryable_error(exc)
                result = AttemptResult(
                    timestamp=_now(),
                    task_id=task_id,
                    provider=provider.name,
                    model=provider.model,
                    prompt_hash=prompt_hash,
                    prompt_preview=preview,
                    latency_seconds=round(latency, 3),
                    status="error",
                    error_type=type(exc).__name__,
                    error_message=str(exc)[:500],
                    retry_count=attempt,
                    failover_used=failover_used,
                )
                log_attempt(config.csv_path, result, include_response=config.csv_include_response, batch_size=config.csv_batch_size)
                last_result = result

                # Non-retryable errors: stop hammering this provider, fail over.
                if not retryable:
                    break
                # Retryable, and we have attempts left: back off and retry.
                if attempt < config.retry.max_retries - 1:
                    sleep_fn(backoff_delay(attempt, config.retry))
                # else: exhausted retries for this provider -> fall through to failover

    # Every provider exhausted.
    return last_result if last_result is not None else AttemptResult(
        timestamp=_now(),
        task_id=task_id,
        provider="",
        model="",
        prompt_hash=prompt_hash,
        prompt_preview=preview,
        status="error",
        error_type="NoProviders",
        error_message="no providers attempted",
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# Bangla Bench item handling
# --------------------------------------------------------------------------- #
@dataclass
class MCQItem:
    item_id: str
    passage: str
    question: str
    choices: list[str]          # exactly the answer texts, index 0 -> A
    correct_answer: Optional[str] = None  # normalized to a letter A-D if known


_LETTERS = ["A", "B", "C", "D"]


def normalize_item(raw: dict[str, Any], fallback_id: str) -> MCQItem:
    """Map a Belebele/BanglaBench-style record into a normalized MCQItem."""
    item_id = str(
        raw.get("item_id")
        or raw.get("id")
        or raw.get("task_id")
        or fallback_id
    )
    passage = str(
        raw.get("flores_passage")
        or raw.get("passage")
        or raw.get("context")
        or ""
    )
    question = str(
        raw.get("question")
        or raw.get("question_text")
        or ""
    )

    # Choices: support mc_answer1-4 or a "choices" list.
    choices: list[str] = []
    if isinstance(raw.get("choices"), list):
        choices = [str(c) for c in raw["choices"]]
    else:
        for i in range(1, 5):
            val = raw.get(f"mc_answer{i}")
            if val is not None:
                choices.append(str(val))

    correct = _normalize_correct(raw, len(choices))
    return MCQItem(
        item_id=item_id,
        passage=passage,
        question=question,
        choices=choices,
        correct_answer=correct,
    )


def _normalize_correct(raw: dict[str, Any], n_choices: int) -> Optional[str]:
    """Normalize a gold answer (number 1-4, index, or letter) to a letter."""
    val = (
        raw.get("correct_answer_num")
        if raw.get("correct_answer_num") is not None
        else raw.get("answer")
    )
    if val is None:
        val = raw.get("correct_answer")
    if val is None:
        return None

    s = str(val).strip().upper()
    if s in _LETTERS:
        return s
    if s.isdigit():
        num = int(s)
        # Treat 1-based (1->A) since Belebele uses correct_answer_num in 1..4.
        idx = num - 1 if num >= 1 else num
        if 0 <= idx < len(_LETTERS):
            return _LETTERS[idx]
    return None


def render_prompt(item: MCQItem) -> str:
    """Build the user prompt text from a normalized MCQ item."""
    lines = []
    if item.passage:
        lines.append(f"অনুচ্ছেদ:\n{item.passage}\n")
    lines.append(f"প্রশ্ন: {item.question}\n")
    lines.append("বিকল্পসমূহ:")
    for letter, choice in zip(_LETTERS, item.choices):
        lines.append(f"{letter}. {choice}")
    lines.append("\nউত্তর (শুধু একটি অক্ষর A/B/C/D):")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Evaluation drivers
# --------------------------------------------------------------------------- #
def iter_jsonl(path: str) -> Iterable[dict[str, Any]]:
    """Yield parsed JSON objects from a JSONL file.

    A single malformed line is skipped with a stderr warning rather than
    aborting the whole (potentially long, paid) run.
    """
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"[iter_jsonl] skipping malformed JSON at {path}:{line_no}: {exc}",
                    file=sys.stderr,
                )


def evaluate_file(
    config: RunnerConfig,
    input_path: str,
    output_path: str,
    max_workers: int = 1,
) -> dict[str, Any]:
    """Evaluate every MCQ item in a JSONL file; write JSONL raw results.

    Resumable: if output_path already exists, previously scored item_ids are
    loaded and skipped, and new results are APPENDED. This lets an interrupted
    900-item paid run be restarted without re-spending on completed items. The
    returned summary reflects the FULL set scored so far (prior + new).

    When max_workers is 1, items are processed sequentially. When max_workers
    is greater than 1, items are processed in parallel via a thread pool.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    correct = 0
    parsed = 0
    skipped = 0
    errors = 0
    scored_ids: set[str] = set()

    # Seed counters/ids from any prior progress so a resumed run keeps
    # accumulating instead of starting the totals over from zero.
    if out.exists():
        for prior in iter_jsonl(str(out)):
            item_id = str(prior.get("item_id", ""))
            if item_id:
                scored_ids.add(item_id)
            total += 1
            if prior.get("predicted") is not None:
                parsed += 1
            if prior.get("is_correct"):
                correct += 1
            if prior.get("status") != "success":
                errors += 1

    def process_item(idx: int, item: MCQItem):
        prompt = render_prompt(item)
        result = call_with_failover(config, prompt, task_id=item.item_id)
        is_correct = (
            result.parsed_answer is not None
            and item.correct_answer is not None
            and result.parsed_answer == item.correct_answer
        )
        record = {
            "item_id": item.item_id,
            "provider": result.provider,
            "model": result.model,
            "predicted": result.parsed_answer,
            "gold": item.correct_answer,
            "is_correct": is_correct,
            "status": result.status,
            "response_text": result.response_text,
            "tokens_used": result.tokens_used,
            "latency_seconds": result.latency_seconds,
            "retry_count": result.retry_count,
            "failover_used": result.failover_used,
            "error_type": result.error_type,
        }
        return (idx, item.item_id, result, is_correct, record)

    with open(out, "a", encoding="utf-8") as out_fh:
        if max_workers > 1:
            items_to_process = []
            for idx, raw in enumerate(iter_jsonl(input_path)):
                item = normalize_item(raw, fallback_id=f"item_{idx}")
                if item.item_id in scored_ids:
                    skipped += 1
                    continue
                items_to_process.append((idx, item))

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_item = {
                    executor.submit(process_item, idx, item): (idx, item)
                    for idx, item in items_to_process
                }
                for future in as_completed(future_to_item):
                    idx, item_id, result, is_correct, record = future.result()
                    total += 1
                    if result.parsed_answer is not None:
                        parsed += 1
                    if is_correct:
                        correct += 1
                    if result.status != "success":
                        errors += 1
                    # Durable write as each item completes. The as_completed loop
                    # runs in this single (consumer) thread, so writes are already
                    # serialized -> an interrupted run stays resumable instead of
                    # losing every buffered-in-memory result.
                    out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_fh.flush()
                    scored_ids.add(item_id)
        else:
            for idx, raw in enumerate(iter_jsonl(input_path)):
                item = normalize_item(raw, fallback_id=f"item_{idx}")
                if item.item_id in scored_ids:
                    skipped += 1
                    continue
                idx, item_id, result, is_correct, record = process_item(idx, item)
                total += 1
                if result.parsed_answer is not None:
                    parsed += 1
                if is_correct:
                    correct += 1
                if result.status != "success":
                    errors += 1
                out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                # Flush per item so a crash mid-run leaves a resumable file.
                out_fh.flush()
                scored_ids.add(item_id)

    # Flush the buffered CSV audit log (the last <batch_size rows would
    # otherwise stay stuck in memory and never reach disk).
    log_attempt_csv_flush_all()

    accuracy = (correct / total) if total else 0.0
    return {
        "total": total,
        "parsed": parsed,
        "correct": correct,
        "accuracy": accuracy,
        "output_path": str(out),
        "csv_log": config.csv_path,
        "skipped": skipped,
        "errors": errors,
    }


def evaluate_file_concurrent(
    config: RunnerConfig,
    input_path: str,
    output_path: str,
    max_workers: int = 6,
) -> dict[str, Any]:
    """Evaluate MCQ items in parallel using a thread pool.

    Resumable: if output_path already exists, previously scored item_ids are
    loaded and skipped, and new results are APPENDED.
    """
    return evaluate_file(config, input_path, output_path, max_workers=max_workers)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bangla Bench LiteLLM runner with provider failover."
    )
    parser.add_argument(
        "--config", default="config.yaml", help="Path to YAML config (default: config.yaml)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_prompt = sub.add_parser("prompt", help="Run a single raw prompt.")
    p_prompt.add_argument("text", help="The user prompt text.")

    p_eval = sub.add_parser("eval", help="Evaluate a JSONL file of MCQ items.")
    p_eval.add_argument("input", help="Path to input JSONL.")
    p_eval.add_argument(
        "-o", "--output", default="results.jsonl", help="Output JSONL path (default: results.jsonl)."
    )
    p_eval.add_argument(
        "-w", "--workers", type=int, default=1, help="Parallel worker threads (default: 1)."
    )

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = RunnerConfig.load(args.config)

    if args.command == "prompt":
        result = call_with_failover(config, args.text, task_id="cli-prompt")
        print(json.dumps(result.as_row(), ensure_ascii=False, indent=2))
        return 0 if result.status == "success" else 1

    if args.command == "eval":
        summary = evaluate_file(config, args.input, args.output, max_workers=args.workers)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
