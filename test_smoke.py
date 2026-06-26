#!/usr/bin/env python3
"""Lightweight smoke tests that do NOT call any external API.

Run with:  python test_smoke.py
Exits 0 if all checks pass, 1 otherwise.
"""
import json
import os
import tempfile

import bangla_bench_runner as r


def check(cond, label):
    print(f"[{'PASS' if cond else 'FAIL'}] {label}")
    return cond


def main() -> int:
    ok = True

    # --- answer parsing ---------------------------------------------------- #
    ok &= check(r.parse_answer("B") == "B", "parse plain letter")
    ok &= check(r.parse_answer("উত্তর: C") == "C", "parse Bangla 'answer:' prefix")
    ok &= check(r.parse_answer("(A)") == "A", "parse parenthesized letter")
    ok &= check(r.parse_answer("The answer is D.") == "D", "parse English sentence")
    ok &= check(r.parse_answer("") is None, "parse empty -> None")
    ok &= check(r.parse_answer("xyz 123") is None, "parse no-letter -> None")

    # --- adversarial parsing (reasoning / CoT / echo / stray letters) ------ #
    # Each of these returns the WRONG letter under a first-match parser; the
    # corrected parser must score the model's actual final choice.
    ok &= check(
        r.parse_answer("Option A is incorrect. B is also wrong. The final answer is C.") == "C",
        "CoT: take the stated final answer, not the first letter",
    )
    ok &= check(
        r.parse_answer("A is a tempting distractor.\nC") == "C",
        "echo + final line: last line's letter wins, not the first mention",
    )
    ok &= check(
        r.parse_answer("মার্কিন D-Day সম্পর্কে আলোচনা।\nB") == "B",
        "stray passage letter (D-Day) ignored; final-line letter wins",
    )
    ok &= check(
        r.parse_answer("Reasoning weighs A and D...\nউত্তর: D") == "D",
        "Bangla marker beats earlier reasoning letters",
    )
    ok &= check(
        r.parse_answer("Long reasoning about options.\nB\n") == "B",
        "penultimate line letter with trailing blank line",
    )
    ok &= check(
        r.parse_answer("Step 1: eliminate C.\nStep 2: choose B.\n\n") == "B",
        "CoT ending with letter on penultimate line before blank",
    )
    ok &= check(
        r.parse_answer("Analysis of all four options.\n\nB") == "B",
        "letter after blank line separator",
    )

    # --- token extraction -------------------------------------------------- #
    ok &= check(
        r.extract_tokens({"usage": {"total_tokens": 42}}) == 42,
        "extract tokens from dict",
    )
    ok &= check(r.extract_tokens(None) is None, "extract tokens from None")

    # --- text extraction --------------------------------------------------- #
    resp = {"choices": [{"message": {"content": "A"}}]}
    ok &= check(r.extract_text(resp) == "A", "extract text from response dict")
    ok &= check(r.extract_text({}) == "", "extract text from malformed -> ''")

    # --- item normalization (Belebele mc_answer style) --------------------- #
    raw1 = {
        "item_id": "x1",
        "flores_passage": "p",
        "question": "q",
        "mc_answer1": "a",
        "mc_answer2": "b",
        "mc_answer3": "c",
        "mc_answer4": "d",
        "correct_answer_num": 2,
    }
    item1 = r.normalize_item(raw1, "fallback")
    ok &= check(item1.item_id == "x1", "normalize item id")
    ok &= check(item1.choices == ["a", "b", "c", "d"], "normalize mc_answer choices")
    ok &= check(item1.correct_answer == "B", "normalize correct_answer_num 2 -> B")

    # --- item normalization (choices list + 1-based answer) ---------------- #
    raw2 = {"id": "x2", "context": "c", "question_text": "q", "choices": ["w", "x", "y", "z"], "answer": 1}
    item2 = r.normalize_item(raw2, "fallback")
    ok &= check(item2.choices == ["w", "x", "y", "z"], "normalize choices list")
    ok &= check(item2.correct_answer == "A", "normalize answer 1 -> A")

    # --- prompt rendering -------------------------------------------------- #
    prompt = r.render_prompt(item1)
    ok &= check("A. a" in prompt and "D. d" in prompt, "render includes lettered choices")
    ok &= check("প্রশ্ন" in prompt, "render includes Bangla question label")

    # --- retryable error detection ----------------------------------------- #
    class FakeRateLimit(Exception):
        status_code = 429

    class FakeFatal(Exception):
        status_code = 400

    ok &= check(r.is_retryable_error(FakeRateLimit("too many requests")), "429 is retryable")
    ok &= check(not r.is_retryable_error(FakeFatal("bad request")), "400 is not retryable")
    ok &= check(r.is_retryable_error(Exception("Rate limit exceeded")), "msg-based rate limit retryable")

    # --- backoff is bounded ------------------------------------------------ #
    rc = r.RetryConfig(base_delay=1.0, max_delay=5.0, jitter=False)
    ok &= check(r.backoff_delay(10, rc) == 5.0, "backoff capped at max_delay")

    # --- config loads ------------------------------------------------------ #
    cfg = r.RunnerConfig.load("config.yaml")
    ok &= check(len(cfg.providers) == 2, "config loads 2 providers")
    ok &= check(cfg.providers[0].name == "deepseek", "first provider is deepseek (priority)")
    ok &= check("A" in cfg.system_prompt, "system prompt mentions A/B/C/D")

    # --- CSV logging round-trip ------------------------------------------- #
    with tempfile.TemporaryDirectory() as d:
        csv_path = os.path.join(d, "log.csv")
        res = r.AttemptResult(
            timestamp="t", task_id="id", provider="nvidia", model="m",
            prompt_hash="h", prompt_preview="p", status="success",
            parsed_answer="A", tokens_used=10, latency_seconds=0.5,
        )
        r.log_attempt(csv_path, res)
        r.log_attempt(csv_path, res)
        r.log_attempt_csv_flush_all()  # Flush buffered writers
        with open(csv_path, encoding="utf-8") as fh:
            lines = fh.read().strip().splitlines()
        ok &= check(len(lines) == 3, "CSV has header + 2 rows")
        ok &= check(lines[0].startswith("timestamp"), "CSV header present")

    # --- failover with a mocked completion (no network) -------------------- #
    calls = {"n": 0}

    def fake_completion(**kwargs):
        calls["n"] += 1
        # First provider always rate-limits; second succeeds.
        if kwargs["model"] == cfg.providers[0].model:
            err = Exception("rate limit exceeded")
            err.status_code = 429
            raise err
        return {"choices": [{"message": {"content": "B"}}], "usage": {"total_tokens": 7}}

    orig_completion = r.completion
    orig_litellm = r.litellm
    try:
        r.completion = fake_completion
        r.litellm = None  # force message/status-based detection
        # provide fake keys so providers are attempted
        os.environ["DEEPSEEK_API_KEY"] = "test"
        os.environ["NVIDIA_API_KEY"] = "test"
        with tempfile.TemporaryDirectory() as d:
            cfg2 = r.RunnerConfig.load("config.yaml")
            cfg2.csv_path = os.path.join(d, "log.csv")
            cfg2.retry.base_delay = 0.0
            cfg2.retry.max_delay = 0.0
            out = r.call_with_failover(cfg2, "test prompt", task_id="fo", sleep_fn=lambda s: None)
            ok &= check(out.status == "success", "failover reaches a working provider")
            ok &= check(out.provider == "nvidia", "failover lands on second provider")
            ok &= check(out.failover_used is True, "failover_used flag set")
            ok &= check(out.parsed_answer == "B", "failover parses answer")
    finally:
        r.completion = orig_completion
        r.litellm = orig_litellm
        for k in ("NVIDIA_API_KEY", "DEEPSEEK_API_KEY"):
            os.environ.pop(k, None)

    print("\n" + ("ALL PASSED" if ok else "SOME FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
