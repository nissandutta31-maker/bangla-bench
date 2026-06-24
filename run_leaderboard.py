#!/usr/bin/env python3
"""BanglaBench leaderboard driver.

The base runner uses cross-provider *failover* (return the first provider that
answers). That is correct for a production app but WRONG for a benchmark: you
end up with one accuracy number stitched across whichever provider happened to
answer each item. A leaderboard needs each model evaluated INDEPENDENTLY and
scored on its own.

This script reuses bangla_bench_runner.py wholesale, but runs each model as its
own single-provider config, then aggregates the per-model accuracies into a
ranked table (leaderboard.md + leaderboard.csv).

Usage:
    python run_leaderboard.py belebele_ben_sample.jsonl

Only models whose API key env var is actually set are run; the rest are skipped,
so you can benchmark a subset by exporting only the keys you have.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import replace
from datetime import datetime, timezone

import bangla_bench_runner as r

# Record the installed litellm version for provenance in the leaderboard header.
# litellm exposes no __version__ attribute, so read it from package metadata
# (with __version__ as a fallback for older builds).
def _litellm_version() -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("litellm")
        except PackageNotFoundError:
            pass
    except Exception:  # pragma: no cover - importlib.metadata always present on 3.8+
        pass
    try:
        import litellm as _litellm

        return getattr(_litellm, "__version__", "unknown")
    except Exception:  # pragma: no cover - litellm not installed
        return "unknown"


LITELLM_VERSION = _litellm_version()

# --------------------------------------------------------------------------- #
# The model lineup. THIS is the strategic surface: frontier proprietary models
# reached through their *native* API keys, plus Bangla-native models as
# (currently commented) placeholders.
#
# Every entry uses the SAME generous max_tokens (UNIFORM_MAX_TOKENS). Reasoning
# / "thinking" models (GPT-5.5, Claude Opus 4.5, Gemini 3.1 Pro) spend hidden
# tokens before the visible answer, so a small budget truncates them before they
# emit the final A/B/C/D letter. A uniform budget is also required for a *fair*
# comparison: every model gets the same room to answer.
#
# Bangla-native models (TigerLLM, TituLLM) are left commented out — they need an
# HF (or other) API key plus a serving endpoint that isn't wired up yet.
# --------------------------------------------------------------------------- #
UNIFORM_MAX_TOKENS = 2048

MODELS = [
    # label                  litellm model string                     key env var          api_base (None = provider default)         max_tokens
    ("GPT-5.5",              "openai/gpt-5.5",                         "OPENAI_API_KEY",    None,                                       UNIFORM_MAX_TOKENS),
    ("Claude Opus 4.5",      "anthropic/claude-opus-4-5-20251101",     "ANTHROPIC_API_KEY", None,                                       UNIFORM_MAX_TOKENS),
    ("Gemini 3.1 Pro",       "gemini/gemini-3.1-pro-preview",          "GEMINI_API_KEY",    None,                                       UNIFORM_MAX_TOKENS),
    ("Llama 3.3 70B (NIM)",  "openai/meta/llama-3.3-70b-instruct",     "NVIDIA_API_KEY",    "https://integrate.api.nvidia.com/v1",      UNIFORM_MAX_TOKENS),
    # --- Bangla-native models: wire these once you have an HF/other key + endpoint --- #
    # ("TigerLLM",           "<hf/other route>",                       "HF_API_KEY",        "<serving endpoint>",                       UNIFORM_MAX_TOKENS),
    # ("TituLLM",            "<hf/other route>",                       "HF_API_KEY",        "<serving endpoint>",                       UNIFORM_MAX_TOKENS),
]


def run_one(base, dataset, label, model, key_env, api_base, max_tokens):
    prov = r.ProviderConfig(
        name=label, model=model, api_key_env=key_env,
        api_base=api_base, temperature=0.0, max_tokens=max_tokens,
    )
    cfg = replace(
        base,
        providers=[prov],
        csv_path=f"logs/leaderboard_{key_env}.csv",
    )
    out_path = f"results_{label.replace(' ', '_').replace('/', '-')}.jsonl"
    # Use concurrent evaluator for parallel item processing
    summary = r.evaluate_file_concurrent(cfg, dataset, out_path, max_workers=6)
    return summary


def count_dataset_items(path):
    """Count non-empty lines in the dataset file; None if it can't be read."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return sum(1 for line in fh if line.strip())
    except OSError:
        return None


def main(argv):
    dataset = argv[0] if argv else "belebele_ben_sample.jsonl"
    base = r.RunnerConfig.load("config.yaml")  # reuse the Bangla system prompt + retry (loaded ONCE)

    rows = []
    for label, model, key_env, api_base, max_tokens in MODELS:
        if not os.environ.get(key_env):
            print(f"[skip] {label}: {key_env} not set")
            continue
        print(f"[run ] {label} ({model}) ...")
        s = run_one(base, dataset, label, model, key_env, api_base, max_tokens)
        rows.append({
            "model": label,
            "model_id": model,
            "max_tokens": max_tokens,
            "accuracy": round(s["accuracy"], 4),
            "correct": s["correct"],
            "parsed": s["parsed"],
            "total": s["total"],
        })

    # No models ran (e.g. no API keys exported): don't emit an empty leaderboard
    # that looks like a real-but-failed run. Signal failure with a non-zero exit.
    if not rows:
        print(
            "[error] No models ran: none of the API keys in MODELS are set "
            f"({', '.join(k for _, _, k, _, _ in MODELS)}). "
            "Export at least one key and re-run. Leaderboard not written.",
            file=sys.stderr,
        )
        return 1

    rows.sort(key=lambda x: x["accuracy"], reverse=True)

    item_count = count_dataset_items(dataset)
    item_count_str = "unknown" if item_count is None else str(item_count)
    run_date = datetime.now(timezone.utc).isoformat()

    csv_fields = ["model", "model_id", "max_tokens", "accuracy", "correct", "parsed", "total"]
    with open("leaderboard.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=csv_fields)
        w.writeheader()
        w.writerows(rows)

    with open("leaderboard.md", "w", encoding="utf-8") as fh:
        fh.write("# BanglaBench — Belebele (Bengali) Leaderboard\n\n")
        fh.write(f"- Run date (UTC): {run_date}\n")
        fh.write(f"- Dataset: `{dataset}` · {item_count_str} items\n")
        fh.write("- Scoring: 4-way MCQ · temperature 0 · closed-book\n")
        fh.write(
            f"- max_tokens: {UNIFORM_MAX_TOKENS} (uniform across all models — required "
            "for a fair comparison of reasoning/thinking models)\n"
        )
        fh.write(f"- litellm version: {LITELLM_VERSION}\n\n")
        fh.write("| Rank | Model | Model ID | max_tokens | Accuracy | Correct/Total | Parsed |\n")
        fh.write("|---|---|---|---|---|---|---|\n")
        for i, x in enumerate(rows, 1):
            fh.write(f"| {i} | {x['model']} | `{x['model_id']}` | {x['max_tokens']} | "
                     f"{x['accuracy']*100:.1f}% | "
                     f"{x['correct']}/{x['total']} | {x['parsed']}/{x['total']} |\n")

    print("\n=== leaderboard ===")
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    print("\nwrote leaderboard.md + leaderboard.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
