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
    python3 run_leaderboard.py belebele_ben_sample.jsonl

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
# The model lineup. THIS is the strategic surface for BanglaBench v1.
#
# Reasoning / thinking models use UNIFORM_MAX_TOKENS (2048). Non-reasoning
# models that reliably emit a single letter can use SIMPLE_MAX_TOKENS (32).
#
# For Bangla-native HF models, set HF_TITULLM_API_BASE and HF_TIGERLLM_API_BASE
# to your Inference Endpoint URLs (env: prefix resolves at run time).
# --------------------------------------------------------------------------- #
UNIFORM_MAX_TOKENS = 2048
SIMPLE_MAX_TOKENS = 32

# Frontier proprietary models (GPT / Claude / Gemini via native API keys).
FRONTIER_KEY_ENVS = frozenset({"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"})

MODELS = [
    # label                  litellm model string                              key env var          api_base (None, URL, or env:VAR_NAME)              max_tokens
    ("DeepSeek V4 Pro",      "deepseek/deepseek-v4-pro",                        "DEEPSEEK_API_KEY",  None,                                                UNIFORM_MAX_TOKENS),
    ("GPT-5.5",              "openai/gpt-5.5",                                  "OPENAI_API_KEY",    None,                                                UNIFORM_MAX_TOKENS),
    ("Claude Opus 4.8",      "anthropic/claude-opus-4-8",                       "ANTHROPIC_API_KEY", None,                                                UNIFORM_MAX_TOKENS),
    ("Llama 3.3 70B (NIM)",  "openai/meta/llama-3.3-70b-instruct",              "NVIDIA_API_KEY",    "https://integrate.api.nvidia.com/v1",               SIMPLE_MAX_TOKENS),
    ("TituLLM 3B",           "huggingface/hishab/titulm-llama-3.2-3b-v1.1",     "HF_TOKEN",          "env:HF_TITULLM_API_BASE",                           UNIFORM_MAX_TOKENS),
    ("TigerLLM 9B",          "huggingface/md-nishat-008/TigerLLM-9B-it",        "HF_TOKEN",          "env:HF_TIGERLLM_API_BASE",                          UNIFORM_MAX_TOKENS),
]


def resolve_api_base(spec: str | None) -> str | None:
    """Resolve api_base: None, a literal URL, or env:VAR_NAME from the environment."""
    if spec is None:
        return None
    if spec.startswith("env:"):
        return os.environ.get(spec[4:]) or None
    return spec


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
    args = list(argv)
    frontier_only = False
    if "--frontier-only" in args:
        frontier_only = True
        args = [a for a in args if a != "--frontier-only"]
    if "-h" in args or "--help" in args:
        print(__doc__)
        print("Options:")
        print("  --frontier-only   Run only GPT-5.5 + Claude Opus 4.8 (skip DeepSeek, Llama, Bangla-native)")
        return 0

    dataset = args[0] if args else "belebele_ben_sample.jsonl"
    base = r.RunnerConfig.load("config.yaml")  # reuse the Bangla system prompt + retry (loaded ONCE)

    rows = []
    for label, model, key_env, api_base_spec, max_tokens in MODELS:
        if frontier_only and key_env not in FRONTIER_KEY_ENVS:
            print(f"[skip] {label}: --frontier-only (not a frontier proprietary model)")
            continue
        if not os.environ.get(key_env):
            print(f"[skip] {label}: {key_env} not set")
            continue
        api_base = resolve_api_base(api_base_spec)
        if api_base_spec and api_base_spec.startswith("env:") and not api_base:
            env_name = api_base_spec[4:]
            print(f"[skip] {label}: {env_name} not set (HF Inference Endpoint URL)")
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
        fh.write("- max_tokens: 2048 for reasoning models; 32 for Llama 3.3 (NIM)\n")
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
