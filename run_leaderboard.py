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
    python3 run_leaderboard.py                              # belebele_ben_full.jsonl
    python3 run_leaderboard.py belebebele_ben_sample.jsonl
    python3 run_leaderboard.py bangla_mcq_native.jsonl
    python3 run_leaderboard.py --all-benchmarks
    python3 run_leaderboard.py belebele_ben_full.jsonl bangla_mcq_native.jsonl

Only models whose API key env var is actually set are run; the rest are skipped,
so you can benchmark a subset by exporting only the keys you have.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import bangla_bench_runner as r

VALIDATION_STATUS_PATH = "validation_status.json"

# Record the installed litellm version for provenance in the leaderboard header.
def _litellm_version() -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("litellm")
        except PackageNotFoundError:
            pass
    except Exception:  # pragma: no cover
        pass
    try:
        import litellm as _litellm

        return getattr(_litellm, "__version__", "unknown")
    except Exception:  # pragma: no cover
        return "unknown"


LITELLM_VERSION = _litellm_version()

# --------------------------------------------------------------------------- #
# The model lineup. Frontier proprietary via native API keys, open-weight via
# NIM, Bangla-native via Hugging Face Inference (HF_TOKEN) or a custom endpoint
# (HF_API_BASE env var for self-hosted TGI/vLLM).
# --------------------------------------------------------------------------- #
UNIFORM_MAX_TOKENS = 2048
SIMPLE_MAX_TOKENS = 32

# Optional override for self-hosted Bangla-native models (TGI/vLLM OpenAI API).
_HF_API_BASE = os.environ.get("HF_API_BASE") or None

MODELS = [
    # label                  litellm model string                                      key env var          api_base (None = provider default)         max_tokens
    ("DeepSeek V4 Pro",      "deepseek/deepseek-v4-pro",                               "DEEPSEEK_API_KEY",  None,                                       UNIFORM_MAX_TOKENS),
    ("DeepSeek V3",          "deepseek/deepseek-chat",                                 "DEEPSEEK_API_KEY",  None,                                       SIMPLE_MAX_TOKENS),
    ("DeepSeek R1",          "deepseek/deepseek-reasoner",                             "DEEPSEEK_API_KEY",  None,                                       UNIFORM_MAX_TOKENS),
    ("GPT-5.5",              "openai/gpt-5.5",                                         "OPENAI_API_KEY",    None,                                       UNIFORM_MAX_TOKENS),
    ("Claude Opus 4.5",      "anthropic/claude-opus-4-5-20251101",                     "ANTHROPIC_API_KEY", None,                                       UNIFORM_MAX_TOKENS),
    ("Gemini 3.1 Pro",       "gemini/gemini-3.1-pro-preview",                          "GEMINI_API_KEY",    None,                                       UNIFORM_MAX_TOKENS),
    ("Llama 3.3 70B (NIM)",  "openai/meta/llama-3.3-70b-instruct",                     "NVIDIA_API_KEY",    "https://integrate.api.nvidia.com/v1",      SIMPLE_MAX_TOKENS),
    ("TigerLLM 9B",          "huggingface/huggingface/md-nishat-008/TigerLLM-9B-it",   "HF_TOKEN",          _HF_API_BASE,                               UNIFORM_MAX_TOKENS),
    ("TituLLM Gemma 2B",     "huggingface/huggingface/hishab/titulm-gemma-2-2b-v1.1", "HF_TOKEN",         _HF_API_BASE,                               UNIFORM_MAX_TOKENS),
]

BENCHMARKS = {
    "belebele_ben_full.jsonl": {
        "slug": "belebele",
        "title": "Belebele (Bengali) Reading Comprehension",
        "category": "reading_comprehension",
    },
    "belebele_ben_sample.jsonl": {
        "slug": "belebele_sample",
        "title": "Belebele (Bengali) Sample",
        "category": "reading_comprehension",
    },
    "bangla_mcq_native.jsonl": {
        "slug": "native_mcq",
        "title": "Native Bengali MCQ",
        "category": "native_mcq",
    },
}


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
    out_path = f"results_{label.replace(' ', '_').replace('/', '-')}_{Path(dataset).stem}.jsonl"
    summary = r.evaluate_file_concurrent(cfg, dataset, out_path, max_workers=6)
    return summary


def count_dataset_items(path):
    """Count non-empty lines in the dataset file; None if it can't be read."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return sum(1 for line in fh if line.strip())
    except OSError:
        return None


def load_validation_status() -> dict:
    try:
        with open(VALIDATION_STATUS_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}


def validation_line(dataset: str) -> str:
    status = load_validation_status().get(dataset, {})
    if not status:
        return "- Validation: not yet recorded (see `validation_status.json`)\n"
    verified = status.get("verified", 0)
    total = status.get("total", "?")
    review_status = status.get("status", "unknown")
    reviewer = status.get("reviewer") or "pending"
    review_date = status.get("review_date") or "pending"
    return (
        f"- Validation: {verified}/{total} verified · status `{review_status}` · "
        f"reviewer {reviewer} · {review_date}\n"
    )


def benchmark_slug(dataset: str) -> str:
    meta = BENCHMARKS.get(dataset)
    if meta:
        return meta["slug"]
    return Path(dataset).stem.replace(".", "_")


def benchmark_title(dataset: str) -> str:
    meta = BENCHMARKS.get(dataset)
    if meta:
        return meta["title"]
    return Path(dataset).stem


def evaluate_dataset(base, dataset: str) -> list[dict]:
    rows = []
    for label, model, key_env, api_base, max_tokens in MODELS:
        if not os.environ.get(key_env):
            print(f"[skip] {label}: {key_env} not set")
            continue
        print(f"[run ] {label} ({model}) on {dataset} ...")
        s = run_one(base, dataset, label, model, key_env, api_base, max_tokens)
        rows.append({
            "benchmark": benchmark_slug(dataset),
            "dataset": dataset,
            "model": label,
            "model_id": model,
            "max_tokens": max_tokens,
            "accuracy": round(s["accuracy"], 4),
            "correct": s["correct"],
            "parsed": s["parsed"],
            "total": s["total"],
        })
    return rows


def write_leaderboard(dataset: str, rows: list[dict]) -> None:
    slug = benchmark_slug(dataset)
    md_path = f"leaderboard_{slug}.md" if slug != "belebele" else "leaderboard.md"
    csv_path = f"leaderboard_{slug}.csv" if slug != "belebele" else "leaderboard.csv"

    rows.sort(key=lambda x: x["accuracy"], reverse=True)
    item_count = count_dataset_items(dataset)
    item_count_str = "unknown" if item_count is None else str(item_count)
    run_date = datetime.now(timezone.utc).isoformat()

    csv_fields = [
        "model", "model_id", "max_tokens", "accuracy", "correct", "parsed", "total",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=csv_fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row[k] for k in csv_fields})

    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(f"# BanglaBench — {benchmark_title(dataset)} Leaderboard\n\n")
        fh.write(f"- Run date (UTC): {run_date}\n")
        fh.write(f"- Dataset: `{dataset}` · {item_count_str} items\n")
        fh.write("- Scoring: 4-way MCQ · temperature 0 · closed-book\n")
        fh.write(
            "- max_tokens: 2048 for reasoning/frontier models; "
            "32 for non-reasoning (V3, Llama 3.3 NIM)\n"
        )
        fh.write(f"- litellm version: {LITELLM_VERSION}\n")
        fh.write(validation_line(dataset))
        fh.write("\n")
        fh.write("| Rank | Model | Model ID | max_tokens | Accuracy | Correct/Total | Parsed |\n")
        fh.write("|---|---|---|---|---|---|---|\n")
        for i, x in enumerate(rows, 1):
            fh.write(
                f"| {i} | {x['model']} | `{x['model_id']}` | {x['max_tokens']} | "
                f"{x['accuracy']*100:.1f}% | "
                f"{x['correct']}/{x['total']} | {x['parsed']}/{x['total']} |\n"
            )

    print(f"wrote {md_path} + {csv_path}")


def write_summary(all_rows: list[dict]) -> None:
    if not all_rows:
        return
    run_date = datetime.now(timezone.utc).isoformat()
    benchmarks = sorted({r["benchmark"] for r in all_rows})
    with open("leaderboard_summary.md", "w", encoding="utf-8") as fh:
        fh.write("# BanglaBench — Cross-Benchmark Summary\n\n")
        fh.write(f"- Run date (UTC): {run_date}\n")
        fh.write(f"- Benchmarks: {', '.join(benchmarks)}\n")
        fh.write(f"- litellm version: {LITELLM_VERSION}\n\n")
        fh.write("| Benchmark | Model | Accuracy | Correct/Total | Parsed |\n")
        fh.write("|---|---|---|---|---|\n")
        for row in sorted(all_rows, key=lambda x: (x["benchmark"], -x["accuracy"])):
            fh.write(
                f"| {row['benchmark']} | {row['model']} | "
                f"{row['accuracy']*100:.1f}% | "
                f"{row['correct']}/{row['total']} | {row['parsed']}/{row['total']} |\n"
            )
    with open("leaderboard_summary.csv", "w", newline="", encoding="utf-8") as fh:
        fields = [
            "benchmark", "dataset", "model", "model_id", "max_tokens",
            "accuracy", "correct", "parsed", "total",
        ]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)
    print("wrote leaderboard_summary.md + leaderboard_summary.csv")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BanglaBench per-model leaderboard.")
    parser.add_argument(
        "datasets",
        nargs="*",
        help="JSONL dataset path(s). Default: belebele_ben_full.jsonl",
    )
    parser.add_argument(
        "--all-benchmarks",
        action="store_true",
        help="Run all registered benchmark datasets.",
    )
    return parser.parse_args(argv)


def resolve_datasets(args: argparse.Namespace) -> list[str]:
    if args.all_benchmarks:
        return list(BENCHMARKS.keys())
    if args.datasets:
        return args.datasets
    return ["belebele_ben_full.jsonl"]


def main(argv):
    args = parse_args(argv)
    datasets = resolve_datasets(args)
    base = r.RunnerConfig.load("config.yaml")

    all_rows: list[dict] = []
    any_ran = False

    for dataset in datasets:
        if not Path(dataset).exists():
            print(f"[error] dataset not found: {dataset}", file=sys.stderr)
            return 1
        print(f"\n=== benchmark: {dataset} ===")
        rows = evaluate_dataset(base, dataset)
        if not rows:
            print(f"[warn] no models ran for {dataset}")
            continue
        any_ran = True
        write_leaderboard(dataset, rows)
        all_rows.extend(rows)

    if not any_ran:
        key_list = ", ".join(sorted({k for _, _, k, _, _ in MODELS}))
        print(
            "[error] No models ran: none of the API keys in MODELS are set "
            f"({key_list}). Export at least one key and re-run.",
            file=sys.stderr,
        )
        return 1

    if len(datasets) > 1:
        write_summary(all_rows)

    print("\n=== leaderboard ===")
    print(json.dumps(all_rows, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
