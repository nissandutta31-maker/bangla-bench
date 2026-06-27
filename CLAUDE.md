# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

BanglaBench is a native-speaker-maintained LLM leaderboard for Bengali (বাংলা). v0 scores models on the Belebele Bengali split (4-way MCQ reading comprehension). Python CLI only — no web UI, no database, no services to boot.

## Architecture

Two independent tracks, zero shared imports between them:

**Track 1 — LiteLLM leaderboard** (`requirements.txt`: `litellm`, `PyYAML`)
- `bangla_bench_runner.py` — evaluation engine: provider routing, retry with exponential backoff, cross-provider failover, answer parsing, CSV audit log. Used as a library.
- `run_leaderboard.py` — **the leaderboard driver.** Scores each model independently (single-provider config), ranks results, writes `leaderboard.md` + `leaderboard.csv`.
- `config.yaml` — system prompt + retry config + failover providers. The leaderboard driver reuses only `system_prompt` + `retry`; it overrides `providers`.
- `test_smoke.py` — 42 offline unit tests (never calls a live API).

**Track 2 — inspect-ai harness** (`requirements-inspect.txt`: `inspect-ai>=0.3.0`)
- `src/validators/silent_failure.py` — framework-agnostic detector for 5 failure modes.
- `tasks/haystack.py` — builds Bengali needle-in-haystack contexts from Belebele passages.
- `tasks/bangla_needle_haystack.py` — inspect-ai task sweeping context size × needle depth.
- `tasks/scorers.py` — inspect scorer reporting accuracy + silent-failure rate.
- `test_silent_failure.py` — 22 offline tests.

## Commands

```bash
# Setup
pip install -r requirements.txt

# Test (offline, no API keys needed)
python3 test_smoke.py
python3 test_silent_failure.py

# Run leaderboard (keys required for model scoring)
python3 run_leaderboard.py belebele_ben_100.jsonl    # 100-item committed eval set
python3 run_leaderboard.py belebele_ben_full.jsonl   # full 900-item split

# Run a single model (--only flag)
python3 run_leaderboard.py belebele_ben_100.jsonl --only "GPT-5.5"

# Dry-run pipeline without keys (exits 1, writes nothing — expected)
python3 run_leaderboard.py
```

## Critical design decisions

**The MODELS list in `run_leaderboard.py` is the single source of truth for the lineup.** `config.yaml`'s `providers:` section is only used by `bangla_bench_runner.py`'s failover serving mode, which is deliberately NOT how the leaderboard is scored. Never edit `config.yaml` to change the lineup.

**Failover ≠ benchmark.** `bangla_bench_runner.py eval` uses cross-provider failover (returns the first provider that answers). That blends providers into one accuracy number — correct for production, wrong for a leaderboard. `run_leaderboard.py` gives each model its own single-provider config and scores it independently.

**Every model is key-gated.** A model whose env var is unset is skipped with `[skip]`. There is no need to comment lines out of MODELS. Export only the keys you have.

**API keys are never stored.** Only env-var names appear in code. Export before running:
```
DEEPSEEK_API_KEY   — DeepSeek R1
OPENAI_API_KEY     — GPT-5.5
ANTHROPIC_API_KEY  — Claude Opus 4.8
GEMINI_API_KEY     — Gemini 3.1 Pro
NVIDIA_API_KEY     — Llama 3.3 70B (NIM)
HF_TOKEN           — TigerLLM 9B, TituLM 3B
```

## Answer parser precedence

`parse_answer()` in `bangla_bench_runner.py` uses strict precedence to avoid false positives from reasoning models:
1. Last explicit answer marker (`answer: C`, `উত্তর- B`)
2. Whole reply is exactly one bare letter
3. Last standalone `\b([ABCD])\b` on the last non-empty line
4. Last UPPERCASE A-D fused to Bengali script (isolated-letter fallback: `_ISOLATED_LETTER_RE`) — recovers cases like `উত্তরঃC` where `\b` fails because Bengali chars are word chars
5. Same isolated-letter scan walking previous lines bottom-up

Steps 4–5 only fire when steps 1–3 found nothing, so they can never lower a model's accuracy.

## Reproducibility

- `belebele_ben_100.jsonl` is committed — it's the exact eval set the published leaderboard is scored on (`head -n 100 belebele_ben_full.jsonl`).
- Eval is resumable: re-running against existing `results_*.jsonl` skips already-scored items, but only when the file's `dataset_sha256` matches the current dataset (legacy or mismatched files must be deleted or re-run with `--fresh`).
- `leaderboard.md` is tracked; `leaderboard.csv`, `results*.jsonl`, and `logs/` are gitignored.

## Model-specific notes

- **Reasoning/thinking models** (DeepSeek R1, GPT-5.5, Claude, Gemini) need `max_tokens=2048` or the final answer letter gets truncated behind hidden CoT tokens. The `temperature` field is `None` for frontier models that reject `temperature=0`.
- **Non-reasoning models** (Llama NIM) get `max_tokens=32`, temperature `0.0`.
- **Bangla-native models** (TigerLLM, TituLM) default to LiteLLM's serverless `huggingface/<org>/<repo>` route, which only works if HF is currently serving that model on its shared fleet. For repeatable runs, deploy a dedicated HF Inference Endpoint and switch to `model="huggingface/tgi"` + `api_base="https://<your-endpoint>/v1/"`.

## What to avoid

- Don't edit `config.yaml` to change the model lineup — edit `MODELS` in `run_leaderboard.py`.
- Don't add `inspect-ai` to the main `requirements.txt` — it belongs in `requirements-inspect.txt`.
- Don't commit `.env` files or API keys. `.env` is gitignored.
- Don't remove the `.drop_params = True` — reasoning models reject params like `temperature=0` and LiteLLM silently dropping unsupported params keeps a single config working across all models.
