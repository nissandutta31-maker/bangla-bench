# BanglaBench

**A unified, native-speaker-maintained leaderboard for Bengali (বাংলা) LLM evaluation.**

Bengali has 250M+ speakers and almost no consolidated, up-to-date LLM evaluation. Academic Bengali benchmarks are scattered across individual papers, and general "multilingual" leaderboards usually score Bengali through machine translation — which measures the translation as much as the model. BanglaBench is one place to see how frontier proprietary models *and* Bangla-native models actually perform on Bengali tasks: scored directly in Bengali, closed-book, each model on its own, and refreshed as new models ship.

> **Status: v0.** Current benchmark: Belebele (Bengali). More benchmarks and broader model coverage on the [roadmap](#roadmap).

---

## Current standings

The live leaderboard is auto-generated in [`leaderboard.md`](./leaderboard.md) (and `leaderboard.csv`) by `run_leaderboard.py`, which scores **each model independently** — no cross-provider blending, no translation.

| Rank | Model | Model ID | max_tokens | Accuracy | Correct/Total | Parsed |
|------|-------|----------|------------|----------|----------------|--------|
| 1 | GPT-5.5 | `openai/gpt-5.5` | 2048 | 92.0% | 92/100 | 100/100 |
| 2 | Claude Opus 4.8 | `anthropic/claude-opus-4-8` | 2048 | 92.0% | 92/100 | 100/100 |
| 3 | DeepSeek R1 | `deepseek/deepseek-reasoner` | 2048 | 85.0% | 85/100 | 95/100 |
| 4 | Llama 3.3 70B (NIM) | `openai/meta/llama-3.3-70b-instruct` | 32 | 84.0% | 84/100 | 100/100 |

*100-item subset of Belebele Bengali. Full v1 lineup also includes DeepSeek V4 Pro, TituLLM, and TigerLLM — run `run_leaderboard.py` with your keys to refresh. See [`leaderboard.md`](./leaderboard.md) for the latest auto-generated standings.*

For context: the random floor on a 4-way MCQ is 25%, and frontier models score well above it on Belebele reading comprehension. A healthy v0 baseline should look nothing like chance.

---

## Methodology

- **Dataset:** Belebele Bengali split (`ben_Beng`) — 900 multiple-choice reading-comprehension items, included in full ([`belebele_ben_full.jsonl`](./belebele_ben_full.jsonl)). The 100-item dev subset ([`belebele_ben_100.jsonl`](./belebele_ben_100.jsonl)) is the first 100 lines of the full file (`belebele_ben_000`–`belebele_ben_099`). Regenerate any N-item prefix with `./scripts/make_subset.sh N`.
- **Task:** 4-way MCQ. The model receives the **passage**, the question, and four options, and answers with a single letter (A–D).
- **Scoring:** exact-letter match · **temperature 0 where supported** · **closed-book** (no tools, no retrieval).
- **Per-model independence:** every model is scored and ranked on its own. Scores are never averaged or failed-over across providers. (The multi-provider failover in the runner is reliability infrastructure, not part of scoring — see [The runner](#the-runner-infrastructure).)
- **Per-model `max_tokens`:** 2048 for reasoning models (R1, V4 Pro — hidden CoT tokens consume budget), 32 for non-reasoning models (single-letter answer).
- **Closed-book only:** models that perform live web retrieval (e.g. Perplexity Sonar *online* models) are excluded from the leaderboard or reported separately — retrieval breaks the closed-book condition.
- **Native-speaker maintained:** answer keys and item fluency are reviewed by a native Bengali speaker rather than trusted blindly from machine translation. A systematic per-item verification pass is on the roadmap.

---

## Model coverage (v1 lineup)

The active `MODELS` list in [`run_leaderboard.py`](run_leaderboard.py):

| Pole | Model | Key env var |
|------|-------|-------------|
| Open-weight | DeepSeek V4 Pro | `DEEPSEEK_API_KEY` |
| Frontier proprietary | GPT-5.5 | `OPENAI_API_KEY` |
| Frontier proprietary | Claude Opus 4.8 | `ANTHROPIC_API_KEY` |
| Open-weight baseline | Llama 3.3 70B (NVIDIA NIM) | `NVIDIA_API_KEY` |
| Bangla-native | TituLLM 3B | `HF_TOKEN` + `HF_TITULLM_API_BASE` |
| Bangla-native | TigerLLM 9B | `HF_TOKEN` + `HF_TIGERLLM_API_BASE` |

---

## Quickstart — reproduce the leaderboard

```bash
pip install -r requirements.txt

# Export keys for whichever models you want ranked (others are skipped).
export DEEPSEEK_API_KEY=...   # DeepSeek V4 Pro
export OPENAI_API_KEY=...     # GPT-5.5
export ANTHROPIC_API_KEY=...  # Claude Opus 4.8
export NVIDIA_API_KEY=...     # Llama 3.3 70B (NIM)
export HF_TOKEN=...           # TituLLM + TigerLLM
export HF_TITULLM_API_BASE=https://xxxxx.endpoints.huggingface.cloud
export HF_TIGERLLM_API_BASE=https://yyyyy.endpoints.huggingface.cloud

# Full v1 lineup (all keys above):
python3 run_leaderboard.py belebele_ben_sample.jsonl

# Frontier + DeepSeek only (skip Llama / Bangla-native):
python3 run_leaderboard.py --frontier-only belebele_ben_sample.jsonl
```

This evaluates each model listed in **`run_leaderboard.py` → `MODELS`** independently on the dataset and writes ranked `leaderboard.md` and `leaderboard.csv`. Only models whose API key env var is set are run.

### API key reference

| Model | Env var(s) | Notes |
|-------|------------|-------|
| DeepSeek V4 Pro | `DEEPSEEK_API_KEY` | Reasoning model; 2048 max_tokens |
| GPT-5.5 | `OPENAI_API_KEY` | Frontier proprietary |
| Claude Opus 4.8 | `ANTHROPIC_API_KEY` | `anthropic/claude-opus-4-8` |
| Llama 3.3 70B (NIM) | `NVIDIA_API_KEY` | Open-weight baseline via NVIDIA NIM |
| TituLLM 3B | `HF_TOKEN`, `HF_TITULLM_API_BASE` | HF Inference Endpoint URL |
| TigerLLM 9B | `HF_TOKEN`, `HF_TIGERLLM_API_BASE` | HF Inference Endpoint URL |

### Bangla-native models (HF Inference Endpoints)

TituLLM and TigerLLM are wired in `MODELS` but skipped until you set endpoint URLs:

1. Deploy on [Hugging Face Inference Endpoints](https://huggingface.co/inference-endpoints):
   - `hishab/titulm-llama-3.2-3b-v1.1` (T4 or A10G)
   - `md-nishat-008/TigerLLM-9B-it` (A10G or L4)
2. Export:
   ```bash
   export HF_TOKEN=hf_...
   export HF_TITULLM_API_BASE=https://<your-titullm-id>.endpoints.huggingface.cloud
   export HF_TIGERLLM_API_BASE=https://<your-tigerllm-id>.endpoints.huggingface.cloud
   ```
3. Smoke: `python3 run_leaderboard.py belebele_ben_sample.jsonl`

### Dataset sizes

```bash
python3 run_leaderboard.py belebele_ben_sample.jsonl   # 30 items — cheap smoke
python3 run_leaderboard.py belebele_ben_100.jsonl      # 100 items — dev subset
python3 run_leaderboard.py belebele_ben_full.jsonl     # 900 items — full benchmark
```

---

## The runner (infrastructure)

`bangla_bench_runner.py` is the underlying evaluation engine: provider routing, retry with exponential backoff, cross-provider failover for reliability, answer parsing, and a per-attempt CSV audit log. It's usable on its own:

```bash
# single prompt
python3 bangla_bench_runner.py prompt "প্রশ্ন: বাংলাদেশের রাজধানী কোনটি? A. ঢাকা B. খুলনা C. চট্টগ্রাম D. সিলেট"

# evaluate a JSONL file
python3 bangla_bench_runner.py eval sample_items.jsonl -o results.jsonl
```

**Note:** the `eval` subcommand uses cross-provider failover, which *blends* providers and is **not** how the leaderboard is scored. For per-model standings, always use `run_leaderboard.py`.

---

## Configuration

`config.yaml` controls failover serving (used by `bangla_bench_runner.py eval`), retry/backoff, the Bengali system prompt, and the CSV log path. **The leaderboard model lineup is edited in `run_leaderboard.py` → `MODELS`**, not in `config.yaml`. API keys are never stored — only env-var names are referenced.

---

## Input format (Belebele / BanglaBench JSONL)

One JSON object per line. The normalizer accepts common field-name variants:

- **id:** `item_id` · `id` · `task_id`
- **passage:** `flores_passage` · `passage` · `context`
- **question:** `question` · `question_text`
- **choices:** `mc_answer1`…`mc_answer4` *or* a `choices` list
- **gold:** `correct_answer_num` (1–4 → A–D) · `answer` · `correct_answer`

```json
{"item_id": "ex-001", "flores_passage": "…", "question": "…", "mc_answer1": "…", "mc_answer2": "…", "mc_answer3": "…", "mc_answer4": "…", "correct_answer_num": 2}
```

---

## What's included

| File | Purpose |
|------|---------|
| `run_leaderboard.py` | Scores each model independently; generates the leaderboard |
| `leaderboard.md` / `leaderboard.csv` | Auto-generated standings (canonical) |
| `bangla_bench_runner.py` | Evaluation engine: routing, failover, parsing, logging |
| `belebele_ben_full.jsonl` | Full Belebele Bengali split (900 items) |
| `belebele_ben_100.jsonl` | 100-item dev subset (first 100 lines of full) |
| `belebele_ben_sample.jsonl` | 30-item sample for quick runs |
| `scripts/make_subset.sh` | Generate deterministic N-item prefix subsets |
| `config.yaml` | Failover providers, prompts, retry, logging |
| `test_smoke.py` | Offline unit tests (no API calls) |

---

## Roadmap

- Regenerate v0 standings on the full split across frontier + open-weight + Bangla-native models
- Add Bengali benchmarks beyond Belebele (native MCQ, math, reasoning)
- Systematic native-speaker verification pass over items and answer keys
- Continuous refresh as new models ship

---

## Contributing

Issues and PRs welcome — especially: adding a model to the leaderboard, flagging a mistranslated or ambiguous item, or contributing a new Bengali benchmark. To add a model, add a tuple to `MODELS` in `run_leaderboard.py` and open a PR with the regenerated leaderboard.

---

## License

MIT.
