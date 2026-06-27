# BanglaBench

**A unified, native-speaker-maintained leaderboard for Bengali (বাংলা) LLM evaluation.**

Bengali has 250M+ speakers and almost no consolidated, up-to-date LLM evaluation. Academic Bengali benchmarks are scattered across individual papers, and general "multilingual" leaderboards usually score Bengali through machine translation — which measures the translation as much as the model. BanglaBench is one place to see how frontier proprietary models *and* Bangla-native models actually perform on Bengali tasks: scored directly in Bengali, closed-book, each model on its own, and refreshed as new models ship.

> **Status: v0.** Current benchmark: Belebele (Bengali). More benchmarks and broader model coverage on the [roadmap](#roadmap).

---

## Current standings

The live leaderboard is auto-generated in [`leaderboard.md`](./leaderboard.md) (and `leaderboard.csv`) by `run_leaderboard.py`, which scores **each model independently** — no cross-provider blending, no translation.

| Rank | Model | Model ID | max_tokens | Accuracy | Correct/Total | Parsed |
|------|-------|----------|------------|----------|----------------|--------|
| 1 | DeepSeek R1 | `deepseek/deepseek-reasoner` | 2048 | 88.0% | 88/100 | 94/100 |
| 2 | DeepSeek V4 Pro | `deepseek/deepseek-v4-pro` | 2048 | 84.0% | 84/100 | 93/100 |
| 3 | DeepSeek V3 | `deepseek/deepseek-chat` | 32 | 77.0% | 77/100 | 100/100 |

*Scored on [`belebele_ben_100.jsonl`](./belebele_ben_100.jsonl) — a committed, deterministic 100-item subset of Belebele Bengali (`head -n 100 belebele_ben_full.jsonl`), so the numbers are reproducible. Frontier proprietary (GPT-5.x / Claude Opus / Gemini 3.x) and Bangla-native (TigerLLM, TituLM) models are now wired into `run_leaderboard.py`; they appear on the board as soon as their API keys are exported and the eval is re-run.*

For context: the random floor on a 4-way MCQ is 25%, and frontier models score well above it on Belebele reading comprehension. A healthy v0 baseline should look nothing like chance.

---

## Methodology

- **Dataset:** Belebele Bengali split (`ben_Beng`) — 900 multiple-choice reading-comprehension items, included in full ([`belebele_ben_full.jsonl`](./belebele_ben_full.jsonl)). Belebele is human-translated by its original authors.
- **Task:** 4-way MCQ. The model receives the **passage**, the question, and four options, and answers with a single letter (A–D).
- **Scoring:** exact-letter match · **temperature 0** · **closed-book** (no tools, no retrieval).
- **Per-model independence:** every model is scored and ranked on its own. Scores are never averaged or failed-over across providers. (The multi-provider failover in the runner is reliability infrastructure, not part of scoring — see [The runner](#the-runner-infrastructure).)
- **Per-model `max_tokens`:** 2048 for reasoning models (R1, V4 Pro — hidden CoT tokens consume budget), 32 for non-reasoning models (single-letter answer).
- **Closed-book only:** models that perform live web retrieval (e.g. Perplexity Sonar *online* models) are excluded from the leaderboard or reported separately — retrieval breaks the closed-book condition.
- **Native-speaker maintained:** answer keys and item fluency are reviewed by a native Bengali speaker rather than trusted blindly from machine translation. A systematic per-item verification pass is on the roadmap.
- **Reproducible eval set:** the exact items the board is scored on are committed as [`belebele_ben_100.jsonl`](./belebele_ben_100.jsonl) (deterministically `head -n 100 belebele_ben_full.jsonl`). For a larger run, point the driver at the committed full 900-item split — the eval is **resumable**, so an interrupted (and paid) run restarts mid-way without re-spending on completed items.

---

## Model coverage (v0 target)

A meaningful Bengali leaderboard has to span both worlds:

- **Frontier proprietary** — GPT / Claude / Gemini class
- **Strong open-weight** — DeepSeek, Llama-family, Qwen
- **Bangla-native / Indic-tuned** — TituLLM and similar

---

## Quickstart — reproduce the leaderboard

```bash
pip install -r requirements.txt

# Keys are read from env vars; only the var *names* are referenced in code.
# Export only the keys you have — every model whose key is unset is skipped.
export DEEPSEEK_API_KEY=...     # DeepSeek V4 Pro / V3 / R1
export OPENAI_API_KEY=...       # GPT-5.x
export ANTHROPIC_API_KEY=...    # Claude Opus
export GEMINI_API_KEY=...       # Gemini 3.x
export HF_TOKEN=...             # Bangla-native: TigerLLM, TituLM (HuggingFace)

# Score every keyed model independently on the committed 100-item eval set.
python3 run_leaderboard.py belebele_ben_100.jsonl

# ...or run the full 900-item split (resumable; safe to re-run after interruption).
python3 run_leaderboard.py belebele_ben_full.jsonl
```

This evaluates each model in the `MODELS` list in `run_leaderboard.py` independently on the dataset and writes ranked `leaderboard.md` and `leaderboard.csv`. To add or drop a model, edit that list (not `config.yaml`).

> **Bangla-native models (HuggingFace):** the two native entries default to LiteLLM's serverless HuggingFace route, which only works if the model is currently on HF's shared inference fleet. For a real, repeatable run, deploy each repo as a [dedicated Inference Endpoint](https://endpoints.huggingface.co) and switch the entry to `model="huggingface/tgi"` with `api_base="https://<your-endpoint>/v1/"` — see the inline note in `run_leaderboard.py`.

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

`config.yaml` controls the model list (and order), per-model `model` / `api_base` / `max_tokens`, retry / backoff, the Bengali system prompt, and the CSV log path. API keys are never stored — only env-var names are referenced.

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
| `belebele_ben_100.jsonl` | Committed 100-item eval set the leaderboard is scored on (`head -n 100` of the full split) |
| `belebele_ben_sample.jsonl` | 30-item sample for quick runs |
| `config.yaml` | Models, prompts, retry, logging |
| `test_smoke.py` | Offline unit tests (no API calls) |
| `test_silent_failure.py` | Offline tests for silent-failure detector + haystack builder |
| `tasks/` | inspect-ai task suite (`bangla_needle_haystack`) |
| `src/validators/silent_failure.py` | Dependency-free "silent failure" detector |

---

## v0.1 harness (inspect-ai track)

Alongside the Belebele leaderboard, v0.1 adds an [inspect-ai](https://inspect.ai-safety-institute.org.uk/)
task suite aimed at **silent failures** — responses a model emits confidently
but that are broken in ways exact-match scoring misses. The first task,
`bangla_needle_haystack`, embeds a Bengali fact inside fluent Bengali filler
(drawn from the project's own Belebele passages) and sweeps context size ×
needle depth to find *where* a model drops context.

```bash
pip install -r requirements-inspect.txt    # pulls in inspect-ai

# Frontier (native key via LiteLLM):
inspect eval tasks/bangla_needle_haystack.py --model openai/gpt-5.5

# Local open-weight (Ollama), custom sweep:
inspect eval tasks/bangla_needle_haystack.py --model ollama/gemma-2-9b \
    -T target_tokens=1000,8000 -T depths=0.0,0.5,1.0
```

The scorer reports accuracy **and** a silent-failure rate; per-sample flags
(`FALSE_NEGATIVE_CONTEXT_DROP`, `REASONING_TRUNCATED`, `MOJIBAKE_DETECTED`,
`SCRIPT_DRIFT_ENGLISH`, `TOKENIZER_EXPLOSION`) are written to the inspect log
for post-mortem analysis. The detector in `src/validators/silent_failure.py` is
dependency-free and unit-tested offline (`python3 test_silent_failure.py`).

This track is **additive** — the LiteLLM leaderboard path does not depend on
inspect-ai.

---

## Roadmap

- Regenerate v0 standings on the full split across frontier + open-weight + Bangla-native models
- Add Bengali benchmarks beyond Belebele (native MCQ, math, reasoning)
- Systematic native-speaker verification pass over items and answer keys
- Continuous refresh as new models ship

---

## Contributing

Issues and PRs welcome — especially: adding a model to the leaderboard, flagging a mistranslated or ambiguous item, or contributing a new Bengali benchmark. To add a model, add it to `config.yaml` and open a PR with the regenerated leaderboard.

---

## License

MIT.
