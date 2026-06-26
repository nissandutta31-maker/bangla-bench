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

*100-item subset of Belebele Bengali. See [`leaderboard.md`](./leaderboard.md) for the latest auto-generated standings.*

For context: the random floor on a 4-way MCQ is 25%, and frontier models score well above it on Belebele reading comprehension. A healthy v0 baseline should look nothing like chance.

---

## Methodology

- **Dataset:** Belebele Bengali split (`ben_Beng`) — 900 multiple-choice reading-comprehension items, included in full ([`belebele_ben_full.jsonl`](./belebele_ben_full.jsonl)). Belebele is human-translated by its original authors.
- **Task:** 4-way MCQ. The model receives the **passage**, the question, and four options, and answers with a single letter (A–D).
- **Scoring:** exact-letter match · **temperature 0 where supported** · **closed-book** (no tools, no retrieval).
- **Per-model independence:** every model is scored and ranked on its own. Scores are never averaged or failed-over across providers. (The multi-provider failover in the runner is reliability infrastructure, not part of scoring — see [The runner](#the-runner-infrastructure).)
- **Per-model `max_tokens`:** 2048 for reasoning models (R1, V4 Pro — hidden CoT tokens consume budget), 32 for non-reasoning models (single-letter answer).
- **Closed-book only:** models that perform live web retrieval (e.g. Perplexity Sonar *online* models) are excluded from the leaderboard or reported separately — retrieval breaks the closed-book condition.
- **Native-speaker maintained:** answer keys and item fluency are reviewed by a native Bengali speaker rather than trusted blindly from machine translation. A systematic per-item verification pass is on the roadmap.

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

# Keys are read from env vars; only the var *names* live in config.yaml.
export OPENAI_API_KEY=...
export DEEPSEEK_API_KEY=...
# (add one key per model you want ranked)

python3 run_leaderboard.py        # run with --help to see options
```

This evaluates each model listed in `config.yaml` independently on the dataset and writes ranked `leaderboard.md` and `leaderboard.csv`.

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
| `belebele_ben_sample.jsonl` | 30-item sample for quick runs |
| `config.yaml` | Models, prompts, retry, logging |
| `test_smoke.py` | Offline unit tests (no API calls) |

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
