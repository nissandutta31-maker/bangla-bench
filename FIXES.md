# BanglaBench runner — review, fixes, and how to run it

## Verdict
The agent built a genuinely sound scaffold. **30/30 offline smoke tests pass**, and
the full eval loop runs end-to-end on **real Belebele-Bengali data** (ingestion →
Bangla prompt rendering → scoring → JSONL/CSV output all verified). Keep the core
`bangla_bench_runner.py` — it's good.

## What was broken / needed fixing

1. **Perplexity model string was dead.** `perplexity/llama-3.1-sonar-large-128k-online`
   was retired Feb 2025 → it 400s. Also it's an *online* model that web-searches at
   query time, which would invalidate a closed-book benchmark (it can look up the
   answer). **Removed from the lineup.** If you ever want a Perplexity-served
   closed-book model, the offline one is `perplexity/r1-1776`.

2. **DeepSeek was actually correct.** `deepseek/deepseek-v4-pro` is a real, current
   model (V4 Pro shipped Apr 2026). Kept it. Caveat: if thinking mode is on, raise
   `max_tokens` (set to 32) or the final answer letter gets truncated.

3. **NVIDIA Llama-3.1-70B works but is a 2024 open model.** Fine as a *baseline*,
   not as the headline. Kept as baseline.

4. **Failover ≠ benchmark (the important one).** The runner returns the *first
   provider that answers* — correct for a production app, wrong for a leaderboard
   (you get one accuracy number stitched across providers). Added
   `run_leaderboard.py`, which evaluates **each model independently** and writes a
   ranked table. This is the piece that turns a "runner" into the report card.

5. **Real data.** `belebele_ben_full.jsonl` = the full 900-item Belebele Bengali
   split, already in the runner's schema. (`belebele_ben_sample.jsonl` = 30-item
   smoke set.)

## How to run

```bash
pip install -r requirements.txt          # litellm + pyyaml

# keys for whatever models you want to benchmark (only set what you have)
export DEEPSEEK_API_KEY=...
export NVIDIA_API_KEY=...
# export OPENAI_API_KEY=...   ANTHROPIC_API_KEY=...   GEMINI_API_KEY=...

# single model sanity check (failover serving mode)
python bangla_bench_runner.py eval belebele_ben_sample.jsonl -o results.jsonl

# THE leaderboard: every model scored separately, ranked -> leaderboard.md/.csv
python run_leaderboard.py belebele_ben_full.jsonl
```

## The one strategic change that matters
The lineup is the product. Right now it's DeepSeek + an old Llama. To be *the
Bengali report card*, the MODELS list in `run_leaderboard.py` should be:

- **Frontier proprietary** — GPT-5, Claude Opus 4.x, Gemini 3 (via their **native**
  keys; Perplexity's API does not serve these).
- **Bangla-native** — TigerLLM, TituLLM (HF / NIM endpoints).

That contrast — "do the frontier models actually understand Bengali, and do the
purpose-built Bangla models beat them?" — is the story nobody has shipped as a
live, native, current leaderboard. Everything above is already wired to produce it.

## Known nuance to watch
- For **reasoning models** (anything with a thinking mode), prefer non-thinking for
  this single-letter MCQ task, or the parser may grab a letter from the reasoning
  trace instead of the final answer. `temperature: 0` + short output = clean parse.
- Run the **full 900** (or a fixed seeded subset) and state the size + date in the
  leaderboard header so results are reproducible and comparable across model drops.
