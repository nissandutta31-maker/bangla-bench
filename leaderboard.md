# BanglaBench — Belebele (Bengali) Leaderboard

- Run date (UTC): 2026-06-27T04:48:02.761138+00:00
- Dataset: `belebele_ben_sample.jsonl` · 30 items
- Scoring: 4-way MCQ · temperature 0 where supported · closed-book
- max_tokens: 2048 for reasoning models, 32 for non-reasoning (Llama NIM)
- litellm version: 1.83.9

| Rank | Model | Model ID | max_tokens | Accuracy | Correct/Total | Parsed |
|---|---|---|---|---|---|---|
| 1 | GPT-5.5 | `openai/gpt-5.5` | 2048 | 92.0% | 92/100 | 100/100 |
| 2 | Claude Opus 4.8 | `anthropic/claude-opus-4-8` | 2048 | 92.0% | 92/100 | 100/100 |
| 3 | DeepSeek R1 | `deepseek/deepseek-reasoner` | 2048 | 85.0% | 85/100 | 95/100 |
| 4 | Llama 3.3 70B (NIM) | `openai/meta/llama-3.3-70b-instruct` | 32 | 84.0% | 84/100 | 100/100 |
