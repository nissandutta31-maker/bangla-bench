# BanglaBench — Belebele (Bengali) Leaderboard

- Run date (UTC): 2026-06-24T20:14:27.496364+00:00
- Dataset: `belebele_ben_100.jsonl` · 100 items
- Scoring: 4-way MCQ · temperature 0 · closed-book
- max_tokens: 2048 for reasoning models (R1, V4 Pro), 32 for non-reasoning (V3)
- litellm version: 1.83.9

| Rank | Model | Model ID | max_tokens | Accuracy | Correct/Total | Parsed |
|---|---|---|---|---|---|---|
| 1 | DeepSeek R1 | `deepseek/deepseek-reasoner` | 2048 | 87.0% | 87/100 | 95/100 |
| 2 | DeepSeek V4 Pro | `deepseek/deepseek-v4-pro` | 2048 | 83.0% | 83/100 | 95/100 |
| 3 | DeepSeek V3 | `deepseek/deepseek-chat` | 32 | 77.0% | 77/100 | 100/100 |
