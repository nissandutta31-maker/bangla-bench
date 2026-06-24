# BanglaBench — Belebele (Bengali) Leaderboard

- Run date (UTC): 2026-06-24T05:52:11.611379+00:00
- Dataset: `belebele_ben_100.jsonl` · 100 items
- Scoring: 4-way MCQ · temperature 0 · closed-book
- max_tokens: 2048 for reasoning models (R1, V4 Pro), 32 for non-reasoning (V3)
- litellm version: 1.83.9

| Rank | Model | Model ID | max_tokens | Accuracy | Correct/Total | Parsed |
|---|---|---|---|---|---|---|
| 1 | DeepSeek R1 | `deepseek/deepseek-reasoner` | 2048 | 88.0% | 88/100 | 94/100 |
| 2 | DeepSeek V4 Pro | `deepseek/deepseek-v4-pro` | 2048 | 84.0% | 84/100 | 93/100 |
| 3 | DeepSeek V3 | `deepseek/deepseek-chat` | 32 | 77.0% | 77/100 | 100/100 |
