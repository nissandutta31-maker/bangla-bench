# AGENTS.md

BanglaBench — a native-speaker-maintained LLM leaderboard for Bengali (বাংলা),
v0 scored on the Belebele Bengali split. It is a Python CLI/library project (no
web UI, no database, no services to boot). See `README.md` for the product story
and `FIXES.md` for design history.

## Cursor Cloud specific instructions

- **Setup / deps:** `pip install -r requirements.txt` (litellm + PyYAML). This is
  handled by the startup update script; no extra system deps.
- **Lint:** the repo ships no linter config; there is no lint step to run.
- **Test (offline, no network/keys):** `python3 test_smoke.py` — prints PASS/FAIL
  per check and `ALL PASSED`, exits non-zero on any failure. This is the only
  automated test suite and it never calls a live API.
- **Run the app (the leaderboard):** `python3 run_leaderboard.py belebele_ben_100.jsonl`
  (or `belebele_ben_full.jsonl` for the 900-item run). It writes ranked
  `leaderboard.md` + `leaderboard.csv`.
- **API keys are required to score real models, and none are present by default.**
  Keys are read from env vars *by name*; export only what you have. Any model
  whose key var is unset is skipped (`[skip] …`). Map: `DEEPSEEK_API_KEY`
  (DeepSeek V4 Pro/V3/R1), `OPENAI_API_KEY` (GPT-5.x), `ANTHROPIC_API_KEY`
  (Claude Opus), `GEMINI_API_KEY` (Gemini 3.x), `HF_TOKEN` (Bangla-native
  TigerLLM/TituLM via HuggingFace). With zero keys set, `run_leaderboard.py`
  exits non-zero and writes nothing — that is expected, not a bug.
- **Verifying the pipeline without keys:** monkeypatch `bangla_bench_runner.completion`
  with a local fake and call `run_leaderboard.main([...])`. This exercises the
  full config→eval→score→rank→output path offline. If you do this, restore the
  tracked `leaderboard.md` afterward (`git checkout -- leaderboard.md`), since the
  driver overwrites it.
- **The lineup lives in `run_leaderboard.py`'s `MODELS` list, NOT `config.yaml`.**
  `config.yaml` only feeds the system prompt + retry config to the leaderboard
  driver (its `providers:` list is used solely by `bangla_bench_runner.py`'s
  failover *serving* mode, which is deliberately NOT how the board is scored).
- **Bangla-native models (HuggingFace) caveat:** the two native entries default to
  LiteLLM's serverless `huggingface/<org>/<repo>` route, which only works if HF is
  currently serving that model on its shared fleet — not guaranteed for these
  small community models. For a repeatable run, deploy a dedicated HF Inference
  Endpoint and switch the entry to `model="huggingface/tgi"` + `api_base=".../v1/"`
  (inline note in `run_leaderboard.py`).
- **Reasoning / "thinking" models** need the large `max_tokens` budget
  (`UNIFORM_MAX_TOKENS=2048`); a small budget truncates the answer before the
  visible A–D letter. `parse_answer` has a stricter fallback that recovers a
  letter fused to Bengali script/punctuation (e.g. `উত্তরঃC`), so reasoning models
  parse cleanly. The fallback only ever converts a previously-unparsed answer, so
  it can never lower a model's accuracy.
- **Reproducibility:** `belebele_ben_100.jsonl` is intentionally committed (it is
  the exact eval set the board is scored on) and is derived deterministically via
  `head -n 100 belebele_ben_full.jsonl`. The eval is resumable: re-running against
  an existing `results_*.jsonl` skips already-scored items.
- **Gitignored artifacts:** `results*.jsonl`, `leaderboard.csv`, and `logs/` are
  ignored; `leaderboard.md` is tracked.
