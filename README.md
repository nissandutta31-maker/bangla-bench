# Bangla Bench LiteLLM Runner

A small, production-usable runner that evaluates Bangla-language multiple-choice
(MCQ) benchmark items (Belebele / BanglaBench style) against multiple LLM
providers via [LiteLLM](https://github.com/BerriAI/litellm), with automatic
retry, exponential backoff, and cross-provider failover.

Providers, in default priority order:

1. **NVIDIA** hosted models (NIM, OpenAI-compatible)
2. **DeepSeek V4 Pro**
3. **Perplexity API**

## Files

| File | Purpose |
| --- | --- |
| `bangla_bench_runner.py` | Main script: routing, failover, parsing, logging, CLI |
| `config.yaml` | Configuration template (providers, prompts, retry, logging) |
| `requirements.txt` | Python dependencies |
| `sample_items.jsonl` | Two example MCQ items |
| `test_smoke.py` | Offline smoke tests (no API calls) |

## Setup

```bash
pip install -r requirements.txt
```

### Environment variables (API keys)

Keys are **never** stored in config or code — only the env-var *names* are
referenced in `config.yaml`. Export the actual secrets before running:

```bash
export NVIDIA_API_KEY=your-nvidia-key
export DEEPSEEK_API_KEY=your-deepseek-key
export PERPLEXITY_API_KEY=your-perplexity-key
```

A provider with a missing key is skipped (logged as `MissingAPIKey`) and the
runner fails over to the next provider.

## Configuration

Edit `config.yaml` to set:

- **Provider priority** — order of the `providers:` list (top = first tried).
- **Model names** and `api_base` per provider.
- **Retry/backoff** — `max_retries`, `base_delay`, `max_delay`, `jitter`.
- **Bangla system prompt** — instructs the model to answer with a single
  `A`/`B`/`C`/`D` letter for deterministic scoring.
- **CSV log path** and prompt-preview length.

## Usage

### Single prompt

```bash
python3 bangla_bench_runner.py prompt "প্রশ্ন: বাংলাদেশের রাজধানী কোনটি? A. ঢাকা B. খুলনা C. চট্টগ্রাম D. সিলেট"
```

### Evaluate a JSONL file

```bash
python3 bangla_bench_runner.py eval sample_items.jsonl -o results.jsonl
```

Outputs:

- `results.jsonl` — one raw result record per item (predicted vs. gold, status,
  tokens, latency, failover info).
- The CSV log at `logging.csv_path` (default `logs/bangla_bench_log.csv`) — one
  row per *attempt*, including retries and failovers.
- A JSON summary on stdout (`total`, `parsed`, `correct`, `accuracy`).

### Custom config path

```bash
python3 bangla_bench_runner.py --config myconfig.yaml eval items.jsonl
```

## Input JSONL format

Compatible with Belebele / BanglaBench style. The normalizer accepts several
field name variants:

- **id**: `item_id` | `id` | `task_id`
- **passage**: `flores_passage` | `passage` | `context`
- **question**: `question` | `question_text`
- **choices**: `mc_answer1`..`mc_answer4` *or* a `choices` list
- **gold answer**: `correct_answer_num` | `answer` | `correct_answer`
  (a number `1`–`4` maps to `A`–`D`; a letter is used directly)

Example:

```json
{"item_id": "ex-001", "flores_passage": "...", "question": "...", "mc_answer1": "...", "mc_answer2": "...", "mc_answer3": "...", "mc_answer4": "...", "correct_answer_num": 2}
```

## CSV log columns

`timestamp, task_id, provider, model, prompt_hash, prompt_preview,
response_text, parsed_answer, tokens_used, latency_seconds, status,
error_type, error_message, retry_count, failover_used`

## Failover behavior

1. Try the first provider.
2. On a **retryable** error (rate limit / 429, timeouts, 5xx, connection
   errors — detected via LiteLLM exception classes, HTTP status codes, and
   message heuristics), retry up to `max_retries` with exponential backoff
   (optionally jittered).
3. On a **non-retryable** error (e.g. 400), stop retrying that provider and
   fail over immediately.
4. Move to the next provider in priority order; repeat.
5. Return the first success, or the last failure if all providers are exhausted.

## Testing

```bash
python3 test_smoke.py
```

These tests run fully offline (LiteLLM `completion` is mocked) and cover answer
parsing, token/text extraction, item normalization, prompt rendering, retryable
error detection, backoff bounding, config loading, CSV logging, and the
failover path.
