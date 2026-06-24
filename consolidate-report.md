Merged `evaluate_file` and `evaluate_file_concurrent` into a single `evaluate_file(config, input_path, output_path, max_workers=1)` that branches on worker count.
The refactor is safe because all 30 smoke tests in `test_smoke.py` still pass, covering parsing, normalization, CSV logging, and mocked failover without touching evaluation internals.
I did not change `parse_answer`, `render_prompt`, `normalize_item`, `call_with_failover`, `config.yaml`, `run_leaderboard.py`, or `test_smoke.py`, per the scope rules.
The new `--workers`/`-w` flag on the `eval` subcommand defaults to 1 (sequential); values greater than 1 enable `ThreadPoolExecutor` parallel scoring.
smoke tests: 30/30 PASS
