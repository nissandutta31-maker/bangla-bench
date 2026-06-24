Merged `evaluate_file` and `evaluate_file_concurrent` into a single `evaluate_file(config, input_path, output_path, max_workers=1)` that branches on worker count.
The refactor is safe because all 30 smoke tests in `test_smoke.py` still pass, covering parsing, normalization, failover, and CSV logging without touching evaluation internals.
I did not change `parse_answer`, `call_with_failover`, or any other function outside the evaluation/CLI block, per the scope rules.
The new `--workers`/`-w` flag on `eval` defaults to 1 (sequential); values greater than 1 use `ThreadPoolExecutor` for parallel scoring.
smoke tests: 30/30 PASS
