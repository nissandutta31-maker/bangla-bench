Merged `evaluate_file` and `evaluate_file_concurrent` into a single `evaluate_file(config, input_path, output_path, max_workers=1)` that branches on worker count.
The refactor preserves resumability, per-item flush, CSV flush, and summary dict keys; `test_smoke.py` exercises parsing, config, CSV logging, and failover without touching evaluation paths, confirming no regressions in shared helpers.
I did not change `run_leaderboard.py` because the task scoped edits to `evaluate_file`, its concurrent twin, and the CLI only; `evaluate_file_concurrent` remains as a thin wrapper so existing callers keep working unchanged.
The new `--workers`/`-w` flag on the `eval` subcommand accepts an integer thread count (default 1); values above 1 use `ThreadPoolExecutor`, while 1 keeps sequential processing.
smoke tests: 30/30 PASS
