#!/usr/bin/env python3
"""Offline project audit — no API calls. Writes NDJSON to debug log."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / ".cursor" / "debug-13e136.log"
SESSION = "13e136"


def log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "sessionId": SESSION,
        "runId": "audit",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def audit_temperature_kwargs() -> None:
    """H-A: Does call_with_failover always include temperature even when None?"""
    sys.path.insert(0, str(ROOT))
    import bangla_bench_runner as r

    prov = r.ProviderConfig(
        name="GPT-5.5", model="openai/gpt-5.5",
        api_key_env="OPENAI_API_KEY", temperature=None, max_tokens=2048,
    )
    # Mirror kwargs construction from call_with_failover
    kwargs = dict(
        model=prov.model,
        messages=[{"role": "user", "content": "test"}],
        api_key="fake",
        max_tokens=prov.max_tokens,
    )
    if prov.temperature is not None:
        kwargs["temperature"] = prov.temperature
    log("A", "project_audit:temperature", "kwargs built for frontier model", {
        "has_temperature_key": "temperature" in kwargs,
        "temperature_value": kwargs.get("temperature"),
        "provider_config_type": str(type(prov.temperature)),
    })


def audit_stale_results_merge() -> None:
    """H-B: Do existing error results get merged without re-run?"""
    sys.path.insert(0, str(ROOT))
    import run_leaderboard as rl

    merged = []
    for label, model, key_env, api_base, max_tokens, _temp in rl.MODELS:
        path = rl.results_path(label)
        if not os.path.exists(path):
            continue
        s = rl.summary_from_results(path)
        errors = 0
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("status") != "success":
                    errors += 1
        merged.append({
            "label": label, "total": s["total"], "parsed": s["parsed"],
            "accuracy": s["accuracy"], "error_rows": errors,
        })
    log("B", "project_audit:stale_merge", "existing results files on disk", {
        "files": merged,
        "would_skip_rerun": [m for m in merged if m["total"] > 0],
    })


def audit_readme_drift() -> None:
    """H-C: README standings vs leaderboard.md."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    lb = (ROOT / "leaderboard.md").read_text(encoding="utf-8")
    readme_has_gpt = "GPT-5.5" in readme and "92.0%" in readme
    lb_has_gpt = "GPT-5.5" in lb and "92.0%" in readme or "92.0%" in lb
    log("C", "project_audit:readme_drift", "README vs leaderboard.md", {
        "readme_has_gpt_92": readme_has_gpt,
        "leaderboard_has_gpt_92": "GPT-5.5" in lb and "92.0%" in lb,
        "readme_still_deepseek_v4": "DeepSeek V4 Pro" in readme,
        "leaderboard_row_count": lb.count("| 1 |"),
    })


def audit_hf_csv_collision() -> None:
    """H-D: Multiple models sharing same audit CSV path."""
    sys.path.insert(0, str(ROOT))
    import run_leaderboard as rl

    paths: dict[str, list[str]] = {}
    for label, _model, _key_env, _base, _mt, _t in rl.MODELS:
        csv_path = f"logs/leaderboard_{label.replace(' ', '_').replace('/', '-')}.csv"
        paths.setdefault(csv_path, []).append(label)
    collisions = {k: v for k, v in paths.items() if len(v) > 1}
    log("D", "project_audit:hf_csv", "CSV path collisions by model label", {
        "collisions": collisions,
    })


def audit_default_dataset() -> None:
    """H-E: Default dataset vs committed eval set."""
    committed = "belebele_ben_100.jsonl"
    log("E", "project_audit:default_dataset", "default dataset", {
        "main_default": "belebele_ben_100.jsonl",
        "committed_100_exists": (ROOT / committed).exists(),
        "sample_exists": (ROOT / "belebele_ben_sample.jsonl").exists(),
    })


def main() -> int:
    audit_temperature_kwargs()
    audit_stale_results_merge()
    audit_readme_drift()
    audit_hf_csv_collision()
    audit_default_dataset()
    print(f"Audit complete. Log: {LOG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
