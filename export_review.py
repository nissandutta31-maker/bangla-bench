#!/usr/bin/env python3
"""Export a Bengali benchmark JSONL to a native-speaker review spreadsheet.

Outputs a CSV with one row per item plus empty review_status / reviewer_notes
columns for a human reviewer to fill in. After review, use --apply to patch gold
keys in the JSONL from rows marked wrong_key with a corrected letter.

Usage:
    python3 export_review.py belebele_ben_full.jsonl review_belebele.csv
    python3 export_review.py bangla_mcq_native.jsonl review_native.csv
    python3 export_review.py --apply review_belebele.csv belebele_ben_full.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import bangla_bench_runner as r

LETTERS = ["A", "B", "C", "D"]
REVIEW_FIELDS = [
    "item_id",
    "passage",
    "question",
    "choice_a",
    "choice_b",
    "choice_c",
    "choice_d",
    "gold_letter",
    "review_status",
    "reviewer_notes",
    "corrected_letter",
]


def export_review(input_path: str, output_path: str) -> int:
    count = 0
    with open(input_path, encoding="utf-8") as inf, open(
        output_path, "w", newline="", encoding="utf-8"
    ) as outf:
        writer = csv.DictWriter(outf, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        for idx, raw in enumerate(r.iter_jsonl(input_path)):
            item = r.normalize_item(raw, fallback_id=f"item_{idx}")
            choices = (item.choices + ["", "", "", ""])[:4]
            writer.writerow({
                "item_id": item.item_id,
                "passage": item.passage,
                "question": item.question,
                "choice_a": choices[0],
                "choice_b": choices[1],
                "choice_c": choices[2],
                "choice_d": choices[3],
                "gold_letter": item.correct_answer or "",
                "review_status": "",
                "reviewer_notes": "",
                "corrected_letter": "",
            })
            count += 1
    print(f"exported {count} items -> {output_path}")
    return count


def apply_review(review_path: str, input_path: str, output_path: str | None) -> int:
    corrections: dict[str, str] = {}
    status_counts: dict[str, int] = {}
    with open(review_path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            status = (row.get("review_status") or "").strip().lower()
            if status:
                status_counts[status] = status_counts.get(status, 0) + 1
            corrected = (row.get("corrected_letter") or "").strip().upper()
            if corrected in LETTERS:
                corrections[row["item_id"]] = corrected

    dest = output_path or input_path
    tmp = Path(dest).with_suffix(".tmp.jsonl")
    updated = 0
    with open(input_path, encoding="utf-8") as inf, open(
        tmp, "w", encoding="utf-8"
    ) as outf:
        for line in inf:
            if not line.strip():
                continue
            obj = json.loads(line)
            item_id = str(obj.get("item_id") or obj.get("id") or "")
            if item_id in corrections:
                letter = corrections[item_id]
                idx = LETTERS.index(letter) + 1
                if "correct_answer_num" in obj:
                    obj["correct_answer_num"] = str(idx)
                elif "answer" in obj:
                    obj["answer"] = idx
                else:
                    obj["correct_answer"] = letter
                updated += 1
            outf.write(json.dumps(obj, ensure_ascii=False) + "\n")
    tmp.replace(dest)
    print(f"applied {updated} gold-key corrections -> {dest}")
    print(f"review status counts: {status_counts}")
    return updated


def update_validation_status(dataset: str, review_path: str) -> None:
    status_path = Path("validation_status.json")
    try:
        data = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        data = {}

    verified = ambiguous = wrong_key = bad_translation = 0
    total = 0
    with open(review_path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            total += 1
            s = (row.get("review_status") or "").strip().lower()
            if s == "verified":
                verified += 1
            elif s == "ambiguous":
                ambiguous += 1
            elif s == "wrong_key":
                wrong_key += 1
            elif s == "bad_translation":
                bad_translation += 1

    data[dataset] = {
        "total": total,
        "verified": verified,
        "ambiguous": ambiguous,
        "wrong_key": wrong_key,
        "bad_translation": bad_translation,
        "status": "reviewed" if verified else "pending_review",
        "reviewer": "native_speaker",
        "review_date": datetime.now(timezone.utc).date().isoformat(),
    }
    status_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"updated {status_path} for {dataset}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export or apply native-speaker review sheets.")
    parser.add_argument("--apply", action="store_true", help="Apply corrected_letter from review CSV.")
    parser.add_argument("--update-status", action="store_true", help="Update validation_status.json counts.")
    parser.add_argument("paths", nargs="+", help="input.jsonl review.csv  OR  review.csv input.jsonl [--apply]")
    args = parser.parse_args(argv)

    if args.apply:
        review_path, input_path = args.paths[0], args.paths[1]
        output_path = args.paths[2] if len(args.paths) > 2 else None
        apply_review(review_path, input_path, output_path)
        if args.update_status:
            update_validation_status(input_path, review_path)
        return 0

    input_path, output_path = args.paths[0], args.paths[1]
    export_review(input_path, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
