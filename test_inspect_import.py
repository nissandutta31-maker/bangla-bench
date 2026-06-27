#!/usr/bin/env python3
"""Import smoke for the inspect-ai track (no API calls).

Run with:  python3 test_inspect_import.py
Exits 0 if task and scorer modules import cleanly, 1 otherwise.
"""
from __future__ import annotations


def main() -> int:
    ok = True

    def check(cond: bool, label: str) -> None:
        nonlocal ok
        print(f"[{'PASS' if cond else 'FAIL'}] {label}")
        ok = ok and cond

    import tasks.bangla_needle_haystack as haystack  # noqa: F401

    check(hasattr(haystack, "bangla_needle_haystack"), "bangla_needle_haystack task exports")

    import tasks.scorers as scorers  # noqa: F401

    check(
        hasattr(scorers, "retrieval_with_silent_failures"),
        "retrieval_with_silent_failures scorer exports",
    )

    print("\n" + ("ALL PASSED" if ok else "SOME FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
