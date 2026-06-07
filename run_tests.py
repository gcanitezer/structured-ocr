#!/usr/bin/env python3
"""Validation suite runner — stdlib only."""
import json
import sys
import time
from datetime import datetime
from pathlib import Path
import unittest


def run_suite():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    targets = [
        "tests.unit.test_types",
        "tests.unit.test_main",
        "tests.unit.test_compiler",
    ]
    for t in targets:
        try:
            suite.addTests(loader.loadTestsFromName(t))
        except Exception as exc:
            print(f"[warn] could not load {t}: {exc}")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result


def main():
    start = time.time()
    result = run_suite()
    elapsed = time.time() - start
    Path("reports").mkdir(exist_ok=True)
    report = {
        "timestamp": datetime.now().isoformat(),
        "tests_run": result.testsRun,
        "errors": len(result.errors),
        "failures": len(result.failures),
        "ok": result.wasSuccessful(),
        "duration_seconds": round(elapsed, 3),
    }
    Path("reports/validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"\n[report] reports/validation_report.json written ({elapsed:.2f}s).")
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()