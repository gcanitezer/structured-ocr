"""Smoke test runner that uses the stdlib unittest module.

This runner is intended to exercise the verification module without
requiring pytest to be installed in the development environment. It
deliberately covers a representative subset of the full pytest test
suite; the comprehensive tests live in tests/verification/.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Ensure src/ is on sys.path
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from structured_ocr.verification.compiler import (  # noqa: E402
    CompilationOutcome,
    CompilationResult,
    LaTeXCompiler,
    SUPPORTED_COMPILERS,
    parse_log_errors,
    parse_log_warnings,
)
from structured_ocr.verification.result import (  # noqa: E402
    ComponentResult,
    VerificationResult,
    VerificationSummary,
)
from structured_ocr.verification.verifier import (  # noqa: E402
    LaTeXVerifier,
    VerificationConfig,
    verify_document,
    verify_documents,
)
from structured_ocr.training.reward_functions import (  # noqa: E402
    LaTeXUnitTestFramework,
    set_compiler_required,
)


SAMPLE_DOC = (
    "\\documentclass{article}\n"
    "\\begin{document}\n"
    "Hello. \\section{Intro}\n"
    "\\end{document}\n"
)


class CompilerTests(unittest.TestCase):
    def test_supported_compilers(self):
        self.assertEqual(SUPPORTED_COMPILERS, ("pdflatex", "xelatex", "lualatex"))

    def test_invalid_engine(self):
        with self.assertRaises(ValueError):
            LaTeXCompiler(engine="context")

    def test_invalid_timeout(self):
        with self.assertRaises(ValueError):
            LaTeXCompiler(timeout=0)

    def test_invalid_passes(self):
        with self.assertRaises(ValueError):
            LaTeXCompiler(passes=0)

    def test_empty_source_returns_empty_outcome(self):
        compiler = LaTeXCompiler()
        result = compiler.compile_string("")
        self.assertEqual(result.outcome, CompilationOutcome.EMPTY_SOURCE)
        self.assertFalse(result.succeeded)
        self.assertEqual(result.score, 0.0)

    def test_compiler_not_found_when_unavailable(self):
        compiler = _ForceUnavailableCompiler()
        result = compiler.compile_string(SAMPLE_DOC)
        self.assertEqual(result.outcome, CompilationOutcome.COMPILER_NOT_FOUND)
        self.assertEqual(result.score, 0.5)

    def test_fake_success(self):
        completed = subprocess.CompletedProcess(
            args=["pdflatex"],
            returncode=0,
            stdout="This is pdfTeX\nOutput written on doc.pdf (1 page).",
            stderr="",
        )

        def fake_run(cmd, **kwargs):
            Path(kwargs["cwd"]).joinpath("doc.pdf").write_bytes(b"%PDF-fake")
            Path(kwargs["cwd"]).joinpath("doc.log").write_text("(no errors)\n")
            return completed

        compiler = _FakeSubprocessCompiler(passes=1)
        with patch(
            "structured_ocr.verification.compiler.subprocess.run", side_effect=fake_run
        ):
            result = compiler.compile_string(SAMPLE_DOC)
        self.assertEqual(result.outcome, CompilationOutcome.SUCCESS)
        self.assertEqual(result.score, 1.0)
        self.assertTrue(result.succeeded)
        self.assertIsNotNone(result.output_path)

    def test_fake_failure(self):
        completed = subprocess.CompletedProcess(args=["pdflatex"], returncode=1, stdout="", stderr="")

        def fake_run(cmd, **kwargs):
            Path(kwargs["cwd"]).joinpath("doc.log").write_text(
                "! Undefined control sequence.\nl.2 \\foo\n"
            )
            return completed

        compiler = _FakeSubprocessCompiler(passes=1)
        with patch(
            "structured_ocr.verification.compiler.subprocess.run", side_effect=fake_run
        ):
            result = compiler.compile_string("oops")
        self.assertEqual(result.outcome, CompilationOutcome.FAILED)
        self.assertEqual(result.score, 0.0)
        self.assertTrue(result.errors)
        self.assertIn("Undefined control sequence", result.errors[0])

    def test_fake_timeout(self):
        def fake_run(cmd, **kwargs):
            Path(kwargs["cwd"]).joinpath("doc.log").write_text("! Emergency stop.\n")
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs["timeout"])

        compiler = _FakeSubprocessCompiler(passes=2, timeout=1.0)
        with patch(
            "structured_ocr.verification.compiler.subprocess.run", side_effect=fake_run
        ):
            result = compiler.compile_string("anything")
        self.assertEqual(result.outcome, CompilationOutcome.TIMEOUT)
        self.assertEqual(result.score, 0.0)
        self.assertIn("timed out", result.message)

    def test_parse_log_errors(self):
        log = (
            "This is pdfTeX, Version 3.14\n"
            "! Undefined control sequence.\n"
            "l.10 \\foo\n"
            "\n"
            "! Missing $ inserted.\n"
        )
        errors = parse_log_errors(log)
        self.assertEqual(errors, ["Undefined control sequence.", "Missing $ inserted."])

    def test_parse_log_errors_dedupes(self):
        log = "! Undefined control sequence.\n! Undefined control sequence.\n"
        errors = parse_log_errors(log)
        self.assertEqual(errors, ["Undefined control sequence."])

    def test_parse_log_warnings(self):
        log = "LaTeX Warning: Reference `fig:foo' on page 1 undefined.\n"
        warnings = parse_log_warnings(log)
        self.assertTrue(any("Reference" in w for w in warnings))


class ResultTests(unittest.TestCase):
    def test_pass_rate(self):
        result = VerificationResult(
            source="x",
            components=[
                ComponentResult(name="a", score=1.0, passed=True),
                ComponentResult(name="b", score=1.0, passed=True),
                ComponentResult(name="c", score=0.0, passed=False),
            ],
            total_components=3,
            passed_components=2,
        )
        self.assertEqual(result.total_components, 3)
        self.assertEqual(result.passed_components, 2)
        self.assertAlmostEqual(result.pass_rate, 2 / 3)
        self.assertFalse(result.passed)

    def test_passes_only_when_all_pass(self):
        result = VerificationResult(
            source="x",
            components=[ComponentResult(name="a", score=1.0, passed=True)],
            total_components=1,
            passed_components=1,
        )
        self.assertTrue(result.passed)

    def test_to_dict_json_serializable(self):
        result = VerificationResult(
            source="x",
            compilation=CompilationResult(outcome=CompilationOutcome.SUCCESS, engine="pdflatex"),
            components=[ComponentResult(name="compilation_success", score=1.0, passed=True)],
            total_components=1,
            passed_components=1,
        )
        json.dumps(result.to_dict())


class VerifierTests(unittest.TestCase):
    def test_nine_components_returned(self):
        verifier = LaTeXVerifier()
        verifier._compiler = _FakeCompiler(CompilationResult(outcome=CompilationOutcome.SUCCESS))
        result = verifier.verify(SAMPLE_DOC, reference=SAMPLE_DOC)
        self.assertEqual(len(result.components), 9)

    def test_skip_components(self):
        verifier = LaTeXVerifier(config=VerificationConfig(skip_components=("visual_similarity",)))
        verifier._compiler = _FakeCompiler(CompilationResult(outcome=CompilationOutcome.SUCCESS))
        result = verifier.verify(SAMPLE_DOC, reference=SAMPLE_DOC)
        names = [c.name for c in result.components]
        self.assertNotIn("visual_similarity", names)
        self.assertEqual(len(result.components), 8)

    def test_total_score_is_weighted_sum(self):
        verifier = LaTeXVerifier()
        verifier._compiler = _FakeCompiler(CompilationResult(outcome=CompilationOutcome.SUCCESS))
        result = verifier.verify(SAMPLE_DOC, reference=SAMPLE_DOC)
        total = sum(c.weighted_score for c in result.components)
        self.assertAlmostEqual(result.total_score, total)

    def test_compilation_success_uses_compiler(self):
        verifier = LaTeXVerifier()
        verifier._compiler = _FakeCompiler(CompilationResult(outcome=CompilationOutcome.SUCCESS))
        result = verifier.verify(SAMPLE_DOC, reference=SAMPLE_DOC)
        cs = result.get("compilation_success")
        self.assertIsNotNone(cs)
        self.assertEqual(cs.score, 1.0)
        self.assertTrue(cs.passed)

    def test_compilation_failure_propagates(self):
        verifier = LaTeXVerifier()
        verifier._compiler = _FakeCompiler(
            CompilationResult(outcome=CompilationOutcome.FAILED, errors=["bad"])
        )
        result = verifier.verify(SAMPLE_DOC)
        cs = result.get("compilation_success")
        self.assertIsNotNone(cs)
        self.assertFalse(cs.passed)
        self.assertIn("errors=1", cs.details)

    def test_compiler_not_found_score(self):
        verifier = LaTeXVerifier()
        verifier._compiler = _FakeCompiler(
            CompilationResult(outcome=CompilationOutcome.COMPILER_NOT_FOUND)
        )
        result = verifier.verify(SAMPLE_DOC)
        cs = result.get("compilation_success")
        self.assertIsNotNone(cs)
        self.assertEqual(cs.score, 0.5)

    def test_batch_returns_summary(self):
        verifier = LaTeXVerifier()
        verifier._compiler = _FakeCompiler(CompilationResult(outcome=CompilationOutcome.SUCCESS))
        summary = verifier.verify_batch([SAMPLE_DOC, SAMPLE_DOC], references=[SAMPLE_DOC, SAMPLE_DOC])
        self.assertEqual(summary.num_documents, 2)
        self.assertEqual(summary.num_compiled, 2)

    def test_batch_empty(self):
        verifier = LaTeXVerifier()
        verifier._compiler = _FakeCompiler(CompilationResult(outcome=CompilationOutcome.SUCCESS))
        summary = verifier.verify_batch([])
        self.assertEqual(summary.num_documents, 0)

    def test_write_report(self):
        verifier = LaTeXVerifier()
        verifier._compiler = _FakeCompiler(CompilationResult(outcome=CompilationOutcome.SUCCESS))
        result = verifier.verify(SAMPLE_DOC, reference=SAMPLE_DOC)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            verifier.write_report(result, path)
            payload = json.loads(path.read_text())
            self.assertEqual(payload["compiler"], "pdflatex")
            self.assertEqual(payload["total_components"], 9)

    def test_function_helpers(self):
        result = verify_document(SAMPLE_DOC, engine="pdflatex", timeout=10.0, passes=1)
        self.assertIsNotNone(result)
        summary = verify_documents([SAMPLE_DOC], engine="pdflatex", timeout=10.0, passes=1)
        self.assertIsNotNone(summary)


class RewardFunctionTests(unittest.TestCase):
    def test_compilation_test_routes_through_new_compiler(self):
        fw = LaTeXUnitTestFramework()
        fake = CompilationResult(outcome=CompilationOutcome.SUCCESS, engine="pdflatex", passes=2)
        with patch(
            "structured_ocr.verification.compiler.LaTeXCompiler.compile_string",
            return_value=fake,
        ):
            result = fw.test_compilation_success(SAMPLE_DOC, compiler="pdflatex")
        self.assertTrue(result.passed)
        self.assertEqual(result.score, 1.0)
        self.assertIn("engine=pdflatex", result.details)

    def test_compilation_test_xelatex(self):
        fw = LaTeXUnitTestFramework()
        fake = CompilationResult(outcome=CompilationOutcome.SUCCESS, engine="xelatex", passes=2)
        with patch(
            "structured_ocr.verification.compiler.LaTeXCompiler.compile_string",
            return_value=fake,
        ):
            result = fw.test_compilation_success(SAMPLE_DOC, compiler="xelatex")
        self.assertTrue(result.passed)
        self.assertIn("engine=xelatex", result.details)

    def test_compilation_test_lualatex(self):
        fw = LaTeXUnitTestFramework()
        fake = CompilationResult(outcome=CompilationOutcome.SUCCESS, engine="lualatex", passes=2)
        with patch(
            "structured_ocr.verification.compiler.LaTeXCompiler.compile_string",
            return_value=fake,
        ):
            result = fw.test_compilation_success(SAMPLE_DOC, compiler="lualatex")
        self.assertTrue(result.passed)
        self.assertIn("engine=lualatex", result.details)

    def test_compilation_test_empty_source(self):
        fw = LaTeXUnitTestFramework()
        result = fw.test_compilation_success("")
        self.assertFalse(result.passed)
        self.assertEqual(result.score, 0.0)

    def test_compilation_test_timeout(self):
        fw = LaTeXUnitTestFramework()
        fake = CompilationResult(outcome=CompilationOutcome.TIMEOUT)
        with patch(
            "structured_ocr.verification.compiler.LaTeXCompiler.compile_string",
            return_value=fake,
        ):
            result = fw.test_compilation_success("doc", timeout=5)
        self.assertFalse(result.passed)
        self.assertIn("timed out", result.details)

    def test_compiler_required_flag(self):
        set_compiler_required(True)
        try:
            fw = LaTeXUnitTestFramework()
            fake = CompilationResult(outcome=CompilationOutcome.COMPILER_NOT_FOUND)
            with patch(
                "structured_ocr.verification.compiler.LaTeXCompiler.compile_string",
                return_value=fake,
            ):
                result = fw.test_compilation_success("doc")
            self.assertFalse(result.passed)
        finally:
            set_compiler_required(False)


class _FakeCompiler:
    def __init__(self, result: CompilationResult) -> None:
        self._result = result
        self.engine = result.engine

    @property
    def is_available(self) -> bool:
        return True

    def compile_string(self, source: str):
        return self._result


class _ForceUnavailableCompiler(LaTeXCompiler):
    """A LaTeXCompiler subclass that always reports ``is_available == False``."""

    @property
    def is_available(self) -> bool:
        return False


class _FakeSubprocessCompiler(LaTeXCompiler):
    """A LaTeXCompiler subclass whose ``is_available`` is True.

    Used together with ``unittest.mock.patch`` on
    ``subprocess.run`` to simulate compiler invocations.
    """

    @property
    def is_available(self) -> bool:
        return True


class PipelineIntegrationTests(unittest.TestCase):
    def setUp(self):
        from structured_ocr.training import TrainingConfig, TrainingPipeline
        from unittest.mock import MagicMock

        self._TrainingConfig = TrainingConfig
        self._TrainingPipeline = TrainingPipeline
        self._MagicMock = MagicMock

    def test_sft_runs_verification_by_default(self):
        from structured_ocr.training import TrainingMode

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train_path = tmp_path / "train.jsonl"
            with open(train_path, "w") as f:
                for i in range(4):
                    f.write(json.dumps({"reference": f"ref-{i}", "prompt": f"prompt-{i}"}) + "\n")
            cfg = self._TrainingConfig(
                output_dir=tmp_path / "out",
                train_dataset=train_path,
                mode=TrainingMode.SFT,
                run_verification=True,
            )
            pipeline = self._TrainingPipeline(cfg)
            fake_summary = {"num_documents": 1, "batch_score": 0.8}
            from structured_ocr.training import SFTResult

            sft_result = SFTResult(
                output_dir=tmp_path / "out",
                train_loss=0.5,
                num_steps=2,
                num_train_samples=4,
            )
            with patch.object(pipeline, "_run_sft", return_value=sft_result), patch.object(
                pipeline, "_maybe_run_verification", return_value=fake_summary
            ) as mock_verify:
                result = pipeline.run()
            self.assertTrue(mock_verify.called)
            self.assertIsNotNone(result.sft_verification)

    def test_verification_disabled(self):
        from structured_ocr.training import TrainingMode, SFTResult

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train_path = tmp_path / "train.jsonl"
            with open(train_path, "w") as f:
                for i in range(2):
                    f.write(json.dumps({"reference": f"ref-{i}", "prompt": f"prompt-{i}"}) + "\n")
            cfg = self._TrainingConfig(
                output_dir=tmp_path / "out",
                train_dataset=train_path,
                mode=TrainingMode.SFT,
                run_verification=False,
            )
            pipeline = self._TrainingPipeline(cfg)
            sft_result = SFTResult(
                output_dir=tmp_path / "out",
                train_loss=0.5,
                num_steps=2,
                num_train_samples=2,
            )
            with patch.object(pipeline, "_run_sft", return_value=sft_result), patch.object(
                pipeline, "_maybe_run_verification", return_value=None
            ):
                result = pipeline.run()
            self.assertIsNone(result.sft_verification)


def main() -> int:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in (
        CompilerTests,
        ResultTests,
        VerifierTests,
        RewardFunctionTests,
        PipelineIntegrationTests,
    ):
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
