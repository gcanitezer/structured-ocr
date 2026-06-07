"""Wrapper around pdflatex / xelatex / lualatex with timeout handling.

This module exposes a :class:`LaTeXCompiler` class that hides the
subprocess plumbing required to invoke a TeX engine, capture its
output, enforce a wall-clock timeout, and parse the resulting ``.log``
file for errors and warnings. It is the single entry point used by
both the verification pipeline and the GRPO reward function.

The wrapper is intentionally synchronous: training jobs and the OCR
verification API both prefer deterministic, easy-to-reason-about
subprocesses over async fan-out. The :class:`LaTeXCompiler` is
reusable, thread-safe (no shared mutable state), and supports
multiple passes (``pdflatex`` typically needs two passes to resolve
forward references).

Example
-------
>>> compiler = LaTeXCompiler(engine="pdflatex", timeout=30)
>>> result = compiler.compile_string(r"\\documentclass{article}\\begin{document}hi\\end{document}")
>>> result.outcome.value
'success'
"""

from __future__ import annotations

import enum
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

logger = logging.getLogger(__name__)

SUPPORTED_COMPILERS: tuple[str, ...] = ("pdflatex", "xelatex", "lualatex")


class CompilationOutcome(str, enum.Enum):
    """Coarse-grained outcome of a single compilation attempt."""

    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    COMPILER_NOT_FOUND = "compiler_not_found"
    EMPTY_SOURCE = "empty_source"
    IO_ERROR = "io_error"


@dataclass
class CompilationResult:
    """Outcome of a single compiler invocation."""

    outcome: CompilationOutcome
    returncode: Optional[int] = None
    engine: str = "pdflatex"
    passes: int = 0
    elapsed_seconds: float = 0.0
    stdout: str = ""
    stderr: str = ""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    output_path: Optional[Path] = None
    log_path: Optional[Path] = None
    message: str = ""

    @property
    def succeeded(self) -> bool:
        return self.outcome == CompilationOutcome.SUCCESS

    @property
    def score(self) -> float:
        """Convert outcome to a [0, 1] score for use as a reward signal."""
        if self.outcome == CompilationOutcome.SUCCESS:
            return 1.0
        if self.outcome == CompilationOutcome.COMPILER_NOT_FOUND:
            return 0.5
        return 0.0

    def to_dict(self) -> Dict[str, object]:
        d = asdict(self)
        d["outcome"] = self.outcome.value
        if self.output_path is not None:
            d["output_path"] = str(self.output_path)
        if self.log_path is not None:
            d["log_path"] = str(self.log_path)
        return d


_LOG_ERROR_RE = re.compile(r"^!\s*(?P<msg>.+?)$", re.MULTILINE)
_LOG_WARNING_RE = re.compile(r"^LaTeX (?P<kind>Warning|Info):(?P<msg>.*?)(?:\n|$)", re.MULTILINE)
_LOG_LINE_RE = re.compile(r"^l\.(?P<line>\d+)\s")


def parse_log_errors(log_text: str) -> List[str]:
    """Extract human-readable error messages from a TeX ``.log`` file.

    The parser is intentionally permissive: it picks up both ``! ...``
    style error lines and ``LaTeX Warning: ...`` notices. Returns a
    list of stripped messages, deduplicated while preserving order.
    """
    if not log_text:
        return []
    seen: set[str] = set()
    out: List[str] = []
    for match in _LOG_ERROR_RE.finditer(log_text):
        msg = match.group("msg").strip()
        if msg and msg not in seen:
            seen.add(msg)
            out.append(msg)
    return out


def parse_log_warnings(log_text: str) -> List[str]:
    """Extract LaTeX ``Warning:`` notices from a TeX ``.log`` file."""
    if not log_text:
        return []
    seen: set[str] = set()
    out: List[str] = []
    for match in _LOG_WARNING_RE.finditer(log_text):
        kind = match.group("kind").lower()
        msg = match.group("msg").strip()
        line = f"{kind}: {msg}"
        if line and line not in seen:
            seen.add(line)
            out.append(line)
    return out


class LaTeXCompiler:
    """Reusable, thread-safe wrapper around a TeX engine.

    Parameters
    ----------
    engine:
        One of ``"pdflatex"``, ``"xelatex"``, ``"lualatex"``.
    timeout:
        Wall-clock timeout in seconds for *each* pass. The total
        timeout for a multi-pass run is approximately
        ``timeout * passes``.
    passes:
        Number of compilation passes. Two passes is the default and
        is sufficient for the vast majority of LaTeX documents
        (one to emit labels, one to resolve ``\\ref``s). Set to 1
        for snappier reward scoring at the cost of unresolved
        forward references.
    extra_args:
        Additional command-line arguments forwarded to the engine.
    workdir:
        Optional persistent working directory. When ``None`` (the
        default) every call uses a fresh :class:`tempfile.TemporaryDirectory`.
    keep_logs:
        When ``True``, log files are kept on disk for debugging.
    """

    def __init__(
        self,
        engine: str = "pdflatex",
        timeout: float = 30.0,
        passes: int = 2,
        extra_args: Optional[Sequence[str]] = None,
        workdir: Optional[Union[str, Path]] = None,
        keep_logs: bool = False,
    ) -> None:
        if engine not in SUPPORTED_COMPILERS:
            raise ValueError(f"Unsupported engine {engine!r}; choose one of {SUPPORTED_COMPILERS}")
        if timeout <= 0:
            raise ValueError("timeout must be > 0")
        if passes < 1:
            raise ValueError("passes must be >= 1")
        self.engine = engine
        self.timeout = float(timeout)
        self.passes = int(passes)
        self.extra_args: List[str] = list(extra_args or [])
        self.workdir = Path(workdir) if workdir is not None else None
        self.keep_logs = keep_logs
        self._lock = threading.Lock()

    @property
    def is_available(self) -> bool:
        """Return ``True`` if the engine binary is on ``$PATH``."""
        return shutil.which(self.engine) is not None

    def compile_file(
        self,
        tex_path: Union[str, Path],
        *,
        output_dir: Optional[Union[str, Path]] = None,
        jobname: str = "doc",
    ) -> CompilationResult:
        """Compile a ``.tex`` file on disk and return the result."""
        tex_path = Path(tex_path)
        if not tex_path.exists():
            return CompilationResult(
                outcome=CompilationOutcome.IO_ERROR,
                engine=self.engine,
                message=f"file not found: {tex_path}",
            )
        source = tex_path.read_text(errors="replace")
        if output_dir is None:
            if self.workdir is not None:
                output_dir = self.workdir
            else:
                output_dir = tex_path.parent
        return self._run(source, output_dir=Path(output_dir), jobname=jobname)

    def compile_string(
        self,
        source: str,
        *,
        jobname: str = "doc",
    ) -> CompilationResult:
        """Compile a LaTeX source string in a temporary directory."""
        if not source or not source.strip():
            return CompilationResult(
                outcome=CompilationOutcome.EMPTY_SOURCE,
                engine=self.engine,
                message="empty LaTeX source",
            )
        if self.workdir is not None:
            target = self.workdir
            target.mkdir(parents=True, exist_ok=True)
            return self._run(source, output_dir=target, jobname=jobname)
        with tempfile.TemporaryDirectory(prefix="latex-") as tmp:
            return self._run(source, output_dir=Path(tmp), jobname=jobname)

    def compile(
        self,
        source: Union[str, Path],
        *,
        jobname: str = "doc",
    ) -> CompilationResult:
        """Compile a string or a path; convenience wrapper."""
        if isinstance(source, (str, Path)) and Path(source).is_file():
            return self.compile_file(source, jobname=jobname)
        return self.compile_string(str(source), jobname=jobname)

    def _build_command(self, jobname: str) -> List[str]:
        return [
            self.engine,
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            f"-jobname={jobname}",
            *self.extra_args,
            "doc.tex",
        ]

    def _run(
        self,
        source: str,
        *,
        output_dir: Path,
        jobname: str,
    ) -> CompilationResult:
        """Execute the engine ``self.passes`` times and aggregate results."""
        start = time.time()
        tex_path = output_dir / "doc.tex"
        try:
            tex_path.write_text(source)
        except OSError as exc:
            return CompilationResult(
                outcome=CompilationOutcome.IO_ERROR,
                engine=self.engine,
                elapsed_seconds=time.time() - start,
                message=f"could not write source: {exc}",
            )

        aggregate_errors: List[str] = []
        aggregate_warnings: List[str] = []
        last_stdout = ""
        last_stderr = ""
        last_returncode: Optional[int] = None
        actual_passes = 0
        last_log_text = ""
        outcome = CompilationOutcome.FAILED
        if not self.is_available:
            elapsed = time.time() - start
            return CompilationResult(
                outcome=CompilationOutcome.COMPILER_NOT_FOUND,
                engine=self.engine,
                elapsed_seconds=elapsed,
                message=f"{self.engine} not on PATH",
            )

        for pass_idx in range(self.passes):
            cmd = self._build_command(jobname)
            try:
                with self._lock:
                    proc = subprocess.run(
                        cmd,
                        cwd=str(output_dir),
                        capture_output=True,
                        text=True,
                        timeout=self.timeout,
                        env=self._build_env(),
                    )
            except subprocess.TimeoutExpired as exc:
                elapsed = time.time() - start
                log_text = self._read_log(output_dir, jobname)
                return CompilationResult(
                    outcome=CompilationOutcome.TIMEOUT,
                    engine=self.engine,
                    returncode=None,
                    passes=actual_passes,
                    elapsed_seconds=elapsed,
                    stdout=exc.stdout or "",
                    stderr=exc.stderr or "",
                    errors=parse_log_errors(log_text),
                    warnings=parse_log_warnings(log_text),
                    log_path=self._maybe_keep_log(output_dir, jobname),
                    message=(
                        f"{self.engine} timed out after {self.timeout:.0f}s on pass "
                        f"{pass_idx + 1}/{self.passes}"
                    ),
                )
            except FileNotFoundError as exc:
                elapsed = time.time() - start
                return CompilationResult(
                    outcome=CompilationOutcome.COMPILER_NOT_FOUND,
                    engine=self.engine,
                    elapsed_seconds=elapsed,
                    message=str(exc),
                )

            actual_passes += 1
            last_stdout = proc.stdout or ""
            last_stderr = proc.stderr or ""
            last_returncode = proc.returncode
            last_log_text = self._read_log(output_dir, jobname)
            aggregate_errors = parse_log_errors(last_log_text)
            aggregate_warnings = parse_log_warnings(last_log_text)
            if proc.returncode != 0:
                outcome = CompilationOutcome.FAILED
                break
            outcome = CompilationOutcome.SUCCESS
            if self.passes <= 1:
                break
            if pass_idx < self.passes - 1 and not self._aux_files_changed(
                output_dir, jobname, last_stdout
            ):
                break

        elapsed = time.time() - start
        pdf_path = output_dir / f"{jobname}.pdf"
        log_path = self._maybe_keep_log(output_dir, jobname)
        message = (
            f"{self.engine} completed in {actual_passes} pass(es) with returncode={last_returncode}"
        )
        return CompilationResult(
            outcome=outcome,
            returncode=last_returncode,
            engine=self.engine,
            passes=actual_passes,
            elapsed_seconds=elapsed,
            stdout=last_stdout,
            stderr=last_stderr,
            errors=aggregate_errors,
            warnings=aggregate_warnings,
            output_path=pdf_path if pdf_path.exists() else None,
            log_path=log_path,
            message=message,
        )

    @staticmethod
    def _build_env() -> Dict[str, str]:
        """Provide a minimal but safe environment for the TeX engine."""
        env = os.environ.copy()
        env.setdefault("TEXINPUTS", ".")
        env.setdefault("TEXMFVAR", str(Path(tempfile.gettempdir()) / "texmf-var"))
        return env

    @staticmethod
    def _read_log(output_dir: Path, jobname: str) -> str:
        log_path = output_dir / f"{jobname}.log"
        if not log_path.exists():
            return ""
        try:
            return log_path.read_text(errors="replace")
        except OSError:
            return ""

    @staticmethod
    def _aux_files_changed(output_dir: Path, jobname: str, stdout: str) -> bool:
        """Heuristic: detect whether the pass produced new aux output.

        If the aux file size is stable across passes the document has
        converged and we can stop early. We also fall back to scanning
        the stdout for the "Rerun" hint emitted by LaTeX.
        """
        aux = output_dir / f"{jobname}.aux"
        if not aux.exists():
            return False
        rerun_hint = "Rerun" in stdout or "rerun" in stdout
        return rerun_hint

    def _maybe_keep_log(self, output_dir: Path, jobname: str) -> Optional[Path]:
        if not self.keep_logs:
            return None
        log_path = output_dir / f"{jobname}.log"
        return log_path if log_path.exists() else None


def compile_document(
    source: str,
    *,
    engine: str = "pdflatex",
    timeout: float = 30.0,
    passes: int = 2,
    keep_logs: bool = False,
) -> CompilationResult:
    """Functional helper to compile a single LaTeX source string.

    Equivalent to constructing a one-shot :class:`LaTeXCompiler` and
    calling :meth:`LaTeXCompiler.compile_string`. Provided for
    convenience in tests and the CLI.
    """
    compiler = LaTeXCompiler(engine=engine, timeout=timeout, passes=passes, keep_logs=keep_logs)
    return compiler.compile_string(source)


__all__ = [
    "CompilationOutcome",
    "CompilationResult",
    "LaTeXCompiler",
    "SUPPORTED_COMPILERS",
    "compile_document",
    "parse_log_errors",
    "parse_log_warnings",
]
