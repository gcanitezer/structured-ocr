from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from structured_ocr.verification.compiler import LaTeXCompiler, CompilationResult


@dataclass
class CompilabilityResult:
    compilable: bool
    compiler_used: Optional[str]
    output_path: Optional[str]
    error_log: str
    warnings: list = field(default_factory=list)
    elapsed_seconds: float = 0.0
    attempts: int = 1


class CompilabilityChecker:
    def __init__(self, timeout_seconds: int = 60, enable_image_comparison: bool = True):
        self.timeout_seconds = timeout_seconds
        self.enable_image_comparison = enable_image_comparison
        self._compilers = {}
        for engine in ("pdflatex", "xelatex", "lualatex"):
            self._compilers[engine] = LaTeXCompiler(
                engine=engine, timeout=timeout_seconds, passes=1
            )

    def check(self, latex_source: str, output_dir: Optional[Path] = None) -> CompilabilityResult:
        preferred_order = ["pdflatex", "xelatex", "lualatex"]
        last_result = None

        for engine in preferred_order:
            result = self._compilers[engine].compile_string(latex_source)
            if result.succeeded:
                return CompilabilityResult(
                    compilable=True,
                    compiler_used=engine,
                    output_path=str(result.output_path) if result.output_path else None,
                    error_log="\n".join(result.errors) if result.errors else "",
                    warnings=result.warnings,
                    elapsed_seconds=result.elapsed_seconds,
                    attempts=1,
                )
            last_result = result

        return CompilabilityResult(
            compilable=False,
            compiler_used=None,
            output_path=None,
            error_log="\n".join(last_result.errors) if last_result and last_result.errors else "No compilers available",
            warnings=last_result.warnings if last_result else [],
            elapsed_seconds=last_result.elapsed_seconds if last_result else 0.0,
            attempts=len(preferred_order),
        )

    def compare_rendered_images(
        self,
        generated_latex: str,
        reference_image: Path,
        output_dir: Optional[Path] = None
    ) -> Optional[float]:
        if not self.enable_image_comparison:
            return None

        result = self.check(generated_latex, output_dir)
        if not result.compilable or not result.output_path:
            return None

        try:
            rendered_pdf = Path(result.output_path)
            rendered_image = _pdf_to_image(rendered_pdf)
            reference_img = Image.open(reference_image)

            if rendered_image is None or reference_img is None:
                return None

            similarity = _image_similarity(rendered_image, reference_img)
            return float(similarity)
        except Exception:
            return None


def _pdf_to_image(pdf_path: Path) -> Optional[Image.Image]:
    try:
        from pdf2image import convert_from_path
        images = convert_from_path(str(pdf_path), dpi=150)
        if images:
            return images[0]
        return None
    except ImportError:
        return None


def _image_similarity(img1: Image.Image, img2: Image.Image) -> float:
    arr1 = _image_to_array(img1)
    arr2 = _image_to_array(img2)

    if arr1.shape != arr2.shape:
        h1, w1 = arr1.shape[:2]
        h2, w2 = arr2.shape[:2]
        scale = min(h1 / h2, w1 / w2, 1.0)
        if scale < 1.0:
            new_h, new_w = int(h2 * scale), int(w2 * scale)
            img2 = img2.resize((new_w, new_h), Image.Resampling.LANCZOS)
            arr2 = _image_to_array(img2)

    diff = np.abs(arr1.astype(float) - arr2.astype(float))
    mse = np.mean(diff ** 2)
    if mse == 0:
        return 1.0

    max_mse = 255 ** 2
    similarity = 1.0 - min(mse / max_mse, 1.0)
    return similarity


def _image_to_array(img: Image.Image) -> np.ndarray:
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img)


def compare_rendered_images(generated_latex: str, reference_image: Path) -> Optional[float]:
    checker = CompilabilityChecker(enable_image_comparison=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        return checker.compare_rendered_images(generated_latex, reference_image, Path(tmpdir))