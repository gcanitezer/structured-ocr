"""LaTeX corpus generation with PDF compilation and image conversion."""

import random
import subprocess
from pathlib import Path
from typing import Any


class CorpusGenerator:
    """Generate LaTeX corpus: source → PDF → page images."""

    def __init__(
        self,
        output_dir: str = "./data/corpus",
        dpi: int = 150,
        compilers: list[str] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.compilers = compilers or ["pdflatex", "xelatex", "lualatex"]
        self._ensure_tools()

    def _ensure_tools(self) -> None:
        """Check for required tools."""
        for compiler in self.compilers:
            try:
                subprocess.run(
                    [compiler, "--version"],
                    capture_output=True,
                    check=True,
                )
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass

    def generate(
        self,
        latex_source: str,
        doc_id: str,
        metadata: dict | None = None,
    ) -> dict[str, Any]:
        """Generate a document: compile LaTeX, convert to images.

        Args:
            latex_source: LaTeX source string
            doc_id: Unique document identifier
            metadata: Optional document metadata

        Returns:
            dict with paths to generated files
        """
        doc_dir = self.output_dir / doc_id
        doc_dir.mkdir(parents=True, exist_ok=True)

        tex_path = doc_dir / "document.tex"
        tex_path.write_text(latex_source)

        pdf_path = self._compile_latex(tex_path)
        if pdf_path is None:
            return {"success": False, "error": "LaTeX compilation failed", "doc_id": doc_id}

        images = self._pdf_to_images(pdf_path, doc_dir)

        return {
            "success": True,
            "doc_id": doc_id,
            "latex_path": str(tex_path),
            "pdf_path": str(pdf_path),
            "images": images,
            "num_pages": len(images),
            "metadata": metadata or {},
        }

    def _compile_latex(self, tex_path: Path) -> Path | None:
        """Compile LaTeX to PDF using available compiler."""
        for compiler in self.compilers:
            try:
                result = subprocess.run(
                    [compiler, "-interaction=nonstopmode", "-halt-on-error", str(tex_path)],
                    cwd=tex_path.parent,
                    capture_output=True,
                    timeout=60,
                )
                if result.returncode == 0:
                    return tex_path.parent / "document.pdf"
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
        return None

    def _pdf_to_images(self, pdf_path: Path, output_dir: Path) -> list[str]:
        """Convert PDF pages to PNG images."""
        image_paths = []
        try:
            result = subprocess.run(
                ["gs", "-dQUIET", f"-sDEVICE=png150", f"-r{self.dpi}", "-o",
                 str(output_dir / "page-%d.png"), str(pdf_path)],
                capture_output=True,
                timeout=120,
            )
            if result.returncode == 0:
                for i in range(100):
                    page_path = output_dir / f"page-{i}.png"
                    if page_path.exists():
                        image_paths.append(str(page_path))
                    else:
                        break
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return image_paths

    def bulk_generate(
        self,
        latex_generator_func,
        count: int = 1000,
        seed: int = 42,
    ) -> list[dict[str, Any]]:
        """Generate multiple documents.

        Args:
            latex_generator_func: Function that takes seed and returns latex string
            count: Number of documents to generate
            seed: Random seed

        Returns:
            List of generation results
        """
        results = []
        for i in range(count):
            random.seed(seed + i)
            latex = latex_generator_func(seed=seed + i)
            result = self.generate(latex["latex"], f"doc_{i:05d}", latex.get("metadata"))
            results.append(result)
        return results