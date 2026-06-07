"""LaTeX document templates for corpus generation."""

import random
from typing import Any


class TextbookTemplate:
    """Template for textbook-style LaTeX documents."""

    SUBJECTS = {
        "math": {
            "packages": ["amsmath", "amssymb", "mathtools"],
            "sections": ["Introduction", "Definitions", "Theorems", "Proofs", "Examples", "Exercises"],
        },
        "biology": {
            "packages": ["graphicx", "float", "booktabs"],
            "sections": ["Introduction", "Methods", "Results", "Discussion", "Conclusion"],
        },
        "chemistry": {
            "packages": ["mhchem", "siunitx", "chemfig"],
            "sections": ["Introduction", "Reactions", "Properties", "Applications", "Safety"],
        },
    }

    def __init__(self, subject: str = "math", num_chapters: int = 3):
        self.subject = subject
        self.num_chapters = num_chapters
        self.config = self.SUBJECTS.get(subject, self.SUBJECTS["math"])

    def generate(self, seed: int | None = None) -> dict[str, Any]:
        """Generate a textbook LaTeX document."""
        if seed is not None:
            random.seed(seed)

        latex = [
            "\\documentclass{book}",
            "\\usepackage[utf8]{inputenc}",
        ]
        latex.extend(f"\\usepackage{{{pkg}}}" for pkg in self.config["packages"])
        latex.append("\\begin{document}")

        for ch in range(self.num_chapters):
            latex.append(f"\n\n\\chapter{{Chapter {ch + 1}}}\n")
            sections = random.sample(self.config["sections"], min(3, len(self.config["sections"])))
            for sec in sections:
                latex.append(f"\n\\section{{{sec}}}\n")
                latex.extend(self._generate_section_content(sec))

        latex.append("\n\\end{document}")
        return {
            "latex": "\n".join(latex),
            "metadata": {"type": "textbook", "subject": self.subject, "chapters": self.num_chapters},
        }

    def _generate_section_content(self, section: str) -> list[str]:
        """Generate LaTeX content for a section."""
        content = []

        if self.subject == "math":
            content.extend(self._math_content(section))
        elif self.subject == "biology":
            content.extend(self._biology_content(section))
        elif self.subject == "chemistry":
            content.extend(self._chemistry_content(section))

        return content

    def _math_content(self, section: str) -> list[str]:
        content = []
        if section == "Introduction":
            content.append("This chapter introduces fundamental concepts.")
        elif section == "Definitions":
            content.append("\\begin{definition}")
            content.append("A mathematical object is ...")
            content.append("\\end{definition}")
        elif section == "Theorems":
            content.append("\\begin{theorem}")
            content.append("\\label{thm:main}")
            content.append("The main theorem states that ...")
            content.append("\\end{theorem}")
        elif section == "Proofs":
            content.append("\\begin{proof}")
            content.append("We prove this by contradiction.")
            content.append("\\end{proof}")
        elif section == "Examples":
            content.append("\\begin{example}")
            content.append("Consider the equation:")
            content.append("\\begin{equation}")
            content.append("E = mc^2")
            content.append("\\end{equation}")
            content.append("\\end{example}")
        elif section == "Exercises":
            content.append("\\begin{enumerate}")
            content.append("\\item Prove the theorem.")
            content.append("\\item Solve the equation.")
            content.append("\\end{enumerate}")
        return content

    def _biology_content(self, section: str) -> list[str]:
        content = []
        if section == "Introduction":
            content.append("This study examines biological processes.")
        elif section == "Methods":
            content.append("\\begin{table}[ht]")
            content.append("\\centering")
            content.append("\\begin{tabular}{|c|c|}")
            content.append("\\hline")
            content.append("Organism & Measurement \\\\")
            content.append("\\hline")
            content.append("Homo sapiens & 37$^\\circ$C \\\\")
            content.append("\\hline")
            content.append("\\end{tabular}")
            content.append("\\end{table}")
        elif section == "Results":
            content.append("\\begin{figure}[ht]")
            content.append("\\centering")
            content.append("\\includegraphics[width=0.8\\textwidth]{placeholder}")
            content.append("\\caption{Sample figure}")
            content.append("\\end{figure}")
        elif section == "Discussion":
            content.append("The results demonstrate significant findings.")
        elif section == "Conclusion":
            content.append("This study provides valuable insights.")
        return content

    def _chemistry_content(self, section: str) -> list[str]:
        content = []
        if section == "Introduction":
            content.append("Chemical reactions are fundamental processes.")
        elif section == "Reactions":
            content.append("\\begin{equation}")
            content.append("\\ce{H2 + O2 -> H2O}")
            content.append("\\end{equation}")
        elif section == "Properties":
            content.append("\\begin{table}[ht]")
            content.append("\\centering")
            content.append("\\begin{tabular}{ll}")
            content.append("Substance & Boiling Point \\\\")
            content.append("\\hline")
            content.append("H2O & 100$^\\circ$C")
            content.append("\\end{tabular}")
            content.append("\\end{table}")
        elif section == "Applications":
            content.append("These compounds have industrial applications.")
        elif section == "Safety":
            content.append("Handle with appropriate precautions.")
        return content


class NewspaperTemplate:
    """Template for newspaper-style LaTeX documents."""

    def __init__(self, num_columns: int = 3, num_articles: int = 5):
        self.num_columns = num_columns
        self.num_articles = num_articles

    def generate(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            random.seed(seed)

        latex = [
            "\\documentclass{article}",
            "\\usepackage{multicol}",
            "\\usepackage{graphicx}",
            "\\begin{document}",
        ]

        for i in range(self.num_articles):
            latex.append(f"\n\\section*{{Headline {i + 1}}}\n")
            latex.append("Lorem ipsum dolor sit amet, consectetur adipiscing elit.\n")
            if random.random() > 0.5:
                latex.append("\\begin{figure}[ht]\n\\centering\n\\includegraphics[width=0.5\\textwidth]{placeholder}\n\\end{figure}\n")

        latex.append("\n\\end{document}")
        return {
            "latex": "\n".join(latex),
            "metadata": {"type": "newspaper", "columns": self.num_columns, "articles": self.num_articles},
        }


class LeafletTemplate:
    """Template for brochure/leaflet-style LaTeX documents."""

    def __init__(self, sections: int = 4):
        self.sections = sections

    def generate(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            random.seed(seed)

        latex = [
            "\\documentclass{article}",
            "\\usepackage{geometry}",
            "\\usepackage{graphicx}",
            "\\usepackage{tikz}",
            "\\geometry{a4paper,landscape}",
            "\\begin{document}",
            "\\begin{center}",
            "\\Huge\\textbf{Brochure Title}\n",
            "\\end{center}",
        ]

        for i in range(self.sections):
            latex.append(f"\n\\begin{{minipage}}{{0.23\\textwidth}}\n")
            latex.append(f"\\textbf{{Section {i + 1}}}\n")
            latex.append("Compact information here.\n")
            latex.append("\\end{minipage}\n")

        latex.append("\n\\end{document}")
        return {
            "latex": "\n".join(latex),
            "metadata": {"type": "leaflet", "sections": self.sections},
        }