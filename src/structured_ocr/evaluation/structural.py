import re


class StructuralEvaluator:
    REQUIRED_SECTIONS = {"introduction", "methods", "results", "conclusion"}

    def compute_section_f1(self, predicted, gold) -> float:
        pred_set = {n.content.split()[0].lower() if hasattr(n, "content") and n.content else "" for n in predicted.sections}
        gold_set = {n.content.split()[0].lower() if hasattr(n, "content") and n.content else "" for n in gold.sections}
        pred_set.discard("")
        gold_set.discard("")
        if not pred_set and not gold_set:
            return 1.0
        tp = len(pred_set & gold_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    def verify_section_hierarchy(self, predicted) -> dict:
        section_names = [
            n.content.lower().split()[0] if hasattr(n, "content") and n.content else n.node_type
            for n in predicted.sections
            if hasattr(n, "node_type") and n.node_type == "section"
        ]
        total = len(section_names)
        correct_order = 0
        for i in range(1, total):
            if section_names[i] >= section_names[i - 1]:
                correct_order += 1
        violations = (total - 1) - correct_order if total > 0 else 0
        return {"correct_order": violations == 0, "violations": violations}

    def verify_table_structure(self, predicted_latex: str) -> dict:
        tabulars = re.findall(r"\\begin\{tabular\}\{([^}]*)\}", predicted_latex)
        tables = re.findall(r"\\begin\{table\}", predicted_latex)
        results = []
        for t in tabulars:
            col_count = len(t)
            aligned = any(c in t for c in "lcr")
            results.append({"col_spec": t, "column_count": col_count, "aligned": aligned})
        return {
            "table_count": len(tables),
            "tabular_count": len(tabulars),
            "details": results,
            "has_valid_tabular": all(r["aligned"] for r in results) or len(results) == 0,
        }

    def extract_labels(self, latex: str) -> set:
        return set(re.findall(r"\\label\{([^}]+)\}", latex))

    def extract_refs(self, latex: str) -> set:
        refs = set(re.findall(r"\\(?:ref|eqref|autoref)\{([^}]+)\}", latex))
        pkg_refs = set(re.findall(r"\\[Cc]ref\{([^}]+)\}", latex))
        return refs | pkg_refs

    def evaluate_cross_reference_integrity(self, latex: str) -> dict:
        labels = self.extract_labels(latex)
        refs = self.extract_refs(latex)
        broken = refs - labels
        unused = labels - refs
        total_refs = len(refs)
        coverage = 1.0 - (len(broken) / total_refs) if total_refs > 0 else 1.0
        return {
            "total_labels": len(labels),
            "total_refs": total_refs,
            "broken_refs": len(broken),
            "unused_labels": len(unused),
            "reference_coverage": coverage,
        }