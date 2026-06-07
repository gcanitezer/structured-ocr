class FormulaEvaluator:
    def edit_distance(self, pred: str, gold: str) -> int:
        a, b = pred, gold
        if len(a) < len(b):
            a, b = b, a
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(
                    min(prev[j] + 1, cur[j - 1] + 1,
                        prev[j - 1] + (0 if ca == cb else 1))
                )
            prev = cur
        return prev[-1]

    def normalized_edit_distance(self, pred: str, gold: str) -> float:
        ed = self.edit_distance(pred, gold)
        return ed / max(len(gold), 1)

    def bleu_score(self, predicted: str, reference: str) -> float:
        pred = predicted.replace("=", " ").replace("^", " ").split()
        ref = reference.replace("=", " ").replace("^", " ").split()
        if not pred and not ref:
            return 100.0
        matches = sum(1 for t in pred if t in ref)
        return (matches / max(len(pred), 1)) * 100.0

    def exact_match(self, predicted: str, reference: str, tolerance: float = 0.05) -> bool:
        return self.normalized_edit_distance(predicted, reference) <= tolerance

    def formula_f1(self, predicted_latex: str, gold_latex: str) -> float:
        pred_tokens = set(predicted_latex.replace(" ", "").lower())
        gold_tokens = set(gold_latex.replace(" ", "").lower())
        if not pred_tokens and not gold_tokens:
            return 1.0
        tp = len(pred_tokens & gold_tokens)
        fp = len(pred_tokens - gold_tokens)
        fn = len(gold_tokens - pred_tokens)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if prec + rec == 0:
            return 0.0
        return 2 * prec * rec / (prec + rec)

    def compute_exact_match_accuracy(self, predictions: list[str], references: list[str]) -> float:
        if not predictions:
            return 0.0
        matches = sum(
            1 for p, g in zip(predictions, references)
            if self.exact_match(p, g)
        )
        return matches / len(predictions)

    def compute_traditional_metrics(self, predictions: list[str], references: list[str]) -> dict:
        bleus = [self.bleu_score(p, g) for p, g in zip(predictions, references)]
        f1s = [self.formula_f1(p, g) for p, g in zip(predictions, references)]
        neds = [self.normalized_edit_distance(p, g) for p, g in zip(predictions, references)]
        return {
            "bleu": sum(bleus) / len(bleus) if bleus else 0.0,
            "formula_f1": sum(f1s) / len(f1s) if f1s else 0.0,
            "ned": sum(neds) / len(neds) if neds else 1.0,
            "exact_match": self.compute_exact_match_accuracy(predictions, references),
        }