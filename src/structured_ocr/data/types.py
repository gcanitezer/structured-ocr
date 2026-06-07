"""Data types as simple stdlib classes."""


class BoundingBox:
    __slots__ = ("x1", "y1", "x2", "y2")

    def __init__(self, x1, y1, x2, y2):
        if x2 <= x1:
            raise ValueError("x2 must be greater than x1")
        if y2 <= y1:
            raise ValueError("y2 must be greater than y1")
        self.x1 = float(x1)
        self.y1 = float(y1)
        self.x2 = float(x2)
        self.y2 = float(y2)

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    @property
    def area(self):
        return self.width * self.height

    def __repr__(self):
        return f"BoundingBox(x1={self.x1}, y1={self.y1}, x2={self.x2}, y2={self.y2})"


class TextBlock:
    def __init__(self, text, bbox, confidence=0.0):
        self.text = text
        self.bbox = bbox
        if not (0.0 <= float(confidence) <= 1.0):
            raise ValueError("confidence must be in [0.0, 1.0]")
        self.confidence = float(confidence)


class FormulaBlock:
    def __init__(self, latex, bbox, confidence=0.0):
        self.latex = latex
        self.bbox = bbox
        self.confidence = max(0.0, min(1.0, float(confidence)))


class TableBlock:
    def __init__(self, content, bbox, num_rows, num_cols, confidence=0.0):
        self.content = content
        self.bbox = bbox
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.confidence = max(0.0, min(1.0, float(confidence)))


class ImageBlock:
    def __init__(self, path, bbox=None, caption="", confidence=0.0):
        self.path = path
        self.bbox = bbox
        self.caption = caption
        self.confidence = max(0.0, min(1.0, float(confidence)))


class DocumentNode:
    def __init__(self, node_type, content, bbox=None, children=None, metadata=None):
        self.node_type = node_type
        self.content = content
        self.bbox = bbox
        self.children = children or []
        self.metadata = metadata or {}


class DocumentStructure:
    def __init__(self, title="", sections=None, raw_latex="", page_count=1, metadata=None):
        self.title = title
        self.sections = sections or []
        self.raw_latex = raw_latex
        self.page_count = page_count
        self.metadata = metadata or {}

    def compute_tree_similarity(self, other):
        if self.raw_latex and other.raw_latex:
            s1 = set(self.raw_latex.split())
            s2 = set(other.raw_latex.split())
            if not s1 and not s2:
                return 1.0
            inter = len(s1 & s2)
            union = len(s1 | s2)
            return inter / union if union > 0 else 0.0
        return 0.0

    def structural_overlap(self, other):
        self_types = self._count_types()
        other_types = other._count_types()
        all_types = set(self_types.keys()) | set(other_types.keys())
        return {t: min(self_types.get(t, 0), other_types.get(t, 0)) for t in all_types}

    def _count_types(self):
        counts = {}
        for node in self.sections:
            counts[node.node_type] = counts.get(node.node_type, 0) + 1
        return counts


class LaTeXDocument:
    def __init__(self, source, document_class="article", packages=None, metadata=None, structure=None):
        self.source = source
        self.document_class = document_class
        self.packages = packages or []
        self.metadata = metadata or {}
        self.structure = structure


class OCRResult:
    def __init__(self, latex="", confidence=0.0, processing_time_ms=0.0, model_name="", page_number=1, detected_elements=None, warnings=None):
        self.latex = latex
        if not (0.0 <= float(confidence) <= 1.0):
            raise ValueError("confidence must be in [0.0, 1.0]")
        self.confidence = float(confidence)
        self.processing_time_ms = float(processing_time_ms)
        self.model_name = model_name
        self.page_number = page_number
        self.detected_elements = detected_elements or {}
        self.warnings = warnings or []


class EvaluationResult:
    def __init__(self, test_name, passed, score=0.0, details="", metrics=None, execution_time_ms=0.0):
        self.test_name = test_name
        self.passed = passed
        if not (0.0 <= float(score) <= 1.0):
            raise ValueError("score must be in [0.0, 1.0]")
        self.score = float(score)
        self.details = details
        self.metrics = metrics or {}
        self.execution_time_ms = float(execution_time_ms)


class BenchmarkResult:
    def __init__(self, metric_name, value, unit, threshold, passed, timestamp):
        self.metric_name = metric_name
        self.value = float(value)
        self.unit = unit
        self.threshold = float(threshold)
        self.passed = passed
        self.timestamp = timestamp