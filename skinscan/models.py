from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


SEVERITY = {"suspicious": 0, "medium": 1, "high": 2, "critical": 3}


@dataclass
class Issue:
    type: str
    severity: str
    confidence: float
    bounds: List[int]  # original image coordinates, [left, top, right, bottom)
    rules: List[str]
    message: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    children: List["Issue"] = field(default_factory=list)
    whitelist: List[str] = field(default_factory=list)
    suppressed: bool = False
    requiresReview: bool = False
    id: str = ""

    def to_dict(self):
        result = asdict(self)
        result.update(result.pop("metrics"))
        result["children"] = [child.to_dict() for child in self.children]
        return result


@dataclass
class DetectionResult:
    pageId: str
    image: str
    width: int = 0
    height: int = 0
    status: str = "uncertain"
    reviewStatus: str = "needs_review"
    issues: List[Issue] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    modules: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    effectiveConfig: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    durationMs: float = 0

    def to_dict(self):
        result = asdict(self)
        result["issues"] = [issue.to_dict() for issue in self.issues]
        result["schemaVersion"] = 1
        result["confidenceMeaning"] = "heuristic score, not a calibrated probability"
        return result


def overlap(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1]))
    aa = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    ab = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return intersection / (aa + ab - intersection), intersection / min(aa, ab)


def walk_issues(issues):
    for issue in issues:
        yield issue
        yield from walk_issues(issue.children)
