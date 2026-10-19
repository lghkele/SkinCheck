"""检测结果的数据约定与 JSON 序列化；这里不执行图像算法或文件 I/O。"""
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


SEVERITY = {"suspicious": 0, "medium": 1, "high": 2, "critical": 3}


@dataclass
class Issue:
    """单个机器候选：风险级别、证据可信度和人工复核要求分别记录。

    confidence 是工程评分，不是概率；children 仍是真实候选，必须参与 CI。
    metrics 保存各算法专属证据，序列化时展开到问题顶层，不能与公共字段重名。
    """
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
    """整页机器观察；modules 记录完整性，effectiveConfig 保留本次可复现配置。

    status 不代替 errors 或 reviewStatus；人工审核保存在独立 review.json，
    不回写此结果，也不追溯改变历史 CI 判定。
    """
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
    """返回 (交并比 IoU, 交集／较小框面积)，后者衡量包含关系。"""
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1]))
    aa = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    ab = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return intersection / (aa + ab - intersection), intersection / min(aa, ab)


def walk_issues(issues):
    """按父项在前递归遍历，确保评分、标注和审核不会遗漏嵌套候选。"""
    for issue in issues:
        yield issue
        yield from walk_issues(issue.children)
