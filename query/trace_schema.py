from dataclasses import dataclass, field, is_dataclass
from typing import Any, Dict, List, Optional


def dataclass_to_dict(value: Any) -> Any:
    if is_dataclass(value):
        result: Dict[str, Any] = {}
        for field_name in value.__dataclass_fields__:
            result[field_name] = dataclass_to_dict(getattr(value, field_name))
        return result
    if isinstance(value, list):
        return [dataclass_to_dict(item) for item in value]
    if isinstance(value, dict):
        return {key: dataclass_to_dict(item) for key, item in value.items()}
    return value


@dataclass
class PlaybackSegment:
    nodeIds: List[str] = field(default_factory=list)
    edgeIds: List[str] = field(default_factory=list)
    answerNodeId: Optional[str] = None
    canonicalNodeIds: List[str] = field(default_factory=list)
    canonicalEdgeIds: List[str] = field(default_factory=list)
    canonicalAnswerNodeId: Optional[str] = None


@dataclass
class TraceOperation:
    operationId: str
    kind: str = ""
    primitive: str = ""
    raw: str = ""
    args: List[Any] = field(default_factory=list)
    title: str = ""
    status: str = ""
    resultSummary: Any = None
    focusTargets: List[Dict[str, Any]] = field(default_factory=list)
    details: List[str] = field(default_factory=list)
    eventRange: Dict[str, Any] = field(default_factory=lambda: {"startStep": 0, "endStep": 0})
    playback: PlaybackSegment = field(default_factory=PlaybackSegment)
    semanticTargets: Dict[str, Any] = field(default_factory=dict)
    hasPlayback: bool = False
    hasSubtreePreview: bool = False
    subtreePreviewData: Any = None
    subtreePreviewTitle: str = ""
    logStartTs: Optional[float] = None
    logEndTs: Optional[float] = None

    def to_public_dict(self) -> Dict[str, Any]:
        data = dataclass_to_dict(self)
        data["playback"] = dataclass_to_dict(self.playback)
        data.pop("logStartTs", None)
        data.pop("logEndTs", None)
        return data


@dataclass
class TraceFrame:
    frameId: str
    parentFrameId: Optional[str] = None
    depth: int = 0
    index: int = 0
    title: str = ""
    query: str = ""
    inputSummary: Any = None
    outputSummary: Any = None
    operations: List[TraceOperation] = field(default_factory=list)
    hasSubtreePreview: bool = False
    subtreePreviewData: Any = None
    subtreePreviewTitle: str = ""
    logStartTs: Optional[float] = None
    logEndTs: Optional[float] = None

    def to_public_dict(self) -> Dict[str, Any]:
        data = dataclass_to_dict(self)
        data["operations"] = [operation.to_public_dict() for operation in self.operations]
        data.pop("logStartTs", None)
        data.pop("logEndTs", None)
        return data


@dataclass
class TraceSubquery:
    index: int
    query: str = ""
    needRetrieval: bool = False
    answer: Any = None
    verifier: Any = None
    reasoningType: str = ""
    narrative: List[str] = field(default_factory=list)
    frames: List[TraceFrame] = field(default_factory=list)
    logStartTs: Optional[float] = None
    logEndTs: Optional[float] = None

    def to_public_dict(self) -> Dict[str, Any]:
        data = dataclass_to_dict(self)
        data["frames"] = [frame.to_public_dict() for frame in self.frames]
        data.pop("logStartTs", None)
        data.pop("logEndTs", None)
        return data


@dataclass
class TraceSession:
    version: str = "v3"
    subqueries: List[TraceSubquery] = field(default_factory=list)
    answerAnchorNodeId: Optional[str] = None
    pathNodeOrder: List[str] = field(default_factory=list)
    pathEdgeOrder: List[str] = field(default_factory=list)
    executionEventCount: int = 0

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "subqueries": [subquery.to_public_dict() for subquery in self.subqueries],
            "answerAnchorNodeId": self.answerAnchorNodeId,
            "pathNodeOrder": list(self.pathNodeOrder),
            "pathEdgeOrder": list(self.pathEdgeOrder),
            "executionEventCount": int(self.executionEventCount or 0),
        }
