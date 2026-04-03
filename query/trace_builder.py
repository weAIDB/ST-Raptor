import ast
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from utils.constants import LOG_DIR

from query.trace_schema import PlaybackSegment, TraceFrame, TraceOperation, TraceSession, TraceSubquery


_LOG_LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}) \| (?P<level>[A-Z]+)\s+\| (?P<message>.*)$"
)
_SUBQUERY_RE = re.compile(r"Subquery\s+(\d+)", re.IGNORECASE)
_DEPTH_RE = re.compile(r"Depth\s+(\d+)", re.IGNORECASE)
_FRAME_MARKER_RE = re.compile(r"TRACE_FRAME_(START|END)\s+depth=(\d+)\s+subquery=(\d+)", re.IGNORECASE)
_PRIMITIVE_RE = re.compile(r"\[(.*?)\]")


def _trace_v3_operation_title(kind: str, primitive: str = "", args: Optional[List[Any]] = None) -> str:
    args = list(args or [])
    primitive = str(primitive or "").upper()
    kind = str(kind or "").strip()
    if kind == "pre_cond":
        return "Pre-COND"
    if kind == "pre_math":
        return "Pre-MATH"
    if kind == "child_lookup":
        return f"CHL · {args[0]}" if args else "CHL"
    if kind == "father_lookup":
        return f"FAT · {args[0]}" if args else "FAT"
    if kind == "extract_lookup":
        return f"EXT · {' / '.join(str(x) for x in args[:2])}" if args else "EXT"
    if kind == "conditional_lookup":
        return f"COND · {' / '.join(str(x) for x in args[:2])}" if args else "COND"
    if kind == "compare_lookup":
        return f"CMP · {' / '.join(str(x) for x in args[:3])}" if args else "CMP"
    if kind == "foreach_lookup":
        return f"FOREACH · {' / '.join(str(x) for x in args[:2])}" if args else "FOREACH"
    if primitive:
        return primitive
    return kind or "Operation"


def _trace_kind_from_primitive(primitive: str) -> str:
    primitive = str(primitive or "").upper()
    mapping = {
        "CHL": "child_lookup",
        "FAT": "father_lookup",
        "EXT": "extract_lookup",
        "COND": "conditional_lookup",
        "CMP": "compare_lookup",
        "FOREACH": "foreach_lookup",
    }
    return mapping.get(primitive, "operation")


def _append_unique(seq: List[str], value: Any):
    text = str(value or "").strip()
    if text and text not in seq:
        seq.append(text)


def _append_detail(operation: TraceOperation, text: str):
    text = str(text or "").strip()
    if text and text not in operation.details:
        operation.details.append(text)


def _event_canonical_id(ev: Dict[str, Any]) -> str:
    return str(ev.get("canonical_id", "") or ev.get("canonical_trace_id", "") or "").strip()


def _maybe_parse_literal(text: str) -> Any:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        return ast.literal_eval(raw)
    except Exception:
        return raw


def _parse_log_timestamp(text: str) -> Optional[float]:
    try:
        return datetime.strptime(str(text).strip(), "%Y-%m-%d %H:%M:%S.%f").timestamp()
    except Exception:
        return None


def _normalize_log_window(chain: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    qa = (chain or {}).get("question_answering", {}) or {}
    log_window = qa.get("log_window", {}) or {}
    start_ts = log_window.get("start_ts")
    end_ts = log_window.get("end_ts")
    try:
        start_ts = float(start_ts) if start_ts is not None else None
    except Exception:
        start_ts = None
    try:
        end_ts = float(end_ts) if end_ts is not None else None
    except Exception:
        end_ts = None
    return start_ts, end_ts


def _read_log_records(log_path: str, start_ts: Optional[float], end_ts: Optional[float]) -> List[Dict[str, Any]]:
    if not log_path or not os.path.exists(log_path):
        return []
    records: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    with open(log_path, "r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            match = _LOG_LINE_RE.match(line)
            if match:
                if current:
                    records.append(current)
                ts_text = match.group("ts")
                current = {
                    "timestamp": ts_text,
                    "ts": _parse_log_timestamp(ts_text),
                    "level": match.group("level"),
                    "message": match.group("message"),
                    "lines": [match.group("message")],
                }
            elif current is not None:
                current["lines"].append(line)
        if current:
            records.append(current)

    if start_ts is None and end_ts is None:
        return records

    buffer_sec = 0.35
    filtered: List[Dict[str, Any]] = []
    for record in records:
        ts = record.get("ts")
        if ts is None:
            continue
        if start_ts is not None and ts < (start_ts - buffer_sec):
            continue
        if end_ts is not None and ts > (end_ts + buffer_sec):
            continue
        filtered.append(record)
    return filtered


def _extract_first_bracket_line(lines: List[str]) -> str:
    for line in lines:
        text = str(line or "").strip()
        if text.startswith("["):
            return text
    return ""


def _extract_primitive(raw: str) -> Tuple[str, List[str]]:
    args = re.findall(_PRIMITIVE_RE, str(raw or ""))
    if not args:
        return "", []
    return str(args[0] or "").upper(), [str(item or "") for item in args[1:]]


def _extract_prompt_table(lines: List[str]) -> Any:
    stripped = [str(line or "").strip() for line in lines]
    for idx, line in enumerate(stripped):
        if line == "### Table" and idx + 1 < len(stripped):
            return _maybe_parse_literal(stripped[idx + 1])
    return None


def _group_execution_events_by_primitive(execution_events: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    current_step = None
    for ev in execution_events or []:
        et = str(ev.get("event_type", "") or "")
        if et == "primitive_execute":
            current_step = int(ev.get("step", 0) or 0)
            grouped[current_step] = [ev]
            continue
        if current_step is None:
            continue
        grouped.setdefault(current_step, []).append(ev)
        if et == "primitive_generated":
            grouped[current_step].append(ev)
    return grouped


def build_typed_trace_v2(
    chain: Dict[str, Any],
    strict_trace: Dict[str, Any],
    execution_events: List[Dict[str, Any]],
) -> Dict[str, Any]:
    qa = (chain or {}).get("question_answering", {}) or {}
    primitive_steps = qa.get("primitive_steps", []) or []
    if not isinstance(primitive_steps, list):
        primitive_steps = []
    normalized_steps: List[Dict[str, Any]] = []
    for i, step in enumerate(primitive_steps):
        if not isinstance(step, dict):
            continue
        visited = step.get("visitedNodeIds", [])
        retrieved = step.get("retrievalNodeIds", [])
        normalized_steps.append({
            "index": i + 1,
            "step": int(step.get("step", 0) or 0),
            "type": str(step.get("type", "") or ""),
            "args": list(step.get("args", [])) if isinstance(step.get("args", []), list) else [],
            "visitedNodeIds": [str(x) for x in (visited if isinstance(visited, list) else []) if str(x).strip()],
            "retrievalNodeIds": [str(x) for x in (retrieved if isinstance(retrieved, list) else []) if str(x).strip()],
        })
    return {
        "version": "v2",
        "primitiveSteps": normalized_steps,
        "answerAnchorNodeId": (strict_trace or {}).get("answer_node_id"),
        "pathNodeOrder": list((strict_trace or {}).get("path_node_order", []) or []),
        "pathEdgeOrder": list((strict_trace or {}).get("path_edge_order", []) or []),
        "executionEventCount": len(execution_events or []),
    }


def _build_trace_v3_from_events(
    chain: Dict[str, Any],
    strict_trace: Dict[str, Any],
    execution_events: List[Dict[str, Any]],
) -> Dict[str, Any]:
    qa = (chain or {}).get("question_answering", {}) or {}
    subquery_meta_list = qa.get("subqueries", []) or []
    subquery_meta_map: Dict[int, Dict[str, Any]] = {}
    for item in subquery_meta_list:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("index", 0) or 0)
        except Exception:
            idx = 0
        if idx > 0:
            subquery_meta_map[idx] = item

    subqueries: List[Dict[str, Any]] = []
    subquery_lookup: Dict[int, Dict[str, Any]] = {}

    def get_subquery(subquery_index: int) -> Dict[str, Any]:
        idx = int(subquery_index or 0)
        meta = subquery_meta_map.get(idx, {})
        if idx not in subquery_lookup:
            subquery = {
                "index": idx,
                "query": str(meta.get("query", "") or ""),
                "needRetrieval": bool(meta.get("need_retrieval", False)),
                "answer": meta.get("answer"),
                "verifier": meta.get("verifier_check"),
                "reasoningType": str(meta.get("reasoning_type", "") or ""),
                "frames": [],
                "_frame_lookup": {},
                "_first_step": None,
            }
            subquery_lookup[idx] = subquery
            subqueries.append(subquery)
        return subquery_lookup[idx]

    def get_frame(subquery: Dict[str, Any], ctx: Dict[str, Any], ev: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        depth = int(ctx.get("depth", 0) or 0)
        if depth <= 0:
            return None
        frame_id = f"sq{subquery['index']}_depth_{depth}"
        frame_lookup = subquery.setdefault("_frame_lookup", {})
        if frame_id not in frame_lookup:
            first_step = int(ev.get("step", 0) or 0)
            frame = {
                "frameId": frame_id,
                "parentFrameId": None,
                "depth": depth,
                "index": 0,
                "title": "",
                "query": str(ctx.get("query", "") or subquery.get("query", "") or ""),
                "inputSummary": None,
                "outputSummary": None,
                "operations": [],
                "_first_step": first_step,
                "_operation_lookup": {},
            }
            frame_lookup[frame_id] = frame
            subquery["frames"].append(frame)
        frame = frame_lookup[frame_id]
        frame["_first_step"] = min(int(frame.get("_first_step") or int(ev.get("step", 0) or 0)), int(ev.get("step", 0) or 0))
        if subquery.get("_first_step") is None:
            subquery["_first_step"] = int(ev.get("step", 0) or 0)
        else:
            subquery["_first_step"] = min(int(subquery["_first_step"]), int(ev.get("step", 0) or 0))
        return frame

    def get_operation(frame: Dict[str, Any], ctx: Dict[str, Any], ev: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        extra = ev.get("extra", {}) or {}
        primitive = str(ctx.get("primitive", "") or ev.get("node_value", "") or "")
        if not primitive:
            return None
        operation_id = f"{frame['frameId']}_{primitive}_{int(ev.get('step', 0) or 0)}"
        operation_lookup = frame.setdefault("_operation_lookup", {})
        if operation_id not in operation_lookup:
            args = ctx.get("primitive_args", [])
            if not isinstance(args, list):
                args = extra.get("args", []) if isinstance(extra.get("args", []), list) else []
            operation = {
                "operationId": operation_id,
                "kind": _trace_kind_from_primitive(primitive),
                "primitive": primitive,
                "raw": str(ctx.get("primitive_raw", "") or ""),
                "args": list(args or []),
                "title": _trace_v3_operation_title(_trace_kind_from_primitive(primitive), primitive, args),
                "status": "",
                "resultSummary": None,
                "focusTargets": [],
                "details": [],
                "eventRange": {"startStep": int(ev.get("step", 0) or 0), "endStep": int(ev.get("step", 0) or 0)},
                "playback": {
                    "nodeIds": [],
                    "edgeIds": [],
                    "answerNodeId": None,
                    "canonicalNodeIds": [],
                    "canonicalEdgeIds": [],
                    "canonicalAnswerNodeId": None,
                },
                "semanticTargets": {},
                "hasPlayback": False,
                "hasSubtreePreview": False,
                "subtreePreviewData": None,
                "subtreePreviewTitle": "",
                "_playback_order": [],
                "_canonical_playback_order": [],
                "_first_step": int(ev.get("step", 0) or 0),
            }
            frame["operations"].append(operation)
            operation_lookup[operation_id] = operation
        return operation_lookup[operation_id]

    for ev in execution_events or []:
        ctx = ev.get("context", {}) or {}
        subquery = get_subquery(int(ctx.get("subquery_index", 0) or 0))
        frame = get_frame(subquery, ctx, ev)
        if frame is None:
            continue
        operation = get_operation(frame, ctx, ev)
        if operation is None:
            continue
        et = str(ev.get("event_type", "") or "")
        frontend_node_id = str(ev.get("frontend_node_id", "") or "").strip()
        canonical_trace_id = _event_canonical_id(ev)
        operation["eventRange"]["startStep"] = min(int(operation["eventRange"]["startStep"]), int(ev.get("step", 0) or 0))
        operation["eventRange"]["endStep"] = max(int(operation["eventRange"]["endStep"]), int(ev.get("step", 0) or 0))
        if et == "primitive_generated":
            _append_unique(operation["details"], f"Generate: {operation['raw']}")
        elif et == "primitive_execute":
            _append_unique(operation["details"], f"Execute: {operation['primitive']}")
        elif et == "retrieval_result":
            result_count = int((ev.get("extra", {}) or {}).get("result_count", 0) or 0)
            operation["resultSummary"] = {"resultCount": result_count}
            _append_unique(operation["details"], f"Retrieve: {result_count} result(s)")
        elif et == "visit" and frontend_node_id:
            _append_unique(operation["_playback_order"], frontend_node_id)
            if canonical_trace_id:
                _append_unique(operation["_canonical_playback_order"], canonical_trace_id)
        elif et == "retrieval_item":
            if frontend_node_id:
                _append_unique(operation["_playback_order"], frontend_node_id)
            if canonical_trace_id:
                _append_unique(operation["_canonical_playback_order"], canonical_trace_id)

    ordered_subqueries = sorted(subqueries, key=lambda item: (int(item.get("index", 0) or 0), int(item.get("_first_step") or 0)))
    for subquery in ordered_subqueries:
        frames = sorted(subquery.get("frames", []), key=lambda item: (int(item.get("depth", 0) or 0), int(item.get("_first_step") or 0)))
        for frame_index, frame in enumerate(frames, start=1):
            frame["index"] = frame_index
            frame["title"] = f"Frame {frame_index}"
            operations = sorted(frame.get("operations", []), key=lambda item: int(item.get("_first_step", 0) or 0))
            for operation in operations:
                node_order = list(operation.get("_playback_order", []) or [])
                canonical_node_order = list(operation.get("_canonical_playback_order", []) or [])
                edge_ids: List[str] = []
                canonical_edge_ids: List[str] = []
                for idx in range(1, len(node_order)):
                    edge_ids.append(f"{node_order[idx - 1]}->{node_order[idx]}")
                for idx in range(1, len(canonical_node_order)):
                    canonical_edge_ids.append(f"{canonical_node_order[idx - 1]}->{canonical_node_order[idx]}")
                operation["playback"] = {
                    "nodeIds": node_order,
                    "edgeIds": edge_ids,
                    "answerNodeId": node_order[-1] if node_order else None,
                    "canonicalNodeIds": canonical_node_order,
                    "canonicalEdgeIds": canonical_edge_ids,
                    "canonicalAnswerNodeId": canonical_node_order[-1] if canonical_node_order else None,
                }
                operation["hasPlayback"] = bool(node_order)
                operation.pop("_playback_order", None)
                operation.pop("_canonical_playback_order", None)
                operation.pop("_first_step", None)
            frame["operations"] = operations
            frame.pop("_operation_lookup", None)
        subquery["frames"] = frames
        subquery.pop("_frame_lookup", None)
        subquery.pop("_first_step", None)

    return {
        "version": "v3",
        "subqueries": ordered_subqueries,
        "answerAnchorNodeId": (strict_trace or {}).get("answer_node_id"),
        "pathNodeOrder": list((strict_trace or {}).get("path_node_order", []) or []),
        "pathEdgeOrder": list((strict_trace or {}).get("path_edge_order", []) or []),
        "executionEventCount": len(execution_events or []),
    }


def _ensure_subquery(subquery_map: Dict[int, TraceSubquery], idx: int, meta_map: Dict[int, Dict[str, Any]], ts: Optional[float]) -> TraceSubquery:
    if idx not in subquery_map:
        meta = meta_map.get(idx, {})
        subquery_map[idx] = TraceSubquery(
            index=idx,
            query=str(meta.get("query", "") or ""),
            needRetrieval=bool(meta.get("need_retrieval", False)),
            answer=meta.get("answer"),
            verifier=meta.get("verifier_check"),
            reasoningType=str(meta.get("reasoning_type", "") or ""),
            logStartTs=ts,
            logEndTs=ts,
        )
    subquery = subquery_map[idx]
    if ts is not None:
        if subquery.logStartTs is None:
            subquery.logStartTs = ts
        subquery.logEndTs = ts
    return subquery


def _start_frame(subquery: TraceSubquery, depth: int, ts: Optional[float], query: str = "") -> TraceFrame:
    frame_index = len(subquery.frames) + 1
    frame = TraceFrame(
        frameId=f"sq{subquery.index}_frame_{frame_index}",
        depth=int(depth or 0),
        index=frame_index,
        title=f"Frame {frame_index}",
        query=str(query or subquery.query or ""),
        logStartTs=ts,
        logEndTs=ts,
    )
    subquery.frames.append(frame)
    return frame


def _start_operation(frame: TraceFrame, kind: str, primitive: str, args: List[str], raw: str, ts: Optional[float]) -> TraceOperation:
    if frame.operations:
        frame.operations[-1].logEndTs = ts or frame.operations[-1].logEndTs
    op_index = len(frame.operations) + 1
    operation = TraceOperation(
        operationId=f"{frame.frameId}_op_{op_index}",
        kind=kind,
        primitive=str(primitive or "").upper(),
        raw=str(raw or ""),
        args=list(args or []),
        title=_trace_v3_operation_title(kind, primitive, args),
        logStartTs=ts,
        logEndTs=ts,
    )
    frame.operations.append(operation)
    return operation


def _find_or_open_frame(
    subquery: TraceSubquery,
    current_frame: Optional[TraceFrame],
    target_depth: int,
    ts: Optional[float],
    query: str,
    frame_prompt_seen: Dict[str, bool],
) -> TraceFrame:
    if current_frame and current_frame.depth == target_depth and not frame_prompt_seen.get(current_frame.frameId, False):
        current_frame.logEndTs = ts or current_frame.logEndTs
        return current_frame
    frame = _start_frame(subquery, target_depth, ts, query=query)
    frame_prompt_seen[frame.frameId] = False
    return frame


def _align_operation_with_events(
    operation: TraceOperation,
    subquery_index: int,
    execution_events: List[Dict[str, Any]],
    next_start_ts: Optional[float],
):
    if operation.kind == "pre_cond":
        matched: List[Dict[str, Any]] = []
        for ev in execution_events or []:
            ctx = ev.get("context", {}) or {}
            if int(ctx.get("subquery_index", 0) or 0) != int(subquery_index or 0):
                continue
            extra = ev.get("extra", {}) or {}
            phase = str(extra.get("phase", "") or "")
            if str(ev.get("event_type", "") or "") == "visit" and phase.startswith("condition"):
                matched.append(ev)
        if matched:
            operation.eventRange["startStep"] = int(matched[0].get("step", 0) or 0)
            operation.eventRange["endStep"] = int(matched[-1].get("step", 0) or 0)
        node_ids: List[str] = []
        edge_ids: List[str] = []
        canonical_node_ids: List[str] = []
        canonical_edge_ids: List[str] = []
        for ev in matched:
            frontend_node_id = str(ev.get("frontend_node_id", "") or "").strip()
            canonical_trace_id = _event_canonical_id(ev)
            if frontend_node_id:
                _append_unique(node_ids, frontend_node_id)
            if canonical_trace_id:
                _append_unique(canonical_node_ids, canonical_trace_id)
        for idx in range(1, len(node_ids)):
            edge_ids.append(f"{node_ids[idx - 1]}->{node_ids[idx]}")
        for idx in range(1, len(canonical_node_ids)):
            canonical_edge_ids.append(f"{canonical_node_ids[idx - 1]}->{canonical_node_ids[idx]}")
        operation.playback = PlaybackSegment(
            nodeIds=node_ids,
            edgeIds=edge_ids,
            answerNodeId=node_ids[-1] if node_ids else None,
            canonicalNodeIds=canonical_node_ids,
            canonicalEdgeIds=canonical_edge_ids,
            canonicalAnswerNodeId=canonical_node_ids[-1] if canonical_node_ids else None,
        )
        operation.hasPlayback = bool(node_ids)
        if node_ids:
            _append_detail(operation, f"Visited nodes: {len(node_ids)}")
        if edge_ids:
            _append_detail(operation, f"Path edges: {len(edge_ids)}")
        return

    start_ts = operation.logStartTs
    end_ts = operation.logEndTs if operation.logEndTs is not None else next_start_ts
    if start_ts is None:
        return
    if end_ts is None:
        end_ts = start_ts + 0.001
    epsilon = 0.02
    matched: List[Dict[str, Any]] = []
    for ev in execution_events or []:
        ctx = ev.get("context", {}) or {}
        if int(ctx.get("subquery_index", 0) or 0) != int(subquery_index or 0):
            continue
        ev_ts = ev.get("ts")
        try:
            ev_ts = float(ev_ts)
        except Exception:
            continue
        if ev_ts < (start_ts - epsilon):
            continue
        if next_start_ts is not None:
            if ev_ts >= (next_start_ts - epsilon):
                continue
        elif ev_ts > (end_ts + epsilon):
            continue
        matched.append(ev)

    if operation.kind == "pre_math":
        matched = []

    if matched:
        operation.eventRange["startStep"] = int(matched[0].get("step", 0) or 0)
        operation.eventRange["endStep"] = int(matched[-1].get("step", 0) or 0)

    node_ids: List[str] = []
    edge_ids: List[str] = []
    canonical_node_ids: List[str] = []
    canonical_edge_ids: List[str] = []
    for ev in matched:
        et = str(ev.get("event_type", "") or "")
        extra = ev.get("extra", {}) or {}
        frontend_node_id = str(ev.get("frontend_node_id", "") or "").strip()
        canonical_trace_id = _event_canonical_id(ev)
        if et == "visit" and frontend_node_id:
            _append_unique(node_ids, frontend_node_id)
            if canonical_trace_id:
                _append_unique(canonical_node_ids, canonical_trace_id)
        elif et == "retrieval_item":
            if frontend_node_id:
                _append_unique(node_ids, frontend_node_id)
            if canonical_trace_id:
                _append_unique(canonical_node_ids, canonical_trace_id)
            result_payload = extra.get("result")
            if result_payload is not None:
                operation.focusTargets.append({
                    "type": str(ev.get("node_type", "") or "result"),
                    "label": str(ev.get("node_type", "") or "result"),
                    "payload": result_payload,
                })
        elif et == "retrieval_result" and operation.resultSummary is None:
            operation.resultSummary = {
                "resultCount": int(extra.get("result_count", 0) or 0),
                "results": extra.get("results", []),
            }

    for idx in range(1, len(node_ids)):
        edge_ids.append(f"{node_ids[idx - 1]}->{node_ids[idx]}")
    for idx in range(1, len(canonical_node_ids)):
        canonical_edge_ids.append(f"{canonical_node_ids[idx - 1]}->{canonical_node_ids[idx]}")
    operation.playback = PlaybackSegment(
        nodeIds=node_ids,
        edgeIds=edge_ids,
        answerNodeId=node_ids[-1] if node_ids else None,
        canonicalNodeIds=canonical_node_ids,
        canonicalEdgeIds=canonical_edge_ids,
        canonicalAnswerNodeId=canonical_node_ids[-1] if canonical_node_ids else None,
    )
    operation.hasPlayback = bool(node_ids)
    if node_ids:
        _append_detail(operation, f"Visited nodes: {len(node_ids)}")
    if edge_ids:
        _append_detail(operation, f"Path edges: {len(edge_ids)}")


def build_trace_v3(
    chain: Dict[str, Any],
    strict_trace: Dict[str, Any],
    execution_events: List[Dict[str, Any]],
) -> Dict[str, Any]:
    qa = (chain or {}).get("question_answering", {}) or {}
    meta_list = qa.get("subqueries", []) or []
    meta_map: Dict[int, Dict[str, Any]] = {}
    for item in meta_list:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("index", 0) or 0)
        except Exception:
            idx = 0
        if idx > 0:
            meta_map[idx] = item

    start_ts, end_ts = _normalize_log_window(chain)
    log_path = os.path.join(LOG_DIR, "app.log")
    log_records = _read_log_records(log_path, start_ts, end_ts)
    if not log_records:
        return _build_trace_v3_from_events(chain, strict_trace, execution_events)

    subquery_map: Dict[int, TraceSubquery] = {}
    frame_prompt_seen: Dict[str, bool] = {}
    current_subquery: Optional[TraceSubquery] = None
    current_frame: Optional[TraceFrame] = None
    current_operation: Optional[TraceOperation] = None
    pending_generated = False
    pending_after_embedding = False
    pending_retrieval = False
    pending_answer_subquery = None
    pending_verifier_subquery = None
    pending_child_depth: Optional[int] = None
    pending_cond_result = False

    for record in log_records:
        ts = record.get("ts")
        lines = [str(line or "").strip() for line in (record.get("lines") or []) if str(line or "").strip()]
        if not lines:
            continue
        message = lines[0]
        level = str(record.get("level", "") or "").upper()
        frame_marker = _FRAME_MARKER_RE.search(message)

        if "Answering Subquery" in message:
            match = _SUBQUERY_RE.search(message)
            if not match:
                continue
            idx = int(match.group(1))
            current_subquery = _ensure_subquery(subquery_map, idx, meta_map, ts)
            current_frame = None
            current_operation = None
            pending_generated = False
            pending_after_embedding = False
            pending_retrieval = False
            pending_child_depth = None
            pending_cond_result = False
            _append_unique(current_subquery.narrative, message)
            continue

        if frame_marker:
            marker_kind = str(frame_marker.group(1) or "").upper()
            marker_depth = int(frame_marker.group(2) or 0)
            marker_subquery_idx = int(frame_marker.group(3) or 0)
            current_subquery = _ensure_subquery(subquery_map, marker_subquery_idx, meta_map, ts)
            if marker_kind == "START":
                current_frame = _start_frame(current_subquery, marker_depth, ts, query=current_subquery.query)
                frame_prompt_seen[current_frame.frameId] = False
                current_operation = None
            else:
                target_frame: Optional[TraceFrame] = None
                if (
                    current_frame is not None
                    and int(current_frame.depth or 0) == marker_depth
                    and int(current_subquery.index or 0) == marker_subquery_idx
                ):
                    target_frame = current_frame
                if target_frame is None:
                    for frame in reversed(current_subquery.frames):
                        if int(frame.depth or 0) == marker_depth:
                            target_frame = frame
                            break
                if target_frame is not None:
                    target_frame.logEndTs = ts or target_frame.logEndTs
                current_operation = None
                current_frame = None
                for frame in reversed(current_subquery.frames):
                    if target_frame is not None and frame.frameId == target_frame.frameId:
                        continue
                    if int(frame.depth or 0) < marker_depth:
                        current_frame = frame
                        break
            continue

        if current_subquery is None:
            continue

        current_subquery.logEndTs = ts
        if level == "INFO":
            _append_unique(current_subquery.narrative, message)

        prompt_table = _extract_prompt_table(lines)
        if prompt_table is not None and current_frame is not None and current_frame.inputSummary is None:
            current_frame.inputSummary = prompt_table

        if "Prompt Args for Primitive Depth" in message:
            depth_match = _DEPTH_RE.search(message)
            depth = int(depth_match.group(1)) if depth_match else (current_frame.depth if current_frame else 1)
            current_frame = _find_or_open_frame(
                current_subquery,
                current_frame,
                depth,
                ts,
                query=current_subquery.query,
                frame_prompt_seen=frame_prompt_seen,
            )
            frame_prompt_seen[current_frame.frameId] = True
            current_frame.inputSummary = prompt_table or current_frame.inputSummary
            current_frame.logEndTs = ts
            continue

        if "没有嵌套，尝试使用COND操作" in message:
            target_depth = pending_child_depth or ((current_frame.depth + 1) if current_frame else 1)
            current_frame = _find_or_open_frame(
                current_subquery,
                current_frame,
                target_depth,
                ts,
                query=current_subquery.query,
                frame_prompt_seen=frame_prompt_seen,
            )
            current_operation = _start_operation(current_frame, "pre_cond", "", [], "pre_cond", ts)
            current_frame.logEndTs = ts
            pending_cond_result = False
            continue

        if "使用COND操作" in message and current_operation and current_operation.kind == "pre_cond":
            bracket_line = _extract_first_bracket_line(lines[1:] or [])
            if bracket_line:
                _append_detail(current_operation, f"Apply: {bracket_line}")
            current_operation.logEndTs = ts
            continue

        if current_operation and current_operation.kind == "pre_cond" and message.startswith("[COND]"):
            _append_detail(current_operation, f"Apply: {message}")
            current_operation.logEndTs = ts
            continue

        if "COND操作结果" in message and current_operation and current_operation.kind == "pre_cond":
            pending_cond_result = True
            current_operation.logEndTs = ts
            continue

        if pending_cond_result and current_operation and current_operation.kind == "pre_cond":
            current_operation.resultSummary = _maybe_parse_literal(message)
            _append_detail(current_operation, f"Result: {message}")
            current_operation.logEndTs = ts
            pending_cond_result = False
            continue

        if "没有嵌套，尝试使用MATH操作" in message:
            target_depth = current_frame.depth if current_frame else (pending_child_depth or 1)
            current_frame = _find_or_open_frame(
                current_subquery,
                current_frame,
                target_depth,
                ts,
                query=current_subquery.query,
                frame_prompt_seen=frame_prompt_seen,
            )
            current_operation = _start_operation(current_frame, "pre_math", "", [], "pre_math", ts)
            current_frame.logEndTs = ts
            continue

        if "不需要进行MATH操作" in message and current_operation and current_operation.kind == "pre_math":
            current_operation.status = "skipped"
            current_operation.resultSummary = "None"
            _append_detail(current_operation, message)
            current_operation.logEndTs = ts
            continue

        if "Generated Primitive of" in message:
            pending_generated = True
            pending_after_embedding = False
            pending_retrieval = False
            continue

        if pending_generated:
            raw = _extract_first_bracket_line(lines)
            if raw:
                primitive, args = _extract_primitive(raw)
                kind = _trace_kind_from_primitive(primitive)
                if current_frame is None:
                    current_frame = _start_frame(current_subquery, pending_child_depth or 1, ts, query=current_subquery.query)
                    frame_prompt_seen[current_frame.frameId] = False
                current_operation = _start_operation(current_frame, kind, primitive, args, raw, ts)
                _append_detail(current_operation, f"Generate: {raw}")
                current_frame.logEndTs = ts
                pending_generated = False
                continue

        if "Primitive After Embedding" in message:
            pending_after_embedding = True
            continue

        if pending_after_embedding and current_operation is not None:
            raw = _extract_first_bracket_line(lines)
            if raw:
                primitive, args = _extract_primitive(raw)
                current_operation.primitive = primitive or current_operation.primitive
                current_operation.args = args or current_operation.args
                current_operation.title = _trace_v3_operation_title(current_operation.kind, current_operation.primitive, current_operation.args)
                _append_detail(current_operation, f"Resolved: {raw}")
                current_operation.logEndTs = ts
                pending_after_embedding = False
                continue

        if "Primitive Execution" in message and current_operation is not None:
            _append_detail(current_operation, f"Execute: {current_operation.primitive or current_operation.kind}")
            current_operation.logEndTs = ts
            continue

        if "Retreval Result" in message:
            pending_retrieval = True
            if current_operation is not None:
                current_operation.logEndTs = ts
            continue

        if pending_retrieval and current_operation is not None:
            if message.startswith("Retrieved Schema:"):
                payload = _maybe_parse_literal(message.split(":", 1)[1].strip())
                current_operation.resultSummary = payload
                current_operation.hasSubtreePreview = True
                current_operation.subtreePreviewData = payload
                current_operation.subtreePreviewTitle = current_operation.title or "Subtree Schema"
                current_frame.hasSubtreePreview = True
                current_frame.subtreePreviewData = payload
                current_frame.subtreePreviewTitle = current_operation.subtreePreviewTitle
                pending_child_depth = (current_frame.depth or 0) + 1
                _append_detail(current_operation, f"Retrieved Schema: {message.split(':', 1)[1].strip()}")
                current_operation.status = "non_empty"
                current_operation.logEndTs = ts
                pending_retrieval = False
                continue
            if message.startswith("Retrieved Data:"):
                payload = _maybe_parse_literal(message.split(":", 1)[1].strip())
                current_operation.resultSummary = payload
                current_operation.status = "non_empty"
                _append_detail(current_operation, f"Retrieved Data: {message.split(':', 1)[1].strip()}")
                current_operation.logEndTs = ts
                pending_child_depth = None
                pending_retrieval = False
                continue

        if "Final Retrieved Data for Subquery" in message:
            if current_frame is not None:
                current_frame.outputSummary = None
            continue

        if "Verifier Check Answer for Subquery" in message:
            match = _SUBQUERY_RE.search(message)
            pending_verifier_subquery = int(match.group(1)) if match else current_subquery.index
            continue

        if pending_verifier_subquery is not None:
            subquery = _ensure_subquery(subquery_map, int(pending_verifier_subquery), meta_map, ts)
            if message in {"True", "False"}:
                subquery.verifier = (message == "True")
            else:
                subquery.verifier = message
            pending_verifier_subquery = None
            continue

        if "Answer for Subquery" in message:
            match = _SUBQUERY_RE.search(message)
            pending_answer_subquery = int(match.group(1)) if match else current_subquery.index
            continue

        if pending_answer_subquery is not None:
            subquery = _ensure_subquery(subquery_map, int(pending_answer_subquery), meta_map, ts)
            subquery.answer = message
            pending_answer_subquery = None
            continue

    ordered_subqueries = [subquery_map[key] for key in sorted(subquery_map.keys())]
    if not ordered_subqueries:
        return _build_trace_v3_from_events(chain, strict_trace, execution_events)

    qa_subqueries = qa.get("subqueries", []) if isinstance(qa.get("subqueries", []), list) else []
    qa_map = {int(item.get("index", 0) or 0): item for item in qa_subqueries if isinstance(item, dict)}

    for subquery in ordered_subqueries:
        meta = qa_map.get(subquery.index, {})
        if not subquery.query:
            subquery.query = str(meta.get("query", "") or "")
        if subquery.answer is None:
            subquery.answer = meta.get("answer")
        if subquery.verifier is None:
            subquery.verifier = meta.get("verifier_check")
        if not subquery.reasoningType:
            subquery.reasoningType = str(meta.get("reasoning_type", "") or "")
        if not subquery.frames:
            subquery.frames.append(_start_frame(subquery, 1, subquery.logStartTs, query=subquery.query))

        subquery_end_ts = subquery.logEndTs
        for frame in subquery.frames:
            if not frame.title:
                frame.title = f"Frame {frame.index or 0}"
            if not frame.query:
                frame.query = subquery.query
            frame.logEndTs = frame.logEndTs or subquery_end_ts
            if frame.operations:
                for idx, operation in enumerate(frame.operations):
                    next_start_ts = frame.operations[idx + 1].logStartTs if idx + 1 < len(frame.operations) else None
                    if operation.logEndTs is None:
                        operation.logEndTs = next_start_ts or frame.logEndTs or subquery_end_ts
                    _align_operation_with_events(operation, subquery.index, execution_events, next_start_ts)
            if frame.operations:
                last_with_result = [op.resultSummary for op in frame.operations if op.resultSummary is not None]
                if last_with_result:
                    frame.outputSummary = last_with_result[-1]
                subtree_ops = [op for op in frame.operations if op.hasSubtreePreview and op.subtreePreviewData is not None]
                if subtree_ops and not frame.hasSubtreePreview:
                    frame.hasSubtreePreview = True
                    frame.subtreePreviewData = subtree_ops[-1].subtreePreviewData
                    frame.subtreePreviewTitle = subtree_ops[-1].subtreePreviewTitle

    session = TraceSession(
        version="v3",
        subqueries=ordered_subqueries,
        answerAnchorNodeId=(strict_trace or {}).get("answer_node_id"),
        pathNodeOrder=list((strict_trace or {}).get("path_node_order", []) or []),
        pathEdgeOrder=list((strict_trace or {}).get("path_edge_order", []) or []),
        executionEventCount=len(execution_events or []),
    )
    return session.to_public_dict()
