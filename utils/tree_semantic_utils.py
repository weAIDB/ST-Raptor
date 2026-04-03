import re
from typing import Any, Dict, List, Optional, Tuple

from table2tree.feature_tree import BodyNode, FeatureTree, IndexNode


def safe_trace_token(value: Any) -> str:
    text = str(value if value is not None else "").strip()
    if not text:
        return "empty"
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", text)
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe[:40] or "empty"


def _safe_token(value: Any) -> str:
    return safe_trace_token(value)


def make_canonical_trace_id(path_parts: List[Any], prefix: str = "ct") -> str:
    raw = "|".join(str(part if part is not None else "").strip() for part in (path_parts or []) if str(part if part is not None else "").strip())
    if not raw:
        raw = "root"
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", raw)
    safe = re.sub(r"_+", "_", safe).strip("_")
    return f"{prefix}_{safe or 'root'}"


def build_typed_root_parts(root_name: str) -> List[str]:
    return [f"root_{safe_trace_token(root_name)}"]


def build_typed_index_segment(index: int, label: Any) -> str:
    return f"m_{int(index)}_{safe_trace_token(label)}"


def build_typed_body_label(index_label: str, body_value: Any, is_subtree: bool = False) -> str:
    if is_subtree or isinstance(body_value, FeatureTree):
        return f"{index_label}::subtree"
    return str(body_value if body_value is not None else "")


def build_typed_body_segment(body_index: int, index_label: str, body_value: Any, is_subtree: bool = False) -> str:
    return f"b_{int(body_index)}_{safe_trace_token(build_typed_body_label(index_label, body_value, is_subtree=is_subtree))}"


def build_typed_index_id(path_parts: List[str], index: int, label: Any) -> str:
    return "ft:" + "/".join(list(path_parts or []) + [build_typed_index_segment(index, label)])


def build_typed_body_id(path_parts: List[str], body_index: int, index_label: str, body_value: Any, is_subtree: bool = False) -> str:
    return "ft:" + "/".join(list(path_parts or []) + [build_typed_body_segment(body_index, index_label, body_value, is_subtree=is_subtree)])


def _body_label(body_node: BodyNode, index_label: str) -> str:
    return build_typed_body_label(index_label, body_node.value, is_subtree=isinstance(body_node.value, FeatureTree))


def build_typed_tree_v2(
    f_tree: FeatureTree,
    root_name: str = "ROOT",
    file_scope: str = "",
) -> Tuple[Dict[str, Any], Dict[int, str]]:
    """
    Export FeatureTree into a typed frontend tree and return object-id lookup.

    - IndexNode -> M_NODE
    - BodyNode  -> B_NODE
    """
    node_lookup: Dict[int, str] = {}

    def walk_feature(tree: FeatureTree, path: List[str], owner_name: str) -> List[Dict[str, Any]]:
        roots: List[Dict[str, Any]] = []
        index_children = list((tree.index_tree.root.children if tree and tree.index_tree else []) or [])

        for idx, index_node in enumerate(index_children):
            m_token = _safe_token(index_node.value)
            m_path = path + [build_typed_index_segment(idx, index_node.value)]
            m_id = "ft:" + "/".join(m_path)
            m_label = str(index_node.value if index_node.value is not None else "M")
            m_node: Dict[str, Any] = {
                "id": m_id,
                "canonicalTraceId": make_canonical_trace_id(["typed"] + m_path),
                "name": m_label,
                "nodeType": "M_NODE",
                "sourceKind": "index_node",
                "owner": owner_name,
                "children": [],
            }
            node_lookup[id(index_node)] = m_id

            bodies = list(index_node.body or [])
            for bi, body_node in enumerate(bodies):
                b_label = _body_label(body_node, m_label)
                b_token = _safe_token(b_label)
                is_subtree = isinstance(body_node.value, FeatureTree)
                b_path = m_path + [build_typed_body_segment(bi, m_label, body_node.value, is_subtree=is_subtree)]
                b_id = "ft:" + "/".join(b_path)
                b_node: Dict[str, Any] = {
                    "id": b_id,
                    "canonicalTraceId": make_canonical_trace_id(["typed"] + b_path),
                    "name": b_label if b_label else "B",
                    "nodeType": "B_NODE",
                    "sourceKind": "subtree_ref" if is_subtree else "value_leaf",
                    "owner": owner_name,
                    "children": [],
                }
                node_lookup[id(body_node)] = b_id

                if is_subtree:
                    child_feature: FeatureTree = body_node.value
                    sub_owner = f"{owner_name}.{m_token}.{bi}"
                    b_node["children"] = walk_feature(
                        child_feature,
                        b_path,
                        sub_owner,
                    )

                m_node["children"].append(b_node)

            roots.append(m_node)
        return roots

    root_parts = build_typed_root_parts(root_name)
    scope_text = str(file_scope or "").strip()
    if scope_text:
        root_parts = root_parts + [
            build_typed_index_segment(0, scope_text),
            build_typed_body_segment(0, scope_text, None, is_subtree=True),
        ]
    roots = walk_feature(f_tree, root_parts, root_name)
    payload = {
        "version": "v2",
        "rootName": root_name,
        "roots": roots,
    }
    return payload, node_lookup


def make_tree_canonical_id(path_parts: List[Any]) -> str:
    return make_canonical_trace_id(["tree"] + list(path_parts or []))


def make_tree_group_canonical_id(path_parts: List[Any]) -> str:
    return make_canonical_trace_id(["tree_group"] + list(path_parts or []))


def make_semantic_canonical_id(alias: Any, target_kind: str = "node") -> str:
    alias_text = str(alias or "").strip()
    if not alias_text:
        return ""
    kind = "group" if str(target_kind or "").strip().lower() == "group" else "node"
    return make_canonical_trace_id(["semantic", kind, alias_text])


def _resolve_alias_hits(alias_hits: Dict[str, List[Dict[str, str]]]) -> Dict[str, Dict[str, str]]:
    resolved: Dict[str, Dict[str, str]] = {}
    for alias_text, hits in alias_hits.items():
        concrete = sorted(
            {
                str(item.get("canonical", "") or "").strip()
                for item in (hits or [])
                if str(item.get("canonical", "") or "").strip()
            }
        )
        groups = sorted(
            {
                str(item.get("group", "") or "").strip()
                for item in (hits or [])
                if str(item.get("group", "") or "").strip()
            }
        )
        if len(concrete) == 1:
            resolved[alias_text] = {
                "canonical_id": concrete[0],
                "target_kind": "node",
            }
            continue
        if len(groups) == 1:
            resolved[alias_text] = {
                "canonical_id": groups[0],
                "target_kind": "group",
            }
    return resolved


def build_flat_row_alias_target_map(raw_row: Any, typed_root_name: str = "HO_TREE") -> Dict[str, Dict[str, str]]:
    """
    Build alias -> canonical target mapping without relying on runtime tree nodes.

    Returned shape:
    {
        "ft:...": {"canonical_id": "ct_tree_...", "target_kind": "node|group"}
    }
    """
    if not isinstance(raw_row, dict):
        return {}
    alias_hits: Dict[str, List[Dict[str, str]]] = {}

    def add_alias(alias: Any, canonical_id: Any = "", group_canonical_id: Any = "") -> None:
        alias_text = str(alias or "").strip()
        canonical_text = str(canonical_id or "").strip()
        group_text = str(group_canonical_id or "").strip()
        if not alias_text or (not canonical_text and not group_text):
            return
        alias_hits.setdefault(alias_text, []).append({
            "canonical": canonical_text,
            "group": group_text,
        })

    typed_root_parts = build_typed_root_parts(typed_root_name)
    base = ["root", "flat_row"]

    def walk(value: Any, ct_parts: List[str], ft_parts: List[str]):
        if isinstance(value, dict):
            for idx, (k, v) in enumerate(value.items()):
                k_str = str(k)
                cur_ct_parts = ct_parts + [f"k_{k_str}", f"idx_{idx}"]
                cur_ft_parts = ft_parts + [build_typed_index_segment(idx, k_str)]
                
                add_alias(
                    build_typed_index_id(ft_parts, idx, k_str),
                    canonical_id=make_tree_canonical_id(cur_ct_parts)
                )

                if isinstance(v, dict) or isinstance(v, list):
                    add_alias(
                        build_typed_body_id(cur_ft_parts, 0, k_str, None, is_subtree=True),
                        canonical_id=make_tree_canonical_id(cur_ct_parts + ["body"])
                    )
                    walk(v, cur_ct_parts + ["body"], cur_ft_parts + [build_typed_body_segment(0, k_str, None, is_subtree=True)])
                else:
                    add_alias(
                        build_typed_body_id(cur_ft_parts, 0, k_str, v, is_subtree=False),
                        canonical_id=make_tree_canonical_id(cur_ct_parts + ["body", "v"])
                    )

        elif isinstance(value, list):
            columns = []
            for row in value:
                if isinstance(row, dict):
                    for k in row.keys():
                        if k not in columns:
                            columns.append(k)
            
            for row_idx, row in enumerate(value):
                if not isinstance(row, dict):
                    continue
                cur_ct_parts = ct_parts + [f"i_{row_idx}"]
                group_ct_parts = cur_ct_parts + ["group"]
                
                for k, v in row.items():
                    k_str = str(k)
                    col_idx = columns.index(k)
                    
                    cell_key_ct_parts = group_ct_parts + [f"k_{k_str}", f"idx_{col_idx}"]
                    cell_val_ct_parts = cell_key_ct_parts + ["body", "v"]
                    
                    group_canonical = make_tree_group_canonical_id(
                        ct_parts + ["header_group", f"k_{k_str}", f"idx_{col_idx}"]
                    )

                    col_ft_parts = ft_parts + [build_typed_index_segment(col_idx, k_str)]
                    
                    add_alias(
                        build_typed_index_id(ft_parts, col_idx, k_str),
                        canonical_id=make_tree_canonical_id(cell_key_ct_parts),
                        group_canonical_id=group_canonical
                    )
                    
                    add_alias(
                        build_typed_body_id(col_ft_parts, row_idx, k_str, v, is_subtree=False),
                        canonical_id=make_tree_canonical_id(cell_val_ct_parts)
                    )

    walk(raw_row, base + ["root"], typed_root_parts)

    return _resolve_alias_hits(alias_hits)


def build_flat_column_alias_target_map(raw_column: Any, typed_root_name: str = "HO_TREE") -> Dict[str, Dict[str, str]]:
    """
    Build alias -> canonical target mapping for flat-column view.
    """
    if not isinstance(raw_column, dict):
        return {}
    alias_hits: Dict[str, List[Dict[str, str]]] = {}

    def add_alias(alias: Any, canonical_id: Any = "") -> None:
        alias_text = str(alias or "").strip()
        canonical_text = str(canonical_id or "").strip()
        if not alias_text or not canonical_text:
            return
        alias_hits.setdefault(alias_text, []).append({
            "canonical": canonical_text,
            "group": "",
        })

    typed_root_parts = build_typed_root_parts(typed_root_name)
    base_parts = ["root", "flat_column", "root"]

    def walk_dict(feature_dict: Dict[str, Any], typed_feature_parts: List[str], canonical_feature_parts: List[str]) -> None:
        if not isinstance(feature_dict, dict):
            return
        for idx, (index_key, index_value) in enumerate(feature_dict.items()):
            index_name = str(index_key if index_key is not None else "")
            index_alias = build_typed_index_id(typed_feature_parts, idx, index_name)
            index_parts = list(canonical_feature_parts) + [f"k_{index_name}", f"idx_{idx}"]
            add_alias(index_alias, canonical_id=make_tree_canonical_id(index_parts))

            typed_index_parts = list(typed_feature_parts) + [build_typed_index_segment(idx, index_name)]
            if isinstance(index_value, dict):
                # flat column tree does not keep an explicit subtree body node; map it to the index canonical node.
                subtree_alias = build_typed_body_id(
                    typed_index_parts,
                    0,
                    index_name,
                    None,
                    is_subtree=True,
                )
                add_alias(subtree_alias, canonical_id=make_tree_canonical_id(index_parts))
                walk_dict(
                    index_value,
                    typed_index_parts + [build_typed_body_segment(0, index_name, None, is_subtree=True)],
                    index_parts + ["body"],
                )
                continue

            if isinstance(index_value, list):
                for item_idx, item in enumerate(index_value):
                    item_is_subtree = isinstance(item, dict)
                    item_alias = build_typed_body_id(
                        typed_index_parts,
                        item_idx,
                        index_name,
                        item,
                        is_subtree=item_is_subtree,
                    )
                    item_parts = index_parts + ["body", f"i_{item_idx}"]
                    add_alias(item_alias, canonical_id=make_tree_canonical_id(item_parts))
                    if item_is_subtree:
                        walk_dict(
                            item,
                            typed_index_parts + [build_typed_body_segment(item_idx, index_name, item, is_subtree=True)],
                            item_parts + ["group"],
                        )
                continue

            scalar_alias = build_typed_body_id(
                typed_index_parts,
                0,
                index_name,
                index_value,
                is_subtree=False,
            )
            scalar_parts = index_parts + ["body", "v"]
            add_alias(scalar_alias, canonical_id=make_tree_canonical_id(scalar_parts))

    walk_dict(raw_column, typed_root_parts, base_parts)

    return _resolve_alias_hits(alias_hits)


def build_semantic_projection_bundle(
    raw_row: Any,
    raw_column: Any,
    typed_root_name: str = "HO_TREE",
) -> Dict[str, Any]:
    """
    Build a unified semantic canonical space and its row/column projections.
    """
    row_alias_targets = build_flat_row_alias_target_map(raw_row, typed_root_name=typed_root_name)
    column_alias_targets = build_flat_column_alias_target_map(raw_column, typed_root_name=typed_root_name)
    alias_to_semantic: Dict[str, Dict[str, str]] = {}
    semantic_to_views: Dict[str, Dict[str, Any]] = {}
    row_canonical_to_semantic: Dict[str, str] = {}
    column_canonical_to_semantic: Dict[str, str] = {}

    aliases = sorted(set((row_alias_targets or {}).keys()) | set((column_alias_targets or {}).keys()))
    for alias in aliases:
        row_target = row_alias_targets.get(alias, {}) if isinstance(row_alias_targets, dict) else {}
        column_target = column_alias_targets.get(alias, {}) if isinstance(column_alias_targets, dict) else {}
        row_canonical = str((row_target or {}).get("canonical_id", "") or "").strip()
        row_kind = str((row_target or {}).get("target_kind", "") or "").strip()
        column_canonical = str((column_target or {}).get("canonical_id", "") or "").strip()
        column_kind = str((column_target or {}).get("target_kind", "") or "").strip()
        if not row_canonical and not column_canonical:
            continue
        target_kind = row_kind or column_kind or ("group" if row_canonical.startswith("ct_tree_group_") else "node")
        semantic_id = make_semantic_canonical_id(alias, target_kind=target_kind)
        if not semantic_id:
            continue
        alias_to_semantic[alias] = {
            "canonical_id": semantic_id,
            "target_kind": target_kind,
        }
        entry = semantic_to_views.setdefault(
            semantic_id,
            {
                "row": [],
                "column": [],
                "target_kind": target_kind,
                "aliases": [],
            },
        )
        if alias not in entry["aliases"]:
            entry["aliases"].append(alias)
        if row_canonical and row_canonical not in entry["row"]:
            entry["row"].append(row_canonical)
            row_canonical_to_semantic[row_canonical] = semantic_id
        if column_canonical and column_canonical not in entry["column"]:
            entry["column"].append(column_canonical)
            column_canonical_to_semantic[column_canonical] = semantic_id

    for entry in semantic_to_views.values():
        entry["row"] = sorted({str(x or "").strip() for x in entry.get("row", []) if str(x or "").strip()})
        entry["column"] = sorted({str(x or "").strip() for x in entry.get("column", []) if str(x or "").strip()})
        entry["aliases"] = sorted({str(x or "").strip() for x in entry.get("aliases", []) if str(x or "").strip()})

    return {
        "alias_to_semantic": alias_to_semantic,
        "semantic_to_views": semantic_to_views,
        "row_canonical_to_semantic": row_canonical_to_semantic,
        "column_canonical_to_semantic": column_canonical_to_semantic,
    }


def build_nested_index_projection_map(
    raw_column: Any,
    semantic_bundle: Optional[Dict[str, Any]] = None,
    typed_root_name: str = "HO_TREE",
) -> Dict[str, Dict[str, Any]]:
    """
    Build semantic-id -> nested index projection map for nested feature viewer.

    Rules:
    - Only index node keys are mapped.
    - Body nodes are not mapped directly.
    - Drilldown is considered available only when the index value is a dict.
    """
    if not isinstance(raw_column, dict):
        return {}

    bundle = semantic_bundle if isinstance(semantic_bundle, dict) else {}
    alias_to_semantic: Dict[str, str] = {}
    for alias, target in (bundle.get("alias_to_semantic", {}) or {}).items():
        alias_text = str(alias or "").strip()
        if not alias_text or not isinstance(target, dict):
            continue
        semantic_id = str(target.get("canonical_id", "") or "").strip()
        if semantic_id:
            alias_to_semantic[alias_text] = semantic_id

    if not alias_to_semantic:
        return {}

    projection_map: Dict[str, Dict[str, Any]] = {}
    typed_root_parts = build_typed_root_parts(typed_root_name)

    def normalize_path(path_parts: List[Any]) -> List[str]:
        normalized: List[str] = []
        for part in path_parts or []:
            text = str(part if part is not None else "").strip()
            if text:
                normalized.append(text)
        return normalized

    def make_entry(index_name: Any, index_path: List[Any], index_value: Any) -> Dict[str, Any]:
        path = normalize_path(index_path)
        drillable = isinstance(index_value, dict)
        child_feature_path = list(path) if drillable else []
        drilldown_candidates: List[Dict[str, Any]] = []
        if drillable:
            drilldown_candidates.append({
                "bodyName": "sub feature tree",
                "isDrilldown": True,
                "nextPath": list(child_feature_path),
            })
        return {
            "path": path,
            "indexName": str(index_name if index_name is not None else ""),
            "drillable": bool(drillable),
            "childFeaturePath": child_feature_path,
            "drilldownBodyCandidates": drilldown_candidates,
        }

    def put_entry(semantic_id: str, entry: Dict[str, Any]) -> None:
        if not semantic_id or not isinstance(entry, dict):
            return
        existed = projection_map.get(semantic_id)
        if not existed:
            projection_map[semantic_id] = entry
            return
        old_path = normalize_path(existed.get("path", []))
        new_path = normalize_path(entry.get("path", []))
        if old_path == new_path:
            return
        if len(new_path) < len(old_path):
            projection_map[semantic_id] = entry

    def walk_dict(feature_dict: Dict[str, Any], typed_feature_parts: List[str], path_prefix: List[str]) -> None:
        for idx, (index_key, index_value) in enumerate(feature_dict.items()):
            index_name = str(index_key if index_key is not None else "")
            index_path = list(path_prefix) + [index_name]
            alias = build_typed_index_id(typed_feature_parts, idx, index_name)
            semantic_id = alias_to_semantic.get(alias, "")
            if semantic_id:
                put_entry(semantic_id, make_entry(index_name, index_path, index_value))

            if isinstance(index_value, dict):
                child_typed_parts = list(typed_feature_parts) + [
                    build_typed_index_segment(idx, index_name),
                    build_typed_body_segment(0, index_name, None, is_subtree=True),
                ]
                walk_dict(index_value, child_typed_parts, index_path)

    # Nested viewer wraps column root into {"root": raw_column}
    walk_dict(raw_column, typed_root_parts, ["root"])
    return projection_map

