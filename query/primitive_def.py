import copy
import re
import types
import time
from typing import Any, Dict, List, Optional

from table2tree.feature_tree import *

_EXEC_TRACE_STATE: Dict[str, Any] = {
    "events": [],
    "ctx": {},
    "step": 0,
    "frame_seq": 0,
    "operation_seq": 0,
}
_EXEC_NODE_ID_LOOKUP: Dict[int, str] = {}
_EXEC_NODE_TRACE_LOOKUP: Dict[int, Dict[str, str]] = {}


def reset_execution_trace():
    _EXEC_TRACE_STATE["events"] = []
    _EXEC_TRACE_STATE["ctx"] = {}
    _EXEC_TRACE_STATE["step"] = 0
    _EXEC_TRACE_STATE["frame_seq"] = 0
    _EXEC_TRACE_STATE["operation_seq"] = 0
    _EXEC_NODE_ID_LOOKUP.clear()
    _EXEC_NODE_TRACE_LOOKUP.clear()


def set_execution_trace_context(**kwargs):
    ctx = _EXEC_TRACE_STATE.get("ctx", {})
    for key, value in kwargs.items():
        if value is None:
            ctx.pop(key, None)
        else:
            ctx[key] = value
    _EXEC_TRACE_STATE["ctx"] = ctx


def get_execution_trace_events():
    return list(_EXEC_TRACE_STATE.get("events", []))


def get_execution_trace_context() -> Dict[str, Any]:
    return dict(_EXEC_TRACE_STATE.get("ctx", {}))


def replace_execution_trace_context(ctx: Optional[Dict[str, Any]] = None):
    _EXEC_TRACE_STATE["ctx"] = dict(ctx or {})


def next_execution_frame_id(prefix: str = "frame") -> str:
    _EXEC_TRACE_STATE["frame_seq"] = int(_EXEC_TRACE_STATE.get("frame_seq", 0)) + 1
    return f"{prefix}_{_EXEC_TRACE_STATE['frame_seq']}"


def next_execution_operation_id(prefix: str = "op") -> str:
    _EXEC_TRACE_STATE["operation_seq"] = int(_EXEC_TRACE_STATE.get("operation_seq", 0)) + 1
    return f"{prefix}_{_EXEC_TRACE_STATE['operation_seq']}"


def set_execution_node_id_lookup(lookup: Optional[Dict[int, str]] = None):
    _EXEC_NODE_ID_LOOKUP.clear()
    if lookup:
        _EXEC_NODE_ID_LOOKUP.update(lookup)


def set_execution_node_trace_lookup(lookup: Optional[Dict[int, Dict[str, str]]] = None):
    _EXEC_NODE_TRACE_LOOKUP.clear()
    if lookup:
        for key, value in (lookup or {}).items():
            try:
                obj_id = int(key)
            except Exception:
                continue
            if not isinstance(value, dict):
                continue
            canonical_id = str(value.get("canonical_id", "") or value.get("canonical_trace_id", "") or "").strip()
            target_kind = str(value.get("target_kind", "") or "").strip()
            if not canonical_id:
                continue
            if not target_kind:
                target_kind = "group" if canonical_id.startswith("ct_tree_group_") or canonical_id.startswith("ct_semantic_group_") else "node"
            _EXEC_NODE_TRACE_LOOKUP[obj_id] = {
                "canonical_id": canonical_id,
                "target_kind": target_kind,
            }


def _safe_value_text(value: Any) -> str:
    try:
        text = str(value)
    except Exception:
        text = repr(value)
    return text[:160]


def record_execution_event(
    event_type: str,
    node_type: str = "",
    node_value: Any = None,
    extra: Optional[Dict[str, Any]] = None,
    node_ref: Any = None,
):
    _EXEC_TRACE_STATE["step"] = int(_EXEC_TRACE_STATE.get("step", 0)) + 1
    entry = {
        "step": _EXEC_TRACE_STATE["step"],
        "ts": time.time(),
        "event_type": event_type,
        "node_type": node_type or "",
        "node_value": _safe_value_text(node_value) if node_value is not None else "",
        "context": dict(_EXEC_TRACE_STATE.get("ctx", {})),
    }
    if extra:
        entry["extra"] = dict(extra)
    canonical_id = ""
    target_kind = ""
    if isinstance(extra, dict):
        canonical_id = str(extra.get("canonical_id", "") or extra.get("canonical_trace_id", "") or "").strip()
        target_kind = str(extra.get("target_kind", "") or "").strip()
    if node_ref is not None and (not canonical_id or not target_kind):
        trace_meta = _EXEC_NODE_TRACE_LOOKUP.get(id(node_ref), {})
        if not canonical_id:
            canonical_id = str(trace_meta.get("canonical_id", "") or "").strip()
        if not target_kind:
            target_kind = str(trace_meta.get("target_kind", "") or "").strip()
    if canonical_id:
        entry["canonical_id"] = canonical_id
        entry["canonical_trace_id"] = canonical_id
    if canonical_id and not target_kind:
        target_kind = "group" if canonical_id.startswith("ct_tree_group_") or canonical_id.startswith("ct_semantic_group_") else "node"
    if target_kind:
        entry["target_kind"] = target_kind
    frontend_node_id = None
    if isinstance(extra, dict):
        frontend_node_id = extra.get("frontend_node_id")
    if not frontend_node_id and node_ref is not None:
        frontend_node_id = _EXEC_NODE_ID_LOOKUP.get(id(node_ref))
    if frontend_node_id:
        entry["frontend_node_id"] = str(frontend_node_id)
    _EXEC_TRACE_STATE.setdefault("events", []).append(entry)


def create_function(code_str, func_name):
    module_code = compile(code_str, "<string>", "exec")
    module = types.ModuleType("dynamic_module")
    exec(module_code, module.__dict__)
    return getattr(module, func_name)


def get_func_name(code_str):
    pattern = r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\("
    return re.findall(pattern, code_str)


def get_subtree_column_bnode(f_tree: FeatureTree, b_node_list):
    # 以删除数据的方式获得子数据表
    def body_tree_dfs(root: BodyNode, path: list, b_node_list: list, depth):
        if len(root.children) == 0 or depth == len(f_tree.index_tree.leaf_nodes):
            # 现在的path是一条完整路径
            flag = False    # 路径不保留
            for b_node in b_node_list:
                fflag = False
                for t_node in path:
                    if t_node.value == b_node.value:
                        fflag = True
                if fflag:
                    flag = True
                    break 

            if not flag:    # 删除该路径
                # Print Path
                # for b_node in path:
                #     print(b_node.value, end=', ')
                # print()
                
                for index, b_node in enumerate(path):
                    # 删 father 的孩子
                    try:
                        if len(b_node.father) > 0: b_node.father.remove(b_node)
                    except Exception as e:
                        # import traceback; traceback.print_exc()
                        pass
                    # 删 index_node.body
                    try:
                        if len(f_tree.index_tree.leaf_nodes[index].body) > 0: f_tree.index_tree.leaf_nodes[index].body.remove(b_node)
                    except Exception as e:
                        # import traceback; traceback.print_exc()
                        pass
                    # 删当前的孩子
                    b_node.children = []
            return
        
        # print children
        # for b_node in root.children:
        #     print(b_node.value, end=', ')
        # print()
                        
        for b_node in root.children:
            path.append(b_node)
            body_tree_dfs(b_node, path[:], b_node_list, depth + 1)
            path = path[:-1]
        
    body_tree_dfs(f_tree.body_tree.root, [], b_node_list, 0)
    
    return f_tree

def condition_search(f_tree: FeatureTree, column, op, value):

    # 找到要筛选的列
    index_node = None
    for index, i_node in enumerate(f_tree.index_tree.leaf_nodes):
        record_execution_event("visit", "index_node", i_node.value, {"phase": "condition_search", "index": index}, node_ref=i_node)
        if i_node.value == column:
            index_node: IndexNode = i_node
            break
    
    if index_node == None:
        return f_tree
    b_node_list = []
    if op == '==':
        for b_node in index_node.body:
            record_execution_event("visit", "body_node", b_node.value, {"phase": "condition_match", "op": op, "column": column}, node_ref=b_node)
            if str(b_node.value) == str(value):
                b_node_list.append(b_node)
    elif op == '!=':
        for b_node in index_node.body:
            if str(b_node.value) != str(value):
                b_node_list.append(b_node)
    elif op == '>=':
        for b_node in index_node.body:
            if str(b_node.value) >= str(value):
                b_node_list.append(b_node)
    elif op == '<=':
        for b_node in index_node.body:
            if str(b_node.value) <= str(value):
                b_node_list.append(b_node)
    elif op == '>':
        for b_node in index_node.body:
            if str(b_node.value) > str(value):
                b_node_list.append(b_node)
    elif op == '<':
        for b_node in index_node.body:
            if str(b_node.value) < str(value):
                b_node_list.append(b_node)
    elif op == 'in':
        for b_node in index_node.body:
            if str(value) in str(b_node.value):
                b_node_list.append(b_node)
    if b_node_list != []:
        return get_subtree_column_bnode(copy.deepcopy(f_tree), b_node_list)

    return f_tree


def search_values(f_tree: FeatureTree, s1: str, s2=None):
    tree_dict = f_tree.__json__()
    print(tree_dict)
    if DEFAULT_TABLE_NAME in tree_dict:
        tree_dict = tree_dict[DEFAULT_TABLE_NAME]
    
    ans = ""
    
    def recursive_search(data):
        nonlocal ans
        if isinstance(data, dict):
            # 搜索键和值
            for k, v in data.items():
                if s1 in str(k) or (s2 and s2 in str(k)):
                    ans += str(k) + ":" + str(v) + ","
                recursive_search(v)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    # 处理列表中的字典（典型的表格行）
                    found = False
                    if s1 in str(item) or (s2 and s2 in str(item)):
                        found = True
                    else:
                        for val in item.values():
                            if s1 in str(val) or (s2 and s2 in str(val)):
                                found = True
                                break
                    if found:
                        ans += str(item) + ","
                else:
                    recursive_search(item)
        elif isinstance(data, str):
            if s1 in data or (s2 and s2 in data):
                ans += data + ","

    recursive_search(tree_dict)
    return [ans[:-1]]


def search_case_one(f_tree: FeatureTree, s: str):
    for item in f_tree.index_tree.leaf_nodes:
        if s in item.value:  # 在标签中
            return True
    for item in f_tree.body_tree.root.children:
        if isinstance(item.value, str) and s in item.value:  # 在body中
            return True
        elif isinstance(item.value, FeatureTree):
            if search_case_one(item.value, s):
                return True
    return False


def search_case_two(f_tree: FeatureTree, s: str):
    # 如果直接后继是一颗子树，则返回子树，如果是字符串，则返回字符串
    ans = []
    pos = 0
    if len(f_tree.index_tree.leaf_nodes) == 0:
        return ans
    for item in f_tree.index_tree.leaf_nodes:
        record_execution_event("visit", "index_node", item.value, {"phase": "search_case_two.index"}, node_ref=item)
        if not item.value:
            continue
        if s in item.value:  # 在标签中
            tmp = f_tree.index_tree.leaf_nodes[pos].body
            for item in tmp:
                record_execution_event("visit", "body_node", item.value, {"phase": "search_case_two.body_from_index"}, node_ref=item)
                if isinstance(item.value, str):
                    ans.append([item.value])
                if isinstance(item.value, FeatureTree):
                    # ans_tmp = item.value.index_tree.leaf_nodes
                    # ans_tmp = [x.value for x in ans_tmp]
                    # ans.append(ans_tmp)
                    ans.append(item.value)
        pos += 1

    body = f_tree.body_tree.root.children[:]
    while len(body) > 0:
        record_execution_event("visit", "body_node", body[0].value, {"phase": "search_case_two.body_walk"}, node_ref=body[0])
        if isinstance(body[0].value, FeatureTree):
            ans.extend(search_case_two(body[0].value, s))
        elif isinstance(body[0].value, str):
            if s in body[0].value:
                ans.extend(search_values(f_tree, s))
        body.extend(body[0].children)
        body = list(set(body[1:]))

    string_flag = True
    if len(ans) == 0:
        string_flag = False
    for i in ans:
        if isinstance(i, list):
            if len(i) == 0:
                string_flag = False
            for j in i:
                if not isinstance(j, str):
                    string_flag = False
        else:
            if not isinstance(i, str):
                string_flag = False

    if string_flag:
        return [[f_tree]]
    return ans


def search_case_three(f_tree: FeatureTree, s: str):
    ans = []
    index = f_tree.index_tree.leaf_nodes
    for item in index:
        record_execution_event("visit", "index_node", item.value, {"phase": "search_case_three.index"}, node_ref=item)
        if len(item.body) == 0:
            continue
        for i in item.body:
            record_execution_event("visit", "body_node", i.value, {"phase": "search_case_three.body"}, node_ref=i)
            if isinstance(i.value, str) and s in i.value:
                ans.append([item.value])
            elif isinstance(i.value, FeatureTree):
                index_tmp = i.value.index_tree.leaf_nodes
                for j in index_tmp:
                    if not j.value:
                        continue
                    if s in j.value:
                        ans.append([item.value])
                ans.extend(search_case_three(i.value, s))

    return ans


def search_case_four_iv(f_tree: FeatureTree, s1: str, s2: str):
    ans = []
    pos = 0
    for item in f_tree.index_tree.leaf_nodes:
        record_execution_event("visit", "index_node", item.value, {"phase": "search_case_four_iv.index"}, node_ref=item)
        if not item.value:
            continue
        if s1 == item.value:  # 在标签中
            for t in f_tree.index_tree.leaf_nodes:
                if t.value == s2:
                    return [[f_tree]]  # 解决同一层
            if len(item.body) == 1:
                ans.append(item.body[0].value)
                return [ans]
            tmp = f_tree.body_tree.root.children[:]
            flag = 0
            for sub in tmp:
                if isinstance(sub.value, str) and s2 == sub.value:
                    flag = 1
                    break
            for i in range(pos):
                if flag == 1:
                    new_tmp = []
                    for item in tmp:
                        if item.value == s2:
                            new_tmp.append(item)
                    tmp = [x for sub in new_tmp for x in sub.children]
                    flag = 2
                elif flag == 0:
                    tmp = [x for sub in tmp for x in sub.children]
                    for sub in tmp:
                        if isinstance(sub.value, str) and s2 == sub.value:
                            flag = 1
                            break
                elif flag == 2:
                    tmp = [x for sub in tmp for x in sub.children]
            if flag == 0:
                continue
            tmp = [x.value for x in tmp]
            ans.append(tmp)
        pos += 1

    body = f_tree.body_tree.root.children[:]
    while len(body) > 0:
        if isinstance(body[0].value, FeatureTree):
            ans.extend(search_case_four(body[0].value, s1, s2))
        body.extend(body[0].children)
        body = list(set(body[1:]))

    string_flag = True
    for i in ans:
        if isinstance(i, list):
            for j in i:
                if not isinstance(j, str):
                    string_flag = False
        else:
            if not isinstance(i, str):
                string_flag = False

    if string_flag:
        return search_values(f_tree, s2)
    return ans


def search_case_four(f_tree: FeatureTree, s1: str, s2: str):
    hir = [-1, -1]  # calc hierarchy, compare and output.
    flag = [False, False]
    trees = [f_tree, f_tree]

    for item in f_tree.index_tree.leaf_nodes:
        record_execution_event("visit", "index_node", item.value, {"phase": "search_case_four.index"}, node_ref=item)
        if not item.value:
            continue
        if s1 == item.value:
            hir[0] = 0
            flag[0] = True
        if s2 == item.value:
            hir[1] = 0
            flag[1] = True

    body = f_tree.index_tree.leaf_nodes[:]
    body = [x for sub in body for x in sub.body]
    body_hir = [1] * len(body)
    while flag != [True, True] and len(body) > 0:
        record_execution_event("visit", "body_node", body[0].value, {"phase": "search_case_four.body_walk"}, node_ref=body[0])
        if isinstance(body[0].value, FeatureTree):
            for item in body[0].value.index_tree.leaf_nodes:
                if not item.value:
                    continue
                if s1 == item.value:
                    if flag[0] == False:
                        hir[0] = body_hir[0]
                        flag[0] = True
                        trees[0] = body[0].value
                if s2 == item.value:
                    if flag[1] == False:
                        hir[1] = body_hir[0]
                        flag[1] = True
                        trees[1] = body[0].value
            tmp = body[0].value.index_tree.leaf_nodes[:]
            tmp = [x for sub in tmp for x in sub.body]
            body.extend(tmp)
            body_hir.extend([body_hir[0] + 1] * len(tmp))
        body = body[1:]
        body_hir = body_hir[1:]

    if hir[0] == -1 and hir[1] == -1:
        return search_values(f_tree, s1, s2)
    if hir[0] == -1:
        return search_case_four_iv(trees[1], s2, s1)
    if hir[1] == -1:
        return search_case_four_iv(trees[0], s1, s2)
    if hir[0] < hir[1]:
        return search_case_two(f_tree, s1)
    else:
        return search_case_two(f_tree, s2)


def condition(f_tree: FeatureTree, content: str, func_str: str):
    func_name = get_func_name(func_str)
    func = create_function(func_str, func_name)
    elems = search_case_two(f_tree, content)
    res = []
    for elem in elems:
        try:
            if func(elem):
                res.append(elem)
        except Exception as e:
            import traceback
            traceback.print_exc()
            continue
    return res


def compare(f_tree: FeatureTree, content1: str, content2: str, func_str: str):
    func_name = get_func_name(func_str)
    func = create_function(func_str, func_name)

    try:
        res = func(content1, content2)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None
    return res


def foreach(f_tree: FeatureTree, content: str, func_str):
    func_name = get_func_name(func_str)
    func = create_function(func_str, func_name)
    elems = search_case_two(f_tree, content)
    res = []
    for elem in elems:
        try:
            res.append(func(elem))
        except Exception as e:
            import traceback
            traceback.print_exc()
            continue
    return res


def meta_search(f_tree: FeatureTree, args: list):
    """
    在FeatureTree结构上进行查找, 共4种模式, 分别对应'元语句设计.md'提到的4种基本状态
    1. args==[content] 直接查找存在性
    2. args==[content, ->]查找content内容的直接后继
    3. args==[->, content]查找content内容的直接前驱
    4. args==[content1, ->, content2]用二维结构查找
    """
    ans = []
    record_execution_event("meta_search_start", "primitive", "meta_search", {"args": list(args)})
    if len(args) == 1:  # case 1 Exist
        if search_case_one(f_tree, args[0]):
            ans.append("Exists.")
            record_execution_event("meta_search_end", "primitive", "meta_search", {"result_count": len(ans)})
            return ans
        else:
            ans.append("Does not exist.")
            record_execution_event("meta_search_end", "primitive", "meta_search", {"result_count": len(ans)})
            return ans
    elif len(args) == 2 and args[1] == "n":  # case 2 CHL
        ans = search_case_two(f_tree, args[0])
        record_execution_event("meta_search_end", "primitive", "meta_search", {"result_count": len(ans)})
        return ans
    elif len(args) == 2 and args[0] == "n":  # case 3 FAT
        record_execution_event("meta_search_end", "primitive", "meta_search", {"result_count": len(ans)})
        return ans
    elif len(args) == 3 and args[1] == "n":  # case 4 EXT
        ans = search_case_four(f_tree, args[0], args[2])
        record_execution_event("meta_search_end", "primitive", "meta_search", {"result_count": len(ans)})
        return ans
    elif len(args) == 3 and args[0] == "cond":  # condition
        ans = condition(f_tree, args[1], args[2])
        record_execution_event("meta_search_end", "primitive", "meta_search", {"result_count": len(ans) if isinstance(ans, list) else 1})
        return ans
    elif len(args) == 4 and args[0] == "cmp":  # compare
        ans = compare(f_tree, args[1], args[2], args[3])
        record_execution_event("meta_search_end", "primitive", "meta_search", {"result_count": 1 if ans is not None else 0})
        return ans
    else:
        record_execution_event("meta_search_end", "primitive", "meta_search", {"result_count": len(ans)})
        return ans
