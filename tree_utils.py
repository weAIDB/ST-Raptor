def json_to_tree_data(obj, name=None):
    """将任意 JSON 转换为 pyecharts Tree 数据结构。
    入口处不强制 root，调用方应传入顶层 key 作为 name。
    """
    # 构建当前节点
    node = {"name": "root" if name is None else str(name)}

    # dict -> children 为其项
    if isinstance(obj, dict):
        children = []
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                children.append(json_to_tree_data(v, name=k))
            else:
                # 保留键为父节点，值作为子节点，避免丢失 "key: value"
                children.append({"name": str(k), "children": [{"name": str(v)}]})
        if children:
            node["children"] = children
        return node

    # list -> children 为索引项
    if isinstance(obj, list):
        children = []
        for i, v in enumerate(obj):
            if isinstance(v, (dict, list)):
                children.append(json_to_tree_data(v, name=f"[{i}]"))
            else:
                children.append({"name": f"[{i}]", "children": [{"name": str(v)}]})
        if children:
            node["children"] = children
        return node

    # 原子值
    return {"name": f"{node['name']}: {obj}"}
