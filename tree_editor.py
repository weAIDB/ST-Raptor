import copy
from html import escape

# ---------------- 树数据管理（内存） ----------------

# 默认空数据，等待上传或手动操作
default_data = []


class TreeEditor:
    def __init__(self):
        self.data = copy.deepcopy(default_data)

    def reset(self):
        self.data = copy.deepcopy(default_data)

    def render_html(self):
        if not self.data:
            return "<div style='height:400px;display:flex;align-items:center;justify-content:center;color:#888;'>暂无数据，请上传或手动添加</div>"

        counter = {"i": 0}

        def node_html(node):
            counter["i"] += 1
            node_id = f"n{counter['i']}"
            full_text = escape(str(node.get("name", "")))
            display_text = full_text[:10] + ("..." if len(full_text) > 10 else "")
            children = node.get("children", [])
            if children:
                children_html = "".join(node_html(ch) for ch in children)
                return (
                    "<li>"
                    f"<input type='checkbox' id='{node_id}' class='toggle'>"
                    f"<label class='box' for='{node_id}' data-full='{full_text}'>{display_text}</label>"
                    f"<ul class='children'>{children_html}</ul>"
                    "</li>"
                )
            return (
                "<li>"
                f"<div class='box leaf' data-full='{full_text}'>{display_text}</div>"
                "</li>"
            )

        body = "".join(node_html(n) for n in self.data)
        return (
            "<style>"
            ".tree-wrapper{width:100%;overflow-x:auto;overflow-y:hidden;padding:10px;text-align:center;}"
            ".tree{display:inline-block;}"
            ".tree ul{padding-top:20px;position:relative;padding-left:0;white-space:nowrap;}"
            ".tree li{display:inline-block;text-align:center;list-style-type:none;position:relative;padding:20px 5px 0 5px;white-space:nowrap;}"
            ".tree li::before,.tree li::after{content:'';position:absolute;top:0;right:50%;border-top:2px solid #c4c8d8;width:50%;height:20px;}"
            ".tree li::after{right:auto;left:50%;border-left:2px solid #c4c8d8;}"
            ".tree li:only-child::after,.tree li:only-child::before{display:none;}"
            ".tree li:only-child{padding-top:0;}"
            ".tree li:first-child::before,.tree li:last-child::after{border:0 none;}"
            ".tree li:last-child::before{border-right:2px solid #c4c8d8;border-radius:0 5px 0 0;}"
            ".tree li:first-child::after{border-radius:5px 0 0 0;}"
            ".tree ul ul::before{content:'';position:absolute;top:0;left:50%;border-left:2px solid #c4c8d8;width:0;height:20px;}"
            ".tree .box{cursor:pointer;display:inline-block;padding:10px 14px;border:1px solid #d7d9e0;background:#6a7de3;color:#fff;border-radius:6px;min-width:120px;max-width:220px;font-size:12px;box-shadow:0 2px 8px rgba(0,0,0,0.12);white-space:normal;word-break:break-word;overflow-wrap:break-word;}"
            ".tree .leaf{background:#8f9df0;}"
            ".tree input.toggle{display:none;}"
            ".tree input.toggle ~ ul.children{display:none;}"
            ".tree input.toggle:checked ~ ul.children{display:block;}"
            "</style>"
            "<div class='tree-wrapper'>"
            "<div class='tree'>"
            "<ul>" + body + "</ul>"
            "</div>"
            "</div>"
        )

    def get_node_list(self):
        names = []
        def traverse(nodes):
            for node in nodes:
                names.append(node["name"])
                if "children" in node:
                    traverse(node["children"])
        traverse(self.data)
        return names

    def rename_node(self, old_name, new_name):
        def traverse(nodes):
            for node in nodes:
                if node["name"] == old_name:
                    node["name"] = new_name
                    return True
                if "children" in node and traverse(node["children"]):
                    return True
            return False
        traverse(self.data)

    def add_child(self, parent_name, new_child_name):
        if not new_child_name:
            return
        def traverse(nodes):
            for node in nodes:
                if node["name"] == parent_name:
                    node.setdefault("children", []).append({"name": new_child_name})
                    return True
                if "children" in node and traverse(node["children"]):
                    return True
            return False
        traverse(self.data)

    def delete_node(self, node_name):
        # 根节点保护
        if self.data and self.data[0]["name"] == node_name:
            return False

        def traverse(nodes):
            for i, node in enumerate(nodes):
                if node["name"] == node_name:
                    del nodes[i]
                    return True
                if "children" in node and traverse(node["children"]):
                    return True
            return False
        if self.data and "children" in self.data[0]:
            traverse(self.data[0]["children"])


# 全局编辑器实例
editor = TreeEditor()
