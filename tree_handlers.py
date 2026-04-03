import os
import json
from gradio import update
from tree_editor import editor, default_data
import copy


def persist_tree():
    """将当前树数据写回前端专用树快照，不污染后端 canonical JSON。"""
    os.makedirs("cache", exist_ok=True)
    try:
        with open("cache/temp.ui.tree.json", "w", encoding="utf-8") as f:
            json.dump(editor.data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] 写入 JSON 失败: {e}")


def refresh_ui():
    choices = editor.get_node_list()
    return (
        editor.render_html(),                              # graph
        update(choices=choices, value=None),               # rename_sel
        update(choices=choices, value=None),               # add_parent_sel
        update(choices=choices, value=None),               # del_sel
        "操作成功，视图已更新",
        update(value=""),                                  # rename_inp clear
        update(value=""),                                  # add_name_inp clear
    )


def handle_rename(target, new_name):
    if not target or not new_name:
        return (
            editor.render_html(),
            update(choices=editor.get_node_list(), value=None),
            update(choices=editor.get_node_list(), value=None),
            update(choices=editor.get_node_list(), value=None),
            "请输入节点和新名称",
            update(value=""),
            update(value=""),
        )
    editor.rename_node(target, new_name)
    persist_tree()
    return refresh_ui()


def handle_add(parent, child_name):
    if not parent or not child_name:
        return (
            editor.render_html(),
            update(choices=editor.get_node_list(), value=None),
            update(choices=editor.get_node_list(), value=None),
            update(choices=editor.get_node_list(), value=None),
            "请选择父节点并输入新名称",
            update(value=""),
            update(value=""),
        )
    editor.add_child(parent, child_name)
    persist_tree()
    return refresh_ui()


def handle_delete(target):
    if not target:
        return (
            editor.render_html(),
            update(choices=editor.get_node_list(), value=None),
            update(choices=editor.get_node_list(), value=None),
            update(choices=editor.get_node_list(), value=None),
            "请选择要删除的节点",
            update(value=""),
            update(value=""),
        )
    editor.delete_node(target)
    persist_tree()
    return refresh_ui()
