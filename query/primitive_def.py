import copy
import re
import types

from table2tree.feature_tree import *


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
        if i_node.value == column:
            index_node: IndexNode = i_node
            break
    
    if index_node == None:
        return f_tree
    b_node_list = []
    if op == '==':
        for b_node in index_node.body:
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
    if s2 is None:
        for item in tree_dict:
            if s1 in item or s1 in item.values():
                ans += str(item) + ","
    else:
        for item in tree_dict:
            if s1 in item or s2 in item or s1 in item.values() or s2 in item.values():
                ans += str(item) + ","
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
        if not item.value:
            continue
        if s in item.value:  # 在标签中
            tmp = f_tree.index_tree.leaf_nodes[pos].body
            for item in tmp:
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
        if len(item.body) == 0:
            continue
        for i in item.body:
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
    if len(args) == 1:  # case 1 Exist
        if search_case_one(f_tree, args[0]):
            ans.append("Exists.")
            return ans
        else:
            ans.append("Does not exist.")
            return ans
    elif len(args) == 2 and args[1] == "n":  # case 2 CHL
        return search_case_two(f_tree, args[0])
    elif len(args) == 2 and args[0] == "n":  # case 3 FAT
        return ans
    elif len(args) == 3 and args[1] == "n":  # case 4 EXT
        return search_case_four(f_tree, args[0], args[2])
    elif len(args) == 3 and args[0] == "cond":  # condition
        return condition(f_tree, args[1], args[2])
    elif len(args) == 4 and args[0] == "cmp":  # compare
        return compare(f_tree, args[1], args[2], args[3])
    else:
        return ans
