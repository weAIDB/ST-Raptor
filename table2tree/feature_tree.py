import time
import pickle
from loguru import logger

from utils.sheet_utils import delete_dict_none_none
from utils.constants import *
from table2tree.extract_excel import process_table_vlm, get_xlsx_sheet, get_structured_xlsx_sheet
from table2tree.tree_partition import *

def serial(level_list):
    s = ""
    for i in level_list:
        s += str(i) + "."
    return s[:-1]


class TreeNode:

    def __init__(self, value=None):
        self.value = value  # string or subtree
        self.children = []

    def add_child(self, child_node):
        self.children.append(child_node)

    def remove_child(self, child_node):
        if child_node in self.children:
            self.children.remove(child_node)


class IndexNode(TreeNode):

    def __init__(self, value=None):
        super().__init__(value)
        self.body = []
        self.father = None

        # 以下的属性只有 leaf node 有，并且需要额外的处理，build_split_info() 函数
        self.group_type = (
            None  # 用于为一列的数据分类，是这个分类的类型 离散、连续、embedding三种
        )
        self.group_name_list = None  # 所有的group_name分类类型
        self.group_id_list = None  # 所有的group_id集合
        self.name2id = None  # dict, name2id的映射
        self.id2name = None  # dict, id2name的映射

    def add_body_node(self, node):
        self.body.append(node)


class BodyNode(TreeNode):

    def __init__(self, value=None):
        super().__init__(value)
        self.father = []
        self.x1 = None
        self.y1 = None
        self.x2 = None
        self.y2 = None

        self.group_name = None  # 用于为一列的数据分类，是当前节点的分类标签
        self.group_id = None  # 用于为一列的数据分类，是当前节点的分类标签id

        self.group_class = 0  # 用于在一个FeatureTree里横向编号，解决搜索问题

    def get_pos(self):
        return [self.x1, self.y1, self.x2, self.y2]

    def add_father(self, node):
        self.father.append(node)


def get_leaf_nodes(root: TreeNode):
    """返回树种所有叶子结点的列表"""
    if not root:
        return []

    if len(root.children) <= 0:
        return [root]

    leaves = []
    for child in root.children:
        leaves.extend(get_leaf_nodes(child))

    return leaves


class IndexTree:

    def __init__(self):
        self.root = IndexNode()
        self.leaf_nodes = []

    def add_index(self, node: IndexNode):
        node.father = self.root
        self.root.add_child(node)
        self.leaf_nodes.extend(get_leaf_nodes(node))

    def add_leaf_body_node_by_pos(self, body_node, pos):
        try:
            # 检查索引是否越界
            if pos < 0 or pos >= len(self.leaf_nodes):
                return  # 索引越界，直接返回
            leaf : IndexNode = self.leaf_nodes[pos]
            leaf.add_body_node(body_node)
        except Exception as e:
            print(body_node.value)
            import traceback; traceback.print_exc()

    def value_list(self):
        res_list = []

        node = self.root.children[:]
        while len(node) > 0:
            curr : IndexNode = node[0]
            res_list.append(curr.value)
            node.extend(curr.children)
            node = node[1:]

        return res_list

    def get_flatten_schema(self):
        schema_list = []

        def dfs(i_node: IndexNode, path: list):
            if not i_node:
                return
            if i_node != self.root:
                path.append(str(i_node.value))

            if len(i_node.children) <= 0:
                schema = "-".join(path)
                if schema in schema_list:
                    index = 1
                    while f"{schema}{index}" in schema_list:
                        index += 1
                    schema_list.append(f"{schema}{index}")
                else:
                    schema_list.append(schema)
            else:
                for node in i_node.children:
                    dfs(node, path[:])

        dfs(self.root, [])
        return schema_list


class BodyTree:

    def __init__(self):
        self.root = BodyNode()

    def add_deep(self, node: BodyNode):
        t = self.root
        while len(t.children) > 0:
            t = t.children[0]
        t.add_child(node)
        node.add_father(t)

    def value_list(self):
        res_list = []

        node = self.root.children[:]
        while len(node) > 0:
            curr : BodyNode = node[0]
            if curr.value not in res_list:
                res_list.append(curr.value)
            node.extend(curr.children)
            node = node[1:]

        return res_list


class FeatureTree:

    def __init__(self, index_tree: IndexTree = None, body_tree: BodyTree = None):
        self.index_tree = index_tree
        self.body_tree = body_tree

    def load_from_pkl(self, pkl_file):
        try:
            with open(pkl_file, "rb") as f:
                f_tree: FeatureTree = pickle.load(f)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(e)
            return None
        return f_tree

    def all_value_list(self):
        return self.index_value_list() + self.body_value_list()

    def get_list_value(self, pos):
        ncols = len(self.index_tree.leaf_nodes)
        if pos < 0 or pos >= ncols:
            return []

        value_list = []
        for b_node in self.index_tree.leaf_nodes[pos].body:
            value_list.append(b_node.value)
        # value_list = [x for x in value_list if x is not None and len(str(x)) > 0]
        return value_list

    def index_value_list(self):
        res_list = []

        flag = False
        for index_node in self.index_tree.leaf_nodes:
            if (
                len(index_node.body) >= 1
                and type(index_node.body[0].value) == FeatureTree
            ):
                flag = True
                break

        if flag:
            for index_node in self.index_tree.leaf_nodes:
                for body_node in index_node.body:
                    if type(body_node.value) == FeatureTree:
                        res = body_node.value.index_value_list()
                        res_list.extend(res)
        res_list = [x for x in res_list if x is not None and len(str(x)) > 0]
        res_list.extend(self.index_tree.value_list())
        return res_list

    def body_value_list(self):
        res_list = []

        flag = False
        for index_node in self.index_tree.leaf_nodes:
            if (
                len(index_node.body) >= 1
                and type(index_node.body[0].value) == FeatureTree
            ):
                flag = True
                break
        if flag:
            for index_node in self.index_tree.leaf_nodes:
                if len(index_node.body) >= 1:
                    if type(index_node.body[0].value) == FeatureTree:
                        res = index_node.body[0].value.body_value_list()
                        res_list.extend(res)
                    else:
                        res_list.append(index_node.body[0].value)
        else:
            res_list.extend(self.body_tree.value_list())
        res_list = [x for x in res_list if x is not None and len(str(x)) > 0]
        return res_list

    def get_max_row(self):
        return max([len(leaf.body) for leaf in self.index_tree.leaf_nodes])

    def get_max_col(self):
        return len(self.index_tree.leaf_nodes)

    # TODO
    def get_structured_table(self):
        """return sturctured table in List format"""
        schema = self.index_tree.get_flatten_schema()

        data = []

        return [schema] + data

    # TODO
    def get_subtree_row(self, index_value: list):
        """根据 行号 获取子树"""
        return []

    # TODO
    def get_subtree_col(self, index_value: list):
        """根据 列号 获取子树"""
        return []

    def __index__(self):
        # 仅获得key的json，值都为None
        json_dict = {}
        # 判断当前 Feature 是 kv 形式的还是 list 形式
        flag = False
        for index_node in self.index_tree.leaf_nodes:
            if (
                len(index_node.body) >= 1
                and type(index_node.body[0].value) == FeatureTree
            ):
                flag = True
                break
        if flag:
            for index_node in self.index_tree.leaf_nodes:
                if len(index_node.body) >= 1:
                    if type(index_node.body[0].value) == FeatureTree:
                        res = index_node.body[0].value.__index__()
                        if DEFAULT_TABLE_NAME in res:
                            json_dict[index_node.value] = res[DEFAULT_TABLE_NAME]
                        else:
                            json_dict[index_node.value] = res
                    else:
                        json_dict[index_node.value] = "String"
                    if (
                        type(json_dict[index_node.value]) == list
                        and len(json_dict[index_node.value]) == 1
                    ):
                        json_dict[index_node.value] = json_dict[index_node.value][0]
        else:
            schema = self.index_tree.get_flatten_schema()
            json_dict[DEFAULT_TABLE_NAME] = schema
        return json_dict

    def __json__(self):
        json_dict = {}

        # 判断当前 Feature 是 kv 形式的还是 list 形式
        flag = False
        for index_node in self.index_tree.leaf_nodes:
            if (
                len(index_node.body) >= 1
                and type(index_node.body[0].value) == FeatureTree
            ):
                flag = True
                break
        if flag:  # kv形式
            for index_node in self.index_tree.leaf_nodes:
                if len(index_node.body) >= 1:
                    if type(index_node.body[0].value) == FeatureTree:
                        res = index_node.body[0].value.__json__()
                        if DEFAULT_TABLE_NAME in res:
                            json_dict[index_node.value] = res[DEFAULT_TABLE_NAME]
                        else:
                            json_dict[index_node.value] = res
                    else:
                        json_dict[index_node.value] = index_node.body[0].value
                    if (
                        type(json_dict[index_node.value]) == list
                        and len(json_dict[index_node.value]) == 1
                    ):
                        json_dict[index_node.value] = json_dict[index_node.value][0]
        else:  # list形式
            res = []
            schema_list = self.index_tree.get_flatten_schema()

            def dfs(b_node, path, x1, x2):
                if not b_node:
                    return
                if b_node != self.body_tree.root:
                    if x1 is not None and b_node.x1 is not None:
                        x1 = max(x1, b_node.x1)
                    if x2 is not None and b_node.x2 is not None:
                        x2 = min(x2, b_node.x2)
                    path.append(b_node)

                if len(b_node.children) <= 0:
                    if len(schema_list) == len(path):  # one row
                        tmp_dict = {}
                        for k, v in zip(schema_list, path):
                            tmp_dict[k] = str(v.value)
                        res.append(tmp_dict)
                    return
                else:
                    for node in b_node.children:
                        if (
                            x1 is not None
                            or x2 is not None
                            or node.x1 is not None
                            or node.x2 is not None
                            or (node.x1 <= x2 and node.x2 >= x1)
                        ):
                            dfs(node, path[:], x1, x2)

            max_x2 = 0
            for b_node in self.body_tree.root.children:
                if b_node.x2 is not None:
                    max_x2 = max(max_x2, b_node.x2)

            dfs(self.body_tree.root, [], 1, max_x2)
            json_dict[DEFAULT_TABLE_NAME] = res
            
            json_dict = delete_dict_none_none(json_dict)
        return json_dict

    def __str__(self, level_list=[1]):
        s = ""
        for index in self.index_tree.leaf_nodes:
            s = s + serial(level_list) + " " + str(index.value) + ": "
            for body_node in index.body:
                if isinstance(body_node.value, FeatureTree):
                    s = s + "\n" + body_node.value.__str__(level_list + [1])
                else:
                    s = s + str(body_node.value) + ", "
            s += "\n"
            level_list[-1] += 1
        return s

def construct_index_tree(schema_sheet):
    from utils.sheet_utils import get_merge_cell_size, single_cell, get_sub_sheet

    nrows = schema_sheet.max_row
    ncols = schema_sheet.max_column

    index_tree = IndexTree()

    col = 1
    while col <= ncols:
        cell = schema_sheet.cell(row=1, column=col)
        x1, y1, x2, y2 = get_merge_cell_size(schema_sheet, cell.coordinate)

        if not single_cell(schema_sheet, x1, y1, nrows, y2):
            sub_schema_sheet = get_sub_sheet(schema_sheet, x2 + 1, y1, nrows, y2)
            sub_index_tree = construct_index_tree(sub_schema_sheet)
            index_node = sub_index_tree.root
            index_node.value = schema_sheet.cell(row=x1, column=y1).value
        else:
            index_node = IndexNode(schema_sheet.cell(row=x1, column=y1).value)
        index_tree.add_index(index_node)

        col += y2 - y1 + 1

    return index_tree


def construct_body_tree_dfs(index_tree: IndexTree, data_sheet, x, y, depth):
    from utils.sheet_utils import (
        get_merge_cell_size,
        get_merge_cell_value,
        get_sub_sheet,
    )

    cell = data_sheet.cell(row=x, column=y)
    x1, y1, x2, y2 = get_merge_cell_size(data_sheet, cell.coordinate)

    value = get_merge_cell_value(data_sheet, cell.coordinate)
    root = BodyNode(value)
    root.x1 = x1
    root.x2 = x2
    root.y1 = y1
    root.y2 = y2
    index_tree.add_leaf_body_node_by_pos(root, depth)

    # 判断是否到数据树的底部了
    if y2 >= data_sheet.max_column:
        return root

    row = x1
    while row <= x2:
        cell = data_sheet.cell(row=row, column=y2 + 1)
        xx1, yy1, xx2, yy2 = get_merge_cell_size(data_sheet, cell.coordinate)

        # 判断合并单元格的粒度变化
        if xx1 < x1:  # 后者包含前者，则后者已经被构建过了，不需要重复构建
            if (
                depth + 1 < len(index_tree.leaf_nodes)
                and len(index_tree.leaf_nodes[depth + 1].body) > 0
            ):
                child = index_tree.leaf_nodes[depth + 1].body[-1]
                if child:
                    root.add_child(child)
                    child.add_father(root)
        else:
            body_node = construct_body_tree_dfs(
                index_tree, data_sheet, xx1, yy1, depth + 1
            )
            body_node.add_father(root)
            root.add_child(body_node)

        row += xx2 - xx1 + 1

    return root


def construct_body_tree(index_tree: IndexTree, data_sheet):
    """使用IndexTree获取BodyNode的Sibling构建BodyTree"""
    from utils.sheet_utils import (
        get_merge_cell_size,
        get_merge_cell_value,
        get_sub_sheet,
    )

    # 检查data_sheet是否为None
    if data_sheet is None:
        return None, None

    nrows = data_sheet.max_row
    ncols = data_sheet.max_column

    body_tree = BodyTree()
    root = body_tree.root

    row = 1
    while row <= nrows:
        cell = data_sheet.cell(row=row, column=1)
        x1, y1, x2, y2 = get_merge_cell_size(data_sheet, cell.coordinate)

        body_node = construct_body_tree_dfs(index_tree, data_sheet, x1, y1, 0)
        body_node.add_father(root)
        root.add_child(body_node)

        row += x2 - x1 + 1

    return body_tree, index_tree


def construct_sheet(sheet):
    from utils.split_utils import split_schema_row

    schame_sheet, data_sheet = split_schema_row(sheet)

    # TODO construct tree index [Need Test]
    index_tree = construct_index_tree(schame_sheet)

    # TODO construct tree body, link index tree and body tree [Need Test]
    body_tree, _ = construct_body_tree(index_tree, data_sheet)

    return FeatureTree(index_tree=index_tree, body_tree=body_tree)


def construct_feature_tree(tree_dict):

    logger.info(f"construct_feature_tree() Start to Process!")
    start_time = time.time()

    try:
        index_tree = IndexTree()
        body_tree = BodyTree()

        for index, body in tree_dict.items():
            if isinstance(body, dict):  # dict -> FeatureTree
                body_node = BodyNode(construct_feature_tree(body))
            elif isinstance(body, (str, int, float, list)) or body is None:  # string
                body_node = BodyNode(body)
            else:  # sheet -> FeatureTree
                body_node = BodyNode(construct_sheet(body))

            # link body tree and index tree
            index_node = IndexNode(value=index)
            index_node.body = [body_node]

            # construct index tree
            index_tree.add_index(index_node)
            # construct body tree
            body_tree.add_deep(body_node)
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(e, tree_dict)

    f_tree = FeatureTree(index_tree=index_tree, body_tree=body_tree)

    end_time = time.time()
    logger.info(f"construct_feature_tree() Process File Successfully!")
    logger.info(f"Cost time: {end_time - start_time} ")

    return f_tree


def number_feature_tree(f_tree: FeatureTree):
    """Number the bodynodes in the feature tree for further searching"""
    for i in range(1, len(f_tree.body_tree.children)):
        f_tree.body_tree.root.children[i].group_class = i
    tmp = f_tree.body_tree.root.children[:]
    while len(tmp) > 0:
        if isinstance(tmp[0].value, FeatureTree):
            number_feature_tree(tmp[0])
        c = tmp[0].group_class
        if len(tmp[0].children) == 0:
            continue
        for item in tmp[0].children:
            item.group_class = c
            if item not in tmp:
                tmp.append(item)
        tmp = tmp[1:]
    return f_tree


def build_split_info(f_tree: FeatureTree):
    """为每一列打标签, 存在IndexNode里"""
    index_tree = f_tree.index_tree

    for index, leaf_node in enumerate(index_tree.leaf_nodes):
        value_list = f_tree.get_list_value(index)
        (
            group_name_list,
            group_id_list,
            id2name,
            name2id,
            mapping,
            group_type,
            example_dict,
        ) = tag_one_list(value_list)

        # print(value_list)
        # print(group_name_list)
        # print(group_id_list)
        # print(id2name)
        # print(name2id)
        # print(mapping)
        # print(group_type)
        # print(example_dict)

        leaf_node.group_name_list = group_name_list
        leaf_node.group_id_list = group_id_list
        leaf_node.id2name = id2name
        leaf_node.name2id = name2id
        leaf_node.group_type = group_type
        leaf_node.example_dict = example_dict

        for i, b_node in enumerate(leaf_node.body):
            b_node.group_id = mapping[i]
            b_node.group_name = id2name[mapping[i]]
    return f_tree


def tag_feature_tree(f_tree: FeatureTree):

    flag = True
    for leaf_node in f_tree.index_tree.leaf_nodes:
        for b_node in leaf_node.body:
            if isinstance(b_node.value, FeatureTree):
                b_node.value = tag_feature_tree(b_node.value)
                flag = False
        if len(leaf_node.body) <= 1:
            flag = False

    if flag:
        f_tree = build_split_info(f_tree)  # 为每一列打标签
        # f_tree = number_feature_tree(f_tree)    # 建立 group_class，用于检索数据 没有用处，替换为直接检索

    return f_tree


def get_excel_feature_tree(file: str,                   # 输入表格文件路径
                           structured: bool = False,    # 输入的表格是否是结构化表格
                           log_dir: str = LOG_DIR,      # LOG 日志记录路径
                           vlm_cache: bool = False      # 是否保存转图片的中间结果
                           ):
    ##### 创建 log_file 文件
    log_file = os.path.join(log_dir, f"{os.path.basename(file)}.log")
    log_file_handler = logger.add(log_file)

    ##### Step 1. VLM + Rule 提取表格为类JSON格式
    logger.info(f"process_table_vlm() Start to Process File: {file}")
    start_time = time.time()

    try:    # 防止表格出现结构不符合规定的情况
        if structured:  # 是结构化表格，直接构建即可
            tree_dict = {DEFAULT_TABLE_NAME: get_xlsx_sheet(file)}
        else:  # 如果是半结构化表格，则使用vlm识别构建json
            tree_dict = process_table_vlm(file, get_json=False, cache=vlm_cache)

        end_time = time.time()
        logger.info(f"process_table_vlm() Process File Successfully: {file} ")
        logger.info(tree_dict)
        logger.info(f"Cost time: {end_time - start_time}")

        ##### Step 2. 将提取出的结构构件为FeatureTree
        logger.info(f"construct_feature_tree() Start to Process File: {file}")
        start_time = time.time()

        ho_tree = construct_feature_tree(tree_dict)
        
        end_time = time.time()
        logger.info(f"construct_feature_tree() Process File Successfully: {file} ")
        logger.info(f"Cost time: {end_time - start_time}")

        ##### Step 3. 为数据的每一列打上标签
        logger.info(f"tag_feature_tree() Start to Process File: {file}")
        start_time = time.time()
        
        ho_tree = tag_feature_tree(ho_tree)
        
    except Exception as e:  # 表格不符合规范，使用结构化建树
        import traceback; traceback.print_exc()
        logger.error(f"Fail to convert {file}! Try to construct through structured method.")
        
        tree_dict = {DEFAULT_TABLE_NAME: get_structured_xlsx_sheet(file)}
        ho_tree = construct_feature_tree(tree_dict)
        ho_tree = tag_feature_tree(ho_tree)

    end_time = time.time()
    logger.info(f"tag_feature_tree() Process File Successfully: {file} ")
    logger.info(f"Cost time: {end_time - start_time}")

    ##### 输出字符串和 JSON 格式
    logger.info(f"HO-Tree JSON:")
    logger.info(f"{ho_tree.__json__()}")
    logger.info(f"HO-Tree String:")
    logger.info(f"{ho_tree.__str__([1])}")

    logger.remove(log_file_handler)

    return ho_tree
