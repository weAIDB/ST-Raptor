"""
对 FeatureTree 每列打特征标签，即反映着一列的数据情况
"""

from collections import defaultdict
import random
from datetime import timedelta, date, time

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

from table2tree.feature_tree import *
from utils.constants import *
from utils.data_type_utils import *
from utils.api_utils import embedding_generate


class ClassificationEmbeddingModel:

    _instance = None  # 类变量，用于存储唯一实例

    def __init__(self, model_path=EMBEDDING_MODE_PATH):
        if EMBEDDING_TYPE == 'local':
            self.model_path = model_path
            self.model = SentenceTransformer(model_path)
        else:
            self.model = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance
    def get_entity_embedding(self, entity_list):
        entity_list = ["#" if str(x).strip() == '' else x for x in entity_list]
        if EMBEDDING_TYPE == 'local':
            embeddings = self.model.encode(entity_list)
        else:
            embeddings = embedding_generate(input_texts=entity_list)
        return embeddings

def generate_cluster_labels(texts, cluster_ids):
    """为每个文本聚类生成可解释的标签

    Args:
        texts: 原始文本列表
        cluster_ids: 每个文本对应的簇ID列表

    Returns:
        dict: {cluster_id: 标签字符串}
    """
    # 组织文本到各个簇
    cluster_to_texts = defaultdict(list)
    for text, cid in zip(texts, cluster_ids):
        cluster_to_texts[cid].append(text)

    # 为每个簇生成标签
    label_dict = {}
    example_dict = {}
    for cid, docs in cluster_to_texts.items():
        # 方法1：TF-IDF提取关键词
        try:
            tfidf = TfidfVectorizer(max_features=20)
            tfidf_matrix = tfidf.fit_transform(docs)
            feature_array = np.array(tfidf.get_feature_names_out())
            tfidf_sorting = np.argsort(tfidf_matrix.sum(axis=0).A1)[::-1]
            top_keywords = feature_array[tfidf_sorting][:3]
        except Exception as e:
            top_keywords = docs[0]

        # 方法2：使用嵌入向量找中心文档
        try:
            embeddings = ClassificationEmbeddingModel().get_entity_embedding(docs)
            centroid = embeddings.mean(axis=0)
            sim_scores = embeddings.dot(centroid)
            most_representative = docs[np.argmax(sim_scores)]
        except Exception as e:
            most_representative = docs[0]

        # 组合最终标签
        label_dict[str(cid)] = list(top_keywords)
        example_dict[str(cid)] = most_representative[:30]

    return label_dict, example_dict


def split_into_intervals(data, num_intervals):
    """
    将数值列表划分为多个区间，并确保区间边界为整数。

    :param data: 数值型列表
    :param num_intervals: 区间数量
    :return: 区间列表，每个区间表示为元组 (start, end)
    """
    if len(data) <= 0:
        return []

    # 找到数据的最小值和最大值
    min_value = min(data)
    max_value = max(data)

    # 计算区间大小
    interval_size = (max_value - min_value) / num_intervals

    # 生成区间
    intervals = []
    for i in range(num_intervals):
        start = int(min_value + i * interval_size)
        end = int(min_value + (i + 1) * interval_size)
        intervals.append((start, end))

    # 确保最后一个区间的结束值包含最大值
    intervals[-1] = (intervals[-1][0], max_value)

    return intervals


def split_datetime_intervals(datetime_list, n):
    """
    将 datetime 列表划分为 n 个区间，每个区间是一个 (start_time, end_time) 元组。

    参数:
        datetime_list (list): 包含 datetime 对象的列表。
        n (int): 区间数量。

    返回:
        list: 区间列表，每个区间是一个 (start_time, end_time) 元组。
    """
    # 检查列表是否为空
    if len(datetime_list) == 0:
        return []

    # 找到最小时间和最大时间
    min_time = min(datetime_list)
    max_time = max(datetime_list)

    # 计算总时间差
    total_delta = max_time - min_time

    # 计算每个区间的时间差
    interval_delta = total_delta / n

    # 生成区间列表
    intervals = []
    for i in range(n):
        start_time = min_time + i * interval_delta
        end_time = start_time + interval_delta
        intervals.append((start_time, end_time))

    return intervals


def tag_one_list(value_list, classify_threshold=3, n_bins=3):
    """对数据列表进行分类/分组，返回组信息和每个值的组索引

    Args:
        value_list: 输入数据列表，支持数值类型或文本字符串

    Returns:
        {
            "group_name_list": 组信息列表，
            "group_id_list": 组id列表,
            "id2name": 组id和组名称的映射,
            "name2id": 组名称和组id的映射
            "mapping": 原列表各值对应的组id,
            "group_type": 离散型/连续xing/文本型,
            "example_dict": 分组的样例数据
        }
    """
    if not value_list:
        return [], [], {}, {}, [], 0, {}

    # 判断数据类型
    judge_numeric = all(is_numeric(x) for x in value_list)
    judge_date = all(is_date(str(x)) for x in value_list)
    if judge_numeric:
        value_list = [float(x) for x in value_list]
    elif judge_date:
        value_list = [str_to_date(x) for x in value_list]
    else:
        for i in range(len(value_list)):
            value_list[i] = str(value_list[i])
    group_name_list = []
    mapping = []

    value_array = np.array(value_list)
    unique_values = np.unique(value_array)
    judge_discrete = len(unique_values) <= classify_threshold

    # 1. 离散型
    if judge_discrete:
        group_name_list = [str(x) for x in unique_values.tolist()]
        group_id_list = [x for x in range(len(group_name_list))]
        id2name = {str(id): str(name) for id, name in enumerate(group_name_list)}
        name2id = {str(name): str(id) for id, name in enumerate(group_name_list)}
        mapping = [name2id[str(name)] for name in value_list]
        group_type = TAG_DISCRETE
        example_dict = {str(id): str(name) for id, name in enumerate(group_name_list)}
        # print("当前是离散型！")
    # 2. 连续型
    else:
        
        if judge_numeric or judge_date:  # 数值型 / 日期型
            # print("当前是数值连续型！")
            min_val, max_val = value_array.min(), value_array.max()
            delta = max_val - min_val
            if isinstance(delta, timedelta):
                delta = delta.total_seconds()
            if delta < 1e-9:  # 所有值相同
                group_name_list = [str((min_val, max_val))]
                group_id_list = ["0"]
                id2name = {"0": str((min_val, max_val))}
                name2id = {str((min_val, max_val)): "0"}
                mapping = [0 for _ in range(len(value_list))]
                example_dict = {"0": str(random.choice(value_array))}
            else:
                if judge_numeric:
                    intervals = [
                        (int(x[0]), int(x[1]))
                        for x in split_into_intervals(value_array, n_bins)
                    ]
                    bins = [x[0] for x in intervals]
                else:
                    intervals = [
                        (x[0], x[1])
                        for x in split_datetime_intervals(value_array, n_bins)
                    ]
                    t = time(0, 0, 0)
                    bins = [
                        int(datetime.combine(x[0], t).timestamp()) for x in intervals
                    ]
                    value_array = [
                        int(datetime.combine(x, t).timestamp()) for x in value_array
                    ]
                group_name_list = [str(x) for x in intervals]
                group_id_list = [x for x in range(len(group_name_list))]
                id2name = {
                    str(id): str(name) for id, name in enumerate(group_name_list)
                }
                name2id = {
                    str(name): str(id) for id, name in enumerate(group_name_list)
                }
                # 确定每个值所属的区间
                indices = np.digitize(value_array, bins, right=False)
                mapping = (indices - 1).tolist()  # 转换为0-based索引
                example_dict = {}
                for id in range(len(group_name_list)):
                    target_value_list = []
                    for index, idd in enumerate(mapping):
                        # print(id, idd)
                        if str(id) == str(idd):
                            target_value_list.append(str(value_array[index]))
                    if len(target_value_list) == 0:
                        example_dict[str(id)] = ""
                    else:
                        example_dict[str(id)] = random.choice(target_value_list)
            group_type = TAG_CONTINUOUS

        else:  # 文本型
            # print("当前是文本连续型！")
            embeddings = ClassificationEmbeddingModel().get_entity_embedding(
                value_list
            )  # 生成文本嵌入

            # 动态确定聚类数目（示例设为最多5类）
            n_clusters = n_bins
            kmeans = KMeans(n_clusters=n_clusters)
            cluster_ids = kmeans.fit_predict(embeddings)

            # 获取聚类中心并排序（使组序更稳定）
            # centers = kmeans.cluster_centers_
            # sorted_centers = sorted(enumerate(centers), key=lambda x: x[1][0])
            # group_order = {str(orig_idx): str(new_idx) for new_idx, (orig_idx, _) in enumerate(sorted_centers)}

            # 新增标签生成
            label_dict, example_dict = generate_cluster_labels(value_list, cluster_ids)
            n_clusters = len(np.unique(cluster_ids))
            # print(label_dict, cluster_ids, n_clusters)

            group_name_list = [label_dict[str(cid)] for cid in range(n_clusters)]
            group_id_list = [str(x) for x in range(len(group_name_list))]
            id2name = {str(id): str(name) for id, name in enumerate(group_name_list)}
            name2id = {str(name): str(id) for id, name in enumerate(group_name_list)}
            mapping = [str(cid) for cid in cluster_ids]
            group_type = TAG_TEXT

    mapping = [str(x) for x in mapping]
    return (
        group_name_list,
        group_id_list,
        id2name,
        name2id,
        mapping,
        group_type,
        example_dict,
    )

