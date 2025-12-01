import json

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from torch import Tensor
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, util

from utils.constants import *
from utils.api_utils import embedding_generate
from table2tree.feature_tree import *


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """平均掩码池化，原本一个样本每个token一个vector，对他们做平均，变成一个样本一个vector"""
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery: {query}"

def find_topk_indices(lst, k):
    import heapq

    topk_with_indices = heapq.nlargest(k, enumerate(lst), key=lambda x: x[1])
    indices = [index for index, value in topk_with_indices]
    return indices

class EmbeddingModel:
    """因为需要指定task，因此无法预处理Embedding"""

    _instance = None  # 类变量，用于存储唯一实例

    def __init__(self):
        if EMBEDDING_TYPE == 'local':
            self.model_path = EMBEDDING_MODE_PATH
            self.model = SentenceTransformer(EMBEDDING_MODE_PATH)
        else:
            self.similarity = util.cos_sim

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    # TODO
    def get_entity_embedding(self, entity_list):
        entity_list = ["#" if str(x).strip() == '' else x for x in entity_list]
        if EMBEDDING_TYPE == 'local':
            embeddings = self.model.encode(entity_list)
        else:
            embeddings = embedding_generate(input_texts=entity_list)
        return embeddings

    def get_embedding_dict(self, entity_list):
        embeddings = self.get_entity_embedding(entity_list)
        embedding_dict = {
            str(entity): embedding.tolist()
            for entity, embedding in zip(entity_list, embeddings)
        }
        return embedding_dict

    def save_embedding_dict(self, embedding_dict, output_file):  
        # 显式指定utf-8编码，确保在Windows系统上能正确处理非ASCII字符
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(embedding_dict, f, ensure_ascii=False)    

    def load_embedding_dict(self, input_file):
        # 从 JSON 文件加载
        # 显式指定utf-8编码，确保在Windows系统上能正确处理非ASCII字符
        with open(input_file, "r", encoding="utf-8") as f:
            loaded_embedding_dict = json.load(f)

        # 将列表转换回 NumPy 数组
        loaded_embedding_dict = {
            k: np.array(v) for k, v in loaded_embedding_dict.items()
        }

        return loaded_embedding_dict

    def split_embedding_dict(self, embedding_dict):
        values = []
        embeddings = []
        for (
            k,
            v,
        ) in embedding_dict.items():
            values.append(k)
            embeddings.append(v.tolist())
        return values, embeddings
    
    def one_to_many_semilarity(self, value, value_list=None, embedding_cache_file=None):
        """One of value_list or embedding_cache_file must be specified"""
        if embedding_cache_file is None:  # without cache
            input_texts = [value] + value_list
            input_texts = [str(s) for s in input_texts]

            embeddings = self.get_entity_embedding(input_texts)

            scores = (embeddings[:1] @ embeddings[1:].T) * 100
            scores = scores.tolist()
        else:
            embedding_dict = self.load_embedding_dict(embedding_cache_file)
            value_list, embedding_list = self.split_embedding_dict(embedding_dict)
            value_embedding = self.get_entity_embedding([value]).astype(np.float64)

            if EMBEDDING_TYPE == 'local':
                scores = self.model.similarity(value_embedding, embedding_list)
            else:
                scores = self.similarity(value_embedding, embedding_list)
            scores = scores.tolist()

        return scores

    def topk_match(
        self,
        entities: list,
        table: list = None,
        k=10,
        embedding_cache_file=None,
    ):
        """One of table or embedding_cache_file must be specified"""
        if embedding_cache_file is None:  # without cache
            if table is None or len(table) == 0:
                return [[x] for x in entities]
            input_texts = entities + table
            input_texts = [str(s) for s in input_texts]

            embeddings = self.get_entity_embedding(input_texts)

            scores = (embeddings[: len(entities)] @ embeddings[len(entities) :].T) * 100
            scores = scores.tolist()
            if not isinstance(scores, list): scores = [[scores]]

            # Find Max Top-k Values
            values = []
            for index, score_lst in enumerate(scores):
                indices = find_topk_indices(score_lst, k)
                values.append([table[i] for i in indices])
        else:
            embedding_dict = self.load_embedding_dict(embedding_cache_file)
            value_list, embedding_list = self.split_embedding_dict(embedding_dict)
            value_embedding = self.get_entity_embedding(entities)

            # Find Max Top-k Values
            if EMBEDDING_TYPE == 'local':
                scores = self.model.similarity(value_embedding.tolist(), embedding_list)
            else:
                scores = self.similarity(value_embedding.tolist(), embedding_list)
            scores = scores.tolist()
            if not isinstance(scores, list): scores = [[scores]]
            
            values = []
            for index, score_lst in enumerate(scores):
                indices = find_topk_indices(score_lst, k)
                values.append([value_list[i] for i in indices])

        return values

    def top1_match(self, entities: list, table: list = None, embedding_cache_file=None):
        """One of table or file must be specified"""
        return flatten_nested_list(
            self.topk_match(entities=entities, table=table, k=1, embedding_cache_file=embedding_cache_file)
        )

    # def one_to_many_semilarity(
    #     self,
    #     value,
    #     value_list,
    #     task="Given an sentence, retrieve relevant sentences that relevant to the sentence.",
    # ):
    #     value = get_detailed_instruct(task, value)
    #     input_texts = [value] + value_list
    #     input_texts = [str(s) for s in input_texts]

    #     # Tokenize the input texts
    #     batch_dict = self.tokenizer(
    #         input_texts,
    #         max_length=512,
    #         padding=True,
    #         truncation=True,
    #         return_tensors="pt",
    #     )

    #     outputs = self.model(**batch_dict)
    #     embeddings = average_pool(
    #         outputs.last_hidden_state, batch_dict["attention_mask"]
    #     )

    #     # normalize embeddings
    #     embeddings = F.normalize(embeddings, p=2, dim=1)
    #     scores = (embeddings[:1] @ embeddings[1:].T) * 100
    #     scores = scores.tolist()

    #     return scores

    # def topk_match(
    #     self,
    #     entities: list,
    #     table: list,
    #     k=10,
    #     task="Given an entity, retrieve relevant values that relevant to the entity.",
    #     log_file=None,
    # ):
    #     entities = [get_detailed_instruct(task, query) for query in entities]
    #     input_texts = entities + table
    #     input_texts = [str(s) for s in input_texts]

    #     # Tokenize the input texts
    #     batch_dict = self.tokenizer(
    #         input_texts,
    #         max_length=512,
    #         padding=True,
    #         truncation=True,
    #         return_tensors="pt",
    #     )

    #     outputs = self.model(**batch_dict)
    #     embeddings = average_pool(
    #         outputs.last_hidden_state, batch_dict["attention_mask"]
    #     )

    #     # normalize embeddings
    #     embeddings = F.normalize(embeddings, p=2, dim=1)
    #     scores = (embeddings[: len(entities)] @ embeddings[len(entities) :].T) * 100
    #     scores = scores.tolist()

    #     # Find Max Top-k Values
    #     values = []
    #     for index, score_lst in enumerate(scores):
    #         indices = find_topk_indices(score_lst, k)
    #         values.append([table[i] for i in indices])

    #     if log_file is not None:  # Log
    #         with open(log_file, "a") as file:
    #             file.write(f"{DELIMITER} Top-{k} Match Result {DELIMITER}\n")
    #             for i, (entity, values) in enumerate(zip(entities, values)):
    #                 file.write(f"Entity: {entity}\n")
    #                 file.write(f"Values: {values}\n")

    #     return values

    # def top1_match(
    #     self,
    #     entities: list,
    #     table: list,
    #     task="Given an string, retrieve most relevant value that relevant to the entity.",
    # ):
    #     return flatten_nested_list(
    #         self.topk_match(entities=entities, table=table, k=1, task=task)
    #     )


def calculate_topk_similarity(query_vectors, target_vectors, topk=6):
    """
    计算两个 Embedding 向量列表之间的相似度，并返回 Top-K 最相关的数据。

    参数:
        query_vectors (np.ndarray): 待匹配的向量列表，形状为 (n, embedding_dim)。
        target_vectors (np.ndarray): 被匹配的向量列表，形状为 (m, embedding_dim)。
        topk (int): 返回的 Top-K 最相关结果。

    返回:
        topk_indices (list): Top-K 最相关的索引列表，形状为 (n, topk)。
        topk_scores (list): Top-K 最相关的相似度分数列表，形状为 (n, topk)。
    """
    # 计算余弦相似度矩阵
    similarity_matrix = cosine_similarity(query_vectors, target_vectors)  # 形状 (n, m)

    # 获取 Top-K 的索引和分数
    topk_indices = np.argsort(similarity_matrix, axis=1)[:, -topk:][
        :, ::-1
    ]  # 形状 (n, topk)
    topk_scores = np.take_along_axis(
        similarity_matrix, topk_indices, axis=1
    )  # 形状 (n, topk)

    return topk_indices.tolist(), topk_scores.tolist()


# class EmbeddingModelAllMiniLML6V2:

#     _instance = None  # 类变量，用于存储唯一实例

#     def __init__(self, model_path=ALLMINILM_MODEL_PATH):
#         self.model_path = model_path
#         self.model = SentenceTransformer(model_path)

#     def __new__(cls, *args, **kwargs):
#         if not cls._instance:
#             cls._instance = super().__new__(cls)
#         return cls._instance

#     def get_entity_embedding(self, entity_list):
#         embeddings = self.model.encode(entity_list)
#         return embeddings

#     def get_embedding_dict(self, entity_list):
#         embeddings = self.get_entity_embedding(entity_list)
#         embedding_dict = {
#             str(entity): embedding.tolist()
#             for entity, embedding in zip(entity_list, embeddings)
#         }
#         return embedding_dict

#     def save_embedding_dict(self, embedding_dict, output_file):
#         with open(output_file, "w") as f:
#             json.dump(embedding_dict, f, ensure_ascii=False)

#     def load_embedding_dict(self, input_file):
#         # 从 JSON 文件加载
#         with open(input_file, "r") as f:
#             loaded_embedding_dict = json.load(f)

#         # 将列表转换回 NumPy 数组
#         loaded_embedding_dict = {
#             k: np.array(v) for k, v in loaded_embedding_dict.items()
#         }

#         return loaded_embedding_dict

#     def split_embedding_dict(self, embedding_dict):
#         values = []
#         embeddings = []
#         for (
#             k,
#             v,
#         ) in embedding_dict.items():
#             values.append(k)
#             embeddings.append(v.tolist())
#         return values, embeddings

#     def one_to_many_semilarity(self, value, value_list=None, embedding_cache_file=None):
#         """One of value_list or embedding_cache_file must be specified"""
#         if embedding_cache_file is None:  # without cache
#             input_texts = [value] + value_list
#             input_texts = [str(s) for s in input_texts]

#             embeddings = self.get_entity_embedding(input_texts)

#             scores = (embeddings[:1] @ embeddings[1:].T) * 100
#             scores = scores.tolist()
#         else:
#             embedding_dict = self.load_embedding_dict(embedding_cache_file)
#             value_list, embedding_list = self.split_embedding_dict(embedding_dict)
#             value_embedding = self.get_entity_embedding([value]).astype(np.float64)

#             scores = self.model.similarity(
#                 value_embedding, embedding_dict
#             )
#             scores = scores.tolist()

#         return scores

#     def topk_match(
#         self,
#         entities: list,
#         table: list = None,
#         k=10,
#         threshold=None,
#         embedding_cache_file=None,
#         log_file=None,
#     ):
#         """One of table or embedding_cache_file must be specified"""
#         if embedding_cache_file is None:  # without cache
#             input_texts = entities + table
#             input_texts = [str(s) for s in input_texts]

#             embeddings = self.get_entity_embedding(input_texts)

#             scores = (embeddings[: len(entities)] @ embeddings[len(entities) :].T) * 100
#             scores = scores.tolist()
#             if not isinstance(scores, list): scores = [[scores]]

#             # Find Max Top-k Values
#             values = []
#             for index, score_lst in enumerate(scores):
#                 indices = find_topk_indices(score_lst, k)
#                 values.append([table[i] for i in indices])
#         else:
#             embedding_dict = self.load_embedding_dict(embedding_cache_file)
#             value_list, embedding_list = self.split_embedding_dict(embedding_dict)
#             value_embedding = self.get_entity_embedding(entities)

#             # Find Max Top-k Values
#             scores = self.model.similarity(
#                 value_embedding.tolist(), embedding_list
#             )
#             scores = scores.tolist()
#             if not isinstance(scores, list): scores = [[scores]]
            
#             values = []
#             for index, score_lst in enumerate(scores):
#                 indices = find_topk_indices(score_lst, k)
#                 values.append([value_list[i] for i in indices])

#         if log_file is not None:  # Log
#             with open(log_file, "a") as file:
#                 file.write(f"{DELIMITER} Top-{k} Match Result {DELIMITER}\n")
#                 for i, (entity, values) in enumerate(zip(entities, values)):
#                     file.write(f"Entity: {entity}\n")
#                     file.write(f"Values: {values}\n")

#         return values

#     def top1_match(self, entities: list, table: list = None, embedding_cache_file=None, threshold=None):
#         """One of table or file must be specified"""
#         return flatten_nested_list(
#             self.topk_match(entities=entities, table=table, k=1, embedding_cache_file=embedding_cache_file, threshold=threshold)
#         )


def match_sub_table(entities: list, f_tree):  #: FeatureTree):
    """使用 Embedding Vector 从 FeatureTree 中提取子部分，返回JSON"""
    model = EmbeddingModel()

    values = model.topk_match(entities, f_tree.body_value_list())

    return values


def flatten_nested_list(value):
    res = []
    for x in value:
        if isinstance(x, list):
            res.extend(flatten_nested_list(x))
        else:
            res.append(x)
    return res


def get_sub_json(values: list, json_dict: dict):
    values = list(set(flatten_nested_list(values)))

    def dfs(values: list, j_dict: dict):
        return_dict = {}
        for key, value in j_dict.items():
            if isinstance(value, list):
                tmp_list = []
                for x in value:
                    if isinstance(x, dict):
                        x = dfs(values, x)
                        if len(x) > 0:
                            tmp_list.append(x)
                    else:
                        if x in values:
                            tmp_list.append(x)
                if len(tmp_list) > 0:
                    return_dict[key] = tmp_list
            elif isinstance(value, dict):
                value = dfs(values, value)
                if len(value) > 0:
                    return_dict[key] = value
            else:
                if value in values or key in values:
                    return_dict[key] = value
        return return_dict

    return dfs(values, json_dict)


def demo():
    # Each query must come with a one-sentence instruction that describes the task
    task = "Given a web search query, retrieve relevant passages that answer the query"
    queries = [
        get_detailed_instruct(task, "how much protein should a female eat"),
        get_detailed_instruct(task, "南瓜的家常做法"),
    ]
    # No need to add instruction for retrieval documents
    documents = [
        "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
        "1.清炒南瓜丝 原料:嫩南瓜半个 调料:葱、盐、白糖、鸡精 做法: 1、南瓜用刀薄薄的削去表面一层皮,用勺子刮去瓤 2、擦成细丝(没有擦菜板就用刀慢慢切成细丝) 3、锅烧热放油,入葱花煸出香味 4、入南瓜丝快速翻炒一分钟左右,放盐、一点白糖和鸡精调味出锅 2.香葱炒南瓜 原料:南瓜1只 调料:香葱、蒜末、橄榄油、盐 做法: 1、将南瓜去皮,切成片 2、油锅8成热后,将蒜末放入爆香 3、爆香后,将南瓜片放入,翻炒 4、在翻炒的同时,可以不时地往锅里加水,但不要太多 5、放入盐,炒匀 6、南瓜差不多软和绵了之后,就可以关火 7、撒入香葱,即可出锅",
    ]
    input_texts = queries + documents

    tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large-instruct")
    model = AutoModel.from_pretrained("intfloat/multilingual-e5-large-instruct")

    # Tokenize the input texts
    batch_dict = tokenizer(
        input_texts, max_length=512, padding=True, truncation=True, return_tensors="pt"
    )

    outputs = model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])

    # normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    scores = (embeddings[:2] @ embeddings[2:].T) * 100
    print(scores.tolist())
    # => [[91.92852783203125, 67.580322265625], [70.3814468383789, 92.1330795288086]]


def main():
    model = EmbeddingModel()

    res = model.one_to_many_semilarity(
        "财政供养人数是多少？",
        [
            "湛江市人力资源和社会保障局的下属二级单位数量是多少？",
            "这个部门有多少个下属二级单位？",
            "湛江市人力资源和社会保障局的下属机构有多少个？",
            "该部门下属的二级单位总数是多少？",
        ],
    )

    print(res)


def main2():
    model = EmbeddingModel()

    res = model.topk_match(
        entities=["预算整体情况", "城乡居民养老"],
        embedding_cache_file="/Users/tangzirui/Desktop/SJTU-DB/TaQA/dataset_json/sstqa/table/1.embedding.json",
        k=3,
    )
    print(res)
    with open(
        "/Users/tangzirui/Desktop/SJTU-DB/TaQA/dataset_json/sstqa/table/1_embedding.json",
        "r",
    ) as f:
        data: dict = json.load(f)
    res = model.topk_match(
        entities=["预算整体情况", "城乡居民养老"], table=list(data.keys()), k=3
    )
    print(res)

def main3():
    model = EmbeddingModel()
    res = model.top1_match(["1", "2"], ["1", "2", "3", "4"])
    print(res)

if __name__ == "__main__":
    # main()
    # main2()
    main3()
