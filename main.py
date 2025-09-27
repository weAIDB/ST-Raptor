"""
The Query Answering Pipeline
"""

import os
import re
import json
import glob
import pickle

from tqdm import tqdm
from loguru import logger

from query.primitive_pipeline import *
from table2tree.extract_excel import *
from table2tree.feature_tree import *
from utils.constants import DELIMITER


def answer_question(
    qa_pair: dict,                          # 一条问答对
    table_file: str,                        # 表格原文件路径
    pkl_dir: str,                           # 存储 HO-Tree 中间结果的路径
    enable_query_decompose: bool = True,    # 是否启用 Query Decomposition 机制
    enable_emebdding: bool = True,          # 是否启用 Embedding 机制
    log_dir: str = LOG_DIR                  # Log 日志目录
):
    
    qid = qa_pair["id"]
    tid = qa_pair['table_id']
    query = qa_pair["query"]

    ##### 创建日志文件 命名为 表格id_问题id.log
    log_file = os.path.join(log_dir, f'{tid}_{qid}.log')
    log_file_handler = logger.add(log_file)

    logger.info(f"{DELIMITER} 开始问答问题 {DELIMITER}")

    start_time = time.time()

    logger.info(f"Question ID: {qid}")
    logger.info(f"Table ID: {tid}")
    logger.info(f"Question: {query}")

    ##### 加载 ho_tree
    pkl_file = os.path.join(pkl_dir, f'{tid}.pkl')
    embedding_cache_file = os.path.join(pkl_dir, f'{tid}_embedding.json')
    with open(pkl_file, 'rb') as file:
        ho_tree = pickle.load(file)

    logger.info(f"Loading PKL File: {pkl_file}")
    logger.info(f"Loading Embedding Cache File: {embedding_cache_file}")

    final_answer, _, reliability = qa_RWP(
        query=query,
        ho_tree=ho_tree,
        table_file=table_file,
        embedding_cache_file=embedding_cache_file,
        enable_emebdding=enable_emebdding,
        enable_query_decompose=enable_query_decompose,
    )
    qa_pair["reliability"] = reliability
    qa_pair["model_output"] = final_answer

    end_time = time.time()

    logger.info(f"{DELIMITER} 回答问题成功！ {DELIMITER}")
    logger.info(f"Cost time: {end_time - start_time}")
    
    logger.remove(log_file_handler)
    
    return qa_pair

def benchmark(
    table_dir: str,                     # 保存表格的文件的目录，目录中是一个一个的表格文件，文件名即为表格的唯一标识
    input_jsonl: str,                   # 保存输入 QA Pair 的 JSONL 文件，每条记录的必要字段为 'id', 'table_id', 'query', 'label'
    output_jsonl: str,                  # 保存 QA Pair 模型推理结果的文件
    pkl_dir: str,                       # 保存表格转换后的 HO-Tree 中间结果以及 Embedding Cache 的路径
    enable_emebdding: bool = True,      # 是否启用 Embedding 机制，对应是否使用两阶段验证的 Forward Verification
    cache_dir: str = CACHE_DIR,         # Cache 缓存保存路径
    log_dir: str = LOG_DIR,             # Log 日志保存路径
    process_from_scratch: bool = False, # 是否重新进行 HO-Tree 与处理过程
    qa_from_scratch: bool = False,      # 是否从头进行 QA
):

    if not os.path.exists(pkl_dir): os.makedirs(pkl_dir)
    if not os.path.exists(cache_dir): os.makedirs(cache_dir)
    if not os.path.exists(log_dir): os.makedirs(log_dir)

    ##### 读入待处理 QA Pair
    input_list = []
    with open(input_jsonl, "r", encoding="utf-8") as file:
        for line in file:
            input_list.append(json.loads(line))
    input_list.sort(key=lambda x: x["table_id"])    # 按照 table_id 的顺序排序

    ##### 读入所有 Table 文件列表
    table_files = sorted(glob.glob(table_dir + "/*"))

    # 处理不同格式的输入，即都转换为 Excel 格式
    new_table_files = []
    for table_file in table_files:
        last_dot_idx = os.path.basename(table_file).rfind('.')
        new_table_file = os.path.join(table_dir, os.path.basename(table_file)[:last_dot_idx] + '.xlsx')

        if table_file.endswith(".xlsx"):
            pass
        elif table_file.endswith(".csv"):
            df = pd.read_csv(new_table_file)
            df.to_excel(new_table_file, index=False, engine='openpyxl')
        elif table_file.endswith(".html"):
            html_content = open(table_file).read()
            html2workbook(html_content).save(new_table_file)
        elif table_file.endswith(".md"):
            markdown_content = open(table_file).read()
            table = extract_markdown_tables(markdown_content)
            with pd.ExcelWriter(new_table_file, engine='openpyxl') as writer:
                sheet_name = f'sheet'
                df = pd.DataFrame(table[1:], columns=table[0])
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        new_table_files.append(new_table_file)
    table_files = new_table_files

    ##### 读取已经处理了的 QA Pair
    output_data = []
    qid_set = set()
    if not qa_from_scratch and os.path.exists(output_jsonl):
        with open(output_jsonl, 'r', encoding='utf-8') as file:
            for line in file:
                output_data.append(json.loads(line))
        qid_set = set([r['id'] for r in output_data])

    ##### 尝试读取 HO-Tree 中间结果文件
    pkl_files = []
    embedding_cache_files = []
    if not process_from_scratch and pkl_dir is not None:
        pkl_files = sorted(glob.glob(pkl_dir + "/*.pkl"))
        embedding_cache_files = sorted(glob.glob(pkl_dir + "/*_embedding.json"))

    ##### 逐一处理每一张表格
    for table_file in table_files:
        table_id = os.path.basename(table_file).split('.')[0]

        ##### 表格预处理 Table -> HO-Tree
        if os.path.join(pkl_dir, f'{table_id}.pkl') not in pkl_files:
            try:
                ho_tree = get_excel_feature_tree(table_file, log_dir=log_dir, vlm_cache=False)
                
                tree_json = ho_tree.__json__()
                tree_str = ho_tree.__str__([1])

                with open(os.path.join(pkl_dir, f"{table_id}.pkl"), "wb") as f:
                    pickle.dump(ho_tree, f)
                with open(os.path.join(pkl_dir, f"{table_id}.txt"), "w", encoding='utf-8') as f:
                    f.write(tree_str)
                with open(os.path.join(pkl_dir, f"{table_id}.json"), "w", encoding='utf-8') as f:
                    json.dump(tree_json, f, indent=4, ensure_ascii=False)
            except Exception as e:
                import traceback; traceback.print_exc()
                logger.error(f"File: {table_file} Error: {e}")
                continue
        else:
            logger.info(f"File: {table_file} has already converted into HO-Tre!!!")

        ##### 获得表格内容的 embedding
        if os.path.join(pkl_dir, f'{table_id}_embedding.json') not in embedding_cache_files:
            embedding_dict = EmbeddingModel().get_embedding_dict(ho_tree.all_value_list())

            with open(os.path.join(pkl_dir, f"{table_id}_embedding.json"), "w", encoding='utf-8') as f:
                json.dump(embedding_dict, f, ensure_ascii=False)

    ##### 逐一处理每一条问题
    for qa_pair in input_list:
        table_id = qa_pair['table_id']
        
        ##### 防止重复问答
        if not qa_from_scratch and qa_pair['id'] in qid_set:
            continue

        ##### 执行问答 #####
        record = answer_question(
            qa_pair=qa_pair,
            table_file=table_file,
            pkl_dir=pkl_dir,
            enable_emebdding=enable_emebdding,
            log_dir=log_dir
        )

        ##### 保存 QA 结果
        output_data.append(record)
        qid_set.add(record['id'])
        with open(output_jsonl, "a", encoding="utf-8") as file:
            file.write(f"{json.dumps(record, ensure_ascii=False)}\n")

def main():
    ##### You need to change this
    input_jsonl ="data/SSTQA-zh/test.jsonl"
    table_dir = "data/SSTQA-zh/table"
    pkl_dir = "data/SSTQA-zh/pkl"
    output_jsonl = "SSTQA-zh/output.jsonl"
    log_dir = "data/SSTQA-zh/log"

    benchmark(
        table_dir=table_dir,
        input_jsonl=input_jsonl,
        output_jsonl=output_jsonl,
        pkl_dir=pkl_dir,
        cache_dir=CACHE_DIR,
        log_dir=log_dir,
    )

if __name__ == "__main__":
    main()
    # run()

