import re
import os
import time as tm
import glob
import pickle
import traceback

from tqdm import tqdm

from table2tree.feature_tree import *
from utils.api_utils import llm_generate
from embedding import match_sub_table, EmbeddingModel
from query.primitive_def import *
from verifier.verifier import *
from utils.constants import *
from utils.prompt_template import *
from utils.sheet_utils import get_xlsx_table_string

# 尝试导入API配置
def get_api_config():
    """获取API配置，如果不可用则返回默认值"""
    try:
        from gradio_app import api_config
        return api_config
    except ImportError:
        return {}

def get_llm_generate():
    """获取配置的llm_generate函数"""
    config = get_api_config()
    def configured_llm_generate(prompt, temperature=0.5, max_tokens=2048):
        return llm_generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            key=config.get("llm_api_key", LLM_API_KEY),
            url=config.get("llm_api_url", LLM_API_URL),
            model=config.get("llm_model", LLM_MODEL_TYPE)
        )
    return configured_llm_generate

def flatten_nested_list(value):
    res = []
    for x in value:
        if isinstance(x, list):
            res.extend(flatten_nested_list(x))
        else:
            res.append(x)
    return res


def bool_string(s):
    s = s.lower()
    if s == "true" or s == "t" or s == "yes" or s == "y" or s == "1" or s == "on":
        return True
    return False


def query_decompose(f_tree: FeatureTree, query: str, temperature=0.5):
    prompt = query_decompose_prompt.format(query=query, schema=f_tree.__index__())

    configured_llm_generate = get_llm_generate()
    res = configured_llm_generate(prompt, temperature=temperature)

    queries = []
    retrieve_flags = []
    lines = res.splitlines()
    for line in lines:
        line = line.strip()

        # 提取 Query 后的内容
        query_start = line.find("[Query]") + len("[Query]")
        query_end = line.find("[Retrieve]")
        query = line[query_start:query_end].strip()

        # 提取 Retrieve 后的内容
        retrieve_start = line.find("[Retrieve]") + len("[Retrieve]")
        retrieve = line[retrieve_start:].strip()

        queries.append(query)
        retrieve_flags.append(bool_string(retrieve))

    return queries, retrieve_flags


def entity_extractor(f_tree: FeatureTree, query: str, temperature=0.5):
    prompt = entity_extract_prompt.format(query=query, schema=f_tree.__index__())

    configured_llm_generate = get_llm_generate()
    res = configured_llm_generate(prompt, temperature=temperature)

    # pattern = r"```python\n(.*?)```"
    # matches = re.findall(pattern, res, re.DOTALL)

    # if len(matches) == 0:
        # return eval(matches[0])

    return eval(res)


def semantic_reason(evidence, query, temperature=0.5, max_tokens=8192):
    if isinstance(evidence, FeatureTree):
        evidence = evidence.__json__()
    prompt = semantic_reasoning_prompt.format(evidence=evidence, query=query)

    configured_llm_generate = get_llm_generate()
    res = configured_llm_generate(prompt, temperature=temperature, max_tokens=max_tokens)

    return res


def calc_math(f_tree: FeatureTree, query, temperature=0.5):

    math_prompt = primitive_prompt_math.format(query=query, table=f_tree.__json__())

    retry_cnt = 1
    args = None  # 初始化args变量
    while retry_cnt < MAX_RETRY_PRIMITIVE:
        try:
            configured_llm_generate = get_llm_generate()
            primitive_seq = configured_llm_generate(prompt=math_prompt, temperature=temperature)

            operation = primitive_seq.splitlines()[0].strip()
            if operation == "None":
                logger.info(f'{DELIMITER} 不需要进行MATH操作 {DELIMITER}')
                return f_tree

            args = re.findall(r"\[(.*?)\]", operation, re.DOTALL)
            if len(args) == 0:
                print("No Operation Detect! Regenerate!")
                raise ValueError("No Primitive Operation Detected! Need Regenerate!")
            if args[0]!="MATH" or not args[2] in ["SUM", "MAX", "MIN", "CNT", "AVR"]:
                raise ValueError("False primitive generated!")
            
            logger.info(f'{DELIMITER} 生成的MATH操作为 {DELIMITER}')
            logger.info(f'[MATH] + [{args[1]}] + [{args[2]}]')
                    
            break
        except Exception as e:
            print(e, f"Generated primitive have unknown error, need regenerate!")
            retry_cnt += 1
    
    try:
        # 检查args是否被成功定义
        if args is None:
            logger.error("Failed to generate valid MATH operation after maximum retries")
            return f_tree
            
        type = args[2]
        for item in f_tree.index_tree.leaf_nodes:
                if args[1] in item.value:  # 在标签中
                    ans = item.body[:]
                    if type == "SUM":
                        ans = [x for x in ans if x.value is not None]
                        ans = [float(x.value) for x in ans]
                        return "答案是"+str(sum(ans))
                    elif type == "MAX" or type == "MIN":
                        ans = [x for x in ans if x.value is not None and x.value != '']
                        ans = [float(x.value) for x in ans]
                        if type == "MAX":
                            num = max(ans)
                        else:
                            num = min(ans)
                        tree_dict = f_tree.__json__()
                        if DEFAULT_TABLE_NAME in tree_dict:
                            tree_dict = tree_dict[DEFAULT_TABLE_NAME]
                        ans = ""
                        for i in tree_dict:
                            if i[args[1]] == str(num):
                                ans += str(item) + ","
                        return "答案是"+ans[:-1]
                    elif type == "AVR":
                        ans = [x for x in ans if x.value is not None]
                        ans = [float(x.value) for x in ans]
                        return "答案是"+str(sum(ans)/len(ans))
                    elif type == "CNT":
                        return "答案是"+str(len(ans))
    except Exception as e:
        import traceback; traceback.print_exc()
        return f_tree


def dfs_reasoning(
    query: str,
    f_tree: FeatureTree,
    query_history: list = None,
    iter_history: str = "",
    depth=1,
    enable_embedding_match=False,
    embedding_cache_file=None,
    temperature=0.5,
    max_tokens=2048,
):
    """Reasoning in a depth-first search manner"""
    if depth > MAX_ITER_PRIMITIVE:
        print(
            "生成的原语句步骤达到了最大次数，停止生成，使用当前获得的数据作为回答依据！"
        )
        return None, None

    # Step 1 设计Prompt，为当前子问题生成原语句，可能会有iter_history 记录之前的迭代结果
    # 需要判断，当前的f_tree 下面是否还嵌套有FeatureTree，如果没有嵌套，则使用Embedding匹配相似子子部分，如果有嵌套，则只给Schema
    flag = False  # 默认不嵌套
    for leaf in f_tree.index_tree.leaf_nodes:
        if len(leaf.body) >= 1 and isinstance(leaf.body[0].value, FeatureTree):
            flag = True
            break

    ### Condition
    # 不存在Tree的嵌套，先使用Condition筛选后再Select
    if not flag:    
        cond_prompt = primtive_prompt_condition.format(
            query=query,
            table=f_tree.__json__(),
        )
        
        logger.info(f"{DELIMITER} 没有嵌套，尝试使用COND操作 {DELIMITER}")
        
        configured_llm_generate = get_llm_generate()
        operation = configured_llm_generate(cond_prompt).strip()
        
        if "None" in operation:
            logger.info(f"{DELIMITER} 检测不使用COND操作 {DELIMITER}")
        else:
            args = re.findall(r"\[(.*?)\]", operation, re.DOTALL)
            if len(args) % 4 != 0 or args[0] != 'COND':
                pass
            else:
                while len(args) != 0:
                    column = args[1]
                    op = args[2]
                    value = args[3]
                    f_tree = condition_search(f_tree, column, op, value)
                    logger.info(f"{DELIMITER} 使用COND操作 {DELIMITER}")
                    logger.info(f"[COND] + [{column}] + [{op}] + [{value}]")
                    logger.info(f"{DELIMITER} COND操作结果 {DELIMITER}")
                    logger.info(f"{f_tree.__json__()}")
                    args = args[4:]

        logger.info(f"{DELIMITER} 没有嵌套，尝试使用MATH操作 {DELIMITER}")
        
        ### Math based on the f_tree
        res = calc_math(f_tree, query, temperature=temperature)
        if isinstance(res, FeatureTree):
            pass
        else:
            return [res], ""
    
    ### Retrieval
    if flag:  # 嵌套有FeatureTree，只给Schema
        table_string = f"{f_tree.__index__()}"

    else:  # 不嵌套FeatureTree
        # entities = entity_extractor(f_tree=f_tree, query=query) # 1. LLM 提取 谓词、实体
        # f_tree : FeatureTree = match_sub_table(entities, f_tree)    # 2. Embedding 模型匹配
        table_string = f"{f_tree.__json__()}"  # 3. 拿匹配度最高的子表

    # prompt = primitive_prompt_zeroshot.format(
    #     query=query,
    #     table=table_string,
    #     query_history=query_history,
    # )

    prompt = primitive_prompt_fewshot.format(
        query=query,
        table=table_string,
        query_history=query_history,
        query_example=query_example,
        table_example=table_example,
    )

    print(f"query: {query}\ntable: {table_string}")
    logger.info(f"{DELIMITER} Prompt Args for Primitive Depth {depth} {DELIMITER}")
    logger.info(f"### Query\n{query}\n### Table\n{table_string}\n###Query History\n{query_history}")

    retry_cnt = 1
    while retry_cnt < MAX_RETRY_PRIMITIVE:

        # Generate primitive
        primitive_seq = llm_generate(prompt=prompt, temperature=temperature, max_tokens=max_tokens)
        operation = primitive_seq.splitlines()[0].strip()

        print(operation)
        logger.info(f"{DELIMITER} Generated Primitive of {LLM_MODEL_TYPE} Try {retry_cnt} {DELIMITER}")
        logger.info(f"{operation}")
        logger.info(f"{DELIMITER} Primitive Execution {DELIMITER}")

        try:
            step_info = f"# Iter{depth}\n Operation: {operation}\n"
            retrieved_data = []
            args = re.findall(r"\[(.*?)\]", operation, re.DOTALL)
            if len(args) == 0:
                print("No Operation Detect! Regenerate!")
                raise ValueError("No Primitive Operation Detected! Need Regenerate!")
            if args[0] in ["CHL", "FAT", "EXT", "COND", "FOREACH"]:
                op_res = []  # primitive execution result list
                if (
                    args[0] == "CHL" in operation and len(args) == 2
                ):  # Retrieve Children
                    if enable_embedding_match:  # Embedding Match
                        print(f_tree.all_value_list())
                        # args[1] = EmbeddingModel().top1_match(
                        #     [args[1]], table=f_tree.all_value_list(), embedding_cache_file=embedding_cache_file
                        # )[0]
                        topk_values = EmbeddingModel().topk_match(
                            [args[1]],
                            f_tree.all_value_list(),
                            embedding_cache_file=embedding_cache_file,
                        )
                        args[1] = topk_values[0][0]

                    logger.info(f"{DELIMITER} Primitive After Embedding {DELIMITER}")
                    logger.info(f"[CHL] + [{args[1]}]")

                    op_res = meta_search(f_tree, [args[1], "n"])

                elif args[0] == "FAT" and len(args) == 2:  # Retrieve Father
                    if enable_embedding_match:  # Embedding Match
                        args[1] = EmbeddingModel().top1_match(
                            [args[1]],
                            table=f_tree.all_value_list(),
                            embedding_cache_file=embedding_cache_file,
                        )[0]

                    logger.info(f"{DELIMITER} Primitive After Embedding {DELIMITER}")
                    logger.info(f"[FAT] + [{args[1]}]")

                    op_res = meta_search(f_tree, ["n", args[1]])

                elif args[0] == "EXT" and len(args) == 3:  # Extract Operation
                    if enable_embedding_match:
                        t = EmbeddingModel().top1_match(
                            [args[1], args[2]],
                            table=f_tree.all_value_list(),
                            embedding_cache_file=embedding_cache_file,
                        )
                        args[1] = t[0]
                        args[2] = t[1]

                    logger.info(f"{DELIMITER} Primitive After Embedding {DELIMITER}")
                    logger.info(f"[EXT] + [{args[1]}] + [{args[2]}]")

                    op_res = meta_search(f_tree, [args[1], "n", args[2]])
                    # step_info += f"Extract Result: {op_res}\n"
                    # retrieved_data.append(op_res)

                elif args[0] == "COND" and len(args) == 3:  # Conditional Search
                    if enable_embedding_match:  # Embedding Match
                        t = EmbeddingModel().top1_match(
                            [args[1], args[2]],
                            table=f_tree.all_value_list(),
                            embedding_cache_file=embedding_cache_file,
                        )
                        args[1] = t[0]
                        args[2] = t[1]

                    logger.info(f"{DELIMITER} Primitive After Embedding {DELIMITER}")
                    logger.info(f"[COND] + [{args[1]}] + [{args[2]}]")

                    op_res = meta_search(f_tree, ["cond", args[1], args[2]])

                elif args[0] == "FOREACH" and len(args) == 3:  # Foreach Operation
                    if enable_embedding_match:  # Embedding Match
                        args[1] = EmbeddingModel().top1_match(
                            [args[1]],
                            table=f_tree.all_value_list(),
                            embedding_cache_file=embedding_cache_file,
                        )[0]

                    logger.info(f"{DELIMITER} Primitive After Embedding {DELIMITER}")
                    logger.info(f"[FOREACH] + [{args[1]}] + [{args[2]}]")

                    op_res = meta_search(f_tree, ["foreach", args[1], args[2]])

                logger.info(f"{DELIMITER} Retreval Result {DELIMITER}")
                if len(op_res) == 0:
                    logger.info(f"Nothing Retrieved")
                else:
                    for subdata in op_res:
                        if isinstance(subdata, FeatureTree):
                            logger.info(f"Retrieved Schema: {subdata.__index__()}")
                        else:
                            logger.info(f"Retrieved Data: {subdata}")

                tree_cnt = 1
                data_cnt = 1
                for subdata in op_res:
                    if isinstance(subdata, FeatureTree):
                        sub_answer, _ = dfs_reasoning(
                            query=query,
                            f_tree=subdata,
                            query_history=query_history,
                            iter_history=iter_history
                            + step_info
                            + f"Retrieve Subtree Schema {tree_cnt}: {subdata.__index__()}\n",
                            depth=depth + 1,
                            embedding_cache_file=embedding_cache_file,
                        )
                        if sub_answer is not None and len(sub_answer) > 0:
                            step_info += f"Retrieve Subtree Schema {tree_cnt}: {subdata.__index__()}\n"
                            step_info += f"Data Retrieved from Subtree{tree_cnt}: {sub_answer[0]}\n"
                            retrieved_data.append(sub_answer)
                            tree_cnt += 1
                        else:
                            step_info += f"Retrieve Subtree Schema {tree_cnt}: {subdata.__index__()}\n"
                            retrieved_data.append(subdata)
                    else:
                        step_info += f"Data Retrieved {data_cnt}: {subdata}\n"
                        retrieved_data.append(subdata)
                        data_cnt += 1

            elif args[0] == "CMP" and len(args) == 4:  # Compare Operation
                if enable_embedding_match:
                    t = EmbeddingModel().top1_match(
                        [args[1], args[2]],
                        table=f_tree.all_value_list(),
                        embedding_cache_file=embedding_cache_file,
                    )
                    args[1] = t[0]
                    args[2] = t[1]

                logger.info(
                    f"{DELIMITER} Primitive After Embedding {DELIMITER}"
                )
                logger.info(f"[CMP] + [{args[1]}] + [{args[2]}] + [{args[3]}]")

                op_res = meta_search(f_tree, ["cmp", args[1], args[2], args[3]])
                step_info += f"Compare Result: {op_res}\n"
                retrieved_data.append(op_res)

            elif args[0] == "END":
                print("问题回答结束!")
                prompt = prompt + step_info

            else:
                print("Unknown Operation! Regenerate!")
                step_info = ""
                raise ValueError("Unknown Primitive Operation!")

            break  # 成功执行了原语句，退出retry循环
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(e, f"Generated primitive have unknown error, need regenerate!")
            logger.error(f"Primitive Need Regenerate! Primitive Execution Error: {e}\n{traceback.print_exc()}")
            retry_cnt += 1

        # 这里 retrieved_data 是最终的答案列表


    # if len(retrieved_data) == 1 and isinstance(retrieved_data[0], FeatureTree):
    r_data = []
    for item in retrieved_data:
        if isinstance(item, list) and len(item) >= 1 and isinstance(item[0], FeatureTree):    item=item[0]
        f_tree = item
        if not isinstance(f_tree, FeatureTree): 
            r_data.append(f_tree)
            continue
        cond_prompt = primtive_prompt_condition.format(
            query=query,
            table=f_tree.__json__(),
        )
        
        logger.info(f"{DELIMITER} 没有嵌套，尝试使用COND操作 {DELIMITER}")
        
        operation = llm_generate(cond_prompt).strip()
        
        if "None" in operation:
            logger.info(f"{DELIMITER} 检测不使用COND操作 {DELIMITER}")
        else:
            args = re.findall(r"\[(.*?)\]", operation, re.DOTALL)
            if len(args) % 4 != 0 or args[0] != 'COND':
                pass
            else:
                while len(args) != 0:
                    column = args[1]
                    op = args[2]
                    value = args[3]
                    f_tree = condition_search(f_tree, column, op, value)
                    logger.info(f"{DELIMITER} 使用COND操作 {DELIMITER}")
                    logger.info(f"[COND] + [{column}] + [{op}] + [{value}]")
                    logger.info(f"{DELIMITER} COND操作结果 {DELIMITER}")
                    logger.info(f"{f_tree.__json__()}")
                    args = args[4:]

        logger.info(f"{DELIMITER} 没有嵌套，尝试使用MATH操作 {DELIMITER}")

        ### Math based on the f_tree
        res = calc_math(f_tree, query, temperature=0.5)  # 使用默认值，因为调用此函数的地方没有temperature参数
        r_data.append(res)
        # if isinstance(res, FeatureTree):
            # pass
        # else:
            # return [res], ""
    
    return r_data, prompt
    # return retrieved_data, prompt


# TODO Bottom Up Reasoning
def get_subtree_by_content(f_tree: FeatureTree, entity):
    
    b_node_list = f_tree.body_tree.root.children[:]
    f_tree_index = [f_tree for _ in range(len(b_node_list))]
    
    while len(b_node_list) > 0:
        b_node: BodyNode = b_node_list[0]
        f_tree = f_tree_index[0]
        
        if isinstance(b_node.value, FeatureTree):
            b_node_list.extend(b_node.value.body_tree.root.children)
            f_tree_index.extend([b_node.value for _ in range(len(b_node.value.body_tree.root.children))])
        elif b_node.value == entity:
            return f_tree            
            
        b_node_list = b_node_list[1:]
        f_tree_index = f_tree_index[1:]    
        
    return None

def bottom_up_reasoning(
    query: str,
    f_tree: FeatureTree,
    embedding_cache_file=None,
    temperature=0.5,
):
    """
    return: FeatureTree / String
    """
    entities = entity_extractor(f_tree, query, temperature=temperature)
    start_from = EmbeddingModel().top1_match(
        entities, f_tree.all_value_list(), embedding_cache_file
    )

    logger.info(f"{DELIMITER} Buttom Up Reasoning {DELIMITER}")
    logger.info(f"Extracted Entities: {entities}")
    logger.info(f"Matched Table Content: {start_from}")

    ##### 获得 embedding 结果所在的 subdata
    subdata_list = []
    retrieved_data_list = []
    for index, content in enumerate(start_from):
        subdata = get_subtree_by_content(f_tree, content)   # 仅匹配BodyNode，如果content在IndexNode中，则会返回None
        if subdata is None: continue
        subdata_list.append(subdata)
        
        if isinstance(subdata, FeatureTree):
            retrieved_data, prompt = dfs_reasoning(
                query=query,
                f_tree=subdata,
                enable_embedding_match=True,
                embedding_cache_file=embedding_cache_file,
            )
            retrieved_data_list.append(retrieved_data)
        else:
            retrieved_data_list.append(subdata)

        logger.info(f"Subdata for {content}:\n {retrieved_data_list[-1]}")

    return retrieved_data_list  # 可能为空 []，则最终基于整表进行reasoning


def delete_list_empty_elem(data: list):
    new_data = []
    for r in data:
        if len(str(r).strip()) != 0:
            new_data.append(r)
    return new_data        


def qa_RWP(query: str, 
           ho_tree: FeatureTree, 
           table_file: str,
           embedding_cache_file=None, 
           enable_emebdding=False, 
           enable_query_decompose=True,
           temperature=0.5,
           max_tokens=8192):
    """Reason-while-Planning"""

    try:
        ##### Step 1. 问题分解
        raw_query = query
        if enable_query_decompose:
            decomposed_queries, retrieve_flag = query_decompose(f_tree=ho_tree, query=query, temperature=temperature)
        else:
            decomposed_queries = [query]
            retrieve_flag = [True]

        logger.info(f"{DELIMITER} Query Decompose {DELIMITER}")
        logger.info(f"{decomposed_queries}")
        logger.info(f"{DELIMITER} Retreive Flag {DELIMITER}")
        logger.info(f"{retrieve_flag}")

        logger.info(f"{DELIMITER} 开始遍历解决分解后的每一个子问题 {DELIMITER}")
        
        ##### 依次遍历每一个子问题
        qa_pair = []
        subquery_index = 1
        for query, flag in zip(decomposed_queries, retrieve_flag):
            logger.info(f"{DELIMITER} Answering Subquery {subquery_index} {DELIMITER}")
            logger.info(f"{query}")

            answer = None
            if flag:  # Need Retrieval
                retrieved_data, reasoning_path = dfs_reasoning(
                    query=query,
                    f_tree=ho_tree,
                    query_history=qa_pair,
                    iter_history="",
                    depth=1,
                    enable_embedding_match=enable_emebdding,
                    embedding_cache_file=embedding_cache_file,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                retrieved_data = flatten_nested_list(retrieved_data)
                for index, data in enumerate(retrieved_data):
                    if isinstance(data, FeatureTree):
                        retrieved_data[index] = data.__json__()

                retrieved_data = delete_list_empty_elem(retrieved_data)
                if len(retrieved_data) == 0:
                    prompt = direct_table_reasoning_prompt.format(table=ho_tree.__json__(), query=query)
                    configured_llm_generate = get_llm_generate()
                    answer = configured_llm_generate(prompt, temperature=temperature, max_tokens=max_tokens)
                else:
                    answer = semantic_reason(retrieved_data, query, temperature=temperature, max_tokens=max_tokens)
                check_res = Verifier().check_answer(query, answer)

                logger.info(f"{DELIMITER} Final Retrieved Data for Subquery {subquery_index} {DELIMITER}")
                logger.info(f"{retrieved_data}")
                logger.info(f"{DELIMITER} Answer for Subquery {subquery_index} {DELIMITER}")
                logger.info(f"{answer}")
                logger.info(f"{DELIMITER} Verifier Check Answer for Subquery {subquery_index} {DELIMITER}")
                logger.info(f"{check_res}")

                if check_res is False:
                    st = tm.perf_counter()
                    retrieved_data = bottom_up_reasoning(query, ho_tree, embedding_cache_file, temperature=temperature)
                    et = tm.perf_counter()
                    retrieved_data = flatten_nested_list(retrieved_data)
                    for index, data in enumerate(retrieved_data):
                        if isinstance(data, FeatureTree):
                            retrieved_data[index] = data.__json__()
                    
                    retrieved_data = delete_list_empty_elem(retrieved_data)
                    if len(retrieved_data) == 0:
                        # Direct Reasoning
                        prompt = direct_table_reasoning_prompt.format(table=ho_tree.__json__(), query=query)
                        configured_llm_generate = get_llm_generate()
                        answer = configured_llm_generate(prompt, temperature=temperature, max_tokens=max_tokens)

                        logger.info(f"{DELIMITER} Bottom Up Reaoning Failed, Direct Reasoning {subquery_index} {DELIMITER}")
                        
                    else:
                        answer = semantic_reason(retrieved_data, query, temperature=temperature, max_tokens=max_tokens)
                        
                        logger.info(f"{DELIMITER} Bottom Up Reaoning Answer for Query {subquery_index} {DELIMITER}")
                    logger.info(f"{answer}")
                    logger.info(f"Time: {et - st}")

            else:  # Do not need retrieval, directly Semantic Reasoning
                answer = semantic_reason(qa_pair, query, temperature=temperature, max_tokens=max_tokens)

                logger.info(f"{DELIMITER} Answer for Subquery {subquery_index} {DELIMITER}")
                logger.info(f"{answer}")

            qa_pair.append({"query": query, "answer": answer})

            subquery_index += 1

        # final_answer = semantic_reason(qa_pair, query)
        final_answer = qa_pair[-1]["answer"]

    except Exception as e:
        import traceback; 
        traceback.print_exc()
        table_str = get_xlsx_table_string(table_file)

        logger.error(f"{DELIMITER} DFS Reasoning Fail! Try to Reason from Scratch! {DELIMITER}")
        prompt = direct_table_reasoning_prompt.format(table=table_str, query=raw_query)
        configured_llm_generate = get_llm_generate()
        final_answer = configured_llm_generate(prompt, temperature=temperature, max_tokens=max_tokens)
        logger.info(f"{final_answer}")
    
    ##### Final Check
    logger.info(f"{DELIMITER} Final Check for Query {DELIMITER}")
    final_check = Verifier().check_answer(query=raw_query, answer=final_answer)
    logger.info(f"{final_check}")
    if final_check:
        logger.info(f"{DELIMITER} Final Check Passed and Final Answer{DELIMITER}")
        logger.info(f"{final_answer}")
    else:
        prompt = direct_table_reasoning_prompt.format(table=ho_tree.__json__(), query=raw_query)
        configured_llm_generate = get_llm_generate()
        final_answer = configured_llm_generate(prompt, temperature=temperature, max_tokens=max_tokens)
        logger.info(f"{DELIMITER} Final Answer for Whole Table Reasoning {DELIMITER}")
        logger.info(f"{final_answer}")
    
    ##### Back Verification
    reliability, query_list = Verifier().back_verify(f_tree=ho_tree, query=raw_query, answer=final_answer)
    logger.info(f"{DELIMITER} Back Verification {DELIMITER}")
    logger.info(f"Query List: {query_list}")
    logger.info(f"Reliabillity: {reliability}")

    return final_answer, qa_pair, reliability
