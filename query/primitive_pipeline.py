import re
import os
import time as tm
import glob
import pickle
import traceback

from tqdm import tqdm

from table2tree.feature_tree import *
from utils.api_utils import generate_deepseek, generate_text_third_party
from embedding import match_sub_table, EmbeddingModelMultilingualE5
from query.primitive_def import *
from verifier.verifier import *
from utils.constants import *
from utils.prompt_template import *


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


def query_decompose(f_tree: FeatureTree, query: str):
    prompt = query_decompose_prompt.format(query=query, schema=f_tree.__index__())

    res = generate_deepseek(prompt, key=API_KEY, url=API_URL, temperature=0.7)

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


def entity_extractor(f_tree: FeatureTree, query: str):
    prompt = entity_extract_prompt.format(query=query, schema=f_tree.__index__())

    res = generate_deepseek(prompt, key=API_KEY, url=API_URL, temperature=1.0)

    # pattern = r"```python\n(.*?)```"
    # matches = re.findall(pattern, res, re.DOTALL)

    # if len(matches) == 0:
        # return eval(matches[0])

    return eval(res)


def semantic_reason(evidence, query):
    if isinstance(evidence, FeatureTree):
        evidence = evidence.__json__()
    prompt = semantic_reasoning_prompt.format(evidence=evidence, query=query)

    res = generate_deepseek(prompt, key=API_KEY, url=API_URL, temperature=1.0)

    return res


def calc_math(f_tree: FeatureTree, query, log_file=None):

    math_prompt = primitive_prompt_math.format(query=query, table=f_tree.__json__())

    retry_cnt = 1
    while retry_cnt < MAX_RETRY_PRIMITIVE:
        try:
            primitive_seq = generate_deepseek(
                prompt = math_prompt,
                key = API_KEY,
                url = API_URL,
                temperature = 1.0,
            )
            operation = primitive_seq.splitlines()[0].strip()
            if operation == "None":
                if log_file is not None:
                    with open(log_file, 'a') as file:
                        file.write(f'{DELIMITER} 不需要进行MATH操作 {DELIMITER}\n')
                return f_tree

            args = re.findall(r"\[(.*?)\]", operation, re.DOTALL)
            if len(args) == 0:
                print("No Operation Detect! Regenerate!")
                raise ValueError("No Primitive Operation Detected! Need Regenerate!")
            if args[0]!="MATH" or not args[2] in ["SUM", "MAX", "MIN", "CNT", "AVR"]:
                raise ValueError("False primitive generated!")
            
            if log_file is not None:
                with open(log_file, 'a') as file:
                    file.write(f'{DELIMITER} 生成的MATH操作为 {DELIMITER}\n')
                    file.write(f'[MATH] + [{args[1]}] + [{args[2]}]\n')
                    
            break
        except Exception as e:
            print(e, f"Generated primitive have unknown error, need regenerate!")
            retry_cnt += 1
    
    try:
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
    log_file=None,
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
        
        if log_file is not None:    # Log
            with open(log_file, 'a') as file:
                file.write(f"{DELIMITER} 没有嵌套，尝试使用COND操作 {DELIMITER}\n")
        
        operation = generate_deepseek(cond_prompt, API_KEY, API_URL).strip()
        
        if "None" in operation:
            if log_file is not None:    # Log
                with open(log_file, 'a') as file:
                    file.write(f"{DELIMITER} 检测不使用COND操作 {DELIMITER}\n")
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
                    if log_file is not None:    # Log
                        with open(log_file, 'a') as file:
                            file.write(f"{DELIMITER} 使用COND操作 {DELIMITER}\n")
                            file.write(f"[COND] + [{column}] + [{op}] + [{value}]\n")
                            file.write(f"{DELIMITER} COND操作结果 {DELIMITER}\n")
                            file.write(f"{f_tree.__json__()}\n")
                    args = args[4:]

        if log_file is not None:    # Log
            with open(log_file, 'a') as file:
                file.write(f"{DELIMITER} 没有嵌套，尝试使用MATH操作 {DELIMITER}\n")
        
        ### Math based on the f_tree
        res = calc_math(f_tree, query, log_file)
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
    if log_file is not None:  # Log
        with open(log_file, "a") as file:
            file.write(
                f"{DELIMITER} Prompt Args for Primitive Depth {depth} {DELIMITER}\n"
            )
            file.write(
                f"### Query\n{query}\n### Table\n{table_string}\n###Query History\n{query_history}\n"
            )

    retry_cnt = 1
    while retry_cnt < MAX_RETRY_PRIMITIVE:

        # Generate primitive
        primitive_seq = generate_deepseek(
            prompt=prompt,
            key=API_KEY,
            url=API_URL,
            temperature=1.0,
        )
        operation = primitive_seq.splitlines()[0].strip()

        print(operation)
        if log_file is not None:  # Log
            with open(log_file, "a") as file:
                file.write(
                    f"{DELIMITER} Generated Primitive of {LLM_MODEL_TYPE} Try {retry_cnt} {DELIMITER}\n"
                )
                file.write(f"{operation}\n")
                file.write(f"{DELIMITER} Primitive Execution {DELIMITER}\n")

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
                        # args[1] = EmbeddingModelMultilingualE5().top1_match(
                        #     [args[1]], table=f_tree.all_value_list(), embedding_cache_file=embedding_cache_file
                        # )[0]
                        topk_values = EmbeddingModelMultilingualE5().topk_match(
                            [args[1]],
                            f_tree.all_value_list(),
                            embedding_cache_file=embedding_cache_file,
                        )
                        args[1] = topk_values[0][0]

                    if log_file is not None:  # Log
                        with open(log_file, "a") as file:
                            file.write(
                                f"{DELIMITER} Primitive After Embedding {DELIMITER}\n"
                            )
                            file.write(f"[CHL] + [{args[1]}]\n")

                    op_res = meta_search(f_tree, [args[1], "n"])

                elif args[0] == "FAT" and len(args) == 2:  # Retrieve Father
                    if enable_embedding_match:  # Embedding Match
                        args[1] = EmbeddingModelMultilingualE5().top1_match(
                            [args[1]],
                            table=f_tree.all_value_list(),
                            embedding_cache_file=embedding_cache_file,
                        )[0]

                    if log_file is not None:  # Log
                        with open(log_file, "a") as file:
                            file.write(
                                f"{DELIMITER} Primitive After Embedding {DELIMITER}\n"
                            )
                            file.write(f"[FAT] + [{args[1]}]\n")

                    op_res = meta_search(f_tree, ["n", args[1]])

                elif args[0] == "EXT" and len(args) == 3:  # Extract Operation
                    if enable_embedding_match:
                        # 注释掉Embedding匹配代码以避免CUDA内存不足
                        # t = EmbeddingModelMultilingualE5().top1_match(
                        #     [args[1], args[2]],
                        #     table=f_tree.all_value_list(),
                        #     embedding_cache_file=embedding_cache_file,
                        # )
                        # args[1] = t[0]
                        # args[2] = t[1]
                        pass  # 直接跳过Embedding匹配

                    if log_file is not None:  # Log
                        with open(log_file, "a") as file:
                            file.write(
                                f"{DELIMITER} Primitive After Embedding {DELIMITER}\n"
                            )
                            file.write(f"[EXT] + [{args[1]}] + [{args[2]}] \n")

                    op_res = meta_search(f_tree, [args[1], "n", args[2]])
                    # step_info += f"Extract Result: {op_res}\n"
                    # retrieved_data.append(op_res)

                elif args[0] == "COND" and len(args) == 3:  # Conditional Search
                    if enable_embedding_match:  # Embedding Match
                        # 注释掉Embedding匹配代码以避免CUDA内存不足
                        # t = EmbeddingModelMultilingualE5().top1_match(
                        #     [args[1], args[2]],
                        #     table=f_tree.all_value_list(),
                        #     embedding_cache_file=embedding_cache_file,
                        # )
                        # args[1] = t[0]
                        # args[2] = t[1]
                        pass  # 直接跳过Embedding匹配

                    if log_file is not None:  # Log
                        with open(log_file, "a") as file:
                            file.write(
                                f"{DELIMITER} Primitive After Embedding {DELIMITER}\n"
                            )
                            file.write(f"[COND] + [{args[1]}] + [{args[2]}]\n")

                    op_res = meta_search(f_tree, ["cond", args[1], args[2]])

                elif args[0] == "FOREACH" and len(args) == 3:  # Foreach Operation
                    if enable_embedding_match:  # Embedding Match
                        # 注释掉Embedding匹配代码以避免CUDA内存不足
                        # args[1] = EmbeddingModelMultilingualE5().top1_match(
                        #     [args[1]],
                        #     table=f_tree.all_value_list(),
                        #     embedding_cache_file=embedding_cache_file,
                        # )[0]
                        pass  # 直接跳过Embedding匹配

                    if log_file is not None:  # Log
                        with open(log_file, "a") as file:
                            file.write(
                                f"{DELIMITER} Primitive After Embedding {DELIMITER}\n"
                            )
                            file.write(f"[FOREACH] + [{args[1]}] + [{args[2]}]\n")

                    op_res = meta_search(f_tree, ["foreach", args[1], args[2]])

                if log_file is not None:  # Log
                    with open(log_file, "a") as file:
                        file.write(f"{DELIMITER} Retreval Result {DELIMITER}\n")
                        if len(op_res) == 0:
                            file.write(f"Nothing Retrieved\n")
                        else:
                            for subdata in op_res:
                                if isinstance(subdata, FeatureTree):
                                    file.write(
                                        f"Retrieved Schema: {subdata.__index__()}\n"
                                    )
                                else:
                                    file.write(f"Retrieved Data: {subdata}\n")

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
                            log_file=log_file,
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
                    t = EmbeddingModelMultilingualE5().top1_match(
                        [args[1], args[2]],
                        table=f_tree.all_value_list(),
                        embedding_cache_file=embedding_cache_file,
                    )
                    args[1] = t[0]
                    args[2] = t[1]

                if log_file is not None:  # Log
                    with open(log_file, "a") as file:
                        file.write(
                            f"{DELIMITER} Primitive After Embedding {DELIMITER}\n"
                        )
                        file.write(f"[CMP] + [{args[1]}] + [{args[2]}] + [{args[3]}]\n")

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
            if log_file is not None:  # Log
                with open(log_file, "a") as file:
                    file.write(
                        f"Primitive Need Regenerate! Primitive Execution Error: {e}\n{traceback.print_exc()}\n"
                    )
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
        
        if log_file is not None:    # Log
            with open(log_file, 'a') as file:
                file.write(f"{DELIMITER} 没有嵌套，尝试使用COND操作 {DELIMITER}\n")
        
        operation = generate_deepseek(cond_prompt, API_KEY, API_URL).strip()
        
        if "None" in operation:
            if log_file is not None:    # Log
                with open(log_file, 'a') as file:
                    file.write(f"{DELIMITER} 检测不使用COND操作 {DELIMITER}\n")
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
                    if log_file is not None:    # Log
                        with open(log_file, 'a') as file:
                            file.write(f"{DELIMITER} 使用COND操作 {DELIMITER}\n")
                            file.write(f"[COND] + [{column}] + [{op}] + [{value}]\n")
                            file.write(f"{DELIMITER} COND操作结果 {DELIMITER}\n")
                            file.write(f"{f_tree.__json__()}\n")
                    args = args[4:]

        if log_file is not None:    # Log
            with open(log_file, 'a') as file:
                file.write(f"{DELIMITER} 没有嵌套，尝试使用MATH操作 {DELIMITER}\n")

        if log_file is not None:    # Log
            with open(log_file, 'a') as file:
                file.write(f"{DELIMITER} 没有嵌套，尝试使用MATH操作 {DELIMITER}\n")
        
        ### Math based on the f_tree
        res = calc_math(f_tree, query, log_file)
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
    log_file=None,
):
    """
    return: FeatureTree / String
    """
    entities = entity_extractor(f_tree, query)
    start_from = EmbeddingModelMultilingualE5().top1_match(
        entities, f_tree.all_value_list(), embedding_cache_file
    )

    if log_file is not None:  # Log
        with open(log_file, "a") as file:
            file.write(f"{DELIMITER} Buttom Up Reasoning {DELIMITER}\n")
            file.write(f"Extracted Entities: {entities}\n")
            file.write(f"Matched Table Content: {start_from}\n")

    # 获得 embedding 结果所在的 subdata
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
                log_file=log_file,
            )
            retrieved_data_list.append(retrieved_data)
        else:
            retrieved_data_list.append(subdata)

        if log_file is not None:    # Log
            with open(log_file, 'a') as file:
                file.write(f"Subdata for {content}:\n {retrieved_data_list[-1]}\n")

    return retrieved_data_list  # 可能为空 []，则最终基于整表进行reasoning


def delete_list_empty_elem(data: list):
    new_data = []
    for r in data:
        if len(str(r).strip()) != 0:
            new_data.append(r)
    return new_data        


def qa_PTR(f_tree, query):
    """Plan-then-Reason
    Challenge: 需要每一个action都可以对应到一个primitive上
    """

    start_time = tm.time()

    # 1. Query Decompose
    decomposed_queries = [query]
    retrieve_flag = [True]
    # decomposed_queries, retrieve_flag = query_decompose(f_tree=f_tree, query=query)
    # print(decomposed_queries)
    # print(retrieve_flag)

    # 2. Query Rewrite: only rewrite queries if retrieval is needed
    # for query, flag in zip(decomposed_queries, retrieve_flag):
    #     if flag:
    #         query = query_rewrite(f_tree=f_tree, query=query)
    # print(decomposed_queries)
    # print(retrieve_flag)

    # 3. Sequence Query Answering

    end_time = tm.time()
    print(f"time: {end_time - start_time}s")


def qa_RWP(f_tree, query, enable_emebdding=False, embedding_cache_file=None, log_file=None, enable_query_decompose=True):
    """Reason-while-Planning"""

    start_time = tm.perf_counter()

    # 1. Query Decompose
    raw_query = query
    if enable_query_decompose:
        decomposed_queries, retrieve_flag = query_decompose(f_tree=f_tree, query=query)
    else:
        decomposed_queries = [query]
        retrieve_flag = [True]

    # Log
    print("Query Decompose:")
    print(f"\t{decomposed_queries}")
    print(f"\t{retrieve_flag}")
    if log_file is not None:
        with open(log_file, "a") as f:
            f.write(f"{DELIMITER} Query Decompose {DELIMITER}\n")
            f.write(f"{decomposed_queries}\n")
            f.write(f"{DELIMITER} Retreive Flag {DELIMITER}\n")
            f.write(f"{retrieve_flag}\n")

    # 3. Sequence Query Answering
    qa_pair = []
    subquery_index = 1
    for query, flag in zip(decomposed_queries, retrieve_flag):

        if log_file is not None: # Log
            with open(log_file, "a") as f:
                f.write(f"{DELIMITER} Answering Subquery {subquery_index} {DELIMITER}\n")
                f.write(f"{query}\n")

        answer = None
        if flag:  # Need Retrieval
            retrieved_data, reasoning_path = dfs_reasoning(
                query=query,
                f_tree=f_tree,
                query_history=qa_pair,
                iter_history="",
                depth=1,
                enable_embedding_match=enable_emebdding,
                embedding_cache_file=embedding_cache_file,
                log_file=log_file,
            )

            retrieved_data = flatten_nested_list(retrieved_data)
            for index, data in enumerate(retrieved_data):
                if isinstance(data, FeatureTree):
                    retrieved_data[index] = data.__json__()

            retrieved_data = delete_list_empty_elem(retrieved_data)
            if len(retrieved_data) == 0:
                prompt = direct_table_reasoning_prompt.format(table=f_tree.__json__(), query=query)
                answer = generate_deepseek(prompt, API_KEY, API_URL)
            else:
                answer = semantic_reason(retrieved_data, query)
            check_res = Verifier().check_answer(query, answer)

            # print(retrieved_data)
            if log_file is not None:  # Log
                with open(log_file, "a") as f:
                    f.write(f"{DELIMITER} Final Retrieved Data for Subquery {subquery_index} {DELIMITER}\n")
                    f.write(f"{retrieved_data}\n")
                    f.write(f"{DELIMITER} Answer for Subquery {subquery_index} {DELIMITER}\n")
                    f.write(f"{answer}\n")
                    f.write(f"{DELIMITER} Verifier Check Answer for Subquery {subquery_index} {DELIMITER}\n")
                    f.write(f"{check_res}\n")

            if check_res is False:
                st = tm.perf_counter()
                retrieved_data = bottom_up_reasoning(query, f_tree, embedding_cache_file, log_file)
                et = tm.perf_counter()
                retrieved_data = flatten_nested_list(retrieved_data)
                for index, data in enumerate(retrieved_data):
                    if isinstance(data, FeatureTree):
                        retrieved_data[index] = data.__json__()
                
                retrieved_data = delete_list_empty_elem(retrieved_data)
                if len(retrieved_data) == 0:
                    # Direct Reasoning
                    prompt = direct_table_reasoning_prompt.format(table=f_tree.__json__(), query=query)
                    answer = generate_deepseek(prompt, API_KEY, API_URL)
                    if log_file is not None:    # Log
                        with open(log_file, 'a') as f:
                            f.write(f"{DELIMITER} Bottom Up Reaoning Failed, Direct Reasoning {subquery_index} {DELIMITER}\n")
                            f.write(f"{answer}\n")
                            f.write(f"Time: {et - st}\n")

                else:
                    answer = semantic_reason(retrieved_data, query)
                    
                    if log_file is not None:    # Log
                        with open(log_file, 'a') as f:
                            f.write(f"{DELIMITER} Bottom Up Reaoning Answer for Query {subquery_index} {DELIMITER}\n")
                            f.write(f"{answer}\n")
                            f.write(f"Time: {et - st}\n")

        else:  # Do not need retrieval, directly Semantic Reasoning
            answer = semantic_reason(qa_pair, query)

            if log_file is not None:  # Log
                with open(log_file, "a") as f:
                    f.write(
                        f"{DELIMITER} Answer for Subquery {subquery_index} {DELIMITER}\n"
                    )
                    f.write(f"{answer}\n")

        
        qa_pair.append({"query": query, "answer": answer})

        subquery_index += 1

    # final_answer = semantic_reason(qa_pair, query)
    print(qa_pair)
    final_answer = qa_pair[-1]["answer"]
    
    # Last Check
    final_check = Verifier().check_answer(query=raw_query, answer=final_answer)
    if log_file is not None:    # Log
        with open(log_file, 'a') as f:
            f.write(f"{DELIMITER} Final Check for Query {DELIMITER}\n")
            f.write(f"{final_check}\n")
    if final_check:
        if log_file is not None:    # Log
            with open(log_file, 'a') as f:
                f.write(f"{DELIMITER} Final Check Passed and Final Answer{DELIMITER}\n")
                f.write(f"{final_answer}\n")
    else:
        prompt = direct_table_reasoning_prompt.format(table=f_tree.__json__(), query=raw_query)
        final_answer = generate_deepseek(prompt, API_KEY, API_URL)
        if log_file is not None:    # Log
            with open(log_file, 'a') as f:
                f.write(f"{DELIMITER} Final Answer for Whole Table Reasoning {DELIMITER}\n")
                f.write(f"{final_answer}\n")

    print(qa_pair)
    print(final_answer)
    
    ### Back Verification
    reliability, query_list = Verifier().back_verify(f_tree=f_tree, query=raw_query, answer=final_answer)
    if log_file is not None:    # Log
        with open(log_file, 'a') as f:
            f.write(f"{DELIMITER} Back Verification {DELIMITER}\n")
            f.write(f"Query List: \n{query_list}\n")
            f.write(f"Reliabillity: {reliability}\n")

    end_time = tm.perf_counter()
    print(f"time: {end_time - start_time}s")
    if log_file is not None:  # Log
        with open(log_file, "a") as f:
            f.write(f"{DELIMITER} Total Process Time {DELIMITER}\n")
            f.write(f"{end_time - start_time}s\n")

    return final_answer, qa_pair, reliability
