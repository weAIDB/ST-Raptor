import os
import json
import time
import shutil
import pickle
import pandas as pd
from datetime import datetime
import gradio as gr
from loguru import logger
from utils.constants import DELIMITER, LOG_DIR
from embedding import EmbeddingModel
from utils.api_utils import vlm_generate, llm_generate, embedding_generate
from query.primitive_pipeline import qa_RWP
from table2tree.feature_tree import get_excel_feature_tree
from config import api_config

def get_llm_generate(prompt, max_tokens=8192, temperature=0.5):
    return llm_generate(
        prompt=prompt,
        key=api_config["llm_api_key"],
        url=api_config["llm_api_url"],
        model=api_config["llm_model"],
        max_tokens=max_tokens,
        temperature=temperature
    )

def get_vlm_generate():
    # 返回一个已经配置好API参数的vlm_generate函数
    def configured_vlm_generate(prompt, image, temperature=0.5):
        return vlm_generate(
            prompt=prompt,
            image=image,
            key=api_config["vlm_api_key"],
            url=api_config["vlm_api_url"],
            model=api_config["vlm_model"],
            temperature=temperature
        )
    return configured_vlm_generate

def get_embedding_generate():
    # 返回一个已经配置好API参数的embedding_generate函数
    def configured_embedding_generate(input_texts, dimensions=1024):
        return embedding_generate(
            input_texts=input_texts,
            key=api_config["embedding_api_key"],
            url=api_config["embedding_api_url"],
            model=api_config["embedding_model"],
            dimensions=dimensions
        )
    return configured_embedding_generate

def answer_question(
    qa_pair: dict,                          # 一条问答对
    table_file: str,                        # 表格原文件路径
    cache_dir: str,                           # 存储 HO-Tree 中间结果的路径
    enable_query_decompose: bool = True,    # 是否启用 Query Decomposition 机制
    enable_emebdding: bool = True,          # 是否启用 Embedding 机制
    log_dir: str = LOG_DIR,                 # Log 日志目录
    temperature: float = 0.5,               # LLM/VLM temperature
    max_tokens: int = 2048                  # LLM/VLM max_tokens
):
    
    query = qa_pair["query"]

    ##### 创建日志文件 命名为 表格id_问题id.log
    log_file = os.path.join(log_dir, f'temp.log')
    log_file_handler = logger.add(log_file)

    logger.info(f"{DELIMITER} 开始问答问题 {DELIMITER}")

    start_time = time.time()

    logger.info(f"Question ID: temp")
    logger.info(f"Table ID: temp")

    logger.info(f"Question: {query}")
    logger.info(f"Temperature: {temperature}")
    logger.info(f"Max tokens: {max_tokens}")

    ##### 加载 ho_tree
    pkl_file = os.path.join(cache_dir, f'temp.pkl')
    embedding_cache_file = os.path.join(cache_dir, f'temp.embedding.json')
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
        temperature=temperature,
        max_tokens=max_tokens
    )
    qa_pair["reliability"] = reliability
    qa_pair["model_output"] = final_answer

    end_time = time.time()

    logger.info(f"{DELIMITER} 回答问题成功！ {DELIMITER}")
    logger.info(f"Cost time: {end_time - start_time}")
    
    logger.remove(log_file_handler)
    
    return qa_pair

def process_table_for_tree(file):
    """专门处理表格，生成H-OTree结构"""
    if file is None:
        return "请先选择表格文件", ""
    clear_all()        
    try:
        cache_dir = "cache"
        log_dir = "log"
        os.makedirs(cache_dir, exist_ok=True)
        source_filename = os.path.splitext(os.path.basename(file.name))[0]
                
        # 创建临时文件
        temp_dir = "data/SSTQA/temp_tables"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file = os.path.join(temp_dir, "temp.xlsx")
        shutil.copy2(file.name, temp_file)
                
        # 读取表格
        df = pd.read_excel(temp_file)
            
        f_tree = get_excel_feature_tree(temp_file, log_dir=log_dir, vlm_cache=False)
        tree_json = f_tree.__json__()
        tree_str = f_tree.__str__([1])
                
        # 保存中间文件
        with open(os.path.join(cache_dir, f"temp.pkl"), "wb") as f:
            pickle.dump(f_tree, f)
        with open(os.path.join(cache_dir, f"temp.txt"), "w", encoding='utf-8') as f:
            f.write(tree_str)
        with open(os.path.join(cache_dir, f"temp.json"), "w", encoding='utf-8') as f:
            json.dump(tree_json, f, indent=4, ensure_ascii=False)
                # 生成嵌入向量
        embedding_dict = EmbeddingModel().get_embedding_dict(f_tree.all_value_list())
        EmbeddingModel().save_embedding_dict(
            embedding_dict, os.path.join(cache_dir, f"temp.embedding.json")
        )
        gr.Info("✅ 表格解析完成！H-OTree结构已生成")
        return tree_json
         
    except Exception as e:
        import traceback
        error_msg = f"处理错误: {str(e)}\n错误详情: {traceback.format_exc()}"
        gr.Warning(f"❌ 生成树失败: {error_msg}")
        return None
    

def process_question_only(question, temperature=0.5, max_tokens=2048):
    """专门处理问题，返回答案"""
    table_file = "data/SSTQA/temp_tables/temp.xlsx"
    if not os.path.exists(table_file):
        gr.Warning("请先上传表格")
        return "请先上传表格"
    if not question.strip():
        gr.Warning("请输入问题")
        return "请输入问题"
    try:
        # 记录参数变更日志（使用 loguru 格式：时间 | 级别 | 内容）
        param_log_file = os.path.join("log", "param_change.log")
        os.makedirs("log", exist_ok=True)
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        msg = f"{timestamp} | PARAM_CHANGE | temperature={temperature}, max_tokens={max_tokens}\n"
        with open(param_log_file, "a", encoding="utf-8") as f:
            f.write(msg)
        qa_pair = {
            "id": "temp",
            "table_id": "temp",
            "query": question.strip()
        }
        cache_dir = "cache"
        result=answer_question(
            qa_pair=qa_pair,
            table_file=table_file,
            cache_dir=cache_dir,
            enable_emebdding=True,
            enable_query_decompose=True,
            log_dir="log",
            temperature=temperature,
            max_tokens=max_tokens
        )
        if result :
            gr.Info("✅ 答案生成成功！")
            return f"答案: {result.get('model_output', '无答案')}\n\n可靠性: {result.get('reliability', '未知')}"
        else:
            gr.Warning("❌ 生成答案失败")
            return "生成答案失败"
    except Exception as e:
        import traceback
        error_msg = f"处理错误: {str(e)}\n错误详情: {traceback.format_exc()}"
        gr.Warning(f"❌ 生成答案失败: {error_msg}")
        return "生成答案失败"

def clear_all():
    """清除所有内容并删除相关文件"""
    import shutil
    import os    
    # 删除临时表格文件
    temp_dir = "data/SSTQA/temp_tables"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)  # 重新创建空目录
    
    # 删除log目录下的所有文件
    log_dir = "log"
    if os.path.exists(log_dir):
        for file in os.listdir(log_dir):
            file_path = os.path.join(log_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

    # 删除cache目录下的所有文件
    cache_dir = "cache"
    if os.path.exists(cache_dir):
        for item in os.listdir(cache_dir):
            item_path = os.path.join(cache_dir, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)  # 递归删除子目录
    
    return None, "", "", {}  # 清空所有界面组件

def read_all_logs(log_dir="log", max_lines=200):
    """合并读取所有日志文件（temp.xlsx.log, param_change.log, temp.log），按时间顺序显示，并添加颜色美化"""
    all_lines = []
    
    log_files = [
        os.path.join(log_dir, "temp.xlsx.log"),
        os.path.join(log_dir, "param_change.log"),
        os.path.join(log_dir, "temp.log"),
    ]
    
    for log_path in log_files:
        if os.path.exists(log_path):
            try:
                with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
                    all_lines.extend(lines)
            except Exception as e:
                all_lines.append(f"[ERROR] 读取 {log_path} 失败: {e}\n")
    
    # 按时间戳排序（loguru 格式：时间 | 级别 | ...）
    try:
        all_lines.sort(key=lambda x: x.split("|")[0].strip() if "|" in x else "")
    except Exception:
        pass
    
    # 取最后 max_lines 行
    log_content = "".join(all_lines[-max_lines:]) if all_lines else "暂无日志"
    
    # 添加颜色美化 - 将日志转换为HTML格式
    # 支持 loguru 格式: 时间 | 级别 | 内容
    html_lines = []
    for line in log_content.split("\n"):
        if "|" in line and len(line.split("|")) >= 3:
            parts = line.split("|", 2)
            timestamp = parts[0].strip()
            level = parts[1].strip()
            content = parts[2].strip()
            # 为时间戳添加蓝色，为日志级别添加绿色
            html_line = f"<span style='color: blue'>{timestamp}</span> | <span style='color: green'>{level}</span> | {content}<br>"
        else:
            # 非标准格式行保持原样
            html_line = line + "<br>"
        html_lines.append(html_line)
    
    # 包装在<pre>标签中以保留格式，但使用HTML允许颜色显示
    return f"<pre style='font-family: monospace; white-space: pre-wrap; word-wrap: break-word;'>{' '.join(html_lines)}</pre>"
