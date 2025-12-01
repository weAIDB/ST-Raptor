import gradio as gr
import pandas as pd
import atexit 
import os
import json
from pathlib import Path
import shutil
import re
import glob
import pickle
import io
import sys
import time

from tqdm import tqdm
from loguru import logger
from embedding import EmbeddingModel
from utils.api_utils import vlm_generate
from table2tree.feature_tree import construct_feature_tree
import re

from query.primitive_pipeline import *
from table2tree.extract_excel import *
from table2tree.feature_tree import *
from utils.constants import DELIMITER, LOG_DIR

def answer_question(
    qa_pair: dict,                          # 一条问答对
    table_file: str,                        # 表格原文件路径
    cache_dir: str,                           # 存储 HO-Tree 中间结果的路径
    enable_query_decompose: bool = True,    # 是否启用 Query Decomposition 机制
    enable_emebdding: bool = True,          # 是否启用 Embedding 机制
    log_dir: str = LOG_DIR,                 # Log 日志目录
    temperature: float = 0.5,               # LLM/VLM temperature
    max_tokens: int = 1024                  # LLM/VLM max_tokens
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
    
        
def process_question_only(question, temperature=0.5, max_tokens=1024):
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
    """合并读取所有日志文件（temp.xlsx.log, param_change.log, temp.log），按时间顺序显示"""
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
    return "".join(all_lines[-max_lines:]) if all_lines else "暂无日志"

def create_interface():
    with gr.Blocks(
        title="ST-Raptor 表格问答系统",
        theme=gr.themes.Soft(),
        css="""
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; padding: 20px; }
        .input-section { background: #f8f9fa; padding: 20px; border-radius: 10px; }
        .output-section { background: white; padding: 20px; border-radius: 10px; margin-top: 20px; }
        .H-OTree-output {
            height:600px !important;
            max-height: 600px !important;
            overflow-y: hidden !important;
            font-size: 13px;
            padding:10px;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
        }
        .H-OTree-output .json-container {
            max-height: 580px !important;
            overflow-y: auto !important;
            interactive: true;
        }
        .question-output {
            max-height: 300px !important;
            overflow-y: auto !important;
        }
        """
    ) as demo:
        
        gr.Markdown("""
        <div class="header">
            <h1>📊 ST-Raptor 表格问答系统</h1>
            <p>上传 Excel 表格并使用自然语言提问，获取智能答案</p>
        </div>
        """)
        
        # 顶部：扁扁的输入框 + 上传和清除按钮
        with gr.Row():
            table_input = gr.File(
                label="上传 Excel 表格",
                file_types=[".xlsx", ".xls"],
                height=150,
                scale=3
            )
            with gr.Column(scale=1):
               upload_btn = gr.Button("📤 上传", variant="primary")
               
               clear_top_btn = gr.Button("🗑️ 清除", variant="secondary")
        
        # 主要内容区域：左右分栏
        with gr.Row():
            # 左侧：H-OTree JSON输出框
            with gr.Column(scale=1):
                gr.Markdown("### 📁 H-OTree 结构")
                tree_output = gr.JSON(
                    label="H-OTree JSON",
                    elem_classes="H-OTree-output"
                )
            
            # 右侧：问题提交区域
            with gr.Column(scale=1):
                # 问题输入框
                gr.Markdown("### ❓ 问题提交")
                question_input = gr.Textbox(
                    label="请输入您的问题",
                    lines=3,
                    placeholder="例如：销售总额是多少？哪个产品销量最高？",
                    show_copy_button=True
                )
                temperature_slider = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.5, step=0.01,
                    label="Temperature (采样多样性)",
                    info="越大越随机，越小越确定"
                )
                max_tokens_box = gr.Number(
                    value=1024, precision=0, label="Max Tokens (最大生成长度)",
                    info="生成答案的最大 token 数"
                )
                # 提交问题按钮
                submit_question_btn = gr.Button(
                    "🚀 提交问题", 
                    variant="primary",
                    size="lg"
                )
                # 问题输出框
                gr.Markdown("### 💬 问题回答")
                answer_output = gr.Textbox(
                    label="AI回答",
                    lines=8,
                    show_copy_button=True,
                    placeholder="AI的回答将在此显示...",
                    interactive=False,
                    elem_classes="question-output"
                )
                # 日志输出框（直接放在问题回答下方，不折叠）
                gr.Markdown("### 📜 实时日志",open=False)
                log_output = gr.Textbox(
                    label="终端日志",
                    lines=18,
                    interactive=False,
                    show_copy_button=True,
                    value=read_all_logs(),
                    elem_id="log-output-box"
                )
                # 注入 JS 使其每次内容变化时自动滚动到底部
                gr.HTML(
                    """
<script>
function scrollLogToBottom() {
    var box = document.querySelector('#log-output-box textarea');
    if (box) {
        box.scrollTop = box.scrollHeight;
    }
}
const observer = new MutationObserver(scrollLogToBottom);
setTimeout(function() {
    var box = document.querySelector('#log-output-box textarea');
    if (box) {
        observer.observe(box, { childList: true, subtree: true, characterData: true });
    }
}, 1000);
</script>
"""
                )
        
        # 示例问题
        gr.Markdown("### 💡 示例问题")
        examples = gr.Examples(
            examples=[
                ["销售总额是多少？"],
                ["哪个产品销量最高？"],
                ["表格有多少行多少列？"]
            ],
            inputs=[question_input],
            label="点击示例快速尝试"
        )
        
        # 上传按钮点击事件 - 处理表格生成H-OTree
        upload_btn.click(
            fn=process_table_for_tree,
            inputs=[table_input],
            outputs=[tree_output]
        )
        
        # 提交问题按钮点击事件 - 处理问题
        submit_question_btn.click(
            fn=process_question_only,
            inputs=[question_input, temperature_slider, max_tokens_box],
            outputs=[answer_output]
        )
        
        # 定时刷新日志窗口（每3秒自动更新）- 使用 Gradio 的 every 参数和 Timer
        def refresh_all_logs_fn():
            return read_all_logs(log_dir="log", max_lines=200)
        
        # 创建隐藏的 Timer 触发器，定时刷新日志
        demo.load(
            fn=refresh_all_logs_fn,
            inputs=[],
            outputs=[log_output],
            every=3
        )
        
        # 清除按钮点击时也清空日志窗口
        def clear_log():
            return ""
        
        # 清除按钮绑定两个事件：清空日志 + 清空所有内容
        clear_top_btn.click(
            fn=clear_log,
            inputs=[],
            outputs=[log_output],
            queue=False
        )
        
        clear_top_btn.click(
            fn=clear_all,
            inputs=[],
            outputs=[table_input, question_input, answer_output, tree_output]
        )
    
    return demo

def main():
    # 启动时仅清理日志文件（保留 cache/temp 等），避免残留旧日志干扰
    def clean_logs(log_dir="log"):
        try:
            if os.path.exists(log_dir):
                for root, dirs, files in os.walk(log_dir):
                    for fname in files:
                        fpath = os.path.join(root, fname)
                        try:
                            os.remove(fpath)
                        except Exception as e:
                            print(f"[WARN] 无法删除日志文件 {fpath}: {e}")
        except Exception as e:
            print(f"[WARN] 清理日志失败: {e}")

    clean_logs("log")

    print("🚀 启动 ST-Raptor Gradio 界面...")
    print("📋 访问地址: http://localhost:7860")
    print("⏹️  按 Ctrl+C 停止服务")
    
    def cleanup():
        clear_all()
    def signal_handler(signum, frame):
        print("🛑 服务已停止，正在清理缓存...")
        cleanup()
        sys.exit(0)
    import signal
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(cleanup)
    
    demo = create_interface()
    demo.close(cleanup)
    demo.queue()  # 启用队列模式
    try:
        demo.launch(
           server_name="0.0.0.0",  # 允许外部访问
           server_port=7860,       
           share=False,            # 不生成公开链接
           debug=True,             
           show_error=True
        )
    except KeyboardInterrupt:
        print("🛑 服务已停止，正在清理缓存...")
        cleanup()
    finally:
        cleanup()

if __name__ == "__main__":
    main()
