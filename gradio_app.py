import gradio as gr
import pandas as pd
import os
import json
from pathlib import Path
import shutil
import re
import glob
import pickle
import io
import sys

from tqdm import tqdm
from loguru import logger

from query.primitive_pipeline import *
from table2tree.extract_excel import *
from table2tree.feature_tree import *
from utils.constants import DELIMITER


class LogCapture:
    def __init__(self):
        self.logs = []
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.log_buffer = io.StringIO()
        
    def start_capture(self):
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

        logger.remove()  # 移除默认处理器
        logger.add(
            self.log_buffer,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} - {message}",
            level="INFO"
        )
        
    def stop_capture(self):
        stdout_content = sys.stdout.getvalue()
        stderr_content = sys.stderr.getvalue()
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        
        loguru_content = self.log_buffer.getvalue()
        
        # 恢复 loguru 默认配置
        logger.remove()
        logger.add(sys.stderr, level="INFO")

        log_messages = []

        print(f"🔍 DEBUG: stdout_content length: {len(stdout_content)}")
        print(f"🔍 DEBUG: stderr_content length: {len(stderr_content)}")
        print(f"🔍 DEBUG: loguru_content length: {len(loguru_content)}")
        if stdout_content:
            for line in stdout_content.strip().split('\n'):
                if line.strip():
                    log_messages.append(f"[STDOUT] {line}")
        
        if stderr_content:
            for line in stderr_content.strip().split('\n'):
                if line.strip():
                    log_messages.append(f"[STDERR] {line}")

        if loguru_content:
            for line in loguru_content.strip().split('\n'):
                if line.strip():
                    log_messages.append(f"[LOG] {line}")
        return '\n'.join(log_messages)

def answer_question(
    f_tree: FeatureTree,
    record,
    enable_emebdding=True,
    embedding_cache_file=None,
    log_dir=None,
):
    query = record["query"]

    # create qa log file
    if log_dir is not None:
        log_file = os.path.join(log_dir, f"{record['id']}.txt")
    else:
        log_file = None

    if log_file is not None:  # Log
        with open(log_file, "a") as f:
            f.write(f"{DELIMITER} Query {DELIMITER}\n")
            f.write(query + "\n")

    try:
        final_answer, qa_pair, reliability = qa_RWP(
            f_tree=f_tree,
            query=query,
            enable_emebdding=enable_emebdding,
            embedding_cache_file=embedding_cache_file,
            log_file=log_file,
        )
        record["reliability"] = reliability
        record["model_output"] = final_answer

    except Exception as e:
        if log_file is not None:  # Log
            with open(log_file, "a") as f:
                f.write(f"{DELIMITER} An Error Occurred {DELIMITER}\n")
                f.write(f"Error: {e}\n")
        print(e)
        import traceback
        traceback.print_exc()
        return None

    if log_file is not None:  # Log
        with open(log_file, "a") as f:
            f.write(f"{DELIMITER} Final Output {DELIMITER}\n")
            f.write(json.dumps(record, ensure_ascii=False, indent=4))

    return record

def process_table_question(table_file, question):
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    if not table_file or not question:
        yield "请上传表格并输入问题", None,""
    
    log_messages = ""
    source_filename = os.path.splitext(os.path.basename(table_file.name))[0]
    log_messages = f"🚀 开始处理表格: {source_filename}\n"
    log_messages += f"🔍 问题: {question}\n\n"
    yield "处理中...", None, log_messages

    log_capture = LogCapture()
    try:
        temp_dir = "data/SSTQA/temp_tables"
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_file = os.path.join(temp_dir, "temp.xlsx")
        shutil.copy2(table_file.name, temp_file)
        df = pd.read_excel(temp_file)

        log_messages += "✅ 临时文件创建完成\n"
        yield "处理中...", df.head(10), log_messages
        
        record={
            "id": "1",
            "table_id": "1",
            "query": question,
            "table_file": table_file.name
        }
        log_capture.start_capture()
        log_messages += "🔍 开始生成特征树...\n"
        yield "处理中...", df.head(10), log_messages
    
        output_data = []
        try:
            f_tree = get_excel_feature_tree(temp_file, log=True, vlm_cache=False)
            tree_json = f_tree.__json__()
            tree_str = f_tree.__str__([1])
            log_messages += "✅ 表格特征树生成成功\n"
            yield "处理中...", df.head(10), log_messages
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"File: {source_filename} Error: {e}")
            with open("./error.txt", "a") as f:
                f.write(f"process_one_table() error: {source_filename}\n")
            log_messages += f"❌ 表格处理错误: {str(e)}\n"
            yield f"❌ 表格处理错误: {str(e)}", df.head(10), log_messages
            return
        
        log_messages += "💾 保存中间文件...\n"
        yield "处理中...", df.head(10), log_messages
        with open(os.path.join(cache_dir, f"{source_filename}.pkl"), "wb") as f:
            pickle.dump(f_tree, f)
        with open(os.path.join(cache_dir, f"{source_filename}.txt"), "w", encoding='utf-8') as f:
            f.write(tree_str)
        with open(os.path.join(cache_dir, f"{source_filename}.json"), "w", encoding='utf-8') as f:
            json.dump(tree_json, f, indent=4, ensure_ascii=False)
        
        log_messages += "✅ 中间文件保存完成\n"
        log_messages += "🔍 生成嵌入向量...\n"
        yield "处理中...", df.head(10), log_messages
        embedding_dict = EmbeddingModelMultilingualE5().get_embedding_dict(
            f_tree.all_value_list()
        )
        EmbeddingModelMultilingualE5().save_embedding_dict(
            embedding_dict, os.path.join(cache_dir, f"{source_filename}.embedding.json")
        )
        embedding_cache_file = os.path.join(cache_dir, f"{source_filename}.embedding.json")


        log_messages += "✅ 嵌入向量生成完成\n"
        log_messages += "🤖 开始调用问答函数...\n"
        yield "处理中...", df.head(10), log_messages
        result = answer_question(
            f_tree=f_tree,
            record=record,
            enable_emebdding=True,
            embedding_cache_file=embedding_cache_file,
            log_dir=None  # 不使用文件日志，而是捕获控制台输出
        )
        log_messages += "✅ 问答函数调用完成\n"
        yield "处理中...", df.head(10), log_messages
        log_messages +=log_capture.stop_capture()
        if result is None:
            log_messages += "❌ 处理失败，请检查表格格式和问题\n"
            yield "❌ 处理失败，请检查表格格式和问题", df.head(10), log_messages
            return
        final_answer = result.get("model_output", "未获取到答案")
        reliability = result.get("reliability", "未知")

        log_messages+=f"\n[SYSTEM] 可靠性: {reliability}"
        log_messages+=f"\n[SYSTEM] 答案: {final_answer}"

        
        try:
            os.remove(temp_file)
        except:
            pass
        
        yield final_answer, df.head(10), log_messages
        return
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
    
        if 'log_messages' not in locals():
            log_messages = ""
    
        try:
            captured_logs = log_capture.stop_capture()
            log_messages += captured_logs
        except:
            pass
    
        log_messages += f"❌ 处理过程中发生错误: {str(e)}\n"
        log_messages += f"错误详情: {traceback_str}\n"
        
        yield f"❌ 处理错误: {str(e)}", df.head(10), log_messages
        return

def create_interface():
    with gr.Blocks(
        title="ST-Raptor 表格问答系统",
        theme=gr.themes.Soft(),
        css="""
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; padding: 20px; }
        .input-section { background: #f8f9fa; padding: 20px; border-radius: 10px; }
        .output-section { background: white; padding: 20px; border-radius: 10px; margin-top: 20px; }
        .log-output {
            max-height: 500px !important;
            overflow-y: auto !important;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            line-height: 1.4;
            resize: none !important;
        }
        """
    ) as demo:
        
        gr.Markdown("""
        <div class="header">
            <h1>📊 ST-Raptor 表格问答系统</h1>
            <p>上传 Excel 表格并使用自然语言提问，获取智能答案</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1, elem_classes="input-section"):
                gr.Markdown("### 📁 输入区域")
                
                table_input = gr.File(
                    label="上传 Excel 表格",
                    file_types=[".xlsx", ".xls"],
                    height=200
                )
                
                question_input = gr.Textbox(
                    label="输入问题",
                    placeholder="例如：销售总额是多少？哪个产品销量最高？",
                    lines=3,
                    max_lines=5
                )
                
                submit_btn = gr.Button(
                    "🚀 获取答案", 
                    variant="primary",
                    size="lg"
                )
                
                clear_btn = gr.Button("🗑️ 清除", variant="secondary")
            
            with gr.Column(scale=2, elem_classes="output-section"):
                gr.Markdown("### 📝 答案输出")
                
                answer_output = gr.Textbox(
                    label="分析结果",
                    interactive=False,
                    lines=6,
                    show_copy_button=True
                )
                
                gr.Markdown("### 📋 表格预览")
                
                table_preview = gr.Dataframe(
                    label="前十行数据",
                    interactive=False,
                    show_copy_button=True,
                    scale=2,
                    wrap=True
                )
                
                gr.Markdown("### 🔍 运行日志")
                
                log_output = gr.Textbox(
                    label="系统日志",
                    lines=20,
                    show_copy_button=True,
                    placeholder="系统运行日志将在此显示...",
                    interactive=False, # 最大行数
                    elem_classes="log-output"
                )
        
        gr.Markdown("### 💡 示例问题")
        
        examples = gr.Examples(
            examples=[
                [None, "销售总额是多少？"],
                [None, "哪个产品销量最高？"],
                [None, "表格有多少行多少列？"]
            ],
            inputs=[table_input, question_input],
            label="点击示例快速尝试"
        )
        
        submit_btn.click(
            fn=process_table_question,
            inputs=[table_input, question_input],
            outputs=[answer_output, table_preview, log_output]
        )
        
        def clear_inputs():
            return None, "", "请上传表格并输入问题", None, None
        
        clear_btn.click(
            fn=clear_inputs,
            inputs=[],
            outputs=[table_input, question_input, answer_output, table_preview, log_output]
        )
    
    return demo

def main():
    print("🚀 启动 ST-Raptor Gradio 界面...")
    print("📋 访问地址: http://localhost:7860")
    print("⏹️  按 Ctrl+C 停止服务")
    
    demo = create_interface()
    
    demo.launch(
        server_name="0.0.0.0",  # 允许外部访问
        server_port=7860,       
        share=False,            # 不生成公开链接
        debug=True,             
        show_error=True        
    )

if __name__ == "__main__":
    main()
