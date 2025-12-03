import gradio as gr
import os
import sys
import time
import atexit

from loguru import logger
from embedding import EmbeddingModel
from utils.api_utils import vlm_generate, llm_generate, embedding_generate
from query.primitive_pipeline import *
from table2tree.extract_excel import *
from table2tree.feature_tree import *
from utils.constants import DELIMITER, LOG_DIR

# 导入其他子文件中的函数
from config import load_api_config, save_api_config, api_config
from interface import create_interface

# 初始化时加载配置
load_api_config()



# 从core_functions.py导入核心功能函数
from core_functions import answer_question, process_table_for_tree, process_question_only, clear_all, read_all_logs, get_llm_generate, get_vlm_generate, get_embedding_generate



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
