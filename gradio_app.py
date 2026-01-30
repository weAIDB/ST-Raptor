import gradio as gr
import os
import sys
import time
import atexit
import json
import fastapi
import uvicorn
from fastapi import Request

from loguru import logger
from embedding import EmbeddingModel
from utils.api_utils import vlm_generate, llm_generate, embedding_generate
from query.primitive_pipeline import *
from table2tree.extract_excel import *
from table2tree.feature_tree import *
from utils.constants import DELIMITER, LOG_DIR

# 导入其他子文件中的函数
from config import load_api_config, save_api_config, api_config
from new_interface import create_interface

# 初始化时加载配置
load_api_config()

# 从core_functions.py导入核心功能函数
from core_functions import (
    answer_question,
    process_table_for_tree,
    process_question_only,
    clear_all,
    read_all_logs,
    get_llm_generate,
    get_vlm_generate,
    get_embedding_generate,
    rebuild_feature_tree_from_json,
)


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
        try:
            clear_all()
        except Exception as e:
            print(f"[WARN] 清理失败: {e}")

    # 注册退出时的清理函数
    atexit.register(cleanup)
    
    demo = create_interface()
    demo.queue()  # 启用队列模式

    app = fastapi.FastAPI()

    @app.post("/save_tree")
    async def save_tree(request: Request):
        try:
            data = await request.json()
        except Exception as e:
            return {"status": "fail", "msg": f"invalid json: {e}"}
        logger.info(f"[save_tree] received save request, len={len(str(data))}")
        # 前端为选中逻辑添加的 id 不落盘，先剥离
        def strip_ids(obj):
            if isinstance(obj, list):
                return [strip_ids(o) for o in obj]
            if isinstance(obj, dict):
                return {k: strip_ids(v) for k, v in obj.items() if k != "id"}
            return obj

        cleaned = strip_ids(data)
        logger.info(f"[save_tree] saving cleaned tree to cache/temp.json and data/SSTQA/temp_tables/temp.json")
        ok, msg = rebuild_feature_tree_from_json(cleaned)
        if ok:
            return {"status": "ok"}
        return {"status": "fail", "msg": msg}

    # 将 gradio app 挂载到 fastapi
    app = gr.mount_gradio_app(app, demo, path="/")

    try:
        uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")
    except (KeyboardInterrupt, Exception) as e:
        if not isinstance(e, KeyboardInterrupt):
            print(f"服务运行出错: {e}")
    finally:
        print("\n🛑 正在关闭界面并清理环境...")
        try:
            demo.close()  # 显式关闭 Gradio 界面
        except:
            pass
        # 稍微等一下让异步任务取消
        time.sleep(0.5)
        cleanup()
        print("✅ 服务已安全停止。")

if __name__ == "__main__":
    main()
