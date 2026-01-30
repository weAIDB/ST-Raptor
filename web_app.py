import os
import json
import re
import shutil
import sys
from datetime import datetime
import openpyxl
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import uvicorn
from loguru import logger

from config import api_config, save_api_config, load_api_config
from utils.constants import LOG_DIR

# 配置日志系统，将print输出也写入日志文件
def setup_logging():
    """配置日志系统，捕获所有输出到日志文件"""
    # 启动时清空日志目录
    def clean_logs_on_startup():
        try:
            if os.path.exists(LOG_DIR):
                for file in os.listdir(LOG_DIR):
                    if file.endswith('.log'):
                        file_path = os.path.join(LOG_DIR, file)
                        try:
                            os.remove(file_path)
                        except Exception as e:
                            print(f"[WARN] 无法删除日志文件 {file_path}: {e}")
        except Exception as e:
            print(f"[WARN] 清理日志失败: {e}")
    
    # 清空旧日志
    clean_logs_on_startup()
    
    # 移除默认的handler（但保留一个用于控制台输出）
    logger.remove()
    
    # 添加控制台输出（带颜色，用于开发调试）
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",  # 控制台只显示INFO及以上级别
        colorize=True
    )
    
    # 添加日志文件输出（所有级别，包括DEBUG）
    log_file = os.path.join(LOG_DIR, "app.log")
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
        level="DEBUG",  # 文件记录所有级别
        rotation="10 MB",  # 日志文件大小超过10MB时轮转
        retention="7 days",  # 保留7天的日志
        encoding="utf-8",
        enqueue=False,  # 不使用队列，立即写入，避免缓冲
        backtrace=False,
        diagnose=False
    )
    
    # 创建一个辅助函数，将print输出也写入日志文件
    # 注意：我们不重定向全局的stdout/stderr，因为这会影响uvicorn
    # 但可以通过logger.info()来记录重要信息
from core_functions import (
    process_question_only,
    read_all_logs,
    get_thinking_chain,
    process_file_with_route,
    reshape_question_with_context,
    save_conversation_history,
    load_conversation_history,
    get_conversation_records,
    clear_all
)
from new_tree_ui import build_new_tree_iframe_html
from file_handlers import load_from_upload, clear_ui
from tree_handlers import persist_tree

# 初始化配置
load_api_config()

# 设置日志系统
setup_logging()

# 测试日志写入（确保日志系统正常工作）
logger.info("=" * 50)
logger.info("ST-Raptor Web应用启动")
logger.info(f"日志目录: {LOG_DIR}")
logger.info("=" * 50)

app = FastAPI(title="ST-Raptor API")

# CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件服务
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
# 图片文件服务
if os.path.exists("image"):
    app.mount("/image", StaticFiles(directory="image"), name="image")
# 项目资源
if os.path.exists("assets"):
    app.mount("/assets", StaticFiles(directory="assets"), name="assets")

# 根路径返回HTML
@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_path = os.path.join("static", "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>ST-Raptor API</h1><p>请确保 static/index.html 文件存在</p>"


@app.get("/index.html", response_class=HTMLResponse)
async def project_intro():
    base_index = os.path.join("index.html")
    if os.path.exists(base_index):
        with open(base_index, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>ST-Raptor Project Intro</h1><p>请确保根目录下的 index.html 存在</p>"

# 获取配置
@app.get("/api/config")
async def get_config():
    try:
        return JSONResponse({
            "success": True,
            "config": api_config
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

# 保存配置
@app.post("/api/config")
async def save_config(request: Request):
    try:
        data = await request.json()
        save_api_config(data)
        return JSONResponse({
            "success": True,
            "message": "配置保存成功"
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

# 文件上传
@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    try:
        import tempfile
        import types
        
        # 保存上传的文件到临时目录
        temp_files = []
        temp_paths = []
        
        for file in files:
            # 创建临时文件
            suffix = os.path.splitext(file.filename)[1]
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            temp_path = temp_file.name
            temp_paths.append(temp_path)
            
            # 写入文件内容
            with open(temp_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            
            # 创建文件对象（模拟Gradio的文件对象）
            file_obj = types.SimpleNamespace(name=temp_path)
            temp_files.append(file_obj)
        
        # 调用原有的上传处理函数
        if len(temp_files) == 1:
            result = load_from_upload(temp_files[0])
        else:
            result = load_from_upload(temp_files)
        
        # 解析返回结果
        tree_html, chat_messages, conversation_id = result
        logger.debug(f"上传文件返回 - conversation_id: {conversation_id}")
        logger.debug(f"上传文件返回 - chat_messages数量: {len(chat_messages) if chat_messages else 0}")
        
        # 构建返回消息
        message = chat_messages[0]["content"] if chat_messages else "File upload successful"
        
        logger.debug(f"上传文件返回 - 返回的conversation_id: {conversation_id}")
        # 清理临时文件（延迟删除，因为可能还需要使用）
        # 在实际应用中，可以设置一个清理机制
        
        return JSONResponse({
            "success": True,
            "conversation_id": conversation_id,
            "message": message
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

# 聊天接口
@app.post("/api/chat")
async def chat(
    request: Request,
    message: str = Form(...),
    conversation_id: str = Form(""),
    temperature: float = Form(0.5),
    max_tokens: int = Form(1024),
    files: Optional[List[UploadFile]] = File(None)
):
    try:
        logger.debug(f"/api/chat 接收 - conversation_id: '{conversation_id}', message长度: {len(message) if message else 0}")
        
        # 获取聊天历史
        chat_history = []
        if conversation_id and conversation_id.strip():
            try:
                chat_history = load_conversation_history(conversation_id)
                logger.debug(f"/api/chat 加载历史记录 - conversation_id: {conversation_id}, 消息数量: {len(chat_history)}")
            except Exception as e:
                logger.error(f"/api/chat 加载历史记录失败: {e}")
                pass
        else:
            logger.warning(f"/api/chat 警告: conversation_id 为空或无效")
        
        # 确保chat_history是有效的列表
        if not isinstance(chat_history, list):
            chat_history = []
        
        # 确保用户输入不为空
        if not message or message.strip() == "":
            return JSONResponse({
                "success": False,
                "error": "Message cannot be empty"
            })
        
        try:
            # 使用上下文重塑问题
            reshaped_message = reshape_question_with_context(message, chat_history, temperature)
            
            # 根据是否有文件选择处理线路
            if files and len(files) > 0:
                import tempfile
                import types
                
                # 保存文件到临时目录
                temp_files = []
                for file in files:
                    # 创建临时文件
                    suffix = os.path.splitext(file.filename)[1]
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                    temp_path = temp_file.name
                    
                    # 写入文件内容
                    with open(temp_path, "wb") as f:
                        shutil.copyfileobj(file.file, f)
                    
                    # 创建文件对象（模拟Gradio的文件对象）
                    file_obj = types.SimpleNamespace(name=temp_path)
                    temp_files.append(file_obj)
                
                # 有文件上传，使用process_file_with_route处理
                bot_message = process_file_with_route(
                    temp_files[0] if len(temp_files) == 1 else temp_files,
                    reshaped_message,
                    temperature,
                    max_tokens,
                    conversation_id
                )
            else:
                # 无文件上传，使用process_question_only处理（H-OTree线路）
                bot_message = process_question_only(
                    reshaped_message,
                    temperature,
                    max_tokens,
                    conversation_id
                )
            
            # 处理返回值为空的情况
            if bot_message is None or bot_message.strip() == "":
                bot_message = "抱歉，未能获取到有效回答，请检查您的问题或配置。"
        
        except Exception as e:
            # 捕获异常，返回友好提示
            bot_message = f"回答生成失败：{str(e)}"
        
        # 严格按照messages格式添加消息
        user_msg = {"role": "user", "content": message.strip()}
        assistant_msg = {"role": "assistant", "content": bot_message.strip()}
        
        chat_history.append(user_msg)
        chat_history.append(assistant_msg)
        
        logger.debug(f"/api/chat 准备保存 - conversation_id: '{conversation_id}', 类型: {type(conversation_id)}, 是否为空: {not conversation_id or conversation_id.strip() == ''}")
        
        # 保存对话历史到文件
        if conversation_id and conversation_id.strip():
            logger.debug(f"/api/chat 保存历史记录 - conversation_id: {conversation_id}, 消息数量: {len(chat_history)}")
            save_result = save_conversation_history(conversation_id, chat_history)
            logger.debug(f"/api/chat 保存历史记录结果: {save_result}")
            
            # 更新历史记录标题（如果有用户问题）
            try:
                from core_functions import get_conversation_records, generate_history_title_from_questions
                import json
                
                # 检查是否已有记录
                records = get_conversation_records()
                # get_conversation_records返回的是表格格式，需要转换
                record_exists = False
                for r in records:
                    if len(r) > 0 and r[0] == conversation_id:
                        record_exists = True
                        break
                
                if not record_exists:
                    # 创建新记录，使用LLM生成标题
                    from core_functions import create_conversation_record
                    from datetime import datetime
                    file_list = []  # 从对话历史中无法直接获取文件列表，留空
                    upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    default_summary = "新对话"
                    create_conversation_record(conversation_id, file_list, upload_time, default_summary, chat_history)
                else:
                    # 更新现有记录的标题
                    llm_title = generate_history_title_from_questions(chat_history)
                    if llm_title:
                        # 更新记录
                        history_dir = "history"
                        record_file = os.path.join(history_dir, "history_records.json")
                        if os.path.exists(record_file):
                            with open(record_file, 'r', encoding='utf-8') as f:
                                records_data = json.load(f)
                            # 找到对应记录并更新
                            for record in records_data:
                                if record.get("conversation_id") == conversation_id:
                                    record["summary"] = llm_title
                                    break
                            # 保存更新后的记录
                            with open(record_file, 'w', encoding='utf-8') as f:
                                json.dump(records_data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error(f"更新历史记录标题失败: {e}")
                import traceback
                logger.error(traceback.format_exc())
        else:
            logger.warning(f"/api/chat 警告: conversation_id 为空或无效，无法保存历史记录")
            logger.warning(f"/api/chat conversation_id 值: '{conversation_id}'")
        
        return JSONResponse({
            "success": True,
            "message": bot_message,
            "conversation_id": conversation_id
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

# 获取树视图
@app.get("/api/tree")
async def get_tree():
    try:
        html = build_new_tree_iframe_html()
        return JSONResponse({
            "success": True,
            "html": html
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

# 获取日志
@app.get("/api/logs")
async def get_logs():
    try:
        import os
        # 调试：检查日志目录和文件
        log_files = []
        if os.path.exists(LOG_DIR):
            log_files = [f for f in os.listdir(LOG_DIR) if f.endswith('.log')]
        # 不再记录这些定期显示的调试信息，减少日志噪音
        # logger.debug(f"日志目录: {LOG_DIR}, 日志文件数量: {len(log_files)}, 文件列表: {log_files}")
        
        logs = read_all_logs(log_dir=LOG_DIR, max_lines=200)
        # logger.debug(f"读取日志结果长度: {len(logs) if logs else 0}")
        
        return JSONResponse({
            "success": True,
            "logs": logs
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

# 获取思维链
@app.get("/api/chain")
async def get_chain():
    try:
        chain = get_thinking_chain()
        return JSONResponse({
            "success": True,
            "chain": chain
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

# 生成对话标题
@app.post("/api/generate-title")
async def generate_title(request: Request):
    try:
        data = await request.json()
        user_question = data.get("question", "")
        
        if not user_question or not user_question.strip():
            return JSONResponse({
                "success": False,
                "error": "Question cannot be empty"
            })
        
        # 使用LLM生成标题
        from core_functions import generate_history_title_from_questions
        chat_history = [{"role": "user", "content": user_question}]
        title = generate_history_title_from_questions(chat_history)
        
        if title:
            return JSONResponse({
                "success": True,
                "title": title
            })
        else:
            # 如果生成失败，使用问题本身（截断）
            fallback_title = user_question[:30] + "..." if len(user_question) > 30 else user_question
            if re.search(r"[\u4e00-\u9fff]", fallback_title):
                fallback_title = "Conversation Summary"
            return JSONResponse({
                "success": True,
                "title": fallback_title
            })
    except Exception as e:
        logger.error(f"生成标题失败: {e}")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

# 清除日志
@app.post("/api/clear")
async def clear_conversation():
    """清除当前对话，开始新对话"""
    try:
        # 调用clear_all清除所有内容
        clear_all()
        
        # 重置树编辑器
        try:
            from tree_editor import editor
            editor.reset()
        except:
            pass  # 如果导入失败，忽略
        
        return JSONResponse({
            "success": True,
            "message": "Conversation cleared. Start a new chat."
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

# 清空所有历史记录
@app.post("/api/clear-history")
async def clear_all_history():
    """清空所有历史记录"""
    try:
        import shutil
        history_dir = "history"
        
        if os.path.exists(history_dir):
            # 删除history目录下的所有内容，但保留目录本身
            for item in os.listdir(history_dir):
                item_path = os.path.join(history_dir, item)
                try:
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                except Exception as e:
                    logger.error(f"删除历史记录项失败 {item_path}: {e}")
        
        # 重新创建空的history_records.json
        record_file = os.path.join(history_dir, "history_records.json")
        with open(record_file, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        
        logger.info("All history records cleared")
        return JSONResponse({
            "success": True,
            "message": "All history records cleared"
        })
    except Exception as e:
        logger.error(f"清空历史记录失败: {e}")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.post("/api/clear-logs")
async def clear_logs():
    try:
        # 清除日志文件内容
        if os.path.exists(LOG_DIR):
            for log_file in os.listdir(LOG_DIR):
                file_path = os.path.join(LOG_DIR, log_file)
                if os.path.isfile(file_path):
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write("")
        return JSONResponse({
            "success": True,
            "message": "日志已清除"
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

# 获取历史记录
@app.get("/api/history")
async def get_history():
    try:
        records = get_conversation_records()
        return JSONResponse({
            "success": True,
            "records": records
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

# 加载历史记录
@app.get("/api/history/{conversation_id}")
async def load_history(conversation_id: str):
    try:
        logger.debug(f"加载历史记录请求 - conversation_id: {conversation_id}")
        history = load_conversation_history(conversation_id)
        logger.debug(f"加载的历史记录数量: {len(history) if isinstance(history, list) else 0}")
        logger.debug(f"历史记录内容: {history}")
        
        # 确保历史记录格式正确
        if not isinstance(history, list):
            logger.warning(f"警告: 历史记录不是列表格式，类型: {type(history)}")
            history = []
        
        return JSONResponse({
            "success": True,
            "history": history
        })
    except Exception as e:
        logger.error(f"加载历史记录异常: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

def preview_excel_rows(file_path: str, max_rows: int = 16, max_cols: int = 12):
    try:
        wb = openpyxl.load_workbook(file_path, data_only=True, read_only=True)
        sheet = wb.active
        rows = []
        for row_cells in sheet.iter_rows(max_row=max_rows, max_col=max_cols, values_only=True):
            if not any(cell is not None and str(cell).strip() for cell in row_cells):
                # skip fully empty rows to keep preview concise
                continue
            rows.append([str(cell) if cell not in (None, "") else "" for cell in row_cells])
        wb.close()
        return {
            "sheet_name": sheet.title,
            "rows": rows
        }
    except Exception as exc:
        logger.warning(f"Cannot preview Excel {file_path}: {exc}")
        return None


@app.get("/api/history/{conversation_id}/table")
async def get_history_table(conversation_id: str):
    history_dir = os.path.join("history", conversation_id)
    if not os.path.isdir(history_dir):
        return JSONResponse({
            "success": False,
            "error": "History directory not found"
        }, status_code=404)

    preferred_temp = os.path.join(history_dir, "temp.xlsx")
    candidates = []
    if os.path.exists(preferred_temp):
        candidates.append(preferred_temp)
    else:
        for name in sorted(os.listdir(history_dir)):
            if name.lower().endswith(".xlsx"):
                candidates.append(os.path.join(history_dir, name))
    if not candidates:
        return JSONResponse({
            "success": False,
            "error": "No Excel files found for this conversation"
        }, status_code=404)

    tables = []
    for path in candidates[:2]:
        preview = preview_excel_rows(path)
        if preview and preview["rows"]:
            tables.append({
                "file": os.path.basename(path),
                "sheet": preview["sheet_name"],
                "rows": preview["rows"]
            })
    if not tables:
        return JSONResponse({
            "success": False,
            "error": "Unable to read table preview"
        }, status_code=500)

    return JSONResponse({
        "success": True,
        "tables": tables
    })

# 保存树结构（用于前端保存功能）
@app.post("/api/save_tree")
async def save_tree(request: Request):
    try:
        from core_functions import rebuild_feature_tree_from_json
        data = await request.json()
        
        # 前端为选中逻辑添加的 id 不落盘，先剥离
        def strip_ids(obj):
            if isinstance(obj, list):
                return [strip_ids(o) for o in obj]
            if isinstance(obj, dict):
                return {k: strip_ids(v) for k, v in obj.items() if k != "id"}
            return obj
        
        cleaned = strip_ids(data)
        ok, msg = rebuild_feature_tree_from_json(cleaned)
        
        if ok:
            return JSONResponse({
                "success": True,
                "message": "树结构保存成功"
            })
        else:
            return JSONResponse({
                "success": False,
                "error": msg
            }, status_code=500)
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

if __name__ == "__main__":
    # 启动时清理日志
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
    
    print("🚀 启动 ST-Raptor Web 界面...")
    print("📋 访问地址: http://localhost:7860")
    print("⏹️  按 Ctrl+C 停止服务")
    
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")
