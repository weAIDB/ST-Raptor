import os
import json
import re
import ast
import hashlib
import shutil
import sys
import asyncio
import uuid
import pickle
from datetime import datetime
import openpyxl
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import Any, Dict, List, Optional
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
from query.trace_builder import build_typed_trace_v2, build_trace_v3
from utils.tree_semantic_utils import (
    build_flat_column_alias_target_map,
    build_flat_row_alias_target_map,
    build_nested_index_projection_map,
    build_semantic_projection_bundle,
    build_typed_body_id,
    build_typed_body_segment,
    build_typed_index_id,
    build_typed_index_segment,
    build_typed_root_parts,
    build_typed_tree_v2,
    make_canonical_trace_id,
)

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

os.makedirs("history", exist_ok=True)

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
if os.path.exists("history"):
    app.mount("/history-assets", StaticFiles(directory="history"), name="history-assets")

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


def _sync_tree_snapshot_for_history(conversation_id: str):
    """Best-effort sync without overriding canonical history artifacts."""
    if not conversation_id:
        return
    history_dir = os.path.join("history", conversation_id)
    os.makedirs(history_dir, exist_ok=True)
    # 注意：history/<conversation_id> 下的 temp.column.json / temp1.json / temp.artifacts.json / temp.id_mappings.json
    # 是该会话的 canonical 产物，不应被 cache 下的临时前端结构覆盖。
    def _safe_copy_if_missing(src_name: str, dst_name: str, validator=None):
        source_path = os.path.join("cache", src_name)
        target_path = os.path.join(history_dir, dst_name)
        if os.path.exists(target_path):
            logger.info(f"[history_sync] skip existing canonical file: {target_path}")
            return
        if not os.path.exists(source_path):
            return
        if validator is not None:
            try:
                with open(source_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                if not validator(raw):
                    logger.warning(f"[history_sync] source shape mismatch, skip: {source_path}")
                    return
            except Exception as e:
                logger.warning(f"[history_sync] source validate failed {source_path}: {e}")
                return
        try:
            shutil.copy2(source_path, target_path)
            logger.info(f"[history_sync] copied {source_path} -> {target_path}")
        except Exception as e:
            logger.warning(f"[history_sync] copy failed {source_path} -> {target_path}: {e}")

    _safe_copy_if_missing("temp.column.json", "temp.column.json", validator=lambda x: isinstance(x, dict))
    _safe_copy_if_missing("temp1.json", "temp1.json", validator=lambda x: isinstance(x, dict))
    _safe_copy_if_missing("temp.artifacts.json", "temp.artifacts.json", validator=lambda x: isinstance(x, dict))
    _safe_copy_if_missing("temp.id_mappings.json", "temp.id_mappings.json", validator=lambda x: isinstance(x, dict))


def _history_tree_chat_path(conversation_id: str) -> str:
    history_dir = os.path.join("history", conversation_id)
    os.makedirs(history_dir, exist_ok=True)
    return os.path.join(history_dir, "tree_chat.json")


def _history_images_dir(conversation_id: str) -> str:
    history_dir = os.path.join("history", conversation_id)
    images_dir = os.path.join(history_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    return images_dir


def _history_tree_images_path(conversation_id: str) -> str:
    history_dir = os.path.join("history", conversation_id)
    os.makedirs(history_dir, exist_ok=True)
    return os.path.join(history_dir, "tree_images.json")


def _looks_like_empty_tree_payload(raw: Any) -> bool:
    if raw is None:
        return True
    if isinstance(raw, list):
        return len(raw) == 0
    if isinstance(raw, dict):
        return len(raw) == 0
    return False


def _collect_frontend_tree_node_catalog(tree_node: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not isinstance(tree_node, dict):
        return rows

    def walk(node: Dict[str, Any], parent_id: str = "", depth: int = 0) -> None:
        if not isinstance(node, dict):
            return
        node_id = str(node.get("id", "") or "").strip()
        canonical_id = str(node.get("canonicalId") or node.get("canonicalTraceId") or "").strip()
        group_canonical_id = str(node.get("groupCanonicalId") or node.get("traceGroupCanonicalId") or "").strip()
        rows.append({
            "id": node_id,
            "canonicalId": canonical_id,
            "groupCanonicalId": group_canonical_id,
            "parentId": str(parent_id or ""),
            "nodeType": str(node.get("nodeType", "") or ""),
            "sourceKind": str(node.get("sourceKind", "") or ""),
            "name": str(node.get("name", "") or ""),
            "depth": int(depth),
        })
        for child in node.get("children", []) or []:
            if isinstance(child, dict):
                walk(child, node_id, depth + 1)

    walk(tree_node, "", 0)
    return rows


def _write_debug_frontend_node_catalog(conversation_id: str, view_mode: str, tree_node: Optional[Dict[str, Any]]) -> str:
    conversation_id = str(conversation_id or "").strip()
    if not conversation_id or not isinstance(tree_node, dict):
        return ""
    try:
        debug_dir = os.path.join("history", conversation_id)
        os.makedirs(debug_dir, exist_ok=True)
        normalized_mode = str(view_mode or "").strip().lower() or "unknown"
        file_name = f"[debug]frontend.node.catalog.{normalized_mode}.json"
        path = os.path.join(debug_dir, file_name)
        nodes = _collect_frontend_tree_node_catalog(tree_node)
        payload = {
            "view_mode": normalized_mode,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "root_id": str(tree_node.get("id", "") or ""),
            "root_canonical_id": str(tree_node.get("canonicalId") or tree_node.get("canonicalTraceId") or ""),
            "node_count": len(nodes),
            "nodes": nodes,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info(
            f"【debug】frontend_node_catalog saved mode={normalized_mode}, "
            f"path={path}, node_count={len(nodes)}"
        )
        return path
    except Exception as e:
        logger.warning(f"【debug】frontend_node_catalog save failed mode={view_mode}: {e}")
        return ""


def _build_flat_tree_node_for_view_mode(conversation_id: str, view_mode: str = "row") -> Optional[Dict[str, Any]]:
    normalized_mode = str(view_mode or "").strip().lower()
    if normalized_mode not in {"row", "column"}:
        normalized_mode = "row"
    if normalized_mode == "column":
        raw_column = _load_column_view_payload(conversation_id)
        if raw_column is None:
            return None
        return _build_flat_index_body_tree(raw_column, root_name="flat column view", path_parts=["root", "flat_column"])
    raw_row = _load_row_view_payload(conversation_id)
    if raw_row is None:
        return None
    return _build_flat_row_tree_with_trace_metadata(raw_row)


def _rebuild_history_tree_snapshot_from_files(conversation_id: str) -> bool:
    """
    Rebuild history/<conversation_id>/temp.column.json and temp1.json from raw files.
    """
    conversation_id = str(conversation_id or "").strip()
    if not conversation_id:
        return False
    history_dir = os.path.join("history", conversation_id)
    if not os.path.isdir(history_dir):
        return False
    try:
        from file_handlers import merge_multiple_tables_to_tree
        import types
    except Exception as e:
        logger.error(f"重建历史树失败（导入模块失败）: {e}")
        return False

    candidate_files: List[Any] = []
    supported_ext = {".xlsx", ".xls", ".docx", ".doc", ".txt", ".md", ".json"}
    for name in os.listdir(history_dir):
        full_path = os.path.join(history_dir, name)
        if not os.path.isfile(full_path):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext in supported_ext:
            candidate_files.append(types.SimpleNamespace(name=full_path))

    if not candidate_files:
        return False

    try:
        merged_data, processed_files, _failed_files = merge_multiple_tables_to_tree(candidate_files, conversation_id=conversation_id)
        if not processed_files:
            return False
        return bool(merged_data)
    except Exception as e:
        logger.error(f"重建历史树失败: {e}")
        return False

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
        _sync_tree_snapshot_for_history(conversation_id)
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
def _chat_core(
    message: str,
    conversation_id: str = "",
    temperature: float = 0.5,
    max_tokens: int = 1024,
    files: Optional[List[UploadFile]] = None
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
            _sync_tree_snapshot_for_history(conversation_id)
            
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
                    default_summary = "New Conversation"
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
        
        return {
            "success": True,
            "message": bot_message,
            "conversation_id": conversation_id
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.post("/api/chat")
async def chat(
    request: Request,
    message: str = Form(...),
    conversation_id: str = Form(""),
    temperature: float = Form(0.5),
    max_tokens: int = Form(1024),
    files: Optional[List[UploadFile]] = File(None)
):
    result = _chat_core(
        message=message,
        conversation_id=conversation_id,
        temperature=temperature,
        max_tokens=max_tokens,
        files=files
    )
    if result.get("success"):
        return JSONResponse(result)
    return JSONResponse(result, status_code=500)


@app.post("/api/chat-stream")
async def chat_stream(
    request: Request,
    message: str = Form(...),
    conversation_id: str = Form(""),
    temperature: float = Form(0.5),
    max_tokens: int = Form(1024),
    files: Optional[List[UploadFile]] = File(None)
):
    def _logs_html_to_lines(logs_html: str):
        if not logs_html:
            return []
        # read_all_logs 返回的是 HTML，需要转为纯文本行
        text = re.sub(r"<br\s*/?>", "\n", str(logs_html), flags=re.IGNORECASE)
        text = re.sub(r"</pre>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", "", text)
        return [line.strip() for line in text.split("\n") if line and line.strip()]

    async def event_generator():
        # 在后台线程执行耗时问答，主协程持续推送日志
        chat_task = asyncio.create_task(asyncio.to_thread(
            _chat_core,
            message,
            conversation_id,
            temperature,
            max_tokens,
            files
        ))
        last_lines = []

        try:
            while not chat_task.done():
                try:
                    logs_html = read_all_logs(log_dir=LOG_DIR, max_lines=260)
                except Exception:
                    logs_html = ""
                current_lines = _logs_html_to_lines(logs_html)
                if current_lines:
                    # 如果日志轮转/截断，通知前端重置
                    if len(current_lines) < len(last_lines) or current_lines[:len(last_lines)] != last_lines:
                        payload = {"type": "log_lines", "reset": True, "lines": current_lines[-120:]}
                        yield (json.dumps(payload, ensure_ascii=False) + "\n")
                        last_lines = current_lines
                    elif len(current_lines) > len(last_lines):
                        new_lines = current_lines[len(last_lines):]
                        payload = {"type": "log_lines", "reset": False, "lines": new_lines}
                        yield (json.dumps(payload, ensure_ascii=False) + "\n")
                        last_lines = current_lines
                await asyncio.sleep(0.9)

            result = await chat_task
            try:
                final_logs = read_all_logs(log_dir=LOG_DIR, max_lines=320)
                final_lines = _logs_html_to_lines(final_logs)
                if final_lines:
                    if len(final_lines) < len(last_lines) or final_lines[:len(last_lines)] != last_lines:
                        payload = {"type": "log_lines", "reset": True, "lines": final_lines[-160:]}
                        yield (json.dumps(payload, ensure_ascii=False) + "\n")
                    elif len(final_lines) > len(last_lines):
                        payload = {"type": "log_lines", "reset": False, "lines": final_lines[len(last_lines):]}
                        yield (json.dumps(payload, ensure_ascii=False) + "\n")
            except Exception:
                pass

            done_payload = {"type": "done", **result}
            yield (json.dumps(done_payload, ensure_ascii=False) + "\n")
        except Exception as e:
            err_payload = {"type": "done", "success": False, "error": str(e)}
            yield (json.dumps(err_payload, ensure_ascii=False) + "\n")

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")

# 获取树视图
@app.get("/api/tree")
async def get_tree(conversation_id: str = ""):
    try:
        data_path = _resolve_column_view_path(conversation_id)
        if conversation_id and not os.path.exists(data_path):
            _rebuild_history_tree_snapshot_from_files(conversation_id)
            data_path = _resolve_column_view_path(conversation_id)
        trace_table_scope = ""
        try:
            chain_data = get_thinking_chain() or {}
            qa_info = (chain_data.get("question_answering", {}) if isinstance(chain_data, dict) else {}) or {}
            trace_table_scope = str(qa_info.get("table_scope", "") or "").strip()
        except Exception:
            trace_table_scope = ""
        typed_payload = _load_typed_tree_v2_payload(conversation_id, table_scope=trace_table_scope)
        if typed_payload:
            html = build_new_tree_iframe_html(initial_data_path=data_path, initial_data=typed_payload)
        else:
            raw_column = _load_column_view_payload(conversation_id)
            if raw_column is not None:
                tree_node = _build_column_feature_tree_node(raw_column, ["root", "column_view"])
                html = build_new_tree_iframe_html(initial_data_path=data_path, initial_data=[tree_node])
            else:
                html = build_new_tree_iframe_html(initial_data_path=data_path)
        return JSONResponse({
            "success": True,
            "html": html
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


@app.get("/api/tree-column")
async def get_tree_column(conversation_id: str = ""):
    # backward-compatible alias: nested view from column JSON
    return await get_tree_nested(conversation_id)


@app.get("/api/tree-nested")
async def get_tree_nested(conversation_id: str = ""):
    try:
        raw_column = _load_column_view_payload(conversation_id)
        if raw_column is None:
            return JSONResponse({
                "success": False,
                "error": "Column view JSON not found"
            }, status_code=404)

        data_path = _resolve_column_view_path(conversation_id)
        tree_node = _build_column_feature_tree_node(raw_column, ["root", "column_view"])
        html = build_new_tree_iframe_html(initial_data_path=data_path, initial_data=[tree_node])
        return JSONResponse({
            "success": True,
            "html": html
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


@app.get("/api/tree-nested-data")
async def get_tree_nested_data(conversation_id: str = ""):
    """
    Return raw column-view JSON for custom nested renderer.
    """
    try:
        raw_column = _load_column_view_payload(conversation_id)
        if raw_column is None:
            return JSONResponse({
                "success": False,
                "error": "Column view JSON not found"
            }, status_code=404)
        return JSONResponse({
            "success": True,
            "data": raw_column
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


@app.get("/api/tree-flat-column")
async def get_tree_flat_column(conversation_id: str = ""):
    try:
        raw_column = _load_column_view_payload(conversation_id)
        if raw_column is None:
            return JSONResponse({
                "success": False,
                "error": "Column view JSON not found"
            }, status_code=404)

        data_path = _resolve_column_view_path(conversation_id)
        tree_node = _build_flat_index_body_tree(raw_column, root_name="flat column view", path_parts=["root", "flat_column"])
        debug_catalog_path = _write_debug_frontend_node_catalog(conversation_id, "column", tree_node)
        html = build_new_tree_iframe_html(initial_data_path=data_path, initial_data=[tree_node])
        return JSONResponse({
            "success": True,
            "html": html,
            "debug_frontend_node_catalog": debug_catalog_path,
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


@app.get("/api/tree-flat-row")
async def get_tree_flat_row(conversation_id: str = ""):
    try:
        raw_row = _load_row_view_payload(conversation_id)
        if raw_row is None:
            return JSONResponse({
                "success": False,
                "error": "Row view JSON not found"
            }, status_code=404)

        data_path = _resolve_row_view_path(conversation_id)
        tree_node = _build_flat_row_tree_with_trace_metadata(raw_row)
        debug_catalog_path = _write_debug_frontend_node_catalog(conversation_id, "row", tree_node)
        html = build_new_tree_iframe_html(initial_data_path=data_path, initial_data=[tree_node])
        return JSONResponse({
            "success": True,
            "html": html,
            "debug_frontend_node_catalog": debug_catalog_path,
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


def _normalize_text_for_trace(text: Any) -> str:
    return re.sub(r"\s+", "", str(text or "").lower()).strip()


def _make_trace_node_id(path_parts: List[str]) -> str:
    raw = "|".join(path_parts) if path_parts else "root"
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", raw)
    safe = re.sub(r"_+", "_", safe).strip("_")
    return f"n_{safe or 'root'}"


def _make_tree_canonical_id(path_parts: List[str]) -> str:
    return make_canonical_trace_id(["tree"] + list(path_parts or []))


def _make_tree_group_canonical_id(path_parts: List[str]) -> str:
    return make_canonical_trace_id(["tree_group"] + list(path_parts or []))


def _append_trace_alias(node: Dict[str, Any], alias: Any) -> None:
    text = str(alias or "").strip()
    if not text:
        return
    aliases = node.setdefault("traceAliases", [])
    if text not in aliases:
        aliases.append(text)


def _get_node_canonical_id(node: Dict[str, Any]) -> str:
    if not isinstance(node, dict):
        return ""
    return str(node.get("canonicalId") or node.get("canonicalTraceId") or "").strip()


def _get_node_group_canonical_id(node: Dict[str, Any]) -> str:
    if not isinstance(node, dict):
        return ""
    return str(node.get("groupCanonicalId") or node.get("traceGroupCanonicalId") or "").strip()


def _infer_trace_target_kind(canonical_id: Any) -> str:
    text = str(canonical_id or "").strip()
    if not text:
        return ""
    if text.startswith("ct_tree_group_") or text.startswith("ct_semantic_group_"):
        return "group"
    return "node"


def _sync_tree_trace_identity_fields(tree_node: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(tree_node, dict):
        return tree_node

    def sync(node: Dict[str, Any]) -> None:
        canonical_id = _get_node_canonical_id(node)
        group_canonical_id = _get_node_group_canonical_id(node)
        if canonical_id:
            node["canonicalId"] = canonical_id
            node["canonicalTraceId"] = canonical_id
        if group_canonical_id:
            node["groupCanonicalId"] = group_canonical_id
            node["traceGroupCanonicalId"] = group_canonical_id

    _walk_tree_nodes(tree_node, sync)
    return tree_node


def _compute_tree_trace_fingerprint(tree_node: Optional[Dict[str, Any]]) -> str:
    if not isinstance(tree_node, dict):
        return ""
    records: List[Dict[str, str]] = []

    def walk(node: Dict[str, Any], parent_canonical_id: str = "") -> None:
        canonical_id = _get_node_canonical_id(node)
        group_canonical_id = _get_node_group_canonical_id(node)
        if canonical_id or group_canonical_id:
            records.append({
                "canonical_id": canonical_id,
                "group_canonical_id": group_canonical_id,
                "parent_canonical_id": str(parent_canonical_id or ""),
                "node_type": str(node.get("nodeType", "") or ""),
            })
        next_parent = canonical_id or parent_canonical_id
        for child in node.get("children", []) or []:
            if isinstance(child, dict):
                walk(child, next_parent)

    walk(tree_node, "")
    if not records:
        return ""
    records.sort(
        key=lambda item: (
            str(item.get("canonical_id", "") or ""),
            str(item.get("group_canonical_id", "") or ""),
            str(item.get("parent_canonical_id", "") or ""),
            str(item.get("node_type", "") or ""),
        )
    )
    payload = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return f"tf_{digest}"


def _walk_tree_nodes(node: Any, visit) -> None:
    if not isinstance(node, dict):
        return
    visit(node)
    for child in node.get("children", []) or []:
        _walk_tree_nodes(child, visit)


def _index_tree_nodes_by_canonical(tree_node: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    mapping: Dict[str, Dict[str, Any]] = {}

    def collect(node: Dict[str, Any]) -> None:
        canonical_id = _get_node_canonical_id(node)
        if canonical_id:
            mapping[canonical_id] = node

    _walk_tree_nodes(tree_node, collect)
    return mapping


def _build_trace_alias_target_map(tree_node: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    alias_hits: Dict[str, List[Dict[str, str]]] = {}
    if not isinstance(tree_node, dict):
        return {}

    def collect(node: Dict[str, Any]) -> None:
        canonical_id = _get_node_canonical_id(node)
        group_canonical_id = _get_node_group_canonical_id(node)
        if not canonical_id and not group_canonical_id:
            return
        for alias in node.get("traceAliases", []) or []:
            alias_text = str(alias or "").strip()
            if not alias_text:
                continue
            alias_hits.setdefault(alias_text, []).append({
                "canonical": canonical_id,
                "group": group_canonical_id,
            })

    _walk_tree_nodes(tree_node, collect)
    resolved: Dict[str, Dict[str, str]] = {}
    for alias, hits in alias_hits.items():
        concrete = sorted({str(hit.get("canonical", "") or "").strip() for hit in hits if str(hit.get("canonical", "") or "").strip()})
        groups = sorted({str(hit.get("group", "") or "").strip() for hit in hits if str(hit.get("group", "") or "").strip()})
        if len(concrete) == 1:
            resolved[alias] = {
                "canonical_id": concrete[0],
                "target_kind": "node",
            }
            continue
        if len(groups) == 1:
            resolved[alias] = {
                "canonical_id": groups[0],
                "target_kind": "group",
            }
    return resolved


def _extract_primary_result_text(value: Any) -> str:
    if isinstance(value, list):
        for item in value:
            text = _extract_primary_result_text(item)
            if text:
                return text
        return ""
    if isinstance(value, dict):
        results = value.get("results", [])
        if isinstance(results, list):
            text = _extract_primary_result_text(results)
            if text:
                return text
        for key in ("value", "preview", "label"):
            text = str(value.get(key, "") or "").strip()
            if text:
                return text
        return ""
    return str(value or "").strip()


def _flatten_strings_for_trace(value: Any, collector: List[str]) -> None:
    if value is None:
        return
    if isinstance(value, (str, int, float, bool)):
        s = str(value).strip()
        if s:
            collector.append(s)
        return
    if isinstance(value, list):
        for item in value:
            _flatten_strings_for_trace(item, collector)
        return
    if isinstance(value, dict):
        for item in value.values():
            _flatten_strings_for_trace(item, collector)


def _extract_bracket_tokens_for_trace(value: Any) -> List[str]:
    text = str(value or "")
    matches = re.findall(r"\[(.*?)\]", text)
    stop = {"CHL", "FAT", "EXT", "COND", "FOREACH", "CMP", "END", "N", "EQ", "LT", "GT", "LTE", "GTE"}
    tokens: List[str] = []
    for token in matches:
        tk = str(token).strip()
        if not tk or len(tk) <= 1:
            continue
        if tk.upper() in stop:
            continue
        tokens.append(tk)
    return tokens


def _build_trace_node_index(raw_tree: Any) -> Dict[str, Dict[str, Any]]:
    nodes: Dict[str, Dict[str, Any]] = {}

    def add_node(node_id: str, name: str, parent_id: Optional[str]) -> None:
        if node_id in nodes:
            return
        nodes[node_id] = {"id": node_id, "name": str(name or ""), "parent": parent_id}

    def walk(obj: Any, fallback_name: str, path_parts: List[str], parent_id: Optional[str]) -> Optional[str]:
        if obj is None:
            return None

        if isinstance(obj, dict) and ("name" in obj or "children" in obj):
            name = str(obj.get("name", fallback_name) or "未命名节点")
            canonical_id = str(obj.get("canonicalId") or obj.get("canonicalTraceId") or "").strip()
            node_id = canonical_id or _make_trace_node_id(path_parts)
            add_node(node_id, name, parent_id)
            children = obj.get("children", [])
            if isinstance(children, list):
                for idx, child in enumerate(children):
                    walk(child, f"[{idx}]", path_parts + [f"c_{idx}"], node_id)
            return node_id

        if isinstance(obj, dict):
            node_id = _make_trace_node_id(path_parts)
            add_node(node_id, str(fallback_name or "未命名节点"), parent_id)
            for key, value in obj.items():
                walk(value, str(key), path_parts + [f"k_{str(key)}"], node_id)
            return node_id

        if isinstance(obj, list):
            node_id = _make_trace_node_id(path_parts)
            add_node(node_id, str(fallback_name or "未命名节点"), parent_id)
            for idx, item in enumerate(obj):
                walk(item, f"[{idx}]", path_parts + [f"i_{idx}"], node_id)
            return node_id

        has_named_wrapper = bool(fallback_name and fallback_name != "未命名节点")
        if has_named_wrapper:
            wrapper_id = _make_trace_node_id(path_parts)
            add_node(wrapper_id, str(fallback_name), parent_id)
            value_id = _make_trace_node_id(path_parts + ["v"])
            add_node(value_id, str(obj), wrapper_id)
            return wrapper_id

        node_id = _make_trace_node_id(path_parts)
        add_node(node_id, str(obj), parent_id)
        return node_id

    if isinstance(raw_tree, list):
        canonical = all(isinstance(item, dict) and ("name" in item or "children" in item) for item in raw_tree)
        for idx, item in enumerate(raw_tree):
            if canonical:
                walk(item, "未命名节点", ["root", f"r_{idx}"], None)
            else:
                walk(item, f"[{idx}]", ["root", f"r_{idx}"], None)
    elif isinstance(raw_tree, dict):
        canonical = ("name" in raw_tree) or ("children" in raw_tree)
        if canonical:
            walk(raw_tree, "未命名节点", ["root"], None)
        else:
            for key, value in raw_tree.items():
                walk(value, str(key), ["root", f"k_{str(key)}"], None)

    return nodes


def _resolve_trace_tree_path(conversation_id: str = "") -> str:
    path = os.path.join("cache", "temp1.json")
    if conversation_id:
        candidate = os.path.join("history", conversation_id, "temp1.json")
        if os.path.exists(candidate):
            path = candidate
        else:
            fallback = os.path.join("history", conversation_id, "temp.column.json")
            if os.path.exists(fallback):
                path = fallback
    return path


def _resolve_feature_tree_pkl_path(conversation_id: str = "") -> str:
    path = os.path.join("cache", "temp.pkl")
    if conversation_id:
        candidate = os.path.join("history", conversation_id, "temp.pkl")
        if os.path.exists(candidate):
            path = candidate
        else:
            manifest_path = os.path.join("history", conversation_id, "temp.artifacts.json")
            if os.path.exists(manifest_path):
                try:
                    with open(manifest_path, "r", encoding="utf-8") as f:
                        manifest = json.load(f)
                    if isinstance(manifest, dict):
                        for item in manifest.values():
                            if not isinstance(item, dict):
                                continue
                            pkl_name = str(item.get("pkl", "") or "")
                            if not pkl_name:
                                continue
                            pkl_candidate = os.path.join("history", conversation_id, pkl_name)
                            if os.path.exists(pkl_candidate):
                                path = pkl_candidate
                                break
                except Exception:
                    pass
    return path


def _resolve_column_view_path(conversation_id: str = "") -> str:
    path = os.path.join("cache", "temp.column.json")
    if conversation_id:
        candidate = os.path.join("history", conversation_id, "temp.column.json")
        if os.path.exists(candidate):
            path = candidate
    return path


def _resolve_row_view_path(conversation_id: str = "") -> str:
    path = os.path.join("cache", "temp1.json")
    if conversation_id:
        candidate = os.path.join("history", conversation_id, "temp1.json")
        if os.path.exists(candidate):
            path = candidate
    return path


def _load_column_view_payload(conversation_id: str = "") -> Optional[Any]:
    """
    Load temp.column.json only.
    """
    path = _resolve_column_view_path(conversation_id)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"读取列视图JSON失败: {e}")
    return None


def _load_row_view_payload(conversation_id: str = "") -> Optional[Any]:
    path = _resolve_row_view_path(conversation_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"读取行视图JSON失败: {e}")
        return None


def _apply_table_scope_to_payload(payload: Any, table_scope: Any) -> Any:
    scope = str(table_scope or "").strip()
    if not scope or not isinstance(payload, dict):
        return payload
    if scope in payload:
        return payload
    return {scope: payload}


def _build_column_feature_tree_node(data: Any, path_parts: List[str]) -> Dict[str, Any]:
    """
    Convert column-view JSON to a canonical frontend tree.
    Rule:
    - dict => "feature tree" container
      - key => M_NODE(index)
      - value => B_NODE(body), recursively expanded
    """
    def make_id(parts: List[str]) -> str:
        return _make_trace_node_id(parts)

    def make_canonical_id(parts: List[str]) -> str:
        return _make_tree_canonical_id(parts)

    def build_body_nodes(value: Any, parts: List[str]) -> List[Dict[str, Any]]:
        if isinstance(value, dict):
            return [{
                "id": make_id(parts + ["b_subtree"]),
                "canonicalTraceId": make_canonical_id(parts + ["b_subtree"]),
                "name": "feature tree",
                "nodeType": "B_NODE",
                "sourceKind": "subtree_ref",
                "children": [_build_column_feature_tree_node(value, parts + ["feature_tree"])],
            }]
        if isinstance(value, list):
            body_nodes: List[Dict[str, Any]] = []
            for idx, item in enumerate(value):
                item_parts = parts + [f"i_{idx}"]
                if isinstance(item, dict):
                    body_nodes.append({
                        "id": make_id(item_parts + ["b_subtree"]),
                        "canonicalTraceId": make_canonical_id(item_parts + ["b_subtree"]),
                        "name": f"feature tree[{idx}]",
                        "nodeType": "B_NODE",
                        "sourceKind": "subtree_ref",
                        "children": [_build_column_feature_tree_node(item, item_parts + ["feature_tree"])],
                    })
                else:
                    body_nodes.append({
                        "id": make_id(item_parts),
                        "canonicalTraceId": make_canonical_id(item_parts),
                        "name": str(item),
                        "nodeType": "B_NODE",
                        "sourceKind": "value_leaf",
                        "children": [],
                    })
            return body_nodes
        return [{
            "id": make_id(parts + ["v"]),
            "canonicalTraceId": make_canonical_id(parts + ["v"]),
            "name": str(value),
            "nodeType": "B_NODE",
            "sourceKind": "value_leaf",
            "children": [],
        }]

    root_node = {
        "id": make_id(path_parts),
        "canonicalTraceId": make_canonical_id(path_parts),
        "name": "feature tree",
        "nodeType": "FEATURE_TREE",
        "sourceKind": "feature_tree",
        "children": [],
    }
    if not isinstance(data, dict):
        root_node["children"] = build_body_nodes(data, path_parts + ["value"])
        return root_node

    for idx, (key, value) in enumerate(data.items()):
        key_text = str(key)
        index_parts = path_parts + [f"k_{key_text}", f"idx_{idx}"]
        index_node = {
            "id": make_id(index_parts),
            "canonicalTraceId": make_canonical_id(index_parts),
            "name": key_text,
            "nodeType": "M_NODE",
            "sourceKind": "index_node",
            "children": build_body_nodes(value, index_parts + ["body"]),
        }
        root_node["children"].append(index_node)
    return root_node


def _build_flat_index_body_tree(data: Any, root_name: str = "flat view", path_parts: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Flatten any dict/list json into index/body graph:
    - dict key => M_NODE
    - list/scalar entries => B_NODE
    - preserve recursive expansion but hide feature-tree wrapper semantics
    """
    base = path_parts[:] if isinstance(path_parts, list) else ["root", "flat_view"]

    def make_id(parts: List[str]) -> str:
        return _make_trace_node_id(parts)

    def make_canonical_id(parts: List[str]) -> str:
        return _make_tree_canonical_id(parts)

    def build_from_value(value: Any, parts: List[str]) -> List[Dict[str, Any]]:
        if isinstance(value, dict):
            children: List[Dict[str, Any]] = []
            for idx, (k, v) in enumerate(value.items()):
                kp = parts + [f"k_{str(k)}", f"idx_{idx}"]
                children.append({
                    "id": make_id(kp),
                    "canonicalTraceId": make_canonical_id(kp),
                    "name": str(k),
                    "nodeType": "M_NODE",
                    "sourceKind": "index_node",
                    "children": build_from_value(v, kp + ["body"]),
                })
            return children
        if isinstance(value, list):
            nodes: List[Dict[str, Any]] = []
            for idx, item in enumerate(value):
                ip = parts + [f"i_{idx}"]
                if isinstance(item, dict):
                    nodes.append({
                        "id": make_id(ip),
                        "canonicalTraceId": make_canonical_id(ip),
                        "name": f"item[{idx}]",
                        "nodeType": "B_NODE",
                        "sourceKind": "body_group",
                        "children": build_from_value(item, ip + ["group"]),
                    })
                elif isinstance(item, list):
                    nodes.append({
                        "id": make_id(ip),
                        "canonicalTraceId": make_canonical_id(ip),
                        "name": f"item[{idx}]",
                        "nodeType": "B_NODE",
                        "sourceKind": "body_group",
                        "children": build_from_value(item, ip + ["group"]),
                    })
                else:
                    nodes.append({
                        "id": make_id(ip),
                        "canonicalTraceId": make_canonical_id(ip),
                        "name": str(item),
                        "nodeType": "B_NODE",
                        "sourceKind": "value_leaf",
                        "children": [],
                    })
            return nodes
        return [{
            "id": make_id(parts + ["v"]),
            "canonicalTraceId": make_canonical_id(parts + ["v"]),
            "name": str(value),
            "nodeType": "B_NODE",
            "sourceKind": "value_leaf",
            "children": [],
        }]

    tree_node = {
        "id": make_id(base),
        "canonicalTraceId": make_canonical_id(base),
        "name": root_name,
        "nodeType": "FEATURE_TREE",
        "sourceKind": "flat_root",
        "children": build_from_value(data, base + ["root"]),
    }
    return _sync_tree_trace_identity_fields(tree_node) or tree_node


def _attach_flat_row_trace_aliases(tree_node: Dict[str, Any], raw_row: Any, typed_root_name: str = "HO_TREE") -> None:
    if not isinstance(tree_node, dict) or not isinstance(raw_row, dict):
        return
    node_by_canonical = _index_tree_nodes_by_canonical(tree_node)
    typed_root_parts = build_typed_root_parts(typed_root_name)
    base = ["root", "flat_row"]

    def walk(value: Any, ct_parts: List[str], ft_parts: List[str]):
        if isinstance(value, dict):
            for idx, (k, v) in enumerate(value.items()):
                k_str = str(k)
                cur_ct_parts = ct_parts + [f"k_{k_str}", f"idx_{idx}"]
                cur_ft_parts = ft_parts + [build_typed_index_segment(idx, k_str)]
                
                k_node = node_by_canonical.get(_make_tree_canonical_id(cur_ct_parts))
                if k_node:
                    _append_trace_alias(k_node, build_typed_index_id(ft_parts, idx, k_str))
                
                if isinstance(v, dict) or isinstance(v, list):
                    v_node = node_by_canonical.get(_make_tree_canonical_id(cur_ct_parts + ["body"]))
                    if v_node:
                        _append_trace_alias(v_node, build_typed_body_id(cur_ft_parts, 0, k_str, None, is_subtree=True))
                    walk(v, cur_ct_parts + ["body"], cur_ft_parts + [build_typed_body_segment(0, k_str, None, is_subtree=True)])
                else:
                    v_node = node_by_canonical.get(_make_tree_canonical_id(cur_ct_parts + ["body", "v"]))
                    if v_node:
                        _append_trace_alias(v_node, build_typed_body_id(cur_ft_parts, 0, k_str, v, is_subtree=False))

        elif isinstance(value, list):
            columns = []
            for row in value:
                if isinstance(row, dict):
                    for k in row.keys():
                        if k not in columns:
                            columns.append(k)
                            
            for row_idx, row in enumerate(value):
                if not isinstance(row, dict):
                    continue
                cur_ct_parts = ct_parts + [f"i_{row_idx}"]
                group_ct_parts = cur_ct_parts + ["group"]
                
                for k, v in row.items():
                    k_str = str(k)
                    col_idx = columns.index(k)
                    
                    cell_key_ct_parts = group_ct_parts + [f"k_{k_str}", f"idx_{col_idx}"]
                    cell_val_ct_parts = cell_key_ct_parts + ["body", "v"]
                    group_canonical_id = _make_tree_group_canonical_id(
                        ct_parts + ["header_group", f"k_{k_str}", f"idx_{col_idx}"]
                    )
                    
                    k_node = node_by_canonical.get(_make_tree_canonical_id(cell_key_ct_parts))
                    if k_node:
                        k_node["traceGroupCanonicalId"] = group_canonical_id
                        _append_trace_alias(k_node, build_typed_index_id(ft_parts, col_idx, k_str))
                    
                    v_node = node_by_canonical.get(_make_tree_canonical_id(cell_val_ct_parts))
                    if v_node:
                        # Row-view group highlight should light up all cells in this column.
                        # Attach the same header-level group canonical id to each row cell value.
                        v_node["traceGroupCanonicalId"] = group_canonical_id
                        col_ft_parts = ft_parts + [build_typed_index_segment(col_idx, k_str)]
                        _append_trace_alias(v_node, build_typed_body_id(col_ft_parts, row_idx, k_str, v, is_subtree=False))

    walk(raw_row, base + ["root"], typed_root_parts)


def _build_flat_row_tree_with_trace_metadata(raw_row: Any) -> Dict[str, Any]:
    tree_node = _build_flat_index_body_tree(raw_row, root_name="flat row view", path_parts=["root", "flat_row"])
    _attach_flat_row_trace_aliases(tree_node, raw_row)
    _sync_tree_trace_identity_fields(tree_node)
    return tree_node


def _find_flat_row_ext_semantic_targets(raw_row: Any, row_anchor: str, column_anchor: str, result_text: str = "") -> Dict[str, Any]:
    if not isinstance(raw_row, dict):
        return {}
    base = ["root", "flat_row"]
    row_anchor_text = str(row_anchor or "").strip()
    column_anchor_text = str(column_anchor or "").strip()
    result_text = str(result_text or "").strip()
    if not row_anchor_text or not column_anchor_text:
        return {}

    def _same_value(actual_text: str, expected_text: str) -> bool:
        if not expected_text:
            return True
        if actual_text == expected_text:
            return True
        try:
            a = float(str(actual_text).replace(",", ""))
            b = float(str(expected_text).replace(",", ""))
            return abs(a - b) < 1e-9
        except Exception:
            return False

    table_candidates: List[Tuple[List[str], Any]] = []
    def find_tables(value: Any, ct_parts: List[str]):
        if isinstance(value, dict):
            for idx, (k, v) in enumerate(value.items()):
                k_str = str(k)
                cur_ct_parts = ct_parts + [f"k_{k_str}", f"idx_{idx}"]
                if isinstance(v, list):
                    find_tables(v, cur_ct_parts + ["body"])
                elif isinstance(v, dict):
                    find_tables(v, cur_ct_parts + ["body"])
        elif isinstance(value, list):
            table_candidates.append((ct_parts, value))
            
    find_tables(raw_row, base + ["root"])

    for table_parts, table_rows in table_candidates:
        for row_idx, row in enumerate(table_rows):
                if not isinstance(row, dict):
                    continue
                row_items = list(row.items())
                row_anchor_key = None
                row_anchor_key_idx = None
                column_key_idx = None
                for key_idx, (key, value) in enumerate(row_items):
                    if str(key) == column_anchor_text:
                        column_key_idx = key_idx
                    if str(value) == row_anchor_text and row_anchor_key is None:
                        row_anchor_key = str(key)
                        row_anchor_key_idx = key_idx
                if row_anchor_key is None or row_anchor_key_idx is None or column_key_idx is None:
                    continue
                column_value = str(row.get(column_anchor_text, "") or "").strip()
                if result_text and not _same_value(column_value, result_text):
                    continue
                row_parts = table_parts + ["body", f"i_{row_idx}"]
                group_parts = row_parts + ["group"]
                row_anchor_parts = group_parts + [f"k_{row_anchor_key}", f"idx_{row_anchor_key_idx}", "body", "v"]
                column_anchor_parts = group_parts + [f"k_{column_anchor_text}", f"idx_{column_key_idx}"]
                result_parts = column_anchor_parts + ["body", "v"]
                return {
                    "tableNodeId": _make_tree_canonical_id(table_parts),
                    "rowItemNodeId": _make_tree_canonical_id(row_parts),
                    "rowAnchorNodeId": _make_tree_canonical_id(row_anchor_parts),
                    "columnAnchorNodeId": _make_tree_group_canonical_id(
                        table_parts + ["header_group", f"k_{column_anchor_text}", f"idx_{column_key_idx}"]
                    ),
                    "resultNodeId": _make_tree_canonical_id(result_parts),
                    "rowAnchorKey": row_anchor_key,
                    "columnAnchorKey": column_anchor_text,
                    "resultText": column_value,
                }
    return {}


def _annotate_execution_events_with_canonical_ids(
    events: List[Dict[str, Any]],
    alias_to_target: Dict[str, Dict[str, str]],
    canonical_to_semantic: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    annotated: List[Dict[str, Any]] = []
    alias_map = dict(alias_to_target or {})
    alias_exact_keys = set(alias_map.keys())
    alias_legacy_exact_map: Dict[str, str] = {}
    alias_scope_prefixes_by_file_token: Dict[str, List[str]] = {}
    scoped_alias_pattern = re.compile(
        r"^ft:root_HO_TREE/(?P<scope_prefix>m_\d+_[^/]+/b_0_[^/]+_subtree)/(?P<rest>m_\d+_[^/]+(?:/.*)?)$"
    )
    scope_prefix_file_pattern = re.compile(
        r"^(?P<mseg>m_\d+_(?P<file_token>[^/]+))/b_0_[^/]+_subtree$"
    )
    for alias in sorted(alias_exact_keys):
        alias_text = str(alias or "").strip()
        if not alias_text:
            continue
        scoped_match = scoped_alias_pattern.match(alias_text)
        if not scoped_match:
            continue
        prefix_text = str(scoped_match.group("scope_prefix") or "").strip()
        rest_text = str(scoped_match.group("rest") or "").strip()
        if prefix_text and rest_text:
            legacy_text = f"ft:root_HO_TREE/{rest_text}"
            alias_legacy_exact_map.setdefault(legacy_text, alias_text)
            file_match = scope_prefix_file_pattern.match(prefix_text)
            if file_match:
                file_token = str(file_match.group("file_token") or "").strip()
                if file_token:
                    seen_prefixes = alias_scope_prefixes_by_file_token.setdefault(file_token, [])
                    if prefix_text not in seen_prefixes:
                        seen_prefixes.append(prefix_text)

    canonical_remap = {
        str(k or "").strip(): str(v or "").strip()
        for k, v in (canonical_to_semantic or {}).items()
        if str(k or "").strip() and str(v or "").strip()
    }

    def _normalize_frontend_alias(node_id: str) -> str:
        text = str(node_id or "").strip()
        if not text:
            return text
        if text in alias_exact_keys:
            return text
        exact_hit = alias_legacy_exact_map.get(text, "")
        if exact_hit:
            return exact_hit
        prefix = "ft:root_HO_TREE/"
        if not text.startswith(prefix):
            return text
        body = text[len(prefix):]
        parts = [p for p in body.split("/") if str(p or "").strip()]
        if not parts:
            return text
        first = str(parts[0] or "").strip()
        m_first = re.match(r"^m_\d+_(?P<file_token>[^/]+)$", first)
        if not m_first:
            return text
        file_token = str(m_first.group("file_token") or "").strip()
        tail_start = 1
        if len(parts) >= 2 and str(parts[1] or "").strip().startswith("b_0_"):
            tail_start = 2
        tail = parts[tail_start:]
        scope_prefixes = alias_scope_prefixes_by_file_token.get(file_token) or []
        matches: List[str] = []
        for matched_scope_prefix in scope_prefixes:
            scope_parts = [p for p in matched_scope_prefix.split("/") if str(p or "").strip()]
            candidate = f"{prefix}{'/'.join(scope_parts + tail)}"
            if candidate in alias_exact_keys:
                matches.append(candidate)
        if len(matches) == 1:
            return matches[0]
        return text

    for ev in events or []:
        item = dict(ev)
        frontend_node_id = _normalize_frontend_alias(str(item.get("frontend_node_id", "") or "").strip())
        if frontend_node_id:
            item["frontend_node_id"] = frontend_node_id
        canonical_id = str(item.get("canonical_id", "") or item.get("canonical_trace_id", "") or "").strip()
        target_kind = str(item.get("target_kind", "") or "").strip()
        target_info = alias_map.get(frontend_node_id, {}) if frontend_node_id else {}
        alias_canonical_id = ""
        if isinstance(target_info, dict):
            alias_canonical_id = str(target_info.get("canonical_id", "") or "").strip()
        if not canonical_id and isinstance(target_info, dict):
            canonical_id = str(target_info.get("canonical_id", "") or "").strip()
        # Prefer view-space canonical id from current alias map when available.
        # This keeps playback ids in ct_tree_* domain instead of semantic domain.
        if alias_canonical_id:
            canonical_id = alias_canonical_id
        if canonical_id and canonical_remap:
            canonical_id = canonical_remap.get(canonical_id, canonical_id)
        if canonical_id:
            item["canonical_id"] = canonical_id
            item["canonical_trace_id"] = canonical_id
        if not target_kind and isinstance(target_info, dict):
            target_kind = str(target_info.get("target_kind", "") or "").strip()
        if not target_kind:
            target_kind = _infer_trace_target_kind(canonical_id)
        if target_kind:
            item["target_kind"] = target_kind
        annotated.append(item)
    return annotated


def _annotate_execution_events_with_semantic_ids(
    events: List[Dict[str, Any]],
    row_canonical_to_semantic: Optional[Dict[str, str]] = None,
    column_canonical_to_semantic: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    annotated: List[Dict[str, Any]] = []
    row_map = {
        str(k or "").strip(): str(v or "").strip()
        for k, v in (row_canonical_to_semantic or {}).items()
        if str(k or "").strip() and str(v or "").strip()
    }
    col_map = {
        str(k or "").strip(): str(v or "").strip()
        for k, v in (column_canonical_to_semantic or {}).items()
        if str(k or "").strip() and str(v or "").strip()
    }
    for ev in events or []:
        item = dict(ev)
        canonical_id = str(item.get("canonical_id", "") or item.get("canonical_trace_id", "") or "").strip()
        semantic_id = ""
        if canonical_id:
            semantic_id = row_map.get(canonical_id, "") or col_map.get(canonical_id, "")
        if semantic_id:
            item["semantic_id"] = semantic_id
        annotated.append(item)
    return annotated


def _find_flat_column_ext_semantic_targets(raw_column: Any, row_anchor: str, column_anchor: str, result_text: str = "") -> Dict[str, Any]:
    if not isinstance(raw_column, dict):
        return {}
    base = ["root", "flat_column"]
    row_anchor_text = str(row_anchor or "").strip()
    column_anchor_text = str(column_anchor or "").strip()
    result_text = str(result_text or "").strip()
    if not row_anchor_text or not column_anchor_text:
        return {}

    def _same_value(actual_text: str, expected_text: str) -> bool:
        if not expected_text:
            return True
        if actual_text == expected_text:
            return True
        try:
            a = float(str(actual_text).replace(",", ""))
            b = float(str(expected_text).replace(",", ""))
            return abs(a - b) < 1e-9
        except Exception:
            return False

    table_candidates: List[Tuple[List[str], Any]] = []
    
    def _is_column_table(d: dict) -> bool:
        if not d:
            return False
        # If at least one value is a list, consider it a column table
        for v in d.values():
            if isinstance(v, list):
                return True
        return False
        
    def find_column_tables(value: Any, parent_parts: List[str]):
        if isinstance(value, dict):
            for idx, (k, v) in enumerate(value.items()):
                k_str = str(k)
                cur_ct_parts = parent_parts + [f"k_{k_str}", f"idx_{idx}"]
                if isinstance(v, dict):
                    if _is_column_table(v):
                        table_candidates.append((cur_ct_parts, v))
                    else:
                        find_column_tables(v, cur_ct_parts + ["body"])

    if _is_column_table(raw_column):
        table_candidates.append((base + ["root"], raw_column))
    else:
        find_column_tables(raw_column, base + ["root"])

    for table_parts, table_columns in table_candidates:
            column_items = list(table_columns.items())
            column_key_idx = None
            for key_idx, (key, _values) in enumerate(column_items):
                if str(key) == column_anchor_text:
                    column_key_idx = key_idx
                    break
            if column_key_idx is None:
                continue

            column_values = table_columns.get(column_anchor_text)
            if not isinstance(column_values, list):
                continue

            for row_anchor_key_idx, (row_anchor_key, row_values) in enumerate(column_items):
                if not isinstance(row_values, list):
                    continue
                for row_idx, row_value in enumerate(row_values):
                    if str(row_value) != row_anchor_text:
                        continue
                    if row_idx >= len(column_values):
                        continue
                    column_value = str(column_values[row_idx] or "").strip()
                    if result_text and not _same_value(column_value, result_text):
                        continue
                    row_anchor_parts = table_parts + [
                        "body",
                        f"k_{str(row_anchor_key)}",
                        f"idx_{row_anchor_key_idx}",
                        "body",
                        f"i_{row_idx}",
                    ]
                    column_anchor_parts = table_parts + ["body", f"k_{column_anchor_text}", f"idx_{column_key_idx}"]
                    result_parts = column_anchor_parts + ["body", f"i_{row_idx}"]
                    return {
                        "tableNodeId": _make_tree_canonical_id(table_parts),
                        "rowAnchorNodeId": _make_tree_canonical_id(row_anchor_parts),
                        "columnAnchorNodeId": _make_tree_canonical_id(column_anchor_parts),
                        "resultNodeId": _make_tree_canonical_id(result_parts),
                        "rowAnchorKey": str(row_anchor_key),
                        "columnAnchorKey": column_anchor_text,
                        "resultText": column_value,
                        "rowIndex": row_idx,
                    }
    return {}


def _build_semantic_compact_projection_maps(semantic_to_views: Any) -> Dict[str, Any]:
    """
    Build compact semantic-id projection maps for replay.
    """
    semantic_map = semantic_to_views if isinstance(semantic_to_views, dict) else {}
    legacy_to_compact: Dict[str, str] = {}
    compact_to_legacy: Dict[str, str] = {}
    compact_to_views: Dict[str, Dict[str, Any]] = {}
    row_to_compact: Dict[str, str] = {}
    column_to_compact: Dict[str, str] = {}
    compact_to_groups: Dict[str, List[str]] = {}
    group_to_compacts: Dict[str, List[str]] = {}

    used_compact_ids: Dict[str, int] = {}

    def _safe_token(text: Any) -> str:
        value = str(text or "").strip()
        value = re.sub(r"[^a-zA-Z0-9_-]+", "_", value)
        value = re.sub(r"_+", "_", value).strip("_")
        return value or "root"

    def _pick_anchor_token(legacy_semantic_id: str, entry: Dict[str, Any]) -> str:
        aliases = entry.get("aliases", []) if isinstance(entry, dict) else []
        for alias in aliases or []:
            alias_text = str(alias or "").strip()
            m = re.search(r"m_\d+_([^/]+)", alias_text)
            if m:
                return _safe_token(m.group(1))
        m = re.search(r"m_\d+_([^_/]+)", str(legacy_semantic_id or ""))
        if m:
            return _safe_token(m.group(1))
        return "root"

    for legacy_semantic_id, raw_entry in semantic_map.items():
        legacy_id = str(legacy_semantic_id or "").strip()
        if not legacy_id:
            continue
        entry = raw_entry if isinstance(raw_entry, dict) else {}
        row_ids = sorted({str(x or "").strip() for x in (entry.get("row", []) or []) if str(x or "").strip()})
        column_ids = sorted({str(x or "").strip() for x in (entry.get("column", []) or []) if str(x or "").strip()})
        aliases = sorted({str(x or "").strip() for x in (entry.get("aliases", []) or []) if str(x or "").strip()})
        target_kind = str(entry.get("target_kind", "") or "node")

        anchor = _pick_anchor_token(legacy_id, entry)
        digest = hashlib.md5(legacy_id.encode("utf-8")).hexdigest()[:8]
        compact_id = f"ct_tree_root_{anchor}_{digest}"
        if compact_id in used_compact_ids and compact_to_legacy.get(compact_id, "") != legacy_id:
            used_compact_ids[compact_id] = int(used_compact_ids.get(compact_id, 1)) + 1
            compact_id = f"{compact_id}_{used_compact_ids[compact_id]}"
        else:
            used_compact_ids.setdefault(compact_id, 1)

        group_ids = sorted({
            rid for rid in row_ids if rid.startswith("ct_tree_group_")
        })
        compact_to_groups[compact_id] = list(group_ids)
        for gid in group_ids:
            group_to_compacts.setdefault(gid, [])
            if compact_id not in group_to_compacts[gid]:
                group_to_compacts[gid].append(compact_id)

        legacy_to_compact[legacy_id] = compact_id
        compact_to_legacy[compact_id] = legacy_id
        compact_to_views[compact_id] = {
            "row": row_ids,
            "column": column_ids,
            "aliases": aliases,
            "target_kind": target_kind,
        }
        for rid in row_ids:
            row_to_compact[rid] = compact_id
        for cid in column_ids:
            column_to_compact[cid] = compact_id

    for gid, values in list(group_to_compacts.items()):
        group_to_compacts[gid] = sorted({str(x or "").strip() for x in values if str(x or "").strip()})

    return {
        "semantic_legacy_to_compact": legacy_to_compact,
        "semantic_compact_to_legacy": compact_to_legacy,
        "semantic_compact_to_views": compact_to_views,
        "row_canonical_to_semantic_compact": row_to_compact,
        "column_canonical_to_semantic_compact": column_to_compact,
        "semantic_compact_to_group_ids": compact_to_groups,
        "group_to_semantic_compacts": group_to_compacts,
    }


def _enrich_trace_v3_with_flat_semantics(
    trace_v3: Dict[str, Any],
    raw_row: Any,
    raw_column: Any,
    semantic_bundle: Optional[Dict[str, Any]] = None,
    preferred_view_mode: str = "column",
    table_scope: str = "",
) -> Dict[str, Any]:
    if not isinstance(trace_v3, dict):
        return trace_v3
    bundle = semantic_bundle if isinstance(semantic_bundle, dict) else {}
    row_to_semantic = {
        str(k or "").strip(): str(v or "").strip()
        for k, v in (bundle.get("row_canonical_to_semantic", {}) or {}).items()
        if str(k or "").strip() and str(v or "").strip()
    }
    column_to_semantic = {
        str(k or "").strip(): str(v or "").strip()
        for k, v in (bundle.get("column_canonical_to_semantic", {}) or {}).items()
        if str(k or "").strip() and str(v or "").strip()
    }
    canonical_to_semantic = {}
    canonical_to_semantic.update(row_to_semantic)
    canonical_to_semantic.update(column_to_semantic)
    semantic_to_views = dict((bundle.get("semantic_to_views", {}) or {}))
    alias_to_semantic_map = dict((bundle.get("alias_to_semantic", {}) or {}))
    semantic_view_index: Dict[str, Dict[str, Any]] = {}
    scope_text = str(table_scope or "").strip()
    scope_root_prefix = ""
    scope_root_prefix_candidates: List[str] = []
    legacy_ft_exact_map: Dict[str, str] = {}
    scope_prefixes_by_file_token: Dict[str, List[str]] = {}

    def normalize_semantic_projection_key(semantic_id: Any) -> str:
        text = str(semantic_id or "").strip()
        return text

    for semantic_key, entry in semantic_to_views.items():
        norm_key = normalize_semantic_projection_key(semantic_key)
        if not norm_key:
            continue
        semantic_view_index.setdefault(norm_key, entry if isinstance(entry, dict) else {})

    if isinstance(alias_to_semantic_map, dict):
        if scope_text:
            scope_token = re.sub(r"[^a-zA-Z0-9_-]+", "_", scope_text).strip("_") or scope_text
            scope_pattern = re.compile(
                rf"^ft:root_HO_TREE/(?P<mseg>m_\d+_{re.escape(scope_token)})/(?P<bseg>b_0_[^/]+_subtree)(?:/|$)"
            )
            for alias in sorted(alias_to_semantic_map.keys()):
                alias_text = str(alias or "").strip()
                if not alias_text:
                    continue
                match = scope_pattern.match(alias_text)
                if match:
                    scope_root_prefix = f"{match.group('mseg')}/{match.group('bseg')}"
                    break

        scoped_alias_pattern = re.compile(
            r"^ft:root_HO_TREE/(?P<scope_prefix>m_\d+_[^/]+/b_0_[^/]+_subtree)/(?P<rest>m_\d+_[^/]+(?:/.*)?)$"
        )
        scope_prefix_file_pattern = re.compile(
            r"^(?P<mseg>m_\d+_(?P<file_token>[^/]+))/b_0_[^/]+_subtree$"
        )
        for alias in sorted(alias_to_semantic_map.keys()):
            alias_text = str(alias or "").strip()
            if not alias_text:
                continue
            scoped_match = scoped_alias_pattern.match(alias_text)
            if not scoped_match:
                continue
            prefix_text = str(scoped_match.group("scope_prefix") or "").strip()
            rest_text = str(scoped_match.group("rest") or "").strip()
            if not prefix_text or not rest_text:
                continue
            if prefix_text not in scope_root_prefix_candidates:
                scope_root_prefix_candidates.append(prefix_text)
            file_match = scope_prefix_file_pattern.match(prefix_text)
            if file_match:
                file_token = str(file_match.group("file_token") or "").strip()
                if file_token:
                    seen_prefixes = scope_prefixes_by_file_token.setdefault(file_token, [])
                    if prefix_text not in seen_prefixes:
                        seen_prefixes.append(prefix_text)
            legacy_text = f"ft:root_HO_TREE/{rest_text}"
            if legacy_text not in legacy_ft_exact_map:
                legacy_ft_exact_map[legacy_text] = alias_text

    if not scope_root_prefix and len(scope_root_prefix_candidates) == 1:
        scope_root_prefix = scope_root_prefix_candidates[0]

    def normalize_legacy_ft_id(ft_id: Any) -> str:
        text = str(ft_id or "").strip()
        if not text:
            return text
        if text in alias_to_semantic_map:
            return text
        exact_hit = legacy_ft_exact_map.get(text, "")
        if exact_hit:
            return exact_hit
        if not scope_root_prefix:
            # Try token-based completion when scope prefix is absent.
            prefix = "ft:root_HO_TREE/"
            if not text.startswith(prefix):
                return text
            body = text[len(prefix):]
            parts = [p for p in body.split("/") if str(p or "").strip()]
            if not parts:
                return text
            first = str(parts[0] or "").strip()
            m_first = re.match(r"^m_\d+_(?P<file_token>[^/]+)$", first)
            if not m_first:
                return text
            file_token = str(m_first.group("file_token") or "").strip()
            tail_start = 1
            if len(parts) >= 2 and str(parts[1] or "").strip().startswith("b_0_"):
                tail_start = 2
            tail = parts[tail_start:]
            scope_prefixes = scope_prefixes_by_file_token.get(file_token) or []
            matches_ft: List[str] = []
            for matched_scope_prefix in scope_prefixes:
                scope_parts = [p for p in matched_scope_prefix.split("/") if str(p or "").strip()]
                candidate = f"{prefix}{'/'.join(scope_parts + tail)}"
                if candidate in alias_to_semantic_map:
                    matches_ft.append(candidate)
            if len(matches_ft) == 1:
                return matches_ft[0]
            return text
        prefix = "ft:root_HO_TREE/"
        if not text.startswith(prefix):
            return text
        if text.startswith(f"{prefix}{scope_root_prefix}/") or text == f"{prefix}{scope_root_prefix}":
            return text
        body = text[len(prefix):]
        parts = [p for p in body.split("/") if str(p or "").strip()]
        if parts:
            m_first = re.match(r"^m_\d+_(?P<file_token>[^/]+)$", str(parts[0] or "").strip())
            if m_first:
                file_token = str(m_first.group("file_token") or "").strip()
                tail_start = 1
                if len(parts) >= 2 and str(parts[1] or "").strip().startswith("b_0_"):
                    tail_start = 2
                tail = parts[tail_start:]
                scope_prefixes = scope_prefixes_by_file_token.get(file_token) or []
                matches_ft2: List[str] = []
                for matched_scope_prefix in scope_prefixes:
                    scope_parts = [p for p in matched_scope_prefix.split("/") if str(p or "").strip()]
                    candidate = f"{prefix}{'/'.join(scope_parts + tail)}"
                    if candidate in alias_to_semantic_map:
                        matches_ft2.append(candidate)
                if len(matches_ft2) == 1:
                    return matches_ft2[0]
        rest = text[len(prefix):]
        if not rest.startswith("m_"):
            return text
        # Never force default scope prefix; it can wrongly rewrite m_1 to m_0.
        return text

    def normalize_playback_frontend_ids(playback: Dict[str, Any]) -> None:
        if not isinstance(playback, dict):
            return
        raw_nodes = playback.get("nodeIds", []) or []
        raw_edges = playback.get("edgeIds", []) or []
        node_ids = [normalize_legacy_ft_id(x) for x in raw_nodes]
        edge_ids: List[str] = []
        for edge in raw_edges:
            edge_text = str(edge or "").strip()
            if not edge_text:
                continue
            if "->" not in edge_text:
                edge_ids.append(normalize_legacy_ft_id(edge_text))
                continue
            left, right = edge_text.split("->", 1)
            edge_ids.append(f"{normalize_legacy_ft_id(left)}->{normalize_legacy_ft_id(right)}")
        playback["nodeIds"] = [str(x or "").strip() for x in node_ids if str(x or "").strip()]
        playback["edgeIds"] = [str(x or "").strip() for x in edge_ids if str(x or "").strip()]
        normalized_answer = normalize_legacy_ft_id(playback.get("answerNodeId", None))
        playback["answerNodeId"] = normalized_answer or None

    def to_semantic_id(canonical_id: Any) -> str:
        text = str(canonical_id or "").strip()
        if not text:
            return ""
        return canonical_to_semantic.get(text, text)

    def normalize_semantic_ids(values: Any) -> List[str]:
        ordered: List[str] = []
        seen = set()
        for value in values or []:
            semantic_id = to_semantic_id(value)
            if not semantic_id or semantic_id in seen:
                continue
            seen.add(semantic_id)
            ordered.append(semantic_id)
        return ordered

    def normalize_canonical_ids(values: Any) -> List[str]:
        ordered: List[str] = []
        seen = set()
        for value in values or []:
            canonical_id = str(value or "").strip()
            if not canonical_id or canonical_id in seen:
                continue
            seen.add(canonical_id)
            ordered.append(canonical_id)
        return ordered

    def normalize_edge_ids(values: Any) -> List[str]:
        ordered: List[str] = []
        seen = set()
        for value in values or []:
            edge_id = str(value or "").strip()
            if not edge_id or edge_id in seen:
                continue
            seen.add(edge_id)
            ordered.append(edge_id)
        return ordered

    def sync_semantic_playback(playback: Dict[str, Any]) -> None:
        if not isinstance(playback, dict):
            return
        canonical_nodes = normalize_canonical_ids(playback.get("canonicalNodeIds", []))
        canonical_edges = normalize_edge_ids(playback.get("canonicalEdgeIds", []))
        canonical_answer = str(playback.get("canonicalAnswerNodeId") or "").strip()
        playback["canonicalNodeIds"] = canonical_nodes
        playback["canonicalEdgeIds"] = canonical_edges
        playback["canonicalAnswerNodeId"] = canonical_answer or None

        semantic_nodes = normalize_semantic_ids(playback.get("semanticNodeIds", []))
        if not semantic_nodes:
            semantic_nodes = normalize_semantic_ids(canonical_nodes)
        semantic_edges = normalize_edge_ids(playback.get("semanticEdgeIds", []))
        if not semantic_edges and len(semantic_nodes) > 1:
            semantic_edges = [f"{semantic_nodes[i - 1]}->{semantic_nodes[i]}" for i in range(1, len(semantic_nodes))]
        semantic_answer = str(playback.get("semanticAnswerNodeId") or "").strip()
        if not semantic_answer:
            semantic_answer = to_semantic_id(canonical_answer)
        if not semantic_answer and semantic_nodes:
            semantic_answer = semantic_nodes[-1]
        playback["semanticNodeIds"] = semantic_nodes
        playback["semanticEdgeIds"] = semantic_edges
        playback["semanticAnswerNodeId"] = semantic_answer or None

    def project_semantic_ids_to_view(values: Any, view_name: str) -> List[str]:
        ordered: List[str] = []
        seen = set()
        target_view = "row" if str(view_name or "").strip().lower() == "row" else "column"
        for value in values or []:
            raw_id = str(value or "").strip()
            if not raw_id:
                continue
            semantic_id = raw_id if raw_id.startswith("ct_semantic_") else to_semantic_id(raw_id)
            if semantic_id and semantic_id.startswith("ct_semantic_"):
                view_entry = semantic_to_views.get(semantic_id, {}) or {}
                if not view_entry:
                    view_entry = semantic_view_index.get(normalize_semantic_projection_key(semantic_id), {}) or {}
                projected = view_entry.get(target_view, []) if isinstance(view_entry, dict) else []
                for item in projected or []:
                    projected_id = str(item or "").strip()
                    if not projected_id or projected_id in seen:
                        continue
                    seen.add(projected_id)
                    ordered.append(projected_id)
                continue
            if raw_id in seen:
                continue
            seen.add(raw_id)
            ordered.append(raw_id)
        return ordered

    normalized_mode = str(preferred_view_mode or "column").strip().lower()
    mode_key = "flatRow" if normalized_mode == "row" else "flatColumn"

    for subquery in trace_v3.get("subqueries", []) or []:
        for frame in subquery.get("frames", []) or []:
            for operation in frame.get("operations", []) or []:
                if not isinstance(operation, dict):
                    continue
                playback = operation.setdefault("playback", {})
                normalize_playback_frontend_ids(playback)
                operation.setdefault("semanticTargets", {})
                if str(operation.get("kind", "") or "") != "extract_lookup":
                    sync_semantic_playback(playback)
                    continue
                args = operation.get("args", []) or []
                row_anchor = str(args[0] if len(args) > 0 else "" or "").strip()
                column_anchor = str(args[1] if len(args) > 1 else "" or "").strip()
                result_text = _extract_primary_result_text(operation.get("resultSummary"))
                flat_row_targets = _find_flat_row_ext_semantic_targets(raw_row, row_anchor, column_anchor, result_text)
                flat_column_targets = _find_flat_column_ext_semantic_targets(raw_column, row_anchor, column_anchor, result_text)
                if flat_row_targets:
                    operation["semanticTargets"]["flatRow"] = flat_row_targets
                if flat_column_targets:
                    operation["semanticTargets"]["flatColumn"] = flat_column_targets
                semantic_targets: Dict[str, Any] = {}
                for key in ["tableNodeId", "rowItemNodeId", "rowAnchorNodeId", "columnAnchorNodeId", "resultNodeId"]:
                    row_id = str((flat_row_targets or {}).get(key, "") or "").strip()
                    col_id = str((flat_column_targets or {}).get(key, "") or "").strip()
                    semantic_id = to_semantic_id(row_id) or to_semantic_id(col_id)
                    if semantic_id:
                        semantic_targets[key] = semantic_id
                if semantic_targets:
                    operation["semanticTargets"]["semantic"] = semantic_targets

                preferred_targets = operation["semanticTargets"].get(mode_key) or {}
                exact_nodes = normalize_canonical_ids(
                    [
                        preferred_targets.get("tableNodeId", ""),
                        preferred_targets.get("rowItemNodeId", ""),
                        preferred_targets.get("rowAnchorNodeId", ""),
                        preferred_targets.get("columnAnchorNodeId", ""),
                        preferred_targets.get("resultNodeId", ""),
                    ]
                )
                if exact_nodes:
                    playback["canonicalNodeIds"] = exact_nodes
                    playback["canonicalEdgeIds"] = []
                    preferred_answer = str(preferred_targets.get("resultNodeId", "") or "").strip()
                    playback["canonicalAnswerNodeId"] = preferred_answer or exact_nodes[-1]
                else:
                    # No direct flat targets found: project semantic playback ids into current view ids.
                    projected_nodes = project_semantic_ids_to_view(playback.get("canonicalNodeIds", []), normalized_mode)
                    projected_answer = project_semantic_ids_to_view(
                        [playback.get("canonicalAnswerNodeId", None)], normalized_mode
                    )
                    if projected_nodes:
                        playback["canonicalNodeIds"] = projected_nodes
                        # Let frontend derive a stable path in current tree when explicit edges are unavailable.
                        playback["canonicalEdgeIds"] = []
                        playback["canonicalAnswerNodeId"] = (
                            projected_answer[0] if projected_answer else projected_nodes[-1]
                        )
                sync_semantic_playback(playback)
    return trace_v3


def _load_typed_tree_v2_payload(conversation_id: str = "", table_scope: str = "") -> Optional[Dict[str, Any]]:
    pkl_path = _resolve_feature_tree_pkl_path(conversation_id)
    if not os.path.exists(pkl_path):
        return None
    try:
        with open(pkl_path, "rb") as f:
            f_tree = pickle.load(f)
        payload, _lookup = build_typed_tree_v2(
            f_tree,
            root_name="HO_TREE",
            file_scope=str(table_scope or "").strip(),
        )
        return payload
    except Exception as e:
        logger.warning(f"构建 typed tree v2 失败: {e}")
        return None


def _parse_retrieved_row_dicts(chain: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    qa = (chain or {}).get("question_answering", {}) or {}
    subqueries = qa.get("subqueries", []) or []

    def collect(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, dict):
            rows.append(value)
            return
        if isinstance(value, list):
            for item in value:
                collect(item)
            return
        if isinstance(value, str):
            s = value.strip()
            if not s:
                return
            try:
                parsed = ast.literal_eval(s)
            except Exception:
                return
            collect(parsed)

    for sq in subqueries:
        collect(sq.get("retrieved_data"))

    uniq: List[Dict[str, Any]] = []
    seen = set()
    for row in rows:
        try:
            key = json.dumps(row, ensure_ascii=False, sort_keys=True)
        except Exception:
            key = str(row)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(row)
    return uniq


def _collect_table_rows_with_paths(raw_tree: Any) -> List[Dict[str, Any]]:
    """
    Collect rows in dict/list table-like structures with deterministic node paths.
    Example path: ["root", "k_Sheet", "k_TableName", "i_0"]
    """
    rows: List[Dict[str, Any]] = []

    def walk(obj: Any, path_parts: List[str]) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                walk(value, path_parts + [f"k_{str(key)}"])
            return
        if isinstance(obj, list):
            if obj and all(isinstance(item, dict) for item in obj):
                for idx, row in enumerate(obj):
                    rows.append({"path_parts": path_parts + [f"i_{idx}"], "row": row})
                return
            for idx, item in enumerate(obj):
                walk(item, path_parts + [f"i_{idx}"])

    walk(raw_tree, ["root"])
    return rows


def _row_dict_matches(candidate: Dict[str, Any], retrieved: Dict[str, Any]) -> bool:
    if not isinstance(candidate, dict) or not isinstance(retrieved, dict):
        return False
    for k, v in retrieved.items():
        if k not in candidate:
            return False
        if str(candidate.get(k)) != str(v):
            return False
    return True


def _compute_strict_trace(chain: Dict[str, Any], raw_tree: Any, answer_text: str = "") -> Dict[str, Any]:
    node_index = _build_trace_node_index(raw_tree)
    parent_map = {nid: (info.get("parent") or None) for nid, info in node_index.items()}

    token_collector: List[str] = []
    ordered_tokens: List[str] = []
    qa = (chain or {}).get("question_answering", {}) or {}
    subqueries = qa.get("subqueries", []) or []

    for sq in subqueries:
        rp = sq.get("reasoning_path")
        if isinstance(rp, list):
            for item in rp:
                tokens = _extract_bracket_tokens_for_trace(item)
                token_collector.extend(tokens)
                ordered_tokens.extend(tokens)
        else:
            tokens = _extract_bracket_tokens_for_trace(rp)
            token_collector.extend(tokens)
            ordered_tokens.extend(tokens)
        _flatten_strings_for_trace(sq.get("retrieved_data"), token_collector)

    final_answer = str(answer_text or qa.get("final_answer") or "")
    token_collector.extend(_extract_bracket_tokens_for_trace(final_answer))

    cleaned_tokens: List[str] = []
    seen_token = set()
    for tk in token_collector:
        t = str(tk or "").strip()
        if len(t) < 2 or len(t) > 80:
            continue
        if t in seen_token:
            continue
        seen_token.add(t)
        cleaned_tokens.append(t)

    token_norms = [_normalize_text_for_trace(t) for t in cleaned_tokens if _normalize_text_for_trace(t)]

    def _pick_best_answer_node_from_candidates(answer_norm: str, candidates: List[tuple]) -> Optional[str]:
        """
        candidates: List[(node_id, normalized_text)]
        Strategy:
        1) exact match first;
        2) then containment match with min length (avoid matching single-char noise like "1");
        3) choose the longest normalized_text for better specificity.
        """
        if not answer_norm or not candidates:
            return None
        exact = [(nid, txt) for nid, txt in candidates if txt and txt == answer_norm]
        if exact:
            exact.sort(key=lambda x: len(x[1]), reverse=True)
            return exact[0][0]
        contain = [
            (nid, txt)
            for nid, txt in candidates
            if txt and len(txt) >= 4 and (answer_norm.find(txt) >= 0 or txt.find(answer_norm) >= 0)
        ]
        if contain:
            contain.sort(key=lambda x: len(x[1]), reverse=True)
            return contain[0][0]
        return None

    def build_paths_from_targets(target_nodes: List[str]) -> Dict[str, List[str]]:
        path_node_order: List[str] = []
        path_edge_order: List[str] = []

        def add_path_root_to(target_id: str) -> None:
            rev = [target_id]
            cur = target_id
            while cur:
                parent_id = parent_map.get(cur)
                if not parent_id:
                    break
                rev.append(parent_id)
                cur = parent_id
            seq = list(reversed(rev))
            for idx, nid in enumerate(seq):
                if not path_node_order or path_node_order[-1] != nid:
                    path_node_order.append(nid)
                if idx > 0:
                    ek = f"{seq[idx - 1]}->{nid}"
                    if ek not in path_edge_order:
                        path_edge_order.append(ek)

        for nid in target_nodes:
            add_path_root_to(nid)
        return {"nodes": path_node_order, "edges": path_edge_order}

    # Prefer execution-grounded trace:
    # map retrieved rows to exact HO-Tree row paths, then resolve reasoning tokens within those rows.
    retrieved_rows = _parse_retrieved_row_dicts(chain)
    table_rows = _collect_table_rows_with_paths(raw_tree)
    selected_rows: List[Dict[str, Any]] = []
    used_row_paths = set()
    for rr in retrieved_rows:
        matched = None
        for tr in table_rows:
            row_path_key = "|".join(tr["path_parts"])
            if row_path_key in used_row_paths:
                continue
            if _row_dict_matches(tr["row"], rr):
                matched = tr
                break
        if matched:
            used_row_paths.add("|".join(matched["path_parts"]))
            selected_rows.append(matched)

    if selected_rows:
        matched_node_ids = set()
        ordered_target_nodes: List[str] = []
        seen_target = set()

        # Row nodes are considered truly accessed.
        for sr in selected_rows:
            rid = _make_trace_node_id(sr["path_parts"])
            matched_node_ids.add(rid)

        # Use reasoning_path order, but only map inside selected retrieved rows.
        for tk in ordered_tokens:
            tkn = _normalize_text_for_trace(tk)
            if not tkn:
                continue
            picked = None
            for sr in selected_rows:
                row = sr["row"]
                for key, value in row.items():
                    key_norm = _normalize_text_for_trace(key)
                    val_norm = _normalize_text_for_trace(value)
                    key_id = _make_trace_node_id(sr["path_parts"] + [f"k_{str(key)}"])
                    val_id = _make_trace_node_id(sr["path_parts"] + [f"k_{str(key)}", "v"])
                    if key_norm and (key_norm.find(tkn) >= 0 or (len(tkn) >= 4 and tkn.find(key_norm) >= 0)):
                        picked = key_id
                        matched_node_ids.add(key_id)
                        break
                    if val_norm and (val_norm.find(tkn) >= 0 or (len(tkn) >= 4 and tkn.find(val_norm) >= 0)):
                        picked = val_id
                        matched_node_ids.add(val_id)
                        break
                if picked:
                    break
            if picked and picked not in seen_target:
                seen_target.add(picked)
                ordered_target_nodes.append(picked)

        answer_node_id: Optional[str] = None
        answer_norm = _normalize_text_for_trace(final_answer)
        if answer_norm:
            answer_candidates: List[tuple] = []
            for sr in selected_rows:
                for key, value in sr["row"].items():
                    val_norm = _normalize_text_for_trace(value)
                    if not val_norm:
                        continue
                    nid = _make_trace_node_id(sr["path_parts"] + [f"k_{str(key)}", "v"])
                    answer_candidates.append((nid, val_norm))
            answer_node_id = _pick_best_answer_node_from_candidates(answer_norm, answer_candidates)

        if answer_node_id:
            matched_node_ids.add(answer_node_id)
            if answer_node_id not in seen_target:
                ordered_target_nodes.append(answer_node_id)

        if not ordered_target_nodes:
            # fallback to row-level traversal if token mapping is empty
            ordered_target_nodes = [_make_trace_node_id(sr["path_parts"]) for sr in selected_rows]

        path_bundle = build_paths_from_targets(ordered_target_nodes)
        edge_ids = set(path_bundle["edges"])

        return {
            "mode": "strict",
            "source": "retrieval_rows",
            "matched_node_ids": list(matched_node_ids),
            "edge_ids": list(edge_ids),
            "path_node_order": path_bundle["nodes"],
            "path_edge_order": path_bundle["edges"],
            "answer_node_id": answer_node_id,
            "answer_node_name": node_index.get(answer_node_id, {}).get("name") if answer_node_id else None
        }

    matched_ids: List[str] = []
    for node_id, info in node_index.items():
        norm_name = _normalize_text_for_trace(info.get("name", ""))
        if not norm_name:
            continue
        matched = any(norm_name.find(tk) >= 0 or (len(tk) >= 4 and tk.find(norm_name) >= 0) for tk in token_norms)
        if matched:
            matched_ids.append(node_id)

    answer_node_id: Optional[str] = None
    answer_norm = _normalize_text_for_trace(final_answer)
    if answer_norm:
        answer_candidates: List[tuple] = []
        for node_id, info in node_index.items():
            norm_name = _normalize_text_for_trace(info.get("name", ""))
            if not norm_name:
                continue
            answer_candidates.append((node_id, norm_name))
        answer_node_id = _pick_best_answer_node_from_candidates(answer_norm, answer_candidates)
    if not answer_node_id and matched_ids:
        answer_node_id = matched_ids[-1]

    final_node_ids = set(matched_ids)
    if answer_node_id:
        final_node_ids.add(answer_node_id)

    edge_set = set()
    for node_id in final_node_ids:
        cur = node_id
        while cur:
            parent_id = parent_map.get(cur)
            if not parent_id:
                break
            edge_set.add(f"{parent_id}->{cur}")
            cur = parent_id

    def find_best_node_by_token(token: str) -> Optional[str]:
        tk = _normalize_text_for_trace(token)
        if not tk:
            return None
        for node_id, info in node_index.items():
            norm_name = _normalize_text_for_trace(info.get("name", ""))
            if not norm_name:
                continue
            if norm_name.find(tk) >= 0 or (len(tk) >= 4 and tk.find(norm_name) >= 0):
                return node_id
        return None

    unique_ordered_nodes: List[str] = []
    seen_node = set()
    for tk in ordered_tokens:
        nid = find_best_node_by_token(tk)
        if not nid or nid in seen_node:
            continue
        seen_node.add(nid)
        unique_ordered_nodes.append(nid)

    if answer_node_id and answer_node_id not in seen_node:
        unique_ordered_nodes.append(answer_node_id)

    path_node_order: List[str] = []
    path_edge_order: List[str] = []

    def add_path_root_to(target_id: str) -> None:
        rev = [target_id]
        cur = target_id
        while cur:
            parent_id = parent_map.get(cur)
            if not parent_id:
                break
            rev.append(parent_id)
            cur = parent_id
        seq = list(reversed(rev))
        for idx, nid in enumerate(seq):
            if not path_node_order or path_node_order[-1] != nid:
                path_node_order.append(nid)
            if idx > 0:
                ek = f"{seq[idx - 1]}->{nid}"
                if ek not in path_edge_order:
                    path_edge_order.append(ek)

    if unique_ordered_nodes:
        for nid in unique_ordered_nodes:
            add_path_root_to(nid)
    elif answer_node_id:
        add_path_root_to(answer_node_id)

    return {
        "mode": "strict",
        "matched_node_ids": list(final_node_ids),
        "edge_ids": list(edge_set),
        "path_node_order": path_node_order,
        "path_edge_order": path_edge_order,
        "answer_node_id": answer_node_id,
        "answer_node_name": node_index.get(answer_node_id, {}).get("name") if answer_node_id else None
    }


def _annotate_execution_trace(events: List[Dict[str, Any]], raw_tree: Any) -> List[Dict[str, Any]]:
    node_index = _build_trace_node_index(raw_tree)
    normalized_nodes: List[Dict[str, str]] = []
    for nid, info in node_index.items():
        name = str(info.get("name", "")).strip()
        norm = _normalize_text_for_trace(name)
        if not norm:
            continue
        normalized_nodes.append({"id": nid, "name": name, "norm": norm})

    annotated: List[Dict[str, Any]] = []
    for ev in events or []:
        item = dict(ev)
        direct_canonical_id = str(item.get("canonical_id", "") or item.get("canonical_trace_id", "") or "").strip()
        direct_frontend_id = str(item.get("frontend_node_id", "") or "").strip()
        if direct_canonical_id or direct_frontend_id:
            annotated.append(item)
            continue
        node_value = str(item.get("node_value", "")).strip()
        norm_value = _normalize_text_for_trace(node_value)
        frontend_node_id = None
        if norm_value:
            exact_matches = [cand for cand in normalized_nodes if str(cand.get("norm", "")) == norm_value]
            if len(exact_matches) == 1:
                frontend_node_id = exact_matches[0]["id"]
        item["frontend_node_id"] = frontend_node_id
        annotated.append(item)
    return annotated


def _build_typed_frontend_node_index(typed_payload: Any) -> List[Dict[str, str]]:
    normalized_nodes: List[Dict[str, str]] = []

    def walk(nodes: Any) -> None:
        if not isinstance(nodes, list):
            return
        for node in nodes:
            if not isinstance(node, dict):
                continue
            node_id = str(node.get("id", "") or "").strip()
            name = str(node.get("name", "") or "").strip()
            norm = _normalize_text_for_trace(name)
            if node_id and norm:
                normalized_nodes.append({"id": node_id, "name": name, "norm": norm})
            walk(node.get("children"))

    walk((typed_payload or {}).get("roots", []))
    return normalized_nodes


def _annotate_execution_trace_with_typed_tree(events: List[Dict[str, Any]], raw_tree: Any, typed_payload: Any) -> List[Dict[str, Any]]:
    typed_nodes = _build_typed_frontend_node_index(typed_payload)
    if not typed_nodes:
        return _annotate_execution_trace(events, raw_tree)

    annotated: List[Dict[str, Any]] = []
    for ev in events or []:
        item = dict(ev)
        direct_canonical_id = str(item.get("canonical_id", "") or item.get("canonical_trace_id", "") or "").strip()
        direct_frontend_id = str(item.get("frontend_node_id", "") or "").strip()
        if direct_canonical_id or direct_frontend_id:
            annotated.append(item)
            continue
        node_value = str(item.get("node_value", "")).strip()
        norm_value = _normalize_text_for_trace(node_value)
        frontend_node_id = None
        if norm_value:
            exact_matches = [cand for cand in typed_nodes if str(cand.get("norm", "")) == norm_value]
            if len(exact_matches) == 1:
                frontend_node_id = exact_matches[0]["id"]
        item["frontend_node_id"] = frontend_node_id
        annotated.append(item)
    return annotated


# 获取思维链
@app.get("/api/chain")
async def get_chain(conversation_id: str = "", view_mode: str = "row"):
    try:
        chain = get_thinking_chain() or {}
        qa_info = (chain.get("question_answering", {}) if isinstance(chain, dict) else {}) or {}
        trace_table_scope = str(qa_info.get("table_scope", "") or "").strip()
        tree_path = _resolve_trace_tree_path(conversation_id)
        raw_tree: Any = []
        if os.path.exists(tree_path):
            with open(tree_path, "r", encoding="utf-8") as f:
                raw_tree = json.load(f)
        final_answer = str((chain.get("question_answering", {}) or {}).get("final_answer", ""))
        execution_events = (chain.get("question_answering", {}) or {}).get("execution_trace", []) or []
        typed_payload = _load_typed_tree_v2_payload(conversation_id, table_scope=trace_table_scope)
        raw_row = _load_row_view_payload(conversation_id)
        raw_column = _load_column_view_payload(conversation_id)
        raw_row = _apply_table_scope_to_payload(raw_row, trace_table_scope)
        raw_column = _apply_table_scope_to_payload(raw_column, trace_table_scope)

        normalized_mode = str(view_mode or "").strip().lower()
        if normalized_mode not in {"row", "column"}:
            normalized_mode = "row"
        strict_tree_source = "row"
        strict_tree = None
        if normalized_mode == "column" and raw_column is not None:
            strict_tree = _build_flat_index_body_tree(
                raw_column,
                root_name="flat column view",
                path_parts=["root", "flat_column"],
            )
            strict_tree_source = "column"
        elif raw_row is not None:
            strict_tree = _build_flat_row_tree_with_trace_metadata(raw_row)
            strict_tree_source = "row"
        elif raw_column is not None:
            strict_tree = _build_flat_index_body_tree(
                raw_column,
                root_name="flat column view",
                path_parts=["root", "flat_column"],
            )
            strict_tree_source = "column_fallback"

        strict_trace_input = strict_tree if strict_tree is not None else raw_tree
        strict_trace = _compute_strict_trace(chain, strict_trace_input, final_answer)

        semantic_bundle = build_semantic_projection_bundle(raw_row, raw_column, typed_root_name="HO_TREE")
        nested_index_projection_map = build_nested_index_projection_map(
            raw_column,
            semantic_bundle=semantic_bundle,
            typed_root_name="HO_TREE",
        )
        trace_tree_fingerprint = _compute_tree_trace_fingerprint(strict_tree)
        alias_tree = strict_tree if isinstance(strict_tree, dict) else _build_flat_tree_node_for_view_mode(conversation_id, normalized_mode)
        alias_to_target = _build_trace_alias_target_map(alias_tree)
        if not alias_to_target:
            if normalized_mode == "column":
                alias_to_target = build_flat_column_alias_target_map(raw_column, typed_root_name="HO_TREE")
            else:
                alias_to_target = build_flat_row_alias_target_map(raw_row, typed_root_name="HO_TREE")
        # Semantic mapping is for cross-view projection, not direct playback canonical ids.
        if not alias_to_target and isinstance(semantic_bundle, dict):
            alias_to_target = semantic_bundle.get("alias_to_semantic", {}) or {}
        canonical_projection_map = {
            "semanticToViews": dict((semantic_bundle or {}).get("semantic_to_views", {}) or {}),
            "rowCanonicalToSemantic": dict((semantic_bundle or {}).get("row_canonical_to_semantic", {}) or {}),
            "columnCanonicalToSemantic": dict((semantic_bundle or {}).get("column_canonical_to_semantic", {}) or {}),
            "nestedIndexProjectionMap": dict(nested_index_projection_map or {}),
            "nested_index_projection_map": dict(nested_index_projection_map or {}),
        }
        frontend_catalog_rows = _collect_frontend_tree_node_catalog(strict_tree if isinstance(strict_tree, dict) else None)
        current_view_canonical_ids: set = set()
        for row in frontend_catalog_rows:
            canonical_id = str(row.get("canonicalId", "") or "").strip()
            group_canonical_id = str(row.get("groupCanonicalId", "") or "").strip()
            if canonical_id:
                current_view_canonical_ids.add(canonical_id)
            if group_canonical_id:
                current_view_canonical_ids.add(group_canonical_id)
        annotated_execution_events = _annotate_execution_trace_with_typed_tree(execution_events, raw_tree, typed_payload)
        annotated_execution_events = _annotate_execution_events_with_canonical_ids(
            annotated_execution_events,
            alias_to_target,
            None,
        )
        annotated_execution_events = _annotate_execution_events_with_semantic_ids(
            annotated_execution_events,
            (semantic_bundle or {}).get("row_canonical_to_semantic", {}) if isinstance(semantic_bundle, dict) else {},
            (semantic_bundle or {}).get("column_canonical_to_semantic", {}) if isinstance(semantic_bundle, dict) else {},
        )
        trace_v2 = build_typed_trace_v2(chain, strict_trace, annotated_execution_events)
        trace_v3 = build_trace_v3(chain, strict_trace, annotated_execution_events)
        trace_v3 = _enrich_trace_v3_with_flat_semantics(
            trace_v3,
            raw_row,
            raw_column,
            semantic_bundle=semantic_bundle,
            preferred_view_mode=normalized_mode,
            table_scope=trace_table_scope,
        )

        # 【debug】输出当前回传给前端用于高亮的 canonical id 信息，便于排查 trace 对齐问题
        debug_matched_ids = [
            str(x or "").strip()
            for x in (strict_trace.get("matched_node_ids", []) if isinstance(strict_trace, dict) else [])
            if str(x or "").strip()
        ]
        debug_path_node_ids = [
            str(x or "").strip()
            for x in (strict_trace.get("path_node_order", []) if isinstance(strict_trace, dict) else [])
            if str(x or "").strip()
        ]
        debug_path_edge_ids = [
            str(x or "").strip()
            for x in (strict_trace.get("path_edge_order", []) if isinstance(strict_trace, dict) else [])
            if str(x or "").strip()
        ]
        debug_answer_node_id = (
            str((strict_trace.get("answer_node_id", "") if isinstance(strict_trace, dict) else "") or "").strip()
        )
        logger.info(
            f"【debug】trace_highlight_ids view_mode={normalized_mode}, source={strict_tree_source}, "
            f"matched={debug_matched_ids[:20]}, path_nodes={debug_path_node_ids[:20]}, "
            f"path_edges={debug_path_edge_ids[:20]}, answer={debug_answer_node_id or 'N/A'}"
        )
        debug_operation_playbacks: List[Dict[str, Any]] = []
        for subquery in (trace_v3.get("subqueries", []) if isinstance(trace_v3, dict) else []) or []:
            subquery_index = int((subquery or {}).get("index", 0) or 0)
            for frame in (subquery.get("frames", []) if isinstance(subquery, dict) else []) or []:
                frame_id = str((frame or {}).get("frameId", "") or "")
                for operation in (frame.get("operations", []) if isinstance(frame, dict) else []) or []:
                    if not isinstance(operation, dict):
                        continue
                    debug_operation_playbacks.append({
                        "subqueryIndex": subquery_index,
                        "frameId": frame_id,
                        "operationId": str(operation.get("operationId", "") or ""),
                        "kind": str(operation.get("kind", "") or ""),
                        "playback": dict(operation.get("playback", {}) or {}),
                    })
        try:
            debug_payload_text = json.dumps(debug_operation_playbacks, ensure_ascii=False)
            logger.info(
                f"【debug】trace_operation_playbacks total={len(debug_operation_playbacks)}, payload={debug_payload_text}"
            )
        except Exception:
            logger.warning("【debug】trace_operation_playbacks serialize failed")
        try:
            debug_dir = os.path.join("history", conversation_id) if conversation_id else "cache"
            os.makedirs(debug_dir, exist_ok=True)
            debug_path = os.path.join(debug_dir, f"[debug]trace.operation.playback.{normalized_mode}.json")
            with open(debug_path, "w", encoding="utf-8") as f:
                json.dump(debug_operation_playbacks, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"写入 operation playback debug 文件失败: {e}")
        try:
            debug_semantic_rows: List[Dict[str, Any]] = []
            for subquery in (trace_v3.get("subqueries", []) if isinstance(trace_v3, dict) else []) or []:
                subquery_index = int((subquery or {}).get("index", 0) or 0)
                for frame in (subquery.get("frames", []) if isinstance(subquery, dict) else []) or []:
                    frame_id = str((frame or {}).get("frameId", "") or "")
                    for operation in (frame.get("operations", []) if isinstance(frame, dict) else []) or []:
                        if not isinstance(operation, dict):
                            continue
                        playback = dict(operation.get("playback", {}) or {})
                        semantic_ids = [
                            str(x or "").strip()
                            for x in (playback.get("semanticNodeIds", []) or [])
                            if str(x or "").strip()
                        ]
                        debug_semantic_rows.append({
                            "subqueryIndex": subquery_index,
                            "frameId": frame_id,
                            "operationId": str(operation.get("operationId", "") or ""),
                            "kind": str(operation.get("kind", "") or ""),
                            "semanticNodeIds": semantic_ids,
                            "semanticAnswerNodeId": str(playback.get("semanticAnswerNodeId", "") or "").strip(),
                        })
            debug_dir = os.path.join("history", conversation_id) if conversation_id else "cache"
            os.makedirs(debug_dir, exist_ok=True)
            debug_semantic_path = os.path.join(debug_dir, f"[debug]trace.operation.semantic_ids.{normalized_mode}.json")
            with open(debug_semantic_path, "w", encoding="utf-8") as f:
                json.dump(debug_semantic_rows, f, ensure_ascii=False, indent=2)
            logger.info(
                f"【debug】trace_operation_semantic_ids saved mode={normalized_mode}, "
                f"path={debug_semantic_path}, total={len(debug_semantic_rows)}"
            )
        except Exception as e:
            logger.warning(f"写入 operation semantic debug 文件失败: {e}")
        try:
            alias_to_semantic_map = (
                dict((semantic_bundle or {}).get("alias_to_semantic", {}) or {})
                if isinstance(semantic_bundle, dict)
                else {}
            )
            semantic_to_views_map = (
                dict((semantic_bundle or {}).get("semantic_to_views", {}) or {})
                if isinstance(semantic_bundle, dict)
                else {}
            )
            debug_ft_semantic_rows: List[Dict[str, Any]] = []
            for ft_alias in sorted(alias_to_semantic_map.keys()):
                target = alias_to_semantic_map.get(ft_alias, {}) or {}
                semantic_id = str(target.get("canonical_id", "") or "").strip()
                target_kind = str(target.get("target_kind", "") or "").strip()
                view_entry = semantic_to_views_map.get(semantic_id, {}) if semantic_id else {}
                row_projection = [
                    str(x or "").strip()
                    for x in ((view_entry or {}).get("row", []) if isinstance(view_entry, dict) else [])
                    if str(x or "").strip()
                ]
                column_projection = [
                    str(x or "").strip()
                    for x in ((view_entry or {}).get("column", []) if isinstance(view_entry, dict) else [])
                    if str(x or "").strip()
                ]
                active_projection = row_projection if normalized_mode == "row" else column_projection
                active_hits = [cid for cid in active_projection if cid in current_view_canonical_ids]
                view_alias_target = alias_to_target.get(ft_alias, {}) if isinstance(alias_to_target, dict) else {}
                debug_ft_semantic_rows.append({
                    "ftAlias": ft_alias,
                    "semanticId": semantic_id,
                    "targetKind": target_kind,
                    "rowProjection": row_projection,
                    "columnProjection": column_projection,
                    "activeProjection": active_projection,
                    "activeProjectionHitsInCurrentView": active_hits,
                    "hasActiveProjectionHit": bool(active_hits),
                    "inCurrentViewAliasMap": bool(view_alias_target),
                    "currentViewAliasCanonicalId": str((view_alias_target or {}).get("canonical_id", "") or "").strip(),
                })
            missing_rows = [item for item in debug_ft_semantic_rows if not bool(item.get("hasActiveProjectionHit", False))]
            playback_ft_aliases: List[str] = []
            playback_ft_seen = set()
            for op_item in debug_operation_playbacks:
                playback = (op_item or {}).get("playback", {}) if isinstance(op_item, dict) else {}
                for node_id in (playback.get("nodeIds", []) if isinstance(playback, dict) else []) or []:
                    ft_id = str(node_id or "").strip()
                    if not ft_id or not ft_id.startswith("ft:") or ft_id in playback_ft_seen:
                        continue
                    playback_ft_seen.add(ft_id)
                    playback_ft_aliases.append(ft_id)
            playback_mapped = [ft_id for ft_id in playback_ft_aliases if ft_id in alias_to_semantic_map]
            playback_unmapped = [ft_id for ft_id in playback_ft_aliases if ft_id not in alias_to_semantic_map]
            debug_map_payload = {
                "view_mode": normalized_mode,
                "tree_source": strict_tree_source,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "counts": {
                    "ft_alias_total": len(debug_ft_semantic_rows),
                    "ft_alias_missing_active_hit": len(missing_rows),
                    "current_view_canonical_total": len(current_view_canonical_ids),
                    "current_view_alias_total": len(alias_to_target) if isinstance(alias_to_target, dict) else 0,
                    "playback_ft_total": len(playback_ft_aliases),
                    "playback_ft_mapped_to_semantic": len(playback_mapped),
                    "playback_ft_unmapped_to_semantic": len(playback_unmapped),
                },
                "missing_preview": missing_rows[:200],
                "playback_unmapped_preview": playback_unmapped[:200],
                "items": debug_ft_semantic_rows,
            }
            debug_dir = os.path.join("history", conversation_id) if conversation_id else "cache"
            os.makedirs(debug_dir, exist_ok=True)
            debug_map_path = os.path.join(debug_dir, f"[debug]trace.ft.semantic.map.{normalized_mode}.json")
            with open(debug_map_path, "w", encoding="utf-8") as f:
                json.dump(debug_map_payload, f, ensure_ascii=False, indent=2)
            logger.info(
                f"【debug】trace_ft_semantic_map saved mode={normalized_mode}, "
                f"path={debug_map_path}, total={len(debug_ft_semantic_rows)}, "
                f"missing={len(missing_rows)}"
            )
        except Exception as e:
            logger.warning(f"写入 ft->semantic 映射 debug 文件失败: {e}")
        return JSONResponse({
            "success": True,
            "chain": chain,
            "trace": strict_trace,
            "execution_trace": annotated_execution_events,
            "trace_v2": trace_v2,
            "trace_v3": trace_v3,
            "trace_tree_fingerprint": trace_tree_fingerprint,
            "traceTreeFingerprint": trace_tree_fingerprint,
            "trace_view_mode": normalized_mode,
            "traceViewMode": normalized_mode,
            "trace_tree_source": strict_tree_source,
            "traceTreeSource": strict_tree_source,
            "debug_trace_highlight_ids": {
                "matched_node_ids": debug_matched_ids,
                "path_node_order": debug_path_node_ids,
                "path_edge_order": debug_path_edge_ids,
                "answer_node_id": debug_answer_node_id,
                "view_mode": normalized_mode,
                "tree_source": strict_tree_source,
            },
            "nested_index_projection_map": nested_index_projection_map,
            "nestedIndexProjectionMap": nested_index_projection_map,
            "canonical_projection_map": canonical_projection_map,
            "canonicalProjectionMap": canonical_projection_map,
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


@app.get("/api/debug/trace-sample-id")
async def get_trace_sample_id(conversation_id: str = "", view_mode: str = "row"):
    """
    Debug helper: return a canonical trace id sample for frontend highlight testing.
    """
    try:
        conversation_id = str(conversation_id or "").strip()
        normalized_mode = str(view_mode or "").strip().lower()
        if normalized_mode not in {"row", "column"}:
            normalized_mode = "row"

        chain = get_thinking_chain() or {}
        final_answer = str((chain.get("question_answering", {}) or {}).get("final_answer", ""))
        raw_row = _load_row_view_payload(conversation_id)
        raw_column = _load_column_view_payload(conversation_id)

        strict_tree_source = "row"
        strict_tree = None
        if normalized_mode == "column" and raw_column is not None:
            strict_tree = _build_flat_index_body_tree(
                raw_column,
                root_name="flat column view",
                path_parts=["root", "flat_column"],
            )
            strict_tree_source = "column"
        elif raw_row is not None:
            strict_tree = _build_flat_row_tree_with_trace_metadata(raw_row)
            strict_tree_source = "row"
        elif raw_column is not None:
            strict_tree = _build_flat_index_body_tree(
                raw_column,
                root_name="flat column view",
                path_parts=["root", "flat_column"],
            )
            strict_tree_source = "column_fallback"

        strict_trace = _compute_strict_trace(chain, strict_tree or [], final_answer)
        matched = [
            str(x or "").strip()
            for x in (strict_trace.get("matched_node_ids", []) if isinstance(strict_trace, dict) else [])
            if str(x or "").strip()
        ]
        path_nodes = [
            str(x or "").strip()
            for x in (strict_trace.get("path_node_order", []) if isinstance(strict_trace, dict) else [])
            if str(x or "").strip()
        ]
        answer_node = str((strict_trace.get("answer_node_id", "") if isinstance(strict_trace, dict) else "") or "").strip()
        sample_id = answer_node or (path_nodes[0] if path_nodes else (matched[0] if matched else ""))
        logger.info(
            f"【debug】trace_sample_id view_mode={normalized_mode}, source={strict_tree_source}, sample_id={sample_id or 'N/A'}"
        )
        return JSONResponse({
            "success": True,
            "conversation_id": conversation_id,
            "view_mode": normalized_mode,
            "tree_source": strict_tree_source,
            "sample_id": sample_id,
            "answer_node_id": answer_node,
            "path_node_order": path_nodes[:30],
            "matched_node_ids": matched[:30],
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


@app.get("/api/debug/highlight-exact")
async def debug_highlight_exact(
    conversation_id: str = "",
    canonical_id: str = "",
    view_mode: str = "column",
):
    """
    Return a deterministic one-node highlight payload (exact canonical id only).
    """
    try:
        conversation_id = str(conversation_id or "").strip()
        canonical_id = str(canonical_id or "").strip()
        normalized_mode = str(view_mode or "").strip().lower()
        if normalized_mode not in {"row", "column"}:
            normalized_mode = "row"
        if not conversation_id:
            return JSONResponse({"success": False, "error": "conversation_id is required"}, status_code=400)
        if not canonical_id:
            return JSONResponse({"success": False, "error": "canonical_id is required"}, status_code=400)

        tree_node = _build_flat_tree_node_for_view_mode(conversation_id, normalized_mode)
        if not isinstance(tree_node, dict):
            return JSONResponse({"success": False, "error": "tree data not found"}, status_code=404)

        catalog_path = _write_debug_frontend_node_catalog(conversation_id, normalized_mode, tree_node)
        catalog_rows = _collect_frontend_tree_node_catalog(tree_node)
        canonical_set = {
            str(row.get("canonicalId", "")).strip()
            for row in catalog_rows
            if str(row.get("canonicalId", "")).strip()
        }
        resolved_frontend_ids = [
            str(row.get("id", "")).strip()
            for row in catalog_rows
            if str(row.get("canonicalId", "")).strip() == canonical_id and str(row.get("id", "")).strip()
        ]
        exists = canonical_id in canonical_set and len(resolved_frontend_ids) > 0
        target_node_id = resolved_frontend_ids[0] if resolved_frontend_ids else canonical_id
        highlight_payload = {
            "nodeIds": [target_node_id],
            "edgeIds": [],
            "answerNodeId": target_node_id,
            "focusNodeId": target_node_id,
            "resetTraceHits": True,
            "deriveEdges": False,
        }
        logger.info(
            f"【debug】highlight_exact view_mode={normalized_mode}, canonical_id={canonical_id}, "
            f"exists_in_catalog={exists}, resolved_frontend_ids={resolved_frontend_ids[:3]}, "
            f"conversation_id={conversation_id}"
        )
        return JSONResponse({
            "success": True,
            "conversation_id": conversation_id,
            "view_mode": normalized_mode,
            "canonical_id": canonical_id,
            "exists_in_catalog": exists,
            "resolved_frontend_ids": resolved_frontend_ids,
            "target_node_id": target_node_id,
            "debug_frontend_node_catalog": catalog_path,
            "highlight_payload": highlight_payload,
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


@app.get("/api/history/{conversation_id}/tree-structure")
async def get_history_tree_structure(conversation_id: str):
    """Return simplified tree nodes/edges for history thumbnails."""
    history_dir = os.path.join("history", conversation_id)
    tree_path = os.path.join(history_dir, "temp.column.json")
    if not os.path.exists(tree_path):
        _rebuild_history_tree_snapshot_from_files(conversation_id)
    if not os.path.exists(tree_path):
        return JSONResponse({
            "success": False,
            "error": "Tree snapshot not found"
        }, status_code=404)
    try:
        with open(tree_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if _looks_like_empty_tree_payload(raw):
            if _rebuild_history_tree_snapshot_from_files(conversation_id):
                with open(tree_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
        # 兼容两类历史快照：
        # 1) canonical tree: {name, children}
        # 2) plain dict/list table snapshot
        # 统一借助 _build_trace_node_index 解析，再截断为缩略图用的轻量结构
        node_index = _build_trace_node_index(raw)
        parent_map = {nid: (info.get("parent") or None) for nid, info in node_index.items()}
        children_map: Dict[Optional[str], List[str]] = {}
        for nid, parent_id in parent_map.items():
            children_map.setdefault(parent_id, []).append(nid)

        roots = children_map.get(None, [])
        roots.sort(key=lambda nid: str(node_index.get(nid, {}).get("name", "")))

        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        max_nodes = 42
        emitted = 0

        queue: List[tuple[str, Optional[str], int]] = [(rid, None, 0) for rid in roots]
        while queue and emitted < max_nodes:
            nid, parent_thumb_id, depth = queue.pop(0)
            info = node_index.get(nid, {})
            thumb_id = f"n{emitted + 1}"
            nodes.append({
                "id": thumb_id,
                "name": str(info.get("name", "")),
                "depth": depth
            })
            if parent_thumb_id:
                edges.append({"from": parent_thumb_id, "to": thumb_id})
            emitted += 1
            for cid in children_map.get(nid, []):
                if emitted >= max_nodes:
                    break
                queue.append((cid, thumb_id, depth + 1))

        return JSONResponse({
            "success": True,
            "nodes": nodes,
            "edges": edges
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


@app.get("/api/history/{conversation_id}/tree-chat")
async def get_history_tree_chat(conversation_id: str):
    conversation_id = str(conversation_id or "").strip()
    if not conversation_id:
        return JSONResponse({
            "success": False,
            "error": "conversation_id is required"
        }, status_code=400)
    try:
        chat_path = _history_tree_chat_path(conversation_id)
        if not os.path.exists(chat_path):
            return JSONResponse({
                "success": True,
                "chat_html": ""
            })
        with open(chat_path, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
        chat_html = str(data.get("chat_html", "") or "")
        return JSONResponse({
            "success": True,
            "chat_html": chat_html
        })
    except Exception as e:
        logger.error(f"读取树聊天历史失败: {e}")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


@app.post("/api/history/{conversation_id}/tree-chat")
async def save_history_tree_chat(conversation_id: str, request: Request):
    conversation_id = str(conversation_id or "").strip()
    if not conversation_id:
        return JSONResponse({
            "success": False,
            "error": "conversation_id is required"
        }, status_code=400)
    try:
        payload = await request.json()
        chat_html = str((payload or {}).get("chat_html", "") or "")
        chat_path = _history_tree_chat_path(conversation_id)
        with open(chat_path, "w", encoding="utf-8") as f:
            json.dump({
                "conversation_id": conversation_id,
                "chat_html": chat_html,
                "updated_at": datetime.now().isoformat(timespec="seconds")
            }, f, ensure_ascii=False, indent=2)
        return JSONResponse({
            "success": True
        })
    except Exception as e:
        logger.error(f"保存树聊天历史失败: {e}")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


@app.get("/api/history/{conversation_id}/tree-images")
async def get_history_tree_images(conversation_id: str):
    conversation_id = str(conversation_id or "").strip()
    if not conversation_id:
        return JSONResponse({
            "success": False,
            "error": "conversation_id is required"
        }, status_code=400)
    try:
        path = _history_tree_images_path(conversation_id)
        images: List[Dict[str, Any]] = []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
            images = data.get("images", []) if isinstance(data, dict) else []
            if not isinstance(images, list):
                images = []

        # 兼容旧记录：若未保存 tree_images.json，则自动扫描 history/<id>/ 下的图片文件并回传
        if not images:
            history_dir = os.path.join("history", conversation_id)
            exts = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
            discovered: List[str] = []
            if os.path.isdir(history_dir):
                for name in sorted(os.listdir(history_dir)):
                    full = os.path.join(history_dir, name)
                    if not os.path.isfile(full):
                        continue
                    if os.path.splitext(name)[1].lower() not in exts:
                        continue
                    discovered.append(name)
                images_dir = os.path.join(history_dir, "images")
                if os.path.isdir(images_dir):
                    for name in sorted(os.listdir(images_dir)):
                        full = os.path.join(images_dir, name)
                        if not os.path.isfile(full):
                            continue
                        if os.path.splitext(name)[1].lower() not in exts:
                            continue
                        discovered.append(f"images/{name}")
            for idx, rel_name in enumerate(discovered):
                images.append({
                    "id": f"legacy_img_{idx+1}",
                    "name": os.path.basename(rel_name),
                    "url": f"/history-assets/{conversation_id}/{rel_name}",
                    "x": 230 + (idx % 3) * 270,
                    "y": 170 + (idx // 3) * 180,
                    "width": 240,
                    "height": 150
                })
            # 首次发现后落盘，后续可保存拖拽布局
            if images:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump({
                        "conversation_id": conversation_id,
                        "updated_at": datetime.now().isoformat(timespec="seconds"),
                        "images": images
                    }, f, ensure_ascii=False, indent=2)
        return JSONResponse({"success": True, "images": images})
    except Exception as e:
        logger.error(f"读取树图片布局失败: {e}")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


@app.post("/api/history/{conversation_id}/tree-images")
async def save_history_tree_images(conversation_id: str, request: Request):
    conversation_id = str(conversation_id or "").strip()
    if not conversation_id:
        return JSONResponse({
            "success": False,
            "error": "conversation_id is required"
        }, status_code=400)
    try:
        payload = await request.json()
        images = (payload or {}).get("images", [])
        if not isinstance(images, list):
            images = []
        normalized: List[Dict[str, Any]] = []
        for item in images:
            if not isinstance(item, dict):
                continue
            normalized.append({
                "id": str(item.get("id", "") or ""),
                "name": str(item.get("name", "") or ""),
                "url": str(item.get("url", "") or ""),
                "x": float(item.get("x", 0) or 0),
                "y": float(item.get("y", 0) or 0),
                "width": float(item.get("width", 220) or 220),
                "height": float(item.get("height", 140) or 140),
            })
        path = _history_tree_images_path(conversation_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "conversation_id": conversation_id,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "images": normalized
            }, f, ensure_ascii=False, indent=2)
        return JSONResponse({"success": True})
    except Exception as e:
        logger.error(f"保存树图片布局失败: {e}")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


@app.post("/api/history/{conversation_id}/tree-images/upload")
async def upload_history_tree_image(conversation_id: str, file: UploadFile = File(...)):
    conversation_id = str(conversation_id or "").strip()
    if not conversation_id:
        return JSONResponse({
            "success": False,
            "error": "conversation_id is required"
        }, status_code=400)
    if not file or not str(file.content_type or "").startswith("image/"):
        return JSONResponse({
            "success": False,
            "error": "Only image files are allowed"
        }, status_code=400)
    try:
        ext = os.path.splitext(file.filename or "")[1].lower()
        if ext not in [".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"]:
            ext = ".png"
        safe_base = re.sub(r"[^a-zA-Z0-9._-]+", "_", os.path.splitext(file.filename or "image")[0]).strip("_") or "image"
        uid = uuid.uuid4().hex[:10]
        file_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_base}_{uid}{ext}"
        images_dir = _history_images_dir(conversation_id)
        save_path = os.path.join(images_dir, file_name)
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        image_id = f"img_{uid}"
        url = f"/history-assets/{conversation_id}/images/{file_name}"
        return JSONResponse({
            "success": True,
            "image": {
                "id": image_id,
                "name": file.filename or file_name,
                "url": url,
                "x": 240,
                "y": 160,
                "width": 240,
                "height": 150
            }
        })
    except Exception as e:
        logger.error(f"上传树图片失败: {e}")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


@app.delete("/api/history/{conversation_id}/tree-images/{image_id}")
async def delete_history_tree_image(conversation_id: str, image_id: str, request: Request):
    conversation_id = str(conversation_id or "").strip()
    image_id = str(image_id or "").strip()
    if not conversation_id:
        return JSONResponse({
            "success": False,
            "error": "conversation_id is required"
        }, status_code=400)
    if not image_id:
        return JSONResponse({
            "success": False,
            "error": "image_id is required"
        }, status_code=400)

    def _safe_delete_rel(history_dir: str, rel_path: str) -> bool:
        rel = str(rel_path or "").strip().lstrip("/")
        if not rel:
            return False
        abs_target = os.path.normpath(os.path.join(history_dir, rel))
        history_abs = os.path.abspath(history_dir)
        target_abs = os.path.abspath(abs_target)
        if not (target_abs == history_abs or target_abs.startswith(history_abs + os.sep)):
            return False
        if not os.path.isfile(target_abs):
            return False
        try:
            os.remove(target_abs)
            return True
        except Exception:
            return False

    try:
        payload: Dict[str, Any] = {}
        try:
            payload = await request.json()
            if not isinstance(payload, dict):
                payload = {}
        except Exception:
            payload = {}

        history_dir = os.path.join("history", conversation_id)
        os.makedirs(history_dir, exist_ok=True)
        path = _history_tree_images_path(conversation_id)

        images: List[Dict[str, Any]] = []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
            if isinstance(data, dict) and isinstance(data.get("images"), list):
                images = data.get("images", [])

        payload_url = str(payload.get("url", "") or "").strip()
        payload_name = str(payload.get("name", "") or "").strip()

        removed_item: Optional[Dict[str, Any]] = None
        kept: List[Dict[str, Any]] = []
        for item in images:
            if not isinstance(item, dict):
                continue
            iid = str(item.get("id", "") or "")
            iurl = str(item.get("url", "") or "")
            iname = str(item.get("name", "") or "")
            matched = False
            if iid and iid == image_id:
                matched = True
            elif payload_url and iurl and iurl == payload_url:
                matched = True
            elif payload_name and iname and iname == payload_name:
                matched = True
            if matched and removed_item is None:
                removed_item = item
                continue
            kept.append(item)

        # 更新布局文件（即使未命中也保持幂等）
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "conversation_id": conversation_id,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "images": kept
            }, f, ensure_ascii=False, indent=2)

        deleted_file = False
        candidate_urls = []
        candidate_names = []
        if removed_item:
            candidate_urls.append(str(removed_item.get("url", "") or ""))
            candidate_names.append(str(removed_item.get("name", "") or ""))
        if payload_url:
            candidate_urls.append(payload_url)
        if payload_name:
            candidate_names.append(payload_name)

        prefix = f"/history-assets/{conversation_id}/"
        for u in candidate_urls:
            if not u or not u.startswith(prefix):
                continue
            rel = u[len(prefix):]
            if _safe_delete_rel(history_dir, rel):
                deleted_file = True
                break

        if not deleted_file:
            for n in candidate_names:
                name_only = os.path.basename(n)
                if not name_only:
                    continue
                if _safe_delete_rel(history_dir, os.path.join("images", name_only)):
                    deleted_file = True
                    break
                if _safe_delete_rel(history_dir, name_only):
                    deleted_file = True
                    break

        return JSONResponse({
            "success": True,
            "removed": removed_item is not None,
            "file_deleted": deleted_file
        })
    except Exception as e:
        logger.error(f"删除树图片失败: {e}")
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
    candidates: List[str] = []
    fallback_temp: Optional[str] = None
    seen: set[str] = set()

    for name in sorted(os.listdir(history_dir)):
        lower = name.lower()
        if not (lower.endswith(".xlsx") or lower.endswith(".xls")):
            continue
        p = os.path.join(history_dir, name)
        ap = os.path.abspath(p)
        if ap in seen:
            continue
        # temp.xlsx 是处理中间副本，默认不展示在预览 tabs 中
        if lower == "temp.xlsx":
            fallback_temp = p
            seen.add(ap)
            continue
        candidates.append(p)
        seen.add(ap)

    # 只有没有其它 Excel 时，才回退到 temp.xlsx
    if not candidates and fallback_temp and os.path.exists(fallback_temp):
        candidates.append(fallback_temp)
    if not candidates:
        return JSONResponse({
            "success": False,
            "error": "No Excel files found for this conversation"
        }, status_code=404)

    tables = []
    for path in candidates[:8]:
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
        from core_functions import rebuild_feature_views_from_json
        data = await request.json()

        # Backward compatible payload:
        # 1) old: raw tree array
        # 2) new: { conversation_id, tree, canonical_tree }
        conversation_id = ""
        tree_payload = data
        canonical_tree_payload = None
        canonical_tree_mode = ""
        ui_root_name = ""
        if isinstance(data, dict) and "tree" in data:
            tree_payload = data.get("tree", [])
            conversation_id = str(data.get("conversation_id", "") or "").strip()
            canonical_tree_payload = data.get("canonical_tree", None)
            canonical_tree_mode = str(data.get("canonical_tree_mode", "") or "").strip().lower()
            ui_root_name = str(data.get("ui_root_name", "") or "").strip().lower()

        debug_dir = os.path.join("history", conversation_id) if conversation_id else "cache"
        os.makedirs(debug_dir, exist_ok=True)
        try:
            with open(os.path.join(debug_dir, "[debug]save.ui.tree.json"), "w", encoding="utf-8") as f:
                json.dump(tree_payload, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"[save_tree] write [debug]save.ui.tree.json failed: {e}")

        def _extract_root_name(payload):
            if isinstance(payload, list) and payload and isinstance(payload[0], dict):
                return str(payload[0].get("name", "") or "").strip().lower()
            if isinstance(payload, dict):
                return str(payload.get("name", "") or "").strip().lower()
            return ""

        # 后端统一策略：保存重建始终优先使用列模式结构，不依赖当前前端显示模式。
        save_input_payload = tree_payload
        root_name = _extract_root_name(tree_payload)
        column_tree_payload = None
        try:
            raw_column = _load_column_view_payload(conversation_id)
            if raw_column is not None:
                column_tree_node = _build_flat_index_body_tree(
                    raw_column,
                    root_name="flat column view",
                    path_parts=["root", "flat_column"],
                )
                column_tree_payload = [column_tree_node]
                target_dir = os.path.join("history", conversation_id) if conversation_id else "cache"
                os.makedirs(target_dir, exist_ok=True)
                column_ui_path = os.path.join(target_dir, "temp.column.ui.json")
                with open(column_ui_path, "w", encoding="utf-8") as f:
                    json.dump(column_tree_payload, f, ensure_ascii=False, indent=2)
                logger.info(
                    f"[save_tree] built/saved column-ui payload for rebuild: {column_ui_path}, input_root={root_name}"
                )
            else:
                logger.warning(
                    f"[save_tree] column view payload missing, fallback to incoming payload, input_root={root_name}"
                )
        except Exception as e:
            logger.warning(f"[save_tree] build column-ui payload failed: {e}")

        # 优先级：
        # 1) 前端逆函数 canonical_tree（必须是列模式导出）
        # 2) 后端基于 temp.column.json 构建的列模式前端树
        # 3) incoming tree 兜底
        if canonical_tree_payload is not None and canonical_tree_mode == "flat_column":
            logger.info(
                f"[save_tree] using canonical_tree payload from frontend, mode={canonical_tree_mode}, ui_root={ui_root_name}"
            )
            save_input_payload = canonical_tree_payload
        elif canonical_tree_payload is not None:
            logger.warning(
                f"[save_tree] ignore canonical_tree due to mode mismatch: mode={canonical_tree_mode}, ui_root={ui_root_name}"
            )
            if column_tree_payload is not None:
                save_input_payload = column_tree_payload
            else:
                save_input_payload = tree_payload
        else:
            if column_tree_payload is not None:
                save_input_payload = column_tree_payload
                logger.info("[save_tree] using backend column-ui payload for unified column rebuild")
            else:
                save_input_payload = tree_payload

        try:
            with open(os.path.join(debug_dir, "[debug]save.canonical.json"), "w", encoding="utf-8") as f:
                json.dump(save_input_payload, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"[save_tree] write [debug]save.canonical.json failed: {e}")

        ok, msg = rebuild_feature_views_from_json(
            save_input_payload,
            conversation_id=conversation_id,
        )
        
        if ok:
            return JSONResponse({
                "success": True,
                "message": "Tree saved to temp.column.json and temp1.json"
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
