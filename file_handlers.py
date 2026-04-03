import os
import types
import copy
from datetime import datetime
from tree_editor import editor, default_data
from tree_utils import json_to_tree_data
from new_tree_ui import build_new_tree_iframe_html
from core_functions import process_table_for_tree, clear_all, read_all_logs, build_and_save_tree_id_mappings
from utils.constants import LOG_DIR


def generate_keywords_summary(text_content, max_tokens=200, temperature=0.3):
    """Use an LLM to generate a keyword summary for text content."""
    if not text_content:
        return "No text content"
    
    # 截取文本的前1000个字符作为输入，避免超出LLM的上下文限制
    truncated_content = text_content[:1000] if len(text_content) > 1000 else text_content
    
    prompt = f"""
    Generate a concise keyword summary for the following text and capture the document's core content.
    
    Text:
    {truncated_content}
    
    Provide 3-5 keywords, separated by commas. If the content includes a topic or title, prioritize it.
    Return keywords only; do not add any explanations.
    """
    
    try:
        from core_functions import get_llm_generate
        keywords = get_llm_generate(prompt, max_tokens=max_tokens, temperature=temperature)
        # Clean up the output to keep only keywords.
        keywords = keywords.strip().replace("\n", "").strip(' "')
        return keywords if keywords else "Document summary"
    except Exception as e:
        print(f"[WARN] LLM生成关键词摘要失败: {e}")
        # If LLM summary fails, return a safe default.
        return "Document summary"


def merge_multiple_tables_to_tree(file_list, conversation_id=None):
    """将多个表格文件和文本文件的数据合并到一个根节点下"""
    merged_data = {}
    merged_column_data = {}
    merged_row_data = {}
    merged_artifact_manifest = {}
    processed_files = []  # 记录成功处理的文件
    failed_files = []     # 记录处理失败的文件
    
    for f in file_list:
        file_path = f.name
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext in [".xlsx", ".xls", ".docx", ".doc"]:
            # 处理表格文件
            try:
                wrapped_file = types.SimpleNamespace(name=f.name)
                data = process_table_for_tree(wrapped_file, conversation_id=conversation_id)
                
                if isinstance(data, dict) and data:
                    # 使用文件名作为表的根键名
                    file_name = os.path.splitext(os.path.basename(f.name))[0]
                    if file_name in merged_data:
                        # 如果文件名重复，添加序号
                        counter = 1
                        new_key = f"{file_name}_{counter}"
                        while new_key in merged_data:
                            counter += 1
                            new_key = f"{file_name}_{counter}"
                        file_name = new_key
                    
                    # 将当前表格数据添加到合并数据中
                    merged_data[file_name] = data

                    # 收集每个文件对应的 column 视图（显式两层循环：文件 -> sheet）
                    try:
                        if conversation_id:
                            column_path = os.path.join("history", conversation_id, "temp.column.json")
                            if os.path.exists(column_path):
                                import json
                                with open(column_path, "r", encoding="utf-8") as cf:
                                    column_payload = json.load(cf)
                                if isinstance(column_payload, dict):
                                    merged_column_data[file_name] = {}
                                    for sheet_name, sheet_body in column_payload.items():
                                        merged_column_data[file_name][str(sheet_name)] = sheet_body
                                    print(
                                        f"[DEBUG] 列视图聚合 file={file_name}, sheets={list(merged_column_data[file_name].keys())}"
                                    )
                                else:
                                    # 防御：非 dict 形态时按原样挂到文件下
                                    merged_column_data[file_name] = column_payload
                    except Exception as e:
                        print(f"[WARN] 读取列视图失败 {f.name}: {e}")

                    # 收集每个文件对应的 row 视图（显式两层循环：文件 -> sheet）
                    try:
                        if conversation_id:
                            row_path = os.path.join("history", conversation_id, "temp1.json")
                            if os.path.exists(row_path):
                                import json
                                with open(row_path, "r", encoding="utf-8") as rf:
                                    row_payload = json.load(rf)
                                if isinstance(row_payload, dict):
                                    merged_row_data[file_name] = {}
                                    for sheet_name, sheet_body in row_payload.items():
                                        merged_row_data[file_name][str(sheet_name)] = sheet_body
                                    print(
                                        f"[DEBUG] 行视图聚合 file={file_name}, sheets={list(merged_row_data[file_name].keys())}"
                                    )
                                else:
                                    merged_row_data[file_name] = row_payload
                    except Exception as e:
                        print(f"[WARN] 读取行视图失败 {f.name}: {e}")

                    # 收集每个文件对应的 artifact manifest，避免被后续文件覆盖
                    try:
                        if conversation_id:
                            manifest_path = os.path.join("history", conversation_id, "temp.artifacts.json")
                            if os.path.exists(manifest_path):
                                import json
                                with open(manifest_path, "r", encoding="utf-8") as mf:
                                    manifest_payload = json.load(mf)
                                if isinstance(manifest_payload, dict) and manifest_payload:
                                    # 单文件场景一般只有一个 key；这里统一挂到外层 file_name 下
                                    if file_name in manifest_payload and isinstance(manifest_payload[file_name], dict):
                                        merged_artifact_manifest[file_name] = manifest_payload[file_name]
                                    else:
                                        first_key = next(iter(manifest_payload.keys()))
                                        first_item = manifest_payload.get(first_key, {})
                                        if isinstance(first_item, dict):
                                            merged_artifact_manifest[file_name] = first_item
                    except Exception as e:
                        print(f"[WARN] 读取产物清单失败 {f.name}: {e}")

                    processed_files.append(f.name)
                else:
                    failed_files.append(f"表格文件处理为空: {f.name}")
            except Exception as e:
                print(f"[WARN] 处理表格文件 {f.name} 失败: {e}")
                failed_files.append(f"表格文件处理失败: {f.name} - {str(e)}")
        
        elif ext in [".txt", ".md", ".json"]:
            # 处理文本文件
            try:
                with open(f.name, 'r', encoding='utf-8') as txt_file:
                    text_content = txt_file.read()
                
                # 使用LLM生成关键词摘要作为父节点名称
                keywords = generate_keywords_summary(text_content)
                
                # 使用文件名作为基础，结合关键词生成唯一节点名
                base_name = os.path.splitext(os.path.basename(f.name))[0]
                node_name = f"{base_name}[{keywords}]"
                
                # 如果节点名重复，添加序号
                if node_name in merged_data:
                    counter = 1
                    new_key = f"{node_name}_{counter}"
                    while new_key in merged_data:
                        counter += 1
                        new_key = f"{node_name}_{counter}"
                    node_name = new_key
                
                # 创建包含关键词摘要的父节点和文本内容的子节点
                text_node = {
                    "content": text_content,  # 文本内容作为子节点
                    "summary": keywords       # 关键词摘要
                }
                
                merged_data[node_name] = text_node
                processed_files.append(f.name)
                
            except Exception as e:
                print(f"[WARN] 处理文本文件 {f.name} 失败: {e}")
                # 如果处理失败，使用文件名作为节点名
                base_name = os.path.splitext(os.path.basename(f.name))[0]
                node_name = f"{base_name}[处理失败]"
                
                if node_name in merged_data:
                    counter = 1
                    new_key = f"{node_name}_{counter}"
                    while new_key in merged_data:
                        counter += 1
                        new_key = f"{node_name}_{counter}"
                    node_name = new_key
                
                merged_data[node_name] = {"content": f"Unable to read file content: {str(e)}"}
                failed_files.append(f"Text file processing failed: {f.name} - {str(e)}")
        else:
            failed_files.append(f"不支持的文件类型: {f.name} ({ext})")
    
    # 验证步骤：检查是否所有上传的文件都被正确处理
    total_files = len(file_list)
    processed_count = len(processed_files)
    failed_count = len(failed_files)
    
    print(f"[INFO] 文件处理结果 - 总计: {total_files}, 成功: {processed_count}, 失败: {failed_count}")
    if processed_count > 0:
        print(f"[INFO] 成功处理的文件: {processed_files}")
    if failed_count > 0:
        print(f"[INFO] 处理失败的文件: {failed_files}")
    
    # 如果有处理失败的文件，记录警告信息
    if failed_count > 0:
        print(f"[WARN] 有 {failed_count} 个文件未能成功处理")

    # 多文件场景下，将每个文件的列视图汇总到最外层 dict，避免只保留最后一个文件
    if conversation_id and merged_column_data:
        try:
            import json
            column_out = os.path.join("history", conversation_id, "temp.column.json")
            with open(column_out, "w", encoding="utf-8") as f:
                json.dump(merged_column_data, f, ensure_ascii=False, indent=4)
            print(f"[INFO] 已写入多文件列视图: {column_out}, keys={list(merged_column_data.keys())}")
        except Exception as e:
            print(f"[WARN] 写入多文件列视图失败: {e}")

    if conversation_id and merged_row_data:
        try:
            import json
            row_out = os.path.join("history", conversation_id, "temp1.json")
            with open(row_out, "w", encoding="utf-8") as f:
                json.dump(merged_row_data, f, ensure_ascii=False, indent=4)
            print(f"[INFO] 已写入多文件行视图: {row_out}, keys={list(merged_row_data.keys())}")
        except Exception as e:
            print(f"[WARN] 写入多文件行视图失败: {e}")

    if conversation_id and merged_artifact_manifest:
        try:
            import json
            manifest_out = os.path.join("history", conversation_id, "temp.artifacts.json")
            with open(manifest_out, "w", encoding="utf-8") as f:
                json.dump(merged_artifact_manifest, f, ensure_ascii=False, indent=4)
            print(f"[INFO] 已写入多文件产物清单: {manifest_out}, keys={list(merged_artifact_manifest.keys())}")
        except Exception as e:
            print(f"[WARN] 写入多文件产物清单失败: {e}")

    if conversation_id and merged_column_data and merged_row_data:
        try:
            build_and_save_tree_id_mappings(os.path.join("history", conversation_id), typed_root_name="HO_TREE")
        except Exception as e:
            print(f"[WARN] 写入多文件ID映射失败: {e}")
    
    return merged_data, processed_files, failed_files


def load_from_upload(file):
    """上传文件后，根据文件类型处理并刷新界面。支持单个文件或多个文件"""
    if not file:
        return build_new_tree_iframe_html(), [], ""
    
    # 生成对话ID并创建历史记录文件夹
    from core_functions import generate_conversation_id, create_conversation_record
    from tree_handlers import persist_tree
    conversation_id = generate_conversation_id()
    conversation_dir = os.path.join("history", conversation_id)
    os.makedirs(conversation_dir, exist_ok=True)
    
    # 复制原始文件到历史记录文件夹
    if isinstance(file, list):
        # 多个文件
        for f in file:
            original_file_path = f.name
            file_name = os.path.basename(original_file_path)
            dest_file_path = os.path.join(conversation_dir, file_name)
            try:
                import shutil
                shutil.copy2(original_file_path, dest_file_path)
            except Exception as e:
                print(f"[WARN] 复制原始文件 {original_file_path} 到历史记录文件夹失败: {e}")
    else:
        # 单个文件
        original_file_path = file.name
        file_name = os.path.basename(original_file_path)
        dest_file_path = os.path.join(conversation_dir, file_name)
        try:
            import shutil
            shutil.copy2(original_file_path, dest_file_path)
        except Exception as e:
            print(f"[WARN] 复制原始文件 {original_file_path} 到历史记录文件夹失败: {e}")
    
    # 检查是否是多个文件
    if isinstance(file, list):
        # 多个文件处理
        try:
            from core_functions import analyze_multiple_files_for_route
            route = analyze_multiple_files_for_route(file)
            
            # table_processed 标记是否至少有一个文件被处理
            table_processed = False
            processed_files = []
            failed_files = []

            # 无论 route 如何，只要上传列表中包含可构树文件，都应该生成树结构。
            # 这样可以避免“xlsx + 图片”被 route 判到非 hotree 时，编辑器出现空树。
            all_files = []
            for f in file:
                file_path = f.name
                ext = os.path.splitext(file_path)[1].lower()
                if ext in [".xlsx", ".xls", ".docx", ".doc", ".txt", ".md", ".json"]:
                    all_files.append(f)
                elif ext in [".csv"]:
                    # csv 先走预处理缓存（保持原行为）
                    try:
                        wrapped_file = types.SimpleNamespace(name=f.name)
                        process_table_for_tree(wrapped_file, conversation_id=conversation_id)
                        processed_files.append(f.name)
                    except Exception as e:
                        failed_files.append(f"CSV file processing failed: {f.name} - {str(e)}")

            if all_files:
                merged_data, merged_ok_files, merged_failed = merge_multiple_tables_to_tree(all_files, conversation_id=conversation_id)
                processed_files.extend(merged_ok_files)
                failed_files.extend(merged_failed)
                if len(merged_ok_files) > 0:
                    root_name = "All_Documents"
                    root_body = merged_data
                    tree_data = json_to_tree_data(root_body, name=root_name)
                    editor.data = [tree_data] if tree_data else copy.deepcopy(default_data)
                    persist_tree()
                    table_processed = True
                    print(f"[INFO] 树结构构建完成，成功处理 {len(merged_ok_files)} 个文件")
                else:
                    editor.data = copy.deepcopy(default_data)
                    persist_tree()
                    table_processed = len(processed_files) > 0
            else:
                # 没有可构树文件（比如全是图片），保持空树
                editor.data = copy.deepcopy(default_data)
                persist_tree()
                table_processed = len(processed_files) > 0
            
            # 返回树界面、聊天消息和状态
            if table_processed:
                # 根据 route 确定处理的文件数量
                if 'processed_files' in locals():
                    processed_count = len(processed_files)
                else:
                    # 对于非 HOTree 路线，计算实际处理的文件数量
                    processed_count = 0
                    for f in file:
                        ext = os.path.splitext(f.name)[1].lower()
                        if ext in [".xlsx", ".xls", ".csv", ".txt", ".md", ".json", ".docx", ".doc"]:
                            processed_count += 1
                
                # 根据上传的文件类型和数量提供详细消息
                file_details = {}
                for f in file:
                    ext = os.path.splitext(f.name)[1].lower()
                    if ext in file_details:
                        file_details[ext] += 1
                    else:
                        file_details[ext] = 1
                
                # 构建详细的消息
                detail_parts = []
                for ext, count in file_details.items():
                    if ext in [".xlsx", ".xls", ".docx", ".doc"]:
                        detail_parts.append(f"{count} spreadsheet file(s)")
                    elif ext in [".txt", ".md", ".json"]:
                        detail_parts.append(f"{count} text file(s)")
                    elif ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp"]:
                        detail_parts.append(f"{count} image file(s)")
                    else:
                        detail_parts.append(f"{count} other file(s)")
                
                detail_msg = ", ".join(detail_parts)
                detailed_content = f"✅ Successfully uploaded and parsed {processed_count} file(s) ({detail_msg}). You can start asking questions now."
                
                # 创建对话记录
                file_names = [os.path.basename(f.name) for f in file]
                upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                summary = f"Uploaded {len(file)} file(s)"
                create_conversation_record(conversation_id, file_names, upload_time, summary)
                
                return build_new_tree_iframe_html(), [{"role": "assistant", "content": detailed_content}], conversation_id
            else:
                # 如果有文件被上传但没有表格被处理，仍然显示一般性上传消息
                if file:  # 如果有文件上传
                    file_count = len(file) if isinstance(file, list) else 1
                    
                    # 根据文件类型提供更详细的反馈
                    file_details = {}
                    for f in file:
                        ext = os.path.splitext(f.name)[1].lower()
                        if ext in file_details:
                            file_details[ext] += 1
                        else:
                            file_details[ext] = 1
                    
                    # 构建详细的消息
                    detail_parts = []
                    for ext, count in file_details.items():
                        if ext in [".xlsx", ".xls", ".docx", ".doc"]:
                            detail_parts.append(f"{count} spreadsheet file(s)")
                        elif ext in [".txt", ".md", ".json"]:
                            detail_parts.append(f"{count} text file(s)")
                        elif ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp"]:
                            detail_parts.append(f"{count} image file(s)")
                        else:
                            detail_parts.append(f"{count} other file(s)")
                    
                    detail_msg = ", ".join(detail_parts)
                    detailed_content = f"✅ Uploaded {file_count} file(s) ({detail_msg}). Files are ready for further processing."
                    
                    # 创建对话记录
                    file_names = [os.path.basename(f.name) for f in file]
                    upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    summary = f"Uploaded {len(file)} file(s)"
                    create_conversation_record(conversation_id, file_names, upload_time, summary)
                    
                    return build_new_tree_iframe_html(), [{"role": "assistant", "content": detailed_content}], conversation_id
                else:
                    # 创建对话记录
                    file_names = []
                    upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    summary = "No files uploaded"
                    create_conversation_record(conversation_id, file_names, upload_time, summary)
                    
                    return build_new_tree_iframe_html(), [], conversation_id
        except Exception as e:
            print(f"[WARN] 多文件上传解析失败: {e}")
            editor.data = copy.deepcopy(default_data)
            persist_tree()
            
            # 创建对话记录
            file_names = [os.path.basename(f.name) for f in file] if file else []
            upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            summary = f"Upload failed: {str(e)}"
            create_conversation_record(conversation_id, file_names, upload_time, summary)
            
            return build_new_tree_iframe_html(), [{"role": "assistant", "content": f"File processing failed: {str(e)}"}], conversation_id
    else:
        # 单个文件处理
        try:
            # 获取文件信息
            file_path = file.name
            file_size = os.path.getsize(file_path)
            
            # 读取文件内容摘要
            file_content = None
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    file_content = f.read(1000)
            except:
                # 二进制文件无法读取内容
                pass
            
            # 确定处理线路
            from core_functions import determine_processing_route
            route = determine_processing_route(file_path, file_size, file_content)
            
            # 预处理表格文件（无论最终路由如何），以便后续问题回答时不需要重复处理
            table_processed = False
            ext = os.path.splitext(file_path)[1].lower()
            if ext in [".xlsx", ".xls", ".csv", ".txt", ".md", ".json", ".docx", ".doc"]:
                # 对表格文件进行预处理，不管最终路由是什么
                wrapped_file = types.SimpleNamespace(name=file.name)
                process_table_for_tree(wrapped_file, conversation_id=conversation_id)  # 预处理并缓存结果
                table_processed = True
            
            # 根据线路处理文件（用于更新界面显示）
            if route == "hotree":
                # 只有表格文件才处理为HOTree
                wrapped_file = types.SimpleNamespace(name=file.name)
                data = process_table_for_tree(wrapped_file, conversation_id=conversation_id)
                # 取第一层表名作为根节点名（若无键则退回默认），并作为入口 name
                if isinstance(data, dict) and data:
                    root_name = next(iter(data.keys()))
                    root_body = data.get(root_name, data)
                    # 若有多条记录，将第2条及之后封装为 dict 追加到首条的列表中
                    if len(data) > 1:
                        body_list = root_body if isinstance(root_body, list) else [root_body]
                        for k, v in list(data.items())[1:]:
                            body_list.append({k: v})
                        root_body = body_list
                else:
                    root_name = "root"
                    root_body = data
                # 直接传入根节点对应的 children/列表，避免额外包一层
                tree_data = json_to_tree_data(root_body, name=root_name)
                editor.data = [tree_data] if tree_data else copy.deepcopy(default_data)
                persist_tree()
            else:
                # 非表格文件不处理为HOTree，保持空树
                editor.data = copy.deepcopy(default_data)
                persist_tree()
            
            # 返回树界面、聊天消息和状态
            if table_processed:
                # 根据文件类型返回不同的状态消息和详细内容
                ext = os.path.splitext(file_path)[1].lower()
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                
                if ext in [".xlsx", ".xls", ".docx", ".doc"]:
                    detailed_content = f"✅ Spreadsheet '{base_name}' uploaded and parsed successfully! You can start asking questions now."
                elif ext in [".txt", ".md", ".json"]:
                    detailed_content = f"✅ Text file '{base_name}' uploaded and processed successfully! You can start asking questions now."
                else:
                    detailed_content = f"✅ File '{base_name}' uploaded and parsed successfully! You can start asking questions now."
                
                # 创建对话记录
                file_names = [os.path.basename(file.name)]
                upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                summary = "Uploaded 1 file"
                create_conversation_record(conversation_id, file_names, upload_time, summary)
                
                return build_new_tree_iframe_html(), [{"role": "assistant", "content": detailed_content}], conversation_id
            else:
                # 单个非表格文件上传，显示一般性上传消息
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                ext = os.path.splitext(file_path)[1].lower()
                if ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp"]:
                    detailed_content = f"✅ Image file '{base_name}' uploaded. It will be used for VLM processing."
                    
                    # 创建对话记录
                    file_names = [os.path.basename(file.name)]
                    upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    summary = "Uploaded 1 file"
                    create_conversation_record(conversation_id, file_names, upload_time, summary)
                    
                    return build_new_tree_iframe_html(), [{"role": "assistant", "content": detailed_content}], conversation_id
                else:
                    detailed_content = f"✅ File '{base_name}' uploaded and ready for further processing."
                    
                    # 创建对话记录
                    file_names = [os.path.basename(file.name)]
                    upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    summary = "Uploaded 1 file"
                    create_conversation_record(conversation_id, file_names, upload_time, summary)
                    
                    return build_new_tree_iframe_html(), [{"role": "assistant", "content": detailed_content}], conversation_id
        except Exception as e:
            print(f"[WARN] 上传解析失败: {e}")
            
            # 创建对话记录
            file_names = [os.path.basename(file.name)] if file else []
            upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            summary = f"Upload failed: {str(e)}"
            create_conversation_record(conversation_id, file_names, upload_time, summary)
            
            return build_new_tree_iframe_html(), [{"role": "assistant", "content": f"File upload failed: {str(e)}"}], conversation_id


def clear_ui():
    """调用清理逻辑并重置界面输出。"""
    clear_all()
    editor.reset()
    return (
        None,  # table_input
        "",    # question_input
        "",    # answer_output
        build_new_tree_iframe_html(),  # tree_iframe
        read_all_logs(log_dir=LOG_DIR)  # log_output
    )
