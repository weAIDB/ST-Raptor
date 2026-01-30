import gradio as gr
from gradio import update
import os
from config import api_config, save_api_config
from utils.constants import LOG_DIR
from core_functions import process_question_only, read_all_logs, get_thinking_chain, process_file_with_route
from new_tree_ui import build_new_tree_iframe_html
from file_handlers import load_from_upload, clear_ui


def create_interface():
    with gr.Blocks(
        title="ST-Raptor 表格问答系统",
        theme=gr.themes.Soft(),
        css="""
        .container { max-width: 1400px; margin: 0 auto; }
        .header { text-align: center; padding: 20px; }
        .sidebar { background: #f0f2f6; padding: 15px; border-radius: 8px; }
        .main-content { padding: 15px; }
        .tab-button { width: 100%; margin-bottom: 10px; }
        .H-OTree-output {
            height:600px !important;
            max-height: 600px !important;
            overflow-y: auto !important;
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
        .chat-container {
            height: 600px;
            display: flex;
            flex-direction: column;
        }
        .chatbox {
            flex: 1;
            overflow-y: auto;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 10px;
        }
        .log-container {
            height: 600px;
            overflow-y: auto;
            padding: 10px;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            background: #f8f9fa;
        }
        .split-panel {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            height: 600px;
        }

        """
    ) as demo:
        gr.Markdown("""
        <div class="header">
            <h1>📊 ST-Raptor 表格问答系统</h1>
            <p>上传 Excel 表格并使用自然语言提问，获取智能答案</p>
        </div>
        """)
        
        # 创建页面状态变量
        page_state = gr.State("chat")
        file_type_state = gr.State("")  # 记录当前上传的文件类型
        conversation_id_state = gr.State("")  # 记录当前对话ID
        
        # 整体两栏布局
        with gr.Row():
            # 左侧边栏 - 较窄，包含配置和页面切换
            with gr.Column(scale=1, elem_classes="sidebar"):
                gr.Markdown("### 页面导航")
                # 页面切换按钮
                chat_btn = gr.Button("💬 问答", variant="primary", elem_classes="tab-button")
                tree_btn = gr.Button("🌳 树视图", variant="secondary", elem_classes="tab-button")
                logs_btn = gr.Button("📜 日志与思维链", variant="secondary", elem_classes="tab-button")
                history_btn = gr.Button("📚 历史记录", variant="secondary", elem_classes="tab-button")
                
                gr.Markdown("### ⚙️ API配置")
                with gr.Accordion("📝 LLM配置", open=False):
                    with gr.Column(elem_classes="accordion-content"):
                        with gr.Row():
                            llm_api_key = gr.Textbox(
                                label="API Key", 
                                value=api_config["llm_api_key"],
                                placeholder="请输入LLM API密钥",
                                type="password",
                                scale=2
                            )
                            llm_model = gr.Dropdown(
                                choices=["deepseek-v3.1", "deepseek-chat","gpt-4-turbo", "claude-3-opus-20240229", "qwen-max"],
                                value=api_config["llm_model"],
                                label="模型",
                                scale=1
                            )
                        llm_api_url = gr.Textbox(
                            label="API URL", 
                            value=api_config["llm_api_url"],
                            placeholder="请输入LLM API地址",
                            lines=1
                        )
                
                # VLM配置Accordion
                with gr.Accordion("🖼️ VLM配置", open=False):
                    with gr.Column(elem_classes="accordion-content"):
                        with gr.Row():
                            vlm_api_key = gr.Textbox(
                                label="API Key", 
                                value=api_config["vlm_api_key"],
                                placeholder="请输入VLM API密钥",
                                type="password",
                                scale=2
                            )
                            vlm_model = gr.Dropdown(
                                choices=["qwen3-vl-plus", "gpt-4-vision-preview", "claude-3-opus-20240229"],
                                value=api_config["vlm_model"],
                                label="模型",
                                scale=1
                            )
                        vlm_api_url = gr.Textbox(
                            label="API URL", 
                            value=api_config["vlm_api_url"],
                            placeholder="请输入VLM API地址",
                            lines=1
                        )
                
                # Embedding配置Accordion
                with gr.Accordion("📊 Embedding配置", open=False):
                    with gr.Column(elem_classes="accordion-content"):
                        with gr.Row():
                            embedding_api_key = gr.Textbox(
                                label="API Key", 
                                value=api_config["embedding_api_key"],
                                placeholder="请输入Embedding API密钥",
                                type="password",
                                scale=2
                            )
                            embedding_model = gr.Dropdown(
                                choices=["text-embedding-v1", "text-embedding-ada-002", "text-embedding-3-large"],
                                value=api_config["embedding_model"],
                                label="模型",
                                scale=1
                            )
                        embedding_api_url = gr.Textbox(
                            label="API URL", 
                            value=api_config["embedding_api_url"],
                            placeholder="请输入Embedding API地址",
                            lines=1
                        )
                
                # 高级设置Accordion
                with gr.Accordion("🔧 高级设置", open=False):
                    with gr.Column(elem_classes="accordion-content"):
                        with gr.Row():
                            temperature_slider = gr.Slider(
                                minimum=0.0, maximum=1.0, value=0.5, step=0.01,
                                label="Temperature (采样多样性)",
                                info="越大越随机，越小越确定",
                                scale=1
                            )
                            max_tokens_box = gr.Number(
                                value=1024, precision=0, label="Max Tokens (最大生成长度)",
                                info="生成答案的最大 token 数",
                                scale=1
                            )
                
                # 统一保存按钮和状态
                with gr.Row():
                    save_config_btn = gr.Button("💾 保存配置", variant="primary")
                    config_status = gr.HTML("", label="配置状态")
                
                # 绑定保存配置事件
                save_config_btn.click(
                    fn=lambda llm_key, llm_url, llm_m, vlm_key, vlm_url, vlm_m, emb_key, emb_url, emb_m: save_api_config({
                        "llm_api_key": llm_key,
                        "llm_api_url": llm_url,
                        "llm_model": llm_m,
                        "vlm_api_key": vlm_key,
                        "vlm_api_url": vlm_url,
                        "vlm_model": vlm_m,
                        "embedding_api_key": emb_key,
                        "embedding_api_url": emb_url,
                        "embedding_model": emb_m
                    }),
                    inputs=[llm_api_key, llm_api_url, llm_model, vlm_api_key, vlm_api_url, vlm_model, embedding_api_key, embedding_api_url, embedding_model],
                    outputs=[config_status]
                )
            
            # 右侧主内容区 - 可切换的页面
            with gr.Column(scale=4, elem_classes="main-content"):
                # 问答页面
                # 问答页面
                with gr.Column(visible=True) as chat_page:
                    gr.Markdown("### ❓ 表格问答")
                    
                    # 文件上传区域 - 横向布局
                    with gr.Row():
                        table_input = gr.File(
                            label="上传文件",
                            file_types=[".xlsx", ".xls", ".csv", ".txt", ".md", ".json", ".jpg", ".jpeg", ".png", ".gif", ".bmp"],
                            height=100,
                            scale=3,
                            file_count="multiple"
                        )
                        with gr.Column():
                            upload_btn = gr.Button("📤 上传文件", variant="secondary", scale=1)
                            clear_chat_btn = gr.Button("🗑️ 清除聊天记录", variant="secondary", scale=1)
                    
                    # 聊天框区域（核心修改：添加type="messages"）
                    with gr.Column(elem_classes="chat-container"):
                        chatbot = gr.Chatbot(elem_classes="chatbox", type="messages")
                        
                        with gr.Row():
                            question_input = gr.Textbox(
                                label="请输入您的问题",
                                lines=2,
                                placeholder="例如：销售总额是多少？哪个产品销量最高？",
                                scale=4
                            )
                            submit_btn = gr.Button("发送", variant="primary", scale=1)
                    
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

                # 聊天功能处理（适配新格式）
                def respond(message, chat_history, temperature, max_tokens, file=None, conversation_id=""):
                    """
                    适配type='messages'的Chatbot格式，添加严格的格式校验和异常处理
                    根据文件类型自动选择处理线路
                    """
                    from core_functions import reshape_question_with_context
                    
                    # 1. 确保chat_history是有效的列表（防止初始值异常）
                    if not isinstance(chat_history, list):
                        chat_history = []
                    
                    # 2. 确保用户输入不为空
                    if not message or message.strip() == "":
                        return "", chat_history
                    
                    try:
                        # 3. 使用上下文重塑问题
                        reshaped_message = reshape_question_with_context(message, chat_history, temperature)
                        
                        # 4. 根据文件类型选择处理线路
                        if file:
                            # 有文件上传，使用process_file_with_route处理
                            bot_message = process_file_with_route(file, reshaped_message, temperature, max_tokens, conversation_id)
                        else:
                            # 无文件上传，使用process_question_only处理（H-OTree线路）
                            bot_message = process_question_only(reshaped_message, temperature, max_tokens, conversation_id)
                        
                        # 处理返回值为空的情况
                        if bot_message is None or bot_message.strip() == "":
                            bot_message = "抱歉，未能获取到有效回答，请检查您的问题或配置。"
                    
                    except Exception as e:
                        # 捕获异常，返回友好提示
                        bot_message = f"回答生成失败：{str(e)}"
                    
                    # 5. 严格按照messages格式添加消息（确保键名和类型正确）
                    user_msg = {"role": "user", "content": message.strip()}
                    assistant_msg = {"role": "assistant", "content": bot_message.strip()}
                    
                    chat_history.append(user_msg)
                    chat_history.append(assistant_msg)
                    
                    # 6. 保存对话历史到文件
                    if conversation_id:
                        from core_functions import save_conversation_history
                        save_conversation_history(conversation_id, chat_history)
                    
                    # 7. 返回空输入框 + 新的聊天记录
                    return "", chat_history

                # 绑定发送按钮
                submit_btn.click(
                    fn=respond,
                    inputs=[question_input, chatbot, temperature_slider, max_tokens_box, table_input, conversation_id_state],
                    outputs=[question_input, chatbot]
                )

                # 绑定回车发送
                question_input.submit(
                    fn=respond,
                    inputs=[question_input, chatbot, temperature_slider, max_tokens_box, table_input, conversation_id_state],
                    outputs=[question_input, chatbot]
                )

                # 清除聊天记录（适配新格式）
                def clear_chat():
                    return [], None, ""

                clear_chat_btn.click(
                    fn=clear_chat,
                    inputs=[],
                    outputs=[chatbot, table_input, question_input]
                )
                
                # 树视图页面
                with gr.Column(visible=False) as tree_page:
                    gr.Markdown("### 🧭 New Tree（纯前端前端原型）")
                    tree_iframe = gr.HTML(value=build_new_tree_iframe_html(), elem_id="new-tree-html")
                    reload_btn = gr.Button("🔄 重新载入最新 JSON", variant="secondary")
                    reload_btn.click(
                        fn=lambda: build_new_tree_iframe_html(),
                        inputs=[],
                        outputs=[tree_iframe],
                        queue=False,
                    )
                    # 提供给前端 save 成功后的自动刷新接口（通过 event name 触发）
                    tree_reload_event = gr.Button(visible=False, elem_id="tree-reload-event")
                    tree_reload_event.click(
                        fn=lambda: build_new_tree_iframe_html(),
                        inputs=[],
                        outputs=[tree_iframe],
                        queue=False,
                    )
                
                # 日志与思维链页面
                with gr.Column(visible=False) as logs_page:
                    gr.Markdown("### 📜 日志与思维链条")
                    with gr.Row(elem_classes="split-panel"):
                        with gr.Column():
                            gr.Markdown("#### 实时日志")
                            log_output = gr.HTML(
                                label="终端日志",
                                value=read_all_logs(log_dir=LOG_DIR),
                                elem_classes="log-container"
                            )
                        with gr.Column():
                            gr.Markdown("#### 思维链条可视化")
                            thinking_chain_output = gr.JSON(
                                label="思维和检索链条",
                                value=get_thinking_chain(),
                                elem_classes="H-OTree-output"
                            )
                    clear_log_btn = gr.Button("🗑️ 清除日志", variant="secondary")
                
                # 历史记录页面
                with gr.Column(visible=False) as history_page:
                    gr.Markdown("### 📚 对话历史记录")
                    with gr.Row():
                        refresh_history_btn = gr.Button("🔄 刷新历史记录", variant="secondary")
                    
                    # 刷新历史记录的函数 - 添加操作列的按钮
                    def refresh_history():
                        from core_functions import get_conversation_records
                        records = get_conversation_records()
                        # 为每条记录添加一个"加载"按钮文本，但实际的按钮需要在界面上单独处理
                        import pandas as pd
                        # 创建一个新的记录列表，将操作列设置为"加载"文本
                        updated_records = []
                        for record in records:
                            updated_record = list(record)
                            updated_record[4] = "加载"  # 将操作列设为"加载"
                            updated_records.append(updated_record)
                        df = pd.DataFrame(updated_records, columns=["ID", "文件名", "上传时间", "内容摘要", "操作"])
                        return df
                    
                    # 历史记录表格
                    history_table = gr.DataFrame(
                        headers=["ID", "文件名", "上传时间", "内容摘要", "操作"],
                        datatype=["str", "str", "str", "str", "str"],
                        interactive=False,
                        elem_id="history-table"
                    )
                    
                    # 绑定刷新按钮
                    refresh_history_btn.click(
                        fn=refresh_history,
                        inputs=[],
                        outputs=[history_table]
                    )
                    
                    # 加载历史记录的函数
                    def load_history_by_id(conversation_id):
                        # 加载对话历史
                        from core_functions import load_conversation_history
                        history = load_conversation_history(conversation_id)
                        
                        # 添加调试日志
                        import logging
                        logging.info(f"Loaded history for {conversation_id}: {history}")
                        logging.info(f"History type: {type(history)}, History length: {len(history) if isinstance(history, list) else 0}")
                        
                        # 确保历史记录格式正确
                        if not isinstance(history, list):
                            history = []
                        
                        # 切换到问答页面并返回对话历史
                        from gradio import update
                        result = {
                            chat_page: update(visible=True),
                            history_page: update(visible=False),
                            tree_page: update(visible=False),
                            logs_page: update(visible=False),
                            chat_btn: update(variant="primary"),
                            history_btn: update(variant="secondary"),
                            tree_btn: update(variant="secondary"),
                            logs_btn: update(variant="secondary"),
                            page_state: "chat",
                            chatbot: history,
                            conversation_id_state: conversation_id
                        }
                        
                        # 返回结果前再次确认
                        logging.info(f"Returning history to chatbot: {history}")
                        return result
                    
                    # 为表格行选择添加处理函数
                    def on_history_row_select(evt: gr.SelectData, df):
                        # 从选中的行获取对话ID
                        row_idx = evt.index[0]  # 行索引
                        conversation_id = df.iloc[row_idx, 0]  # 第一列是ID
                        return load_history_by_id(conversation_id)
                    
                    # 绑定表格行选择事件
                    history_table.select(
                        fn=on_history_row_select,
                        inputs=[history_table],
                        outputs=[chat_page, history_page, tree_page, logs_page, chat_btn, history_btn, tree_btn, logs_btn, page_state, chatbot, conversation_id_state],
                        show_progress=True
                    )
        
        # 页面切换逻辑
        def show_chat_page():
            return {
                chat_page: update(visible=True),
                tree_page: update(visible=False),
                logs_page: update(visible=False),
                chat_btn: update(variant="primary"),
                tree_btn: update(variant="secondary"),
                logs_btn: update(variant="secondary"),
                page_state: "chat"
            }
        
        # 专门用于保持历史记录的聊天页面显示函数
        def show_chat_page_with_current_state(chat_history, conv_id):
            return {
                chat_page: update(visible=True),
                tree_page: update(visible=False),
                logs_page: update(visible=False),
                chat_btn: update(variant="primary"),
                tree_btn: update(variant="secondary"),
                logs_btn: update(variant="secondary"),
                page_state: "chat",
                chatbot: chat_history,
                conversation_id_state: conv_id
            }
        
        def show_tree_page(chat_history=None, conv_id=None):
            result = {
                chat_page: update(visible=False),
                tree_page: update(visible=True),
                logs_page: update(visible=False),
                chat_btn: update(variant="secondary"),
                tree_btn: update(variant="primary"),
                logs_btn: update(variant="secondary"),
                page_state: "tree"
            }
            
            # 保留聊天历史和对话ID（如果提供的话）
            if chat_history is not None:
                result[chatbot] = chat_history
            if conv_id is not None:
                result[conversation_id_state] = conv_id
                
            return result
        
        def show_logs_page(chat_history=None, conv_id=None):
            result = {
                chat_page: update(visible=False),
                tree_page: update(visible=False),
                logs_page: update(visible=True),
                history_page: update(visible=False),
                chat_btn: update(variant="secondary"),
                tree_btn: update(variant="secondary"),
                logs_btn: update(variant="primary"),
                history_btn: update(variant="secondary"),
                page_state: "logs"
            }
            
            # 保留聊天历史和对话ID（如果提供的话）
            if chat_history is not None:
                result[chatbot] = chat_history
            if conv_id is not None:
                result[conversation_id_state] = conv_id
                
            return result
        
        def show_history_page(chat_history=None, conv_id=None):
            from core_functions import get_conversation_records
            records = get_conversation_records()
            import pandas as pd
            # 为每条记录添加一个"加载"按钮文本
            updated_records = []
            for record in records:
                updated_record = list(record)
                updated_record[4] = "加载"  # 将操作列设为"加载"
                updated_records.append(updated_record)
            df = pd.DataFrame(updated_records, columns=["ID", "文件名", "上传时间", "内容摘要", "操作"])
            
            result = {
                chat_page: update(visible=False),
                tree_page: update(visible=False),
                logs_page: update(visible=False),
                history_page: update(visible=True),
                chat_btn: update(variant="secondary"),
                tree_btn: update(variant="secondary"),
                logs_btn: update(variant="secondary"),
                history_btn: update(variant="primary"),
                page_state: "history",
                history_table: df
            }
            
            # 保留聊天历史和对话ID（如果提供的话）
            if chat_history is not None:
                result[chatbot] = chat_history
            if conv_id is not None:
                result[conversation_id_state] = conv_id
                
            return result
        
        # 绑定页面切换按钮
        chat_btn.click(show_chat_page_with_current_state, inputs=[chatbot, conversation_id_state], outputs=[chat_page, tree_page, logs_page, history_page, chat_btn, tree_btn, logs_btn, history_btn, page_state, chatbot, conversation_id_state])
        tree_btn.click(show_tree_page, inputs=[chatbot, conversation_id_state], outputs=[chat_page, tree_page, logs_page, history_page, chat_btn, tree_btn, logs_btn, history_btn, page_state, chatbot, conversation_id_state])
        logs_btn.click(show_logs_page, inputs=[chatbot, conversation_id_state], outputs=[chat_page, tree_page, logs_page, history_page, chat_btn, tree_btn, logs_btn, history_btn, page_state, chatbot, conversation_id_state])
        history_btn.click(show_history_page, inputs=[chatbot, conversation_id_state], outputs=[chat_page, tree_page, logs_page, history_page, chat_btn, tree_btn, logs_btn, history_btn, page_state, history_table, chatbot, conversation_id_state])
        
        # 上传按钮点击事件 - 处理表格生成并刷新前端树
        upload_btn.click(
            fn=load_from_upload,
            inputs=[table_input],
            outputs=[tree_iframe, chatbot, conversation_id_state]
        )
        
        # 绑定文件自动处理事件（显示文件上传成功消息到聊天界面）
        def on_file_upload(file):
            if file:
                if isinstance(file, list):
                    # 多个文件
                    file_info = []
                    for f in file:
                        ext = os.path.splitext(f.name)[1].lower()
                        file_size = os.path.getsize(f.name)
                        file_info.append(f"  - {os.path.basename(f.name)} ({file_size} 字节, 类型: {ext})")
                    # 返回聊天消息而不是输入框内容
                    chat_msg = [{"role": "assistant", "content": f"✅ {len(file)} 个文件已提交，请按'上传文件'按钮:\n" + "\n".join(file_info)}]
                else:
                    # 单个文件
                    ext = os.path.splitext(file.name)[1].lower()
                    file_size = os.path.getsize(file.name)
                    # 返回聊天消息而不是输入框内容
                    chat_msg = [{"role": "assistant", "content": f"✅ 文件已提交，请按'上传文件'按钮: {os.path.basename(file.name)} ({file_size} 字节)\n文件类型: {ext}"}]
            else:
                # 没有文件时显示默认状态
                chat_msg = []
            
            return chat_msg
        
        table_input.change(
            fn=on_file_upload,
            inputs=[table_input],
            outputs=[chatbot]
        )
        
        # 清除聊天记录
        def clear_chat():
            return [], None, ""
        
        clear_chat_btn.click(
            fn=clear_chat,
            inputs=[],
            outputs=[chatbot, table_input, question_input]
        )
        
        # 定时刷新日志窗口（每3秒自动更新）
        def refresh_all_logs_fn():
            return read_all_logs(log_dir=LOG_DIR, max_lines=200)
        
        # 定时刷新思维链条窗口（每3秒自动更新）
        def refresh_thinking_chain_fn():
            return get_thinking_chain()
        
        # 创建隐藏的 Timer 触发器
        log_timer = gr.Timer(3, active=True)
        log_timer.tick(
            fn=refresh_all_logs_fn,
            outputs=[log_output],
        )
        
        chain_timer = gr.Timer(3, active=True)
        chain_timer.tick(
            fn=refresh_thinking_chain_fn,
            outputs=[thinking_chain_output],
        )
        
        def clear_log():
            # 清除日志文件内容
            if os.path.exists(LOG_DIR):
                for log_file in os.listdir(LOG_DIR):
                    file_path = os.path.join(LOG_DIR, log_file)
                    if os.path.isfile(file_path):
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write("")  # 清空文件
            return ""  # 清空界面显示
        
        clear_log_btn.click(
            fn=clear_log,
            inputs=[],
            outputs=[log_output],
            queue=False
        )
        
        # 清除按钮也清空思维链条
        clear_log_btn.click(
            fn=lambda: {},
            inputs=[],
            outputs=[thinking_chain_output],
            queue=False
        )
    
    return demo