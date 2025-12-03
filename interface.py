import gradio as gr
from config import api_config, save_api_config
from core_functions import process_table_for_tree, process_question_only, clear_all, read_all_logs


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
        .question-output {
            max-height: 300px !important;
            overflow-y: auto !important;
        }
        #log-output-box {
            height: 400px !important;
            max-height: 400px !important;
            overflow-y: auto !important;
            padding: 10px;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            background: #f8f9fa;
        }
        """
    ) as demo:
        
        gr.Markdown("""
        <div class="header">
            <h1>📊 ST-Raptor 表格问答系统</h1>
            <p>上传 Excel 表格并使用自然语言提问，获取智能答案</p>
        </div>
        """)
        
        # 顶部：API配置面板
        with gr.Accordion("⚙️ API配置", open=False):
            with gr.Tabs():
                # LLM配置标签页
                with gr.TabItem("📝 LLM配置"):
                    with gr.Row():
                        llm_api_key = gr.Textbox(
                            label="LLM API Key", 
                            value=api_config["llm_api_key"],
                            placeholder="请输入LLM API密钥",
                            type="password",
                            scale=2
                        )
                        llm_model = gr.Dropdown(
                            choices=["deepseek-v3.1", "gpt-4-turbo", "claude-3-opus-20240229", "qwen-max"],
                            value=api_config["llm_model"],
                            label="LLM 模型",
                            scale=1
                        )
                    llm_api_url = gr.Textbox(
                        label="LLM API URL", 
                        value=api_config["llm_api_url"],
                        placeholder="请输入LLM API地址",
                        lines=1
                    )
                
                # VLM配置标签页
                with gr.TabItem("🖼️ VLM配置"):
                    with gr.Row():
                        vlm_api_key = gr.Textbox(
                            label="VLM API Key", 
                            value=api_config["vlm_api_key"],
                            placeholder="请输入VLM API密钥",
                            type="password",
                            scale=2
                        )
                        vlm_model = gr.Dropdown(
                            choices=["qwen3-vl-plus", "gpt-4-vision-preview", "claude-3-opus-20240229"],
                            value=api_config["vlm_model"],
                            label="VLM 模型",
                            scale=1
                        )
                    vlm_api_url = gr.Textbox(
                        label="VLM API URL", 
                        value=api_config["vlm_api_url"],
                        placeholder="请输入VLM API地址",
                        lines=1
                    )
                
                # Embedding配置标签页
                with gr.TabItem("📊 Embedding配置"):
                    with gr.Row():
                        embedding_api_key = gr.Textbox(
                            label="Embedding API Key", 
                            value=api_config["embedding_api_key"],
                            placeholder="请输入Embedding API密钥",
                            type="password",
                            scale=2
                        )
                        embedding_model = gr.Dropdown(
                            choices=["text-embedding-v1", "text-embedding-ada-002", "text-embedding-3-large"],
                            value=api_config["embedding_model"],
                            label="Embedding 模型",
                            scale=1
                        )
                    embedding_api_url = gr.Textbox(
                        label="Embedding API URL", 
                        value=api_config["embedding_api_url"],
                        placeholder="请输入Embedding API地址",
                        lines=1
                    )
                
                # 保存按钮和状态显示
                save_config_btn = gr.Button("💾 保存配置", variant="primary")
                config_status = gr.HTML("", label="配置状态")
                
                # 绑定保存配置按钮事件
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
                # 日志输出框（放在可折叠面板中，默认隐藏）
                with gr.Accordion("📜 实时日志", open=False):
                    log_output = gr.HTML(
                        label="终端日志",
                        value=read_all_logs(),
                        elem_id="log-output-box"
                    )
                # 注入 JS 使其每次内容变化时自动滚动到底部
                gr.HTML(
                    """
<script>
function scrollLogToBottom() {
    var box = document.querySelector('#log-output-box');
    if (box) {
        box.scrollTop = box.scrollHeight;
    }
}
const observer = new MutationObserver(scrollLogToBottom);
setTimeout(function() {
    var box = document.querySelector('#log-output-box');
    if (box) {
        observer.observe(box, { childList: true, subtree: true, characterData: true });
    }
}, 1000);
</script>
""")
        
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
