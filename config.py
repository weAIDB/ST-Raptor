import os
import json
from utils.constants import DELIMITER, LOG_DIR, LLM_API_URL, LLM_API_KEY, LLM_MODEL_TYPE, VLM_API_URL, VLM_API_KEY, VLM_MODEL_TYPE, EMBEDDING_API_URL, EMBEDDING_API_KEY, EMBEDDING_MODEL_TYPE

# 全局配置字典，用于存储用户设置的API配置
api_config = {
    "llm_api_key": LLM_API_KEY,
    "llm_api_url": LLM_API_URL,
    "llm_model": LLM_MODEL_TYPE,
    "vlm_api_key": VLM_API_KEY,
    "vlm_api_url": VLM_API_URL,
    "vlm_model": VLM_MODEL_TYPE,
    "embedding_api_key": EMBEDDING_API_KEY,
    "embedding_api_url": EMBEDDING_API_URL,
    "embedding_model": EMBEDDING_MODEL_TYPE
}

# 配置文件路径
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_config.json")

# 从文件加载配置
def load_api_config():
    global api_config
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
                api_config.update(loaded_config)
        except Exception as e:
            print(f"加载API配置失败: {e}")

# 保存配置到文件
def save_api_config(config):
    global api_config
    api_config.update(config)
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(api_config, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存API配置失败: {e}")
