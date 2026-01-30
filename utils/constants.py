import os

DELIMITER = "################"

T_LIST = 1
T_ARRT = 2
T_SEMI = 3
T_MIX = 4
T_OTHER = -1

SMALL_TABLE_ROWS = 3
SMALL_TABLE_COLUMNS = 3
BIG_TABLE_ROWS = 8
BIG_TABLE_COLUMNS = 8

DEFAULT_TABLE_NAME = "table"
DEFAULT_SUBTABLE_NAME = "subtable"
DEFAULT_SUBVALUE_NAME = "subvalue"
DEFAULT_SPLIT_SIG = "-"

DIRECTION_KEY = "direction_key"
VLM_SCHEMA_KEY = "vlm_schema_key"
SCHEMA_TOP = True
SCHEMA_LEFT = False
SCHEMA_FAIL = -1

STATUS_END = 1
STATUS_RETRIEVE = 2
STATUS_AGG = 3
STATUS_SPLIT = 4

TAG_DISCRETE = 1
TAG_CONTINUOUS = 2
TAG_TEXT = 3

MAX_ITER_META_INFORMATION_DETECTION = 5  # Meta Information Detection 的最大尝试次数
MAX_ITER_PRIMITIVE = 5  # 生成原语句的最多步骤次数
MAX_RETRY_HOTREE = 3  # 表格转换HO-Tree的最大尝试次数
MAX_RETRY_PRIMITIVE = 5  # 前向验证时重新生成原语句的次数

def delete_dict_none_none(data : dict):
    """将JSON中None:None的项去除"""
    new_dict = {}
    for k, v in data.items():
        if (k is None or k == 'None') and (v is None or v == 'None'):
            pass
        else:
            if isinstance(v, dict):
                new_dict[k] = delete_dict_none_none(v)
            elif isinstance(v, list):
                new_v = []
                for item in v:
                    if isinstance(item, dict):
                        new_v.append(delete_dict_none_none(item))
                    else:
                        new_v.append(item)
                new_dict[k] = new_v
            else:
                new_dict[k] = v
    return new_dict

#################### Change The Directory Path ####################
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE_DIR = os.path.join(BASE_DIR, "cache")
LOG_DIR = os.path.join(BASE_DIR, "log")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
#################### Change The Directory Path ####################“


#################### Change The Model Information ####################
"""Change this for requesting LLM"""
LLM_API_URL = ""
LLM_API_KEY = ""
LLM_MODEL_TYPE = "" 

"""Change this for requesting VLM"""
VLM_API_URL = ""
VLM_API_KEY = ""
VLM_MODEL_TYPE = ""

"""Change this for requesting Embedding Model"""
EMBEDDING_TYPE = "api" # api / local

## If EMBEDDING_TYPE is local
EMBEDDING_MODE_PATH = "YOUR_PATH_TO_MULTILINGULE_E5"

## If EMBEDDING_TYPE is api
EMBEDDING_API_URL = ""
EMBEDDING_API_KEY = ""
EMBEDDING_MODEL_TYPE = ""

#################### Change The Model Information ####################

FONT_PATH = "file://" + os.path.join(BASE_DIR, "static/simfang.ttf")
HTML_CACHE_DIR = os.path.join(CACHE_DIR, "html")
IMAGE_CACHE_DIR = os.path.join(CACHE_DIR, "image")
EXCEL_CACHE_DIR = os.path.join(CACHE_DIR, "excel")
SCHEMA_CACHE_DIR = os.path.join(CACHE_DIR, "schema")
JSON_CACHE_DIR = os.path.join(CACHE_DIR, "json")
OUTPUT_JSON_CACHE_DIR = os.path.join(CACHE_DIR, "output_json")
