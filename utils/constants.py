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


INTERNVL = None
INTERNVL_1B = 8101
INTERNVL_2B = 8102
INTERNVL_4B = 8104
INTERNVL_8B = 8108
INTERNVL_26B = 8126
INTERNVL_38B = 8138
INTERNVL_78B = 8178

INTERNLM = "internlm"
INTERNLM_1B = 8201
INTERNLM_2B = 8202
INTERNLM_4B = 8204
INTERNLM_8B = 8208
INTERNLM_26B = 8226
INTERNLM_38B = 8238
INTERNLM_78B = 8278

LLAMA = "llama"
LLAMA3_8B = 8008
LLAMA3_72B = 8072

QWEN = "qwen"
QWEN2_5_32B = 9032
QWEN2_5_72B = 9072

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

DEEPSEEK = "deepseek-chat"
DEEPSEEK_PORT = ""


#################### Change The Directory Path ####################
BASE_DIR = '/mnt/d/Desktop/st-raptor/ST-Raptor'            # The Project Directory
CACHE_DIR = os.path.join(BASE_DIR, 'cache')     # The Cache Directory
LOG_DIR = os.path.join(BASE_DIR, 'log')
#################### Change The Directory Path ####################


#################### Change The Model Information ####################
"""
If you use external LLM API, change API_URL, API_KEY, and LLM_MODEL_TPYE to your settings.
The LLM_PORT parameter is useless when using external LLM API.

If you use local deployment LLM, change API_URL, LLM_MODEL_TYPE, and LLM_PORT
The API_KEY parameter is useless when using local deployment LLM.
"""
API_URL = "https://api.deepseek.com"#这里默认使用deepseek，也可改成你喜欢的模型url
API_KEY = #在此处填写你的key
LLM_MODEL_TYPE = DEEPSEEK 
LLM_PORT = DEEPSEEK_PORT

"""Set local deployment VLM information"""
VLM_MODEL_TYPE = None   #注意此处是调用外部api的方式，如果使用本地部署，则需要修改为本地部署的模型路径，同时需要修改28行的INTERVAL参数，详情可于github咨询
VLM_PORT = None



"""Set the Embedding model information"""
MULTILINGUAL_E5_MODEL_PATH = "./data/models/multilingual-e5-large-instruct/"
EMBEDDING_MODE_PATH = MULTILINGUAL_E5_MODEL_PATH
CLASSICIFATION_ENBEDDING_MODEL_PATH = MULTILINGUAL_E5_MODEL_PATH
#################### Change The Model Information ####################


FONT_PATH = "file://" + os.path.join(BASE_DIR, "static/simfang.ttf")
HTML_CACHE_DIR = os.path.join(CACHE_DIR, "html")
IMAGE_CACHE_DIR = os.path.join(CACHE_DIR, "image")
EXCEL_CACHE_DIR = os.path.join(CACHE_DIR, "excel")
SCHEMA_CACHE_DIR = os.path.join(CACHE_DIR, "schema")
JSON_CACHE_DIR = os.path.join(CACHE_DIR, "json")
OUTPUT_JSON_CACHE_DIR = os.path.join(CACHE_DIR, "output_json")

ALLMINILM_MODEL_PATH = "" # deprecated

