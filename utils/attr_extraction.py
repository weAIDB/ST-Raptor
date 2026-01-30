import re
import random
import string
import glob
import os
import json

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

import imgkit

# sudo apt-get install wkhtmltopdf
# sudo apt-get install -y fonts-noto-cjk fonts-wqy-microhei
# brew install wkhtmltopdf
from loguru import logger
from PIL import Image
from openpyxl import load_workbook
from bs4 import BeautifulSoup
from bs4.element import NavigableString

from utils.constants import *
from utils.sheet_utils import *

def generate_unique_string(length=10):
    import time
    # 使用当前时间戳
    timestamp = int(time.time() * 1000)  # 获取毫秒级时间戳
    # 生成随机字符串
    random_string = "".join(
        random.choices(string.ascii_letters + string.digits, k=length)
    )
    # 组合时间戳和随机字符串，确保唯一性
    unique_string = f"{timestamp}_{random_string}"
    return unique_string


def sheet2html_file(sheet, output_file=None):

    html = sheet2html(sheet)

    # 解析 HTML
    soup = BeautifulSoup(html, "html.parser")

    # 遍历所有单元格，修改内容
    for td in soup.find_all("td"):
        # 使用 &nbsp; 添加空格
        s = td.text
        td.clear()  # 清空当前内容
        td.append(NavigableString("\u00a0"))  # 添加 non-breaking space
        td.append(NavigableString(s))  # 添加原有内容
        td.append(NavigableString("\u00a0"))  # 再添加 non-breaking space
    # 打印修改后的 HTML
    html = str(soup)

    if output_file is None:
        if not os.path.exists(HTML_CACHE_DIR):
            os.makedirs(HTML_CACHE_DIR)
        output_file = os.path.join(HTML_CACHE_DIR, f"{generate_unique_string()}.html")
    # 将 HTML 保存到文件
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)
    return output_file


def capture_screenshot(html_file, screenshot_file):
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # 无头模式
    chrome_options.add_argument("--no-sandbox")
    # Bypass OS security model
    chrome_options.add_argument("--window-size=1920x1080")
    chrome_options.add_argument("--lang=zh-CN")  # 设置语言为中文（简体）

    driver = webdriver.Chrome(options=chrome_options)
    driver.get(f"file://{os.path.abspath(html_file)}")

    driver.execute_script(
        f"""
        var style = document.createElement('style');
        style.innerHTML = `
            @font-face {{
                font-family: 'MyCustomFont';
                src: url({FONT_PATH}) format('truetype');
            }}
            body {{
                font-family: 'MyCustomFont', sans-serif;
                font-size: 12pt;
            }}
        `;
        document.head.appendChild(style);
    """
    )

    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "td")))
    WebDriverWait(driver, 10).until(
        lambda driver: driver.execute_script("return document.readyState") == "complete"
    )

    time.sleep(0.5)
    table_width = driver.execute_script("return document.body.scrollWidth")
    table_height = driver.execute_script("return document.body.scrollHeight")
    driver.set_window_size(table_width + 100, table_height + 100)
    time.sleep(0.5)

    driver.save_screenshot(screenshot_file)
    driver.quit()

def resize_to_target_size(input_path, output_path, target_kb, quality=90, enable_crop=True):
    original_size = os.path.getsize(input_path)
    
    a = (15 * 1024 * 1024) / original_size

    logger.info(f"Image Crop Ratio for 15MB: {a} Original Size: {original_size}")

    if a < 1 and enable_crop: # Crop when > 15MB
        img = Image.open(input_path)
        width, height = img.size
        
        # 计算裁剪区域：宽80%，高60%
        new_width = int(width * a)
        new_height = int(height * a)
        
        # 裁剪图片
        cropped_img = img.crop((0, 0, new_height, new_width))
        
        # 保存图片
        cropped_img.save(input_path)

    img = Image.open(input_path)
    original_size = os.path.getsize(input_path)

    # 初步调整
    if original_size > target_kb * 1024:
        logger.info(f'{DELIMITER} 压缩图片大小，防止超出 VLM 长度限制 {DELIMITER}')
        logger.info(f'Origin size {original_size // 1024} KB')
        # 根据大小比例初步缩放尺寸
        scale = (target_kb * 1024 / original_size) ** 0.5
        new_width = int(img.width * scale)
        new_height = int(img.height * scale)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # 精细调整质量参数
        temp_path = os.path.dirname(output_path) + '/tmp.png'
        for q in range(quality, 10, -5):
            img.save(temp_path, optimize=True, quality=q)
            if os.path.getsize(temp_path) <= target_kb * 1024:
                break
        
        # 使用os.replace替代os.rename，确保在Windows系统上可以覆盖已存在的文件
        os.replace(temp_path, output_path)
        logger.info(f'Scaled size {os.path.getsize(output_path) // 1024} KB')


def sheet_to_image(sheet, output_image_file=None, enable_crop=True):
    if output_image_file is None:
        basename = generate_unique_string()
        if not os.path.exists(HTML_CACHE_DIR):
            os.makedirs(HTML_CACHE_DIR)
        html_file = os.path.join(HTML_CACHE_DIR, f"{basename}.html")
    else:
        basename = os.path.basename(output_image_file)[:-4]
        html_file = os.path.join(os.path.dirname(output_image_file), f"{basename}.png")
    html_file = sheet2html_file(sheet, html_file)
    if not os.path.exists(IMAGE_CACHE_DIR):
        os.makedirs(IMAGE_CACHE_DIR)
    screenshot_file = os.path.join(IMAGE_CACHE_DIR, f"{basename}.png")

    # capture_screenshot(html_file, screenshot_file)
    options = {
        "encoding": "UTF-8",
        "quality": 100,
        "quiet": "",
    }
    imgkit.from_file(html_file, screenshot_file, options=options)

    resize_to_target_size(screenshot_file, screenshot_file, target_kb=5120, enable_crop=enable_crop)

    return screenshot_file


def get_keys(d, keys=None):
    if keys is None:
        keys = set()
    if isinstance(d, dict):  # 如果是字典
        for key, value in d.items():
            keys.add(key)
            get_keys(value, keys)  # 递归处理值
    elif isinstance(d, list):  # 如果是列表
        for item in d:
            get_keys(item, keys)  # 递归处理列表中的每个项
    elif isinstance(d, str):
        keys.add(d)
    return keys


def vlm_get_direction(sheet, model="internvl", save=False):
    # 导入get_vlm_generate函数，使用全局API配置
    try:
        from gradio_app import get_vlm_generate
        vlm_generate = get_vlm_generate()
    except ImportError:
        # 如果无法导入，使用默认的vlm_generate函数
        from utils.api_utils import vlm_generate
        
    image_file = sheet_to_image(sheet)
    prompt = """
    请告诉我当前表的表头行或者表头列在表格的上部还是左侧，如果表头在上部，则输出“上部”，如果表头在左侧，则输出“左侧”。
    这里的表头意思是表的元信息，是表格转换为JSON后的键的内容！
    注意，仅需要输出“上部”或“左侧”，不要使用任何额外格式！
    """
    
    res = vlm_generate(prompt, image_file, model)
    
    return res
    

def vlm_get_schema(sheet,                   # 需要获取 schema 的 sheet
                   ):
    """使用 VLM 提取表格的 Schema"""
    
    # 导入get_vlm_generate函数，使用全局API配置
    try:
        from gradio_app import get_vlm_generate
        vlm_generate = get_vlm_generate()
    except ImportError:
        # 如果无法导入，使用默认的vlm_generate函数
        from utils.api_utils import vlm_generate
        
    image_file = sheet_to_image(sheet)
    prompt = """请将给定的表格图片转换为JSON格式，并且只需要直接返回转化JSON中所有键的内容。请使用python的列表格式返回键的列表！返回的Python列表语法需要准确！不要使用任何Markdown格式，不要修改表格内容。"""

    logger.info(f"{DELIMITER} 表格转换为图片的路径 {DELIMITER}")
    logger.info(f"{image_file}")

    cnt = 0
    while cnt < MAX_ITER_META_INFORMATION_DETECTION:
        res = vlm_generate(prompt, image_file)
        logger.info(f"模型输出: {res}")
        
        flag = False
        match = re.findall(r"```python(.*?)```", res, re.DOTALL)    # 尝试匹配 python 格式
        if len(match) != 0:
            res = match[0].strip()
            if res[0] == '[' and res[-1] != ']': res = res + ']'
            if res[0] == '{' and res[-1] != '}': res = res + '}'
            try:
                res = eval(res)
                flag = True
            except Exception as e:
                pass
                # import traceback; traceback.print_exc()
                # print(f'VLM output: {res}'); print(f'Exception: {e}')

        if not flag:
            match = re.findall(r"```json(.*?)```", res, re.DOTALL)  # 尝试匹配 json 格式
        
            if len(match) != 0:
                res = match[0].strip()
                if res[0] == '[' and res[-1] != ']': res = res + ']'
                if res[0] == '{' and res[-1] != '}': res = res + '}'
                try:
                    res = eval(res)
                    flag = True
                except Exception as e:
                    pass
                    # import traceback; traceback.print_exc()
                    # print(f'VLM output: {res}'); print(f'Exception: {e}')
        if not flag:
            res = res.strip()
            if res[0] == '[' and res[-1] != ']': res = res + ']'
            if res[0] == '{' and res[-1] != '}': res = res + '}'

        if not flag:
            try:
                res = eval(res)   # 可以通过 eval() 转换
                flag = True
            except Exception as e:
                pass
                # import traceback; traceback.print_exc()
                # print(f'VLM output: {res}'); print(f'Exception: {e}')
            
        if not flag:
            try:
                res = {"key": eval(f"[{res}]")} # 可以通过eval转换为列表
                flag = True
            except Exception as e:
                pass
                # import traceback; traceback.print_exc()
                # print(f'VLM output: {res}'); print(f'Exception: {e}')

        if flag:
            break
        else:
            cnt += 1
    if cnt == MAX_ITER_META_INFORMATION_DETECTION - 1:
        print(f'Meta Information Detection reaches max retry.')

    return res

def get_schema_direction_by_vlm(sheet, image_file=None):
    # 导入get_vlm_generate函数，使用全局API配置
    try:
        from gradio_app import get_vlm_generate
        vlm_generate = get_vlm_generate()
    except ImportError:
        # 如果无法导入，使用默认的vlm_generate函数
        from utils.api_utils import vlm_generate
        
    prompt = "表格的Schema是可以总结表格某一列或某一行的单元格，请判断该表格最外层的 SCHEMA 是表格的上部还是表格的左侧，如果是上部，请输出true，如果是左侧，请输出false。注意，你只需要输出true或false，不要有任何其他格式。"

    if image_file is None:
        image_file = sheet_to_image(sheet)
    res = vlm_generate(prompt, image_file).lower()

    if res == 't' or res == 'true' or res == 'yes' or res == '1': return SCHEMA_TOP
    return SCHEMA_LEFT

def vlm_get_json(sheet, save=False, enable_crop=False):
    # 导入get_vlm_generate函数，使用全局API配置
    try:
        from gradio_app import get_vlm_generate
        vlm_generate = get_vlm_generate()
    except ImportError:
        # 如果无法导入，使用默认的vlm_generate函数
        from utils.api_utils import vlm_generate
        
    image_file = sheet_to_image(sheet, enable_crop=enable_crop)
    prompt = "请将给定的表格图片转换为JSON格式，注意，JSON的键与值都必须直接是表格的内容，请直接返回转化后的标准格式的JSON，不要使用任何Markdown格式，不要修改表格内容。"

    cnt = 0
    while cnt < MAX_ITER_META_INFORMATION_DETECTION:
        res = vlm_generate(prompt, image_file)

        flag = False
        if VLM_MODEL_TYPE == VLM_MODEL_TYPE:
            match = re.findall(r"```json(.*?)```", res, re.DOTALL)
        try:
            if len(match) == 0:
                res = eval(res)
            else:
                res = eval(match[0])
            flag = True
        except Exception as e:
            pass
            # import traceback; traceback.print_exc()
            # print(e)

        if flag:
            break
        else:
            cnt += 1
            
    if save:
        basename = os.path.basename(image_file)[:-4]
        if not os.path.exists(JSON_CACHE_DIR):
            os.makedirs(JSON_CACHE_DIR)
        schame_file = os.path.join(JSON_CACHE_DIR, f"{basename}.json")
        with open(schame_file, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=4, ensure_ascii=False)
    return res

