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


def sheet_to_html_file(sheet, output_file=None):

    html = sheet_to_html(sheet)

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


def sheet_to_image(sheet, output_image_file=None):
    if output_image_file is None:
        basename = generate_unique_string()
        if not os.path.exists(HTML_CACHE_DIR):
            os.makedirs(HTML_CACHE_DIR)
        html_file = os.path.join(HTML_CACHE_DIR, f"{basename}.html")
    else:
        basename = os.path.basename(output_image_file)[:-4]
        html_file = os.path.join(os.path.dirname(output_image_file), f"{basename}.png")
    html_file = sheet_to_html_file(sheet, html_file)
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
    image_file = sheet_to_image(sheet)
    prompt = """
    请告诉我当前表的表头行或者表头列在表格的上部还是左侧，如果表头在上部，则输出“上部”，如果表头在左侧，则输出“左侧”。
    这里的表头意思是表的元信息，是表格转换为JSON后的键的内容！
    注意，仅需要输出“上部”或“左侧”，不要使用任何额外格式！
    """
    
    res = vlm_generate(prompt, image_file, model)
    
    return res
    

def vlm_get_schema(sheet, model=INTERNVL, save=False):
    """
    sheet: 获取schema的sheet
    model: 使用的多模态大模型
    save: 是否将获得的schema进行保存至SCHEMA_CACHE_DIR文件夹中
    """
    image_file = sheet_to_image(sheet)
    prompt = """
    请将给定的表格图片转换为JSON格式，并且只需要直接返回转化JSON中所有键的内容。
    请使用python的列表格式返回键的列表！返回的Python列表语法需要准确！不要使用任何Markdown格式，不要修改表格内容。
    """

    cnt = 0
    while cnt < MAX_ITER_META_INFORMATION_DETECTION:
        res = vlm_generate(prompt, image_file, model)

        if VLM_MODEL_TYPE == INTERNVL:
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

    if save:
        basename = os.path.basename(image_file)[:-4]
        if not os.path.exists(SCHEMA_CACHE_DIR):
            os.mkdir(SCHEMA_CACHE_DIR)
        schame_file = os.path.join(SCHEMA_CACHE_DIR, f"{basename}.txt")
        with open(schame_file, "w") as f:
            f.write(str(res))
    return res


def vlm_get_json(sheet, model=INTERNVL, save=False):
    image_file = sheet_to_image(sheet)
    prompt = "请将给定的表格图片转换为JSON格式，注意，JSON的键与值都必须直接是表格的内容，请直接返回转化后的标准格式的JSON，不要使用任何Markdown格式，不要修改表格内容。"

    cnt = 0
    while cnt < MAX_ITER_META_INFORMATION_DETECTION:
        res = vlm_generate(prompt, image_file, model)

        flag = False
        if VLM_MODEL_TYPE == INTERNVL:
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

