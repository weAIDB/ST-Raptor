import re
from datetime import datetime
from typing import Union

# 常见日期格式模式（按优先级排序）
DATE_PATTERNS = [
    # ISO 格式
    "%Y-%m-%d",  # 2023-10-05
    "%Y%m%d",  # 20231005
    "%Y-%m",  # 2023-10 (自动补当月1日)
    # 带英文月份的格式
    "%d %B %Y",  # 05 October 2023
    "%d-%b-%Y",  # 05-Oct-2023
    "%b %d, %Y",  # Oct 05, 2023
    "%B %d, %Y",  # October 05, 2023
    # 数字格式（处理不同分隔符和顺序）
    "%m/%d/%Y",  # 10/05/2023 (美式)
    "%d/%m/%Y",  # 05/10/2023 (欧式)
    "%Y/%m/%d",  # 2023/10/05
    "%m-%d-%Y",  # 10-05-2023
    "%d-%m-%Y",  # 05-10-2023
    "%Y-%d-%m",  # 2023-05-10
    "%m.%d.%Y",  # 10.05.2023
    "%d.%m.%Y",  # 05.10.2023
    "%Y%m",  # 202310 (自动补1日)
    # 处理不带前导零的情况
    "%m/%d/%y",  # 10/5/23
    "%d/%m/%y",  # 5/10/23
    "%m-%d-%y",  # 10-5-23
    "%d-%m-%y",  # 5-10-23
    # 中文格式
    "%Y年%m月%d日",  # 2023年10月05日
    "%Y年%m月",  # 2023年10月
    "%m月%d日",  # 10月05日 (自动补当年)
    # 特殊格式处理
    "%Y%j",  # 年+儒略日 2023278
    "%d-%b-%y",  # 05-Oct-23
    "%b-%d-%y",  # Oct-05-23
]


def is_integer(s):
    """判断字符串是否是整数"""
    try:
        int(s)
        return True
    except Exception as e:
        return False


def is_float(s):
    try:
        float(s)
        return True
    except Exception as e:
        return False


def is_numeric(s):
    try:
        int(s)
        float(s)
        return True
    except Exception as e:
        return False


def is_date(s, date_formats=DATE_PATTERNS):
    """判断字符串是否是日期"""
    for fmt in date_formats:
        try:
            datetime.strptime(s, fmt)
            return True
        except Exception as e:
            pass
    return False


def str_to_date(date_str: str, default_year: int = None) -> Union[datetime.date, None]:
    """
    将字符串转换为日期对象，支持自动识别多种格式

    参数：
        date_str: 日期字符串
        default_year: 当日期缺少年份时的默认年份

    返回：
        datetime.date对象 或 None（解析失败时）
    """
    original_str = date_str.strip()
    if not original_str:
        return None

    # 预处理中文年月日
    date_str = original_str.replace("年", "-").replace("月", "-").replace("日", "")

    for pattern in DATE_PATTERNS:
        try:
            dt = datetime.strptime(date_str, pattern)

            # 处理没有年份的情况
            if "%Y" not in pattern and "%y" not in pattern:
                if not default_year:
                    default_year = datetime.now().year
                dt = dt.replace(year=default_year)

            return dt.date()
        except Exception as e:
            continue

    # 尝试处理不带年份的月份日期（如10-05）
    if "-" in date_str and len(date_str.split("-")) == 2:
        try:
            month, day = map(int, date_str.split("-"))
            year = default_year or datetime.now().year
            return datetime(year, month, day).date()
        except Exception as e:
            pass

    # 特殊处理纯数字格式（如20231005）
    if date_str.isdigit():
        # 处理6位数字（YYMMDD）
        if len(date_str) == 6:
            try:
                return datetime.strptime(date_str, "%y%m%d").date()
            except Exception as e:
                pass

        # 处理8位数字（YYYYMMDD）
        if len(date_str) == 8:
            try:
                return datetime.strptime(date_str, "%Y%m%d").date()
            except Exception as e:
                pass

    # 最终尝试使用dateutil（如果安装）
    try:
        from dateutil.parser import parse

        return parse(original_str).date()
    except ImportError:
        pass
    except Exception as e:
        return None

    return None
