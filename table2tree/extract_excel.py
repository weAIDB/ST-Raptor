import re
import os
import glob
import json
import openpyxl
# import openpyxl.workbook
# import openpyxl.workbook.properties
# import openpyxl.worksheet
# import openpyxl.worksheet.worksheet
import requests

import pandas as pd
import numpy as np
import tqdm as tqdm
from loguru import logger

from utils.api_utils import *
from utils.sheet_utils import *
from utils.split_utils import *
from utils.attr_extraction import *
from utils.constants import *


def logAuto(info, flag=True):
    if flag:
        logger.info(info)


def match_min_table_structure(sheet):
    """匹配最小结构，即一个或两个单元格的时候"""
    nrows = sheet.max_row
    ncols = sheet.max_column

    # if nrows < 1 or ncols < 1 or nrows > 2 or ncols > 2 or (nrows == 2 and ncols == 2):
    # return None
    # if nrows == 1 and ncols == 1:
    # return {sheet.cell(row=1, column=1).value : None}

    first_cell = sheet.cell(row=1, column=1)
    _, _, x2, y2 = get_merge_cell_size(sheet, first_cell.coordinate)

    if x2 == nrows and y2 == ncols:
        return {get_merge_cell_value(sheet, first_cell.coordinate): None}
    if x2 == nrows:
        second_cell = sheet.cell(row=1, column=y2 + 1)
        _, _, xx2, yy2 = get_merge_cell_size(sheet, second_cell.coordinate)
        if yy2 == ncols:
            return {
                get_merge_cell_value(
                    sheet, first_cell.coordinate
                ): get_merge_cell_value(sheet, second_cell.coordinate)
            }
    if y2 == ncols:
        second_cell = sheet.cell(row=x2 + 1, column=1)
        _, _, xx2, yy2 = get_merge_cell_size(sheet, second_cell.coordinate)
        if xx2 == nrows:
            return {
                get_merge_cell_value(
                    sheet, first_cell.coordinate
                ): get_merge_cell_value(sheet, second_cell.coordinate)
            }


def match_attr(sheet, pos_list):
    return_dict = {}

    nrows = sheet.max_row
    ncols = sheet.max_column

    x = 1
    while x <= nrows:
        y = 1
        while y <= ncols:
            cell = sheet.cell(row=x, column=y)
            x1, y1, x2, y2 = get_merge_cell_size(sheet, cell.coordinate)
            if in_pos_list(x, y, pos_list) is not None:  # 是Schema
                key = get_merge_cell_value(sheet, cell.coordinate)
                if in_pos_list(x, y2 + 1, pos_list) is None:  # 若右边不是Schema
                    return_dict[key] = get_merge_cell_value(
                        sheet, sheet.cell(row=x, column=y2 + 1).coordinate
                    )
                if in_pos_list(x2 + 1, y, pos_list) is None:  # 若下面不是Schema
                    if key not in return_dict or (
                        key in return_dict
                        and (return_dict[key] is None or return_dict[key] == "")
                    ):
                        return_dict[key] = get_merge_cell_value(
                            sheet, sheet.cell(row=x2 + 1, column=y).coordinate
                        )
            y += y2 - y1 + 1
        x += 1

    return return_dict


def match_list_column(schema, sheet):
    """属性名在表格左侧的情况下提取子半结构化表格"""
    # Step1 获取Schema
    # 找到第一行中高度最高的单元格，即确定整个Schema行合并了多少行
    nrows = sheet.max_row
    ncols = sheet.max_column

    # Step2 提取Schema
    schema = extract_schema_column(schema)
    flattenned_schema = flatten_schema(schema)  # 展平Schema

    # Step3 预处理切分表格，保证从左向右粒度不增，并记录切分的实体关系
    content = sheet
    col = 1
    while col <= ncols:
        x1, y1, x2, y2 = get_merge_cell_size(
            content, content.cell(row=1, column=col).coordinate
        )
        max_width = y2 - y1 + 1
        row = x2 + 1
        while row <= nrows:
            cell = content.cell(row=row, column=col)
            x1, y1, x2, y2 = get_merge_cell_size(content, cell.coordinate)
            width = y2 - y1 + 1
            if width > max_width:  # split
                value = content.cell(row=x1, column=y1).value
                content.unmerge_cells(get_coordinate_by_cell_pos(x1, y1, x2, y2))
                content[
                    get_coordinate_by_cell_pos(x1, y1 + max_width, x1, y1 + max_width)
                ] = value
                content.merge_cells(
                    get_coordinate_by_cell_pos(x1, y1, x2, y1 + max_width - 1)
                )
                content.merge_cells(
                    get_coordinate_by_cell_pos(x1, y1 + max_width, x2, y2)
                )
            row += x2 - x1 + 1
        col += max_width

    # Step4 开始递归的按列切分表格
    col_content = extract_columns(content)

    print(flattenned_schema, col_content)

    # Step5 构建嵌套的字典格式
    return_list, _ = schema_content_match(flattenned_schema, col_content)

    return return_list


def get_schema_height(sheet, direction):
    height = 1

    nrows = sheet.max_row
    ncols = sheet.max_column

    if direction == SCHEMA_TOP:
        for col in range(1, ncols + 1):
            _, _, x2, _ = get_merge_cell_size(
                sheet, sheet.cell(row=1, column=col).coordinate
            )
            height = max(height, x2)

    elif direction == SCHEMA_LEFT:
        for row in range(1, nrows + 1):
            _, _, _, y2 = get_merge_cell_size(
                sheet, sheet.cell(row=row, column=1).coordinate
            )
            height = max(height, y2)

    return height

def get_nrow_cells(sheet):
    n = 0
    
    nrows = sheet.max_row
    ncols = sheet.max_column
    
    for row in range(1, nrows + 1):
        col = 1
        cnt = 0
        while col <= ncols:
            cnt += 1            
            cell = sheet.cell(row=row, column=col)
            x1, y1, x2, y2 = get_merge_cell_size(sheet, cell.coordinate)
            col += (y2 - y1 + 1)
        n = max(n, cnt)
    return n

def get_ncol_cells(sheet):
    n = 0
    
    nrows = sheet.max_row
    ncols = sheet.max_column
    
    for col in range(1, ncols + 1):
        row = 1
        cnt = 0
        while row <= nrows:
            cnt += 1
            cell = sheet.cell(row=row, column=col)
            x1, y1, x2, y2 = get_merge_cell_size(sheet, cell.coordinate)
            row += (x2 - x1 + 1)
        n = max(n, cnt)
    
    return n

def preprocess_cell(value):

    value = str(value)
    # value = re.sub(r"\s+", "", value)  # 去除所有空白字符

    return value


def preprocess_sheet(sheet):
    for row in sheet.iter_rows():
        for cell in row:
            if cell.value is not None:
                cell.value = preprocess_cell(cell.value)
    return sheet


def get_xlsx_sheet(file):
    wb = openpyxl.load_workbook(file, data_only=True)
    sheet = preprocess_sheet(wb.active)
    return sheet


def process_xlsx_vlm(file, get_json=False, log=False, cache=False):
    logAuto(f"process_xlsx_vlm() Start to Process File: {file}", log)
    wb = openpyxl.load_workbook(file, data_only=True)
    sheet = preprocess_sheet(wb.active)
    res = process_sheet_vlm(sheet, get_json=get_json, log=log, cache=cache)
    logAuto(f"process_xlsx_vlm() Process File Successfully: {file} ", log)
    return res

################################# Main #################################
def process_sheet_vlm(sheet,            # 待处理的Excel工作簿格式的表格
                      get_json=False,   # 获得VLM + Rule识别后的中间结果JSON
                      log=False,        # 在控制台输出日志
                      log_file=None,    # 日志输出文件
                      cache=False       # 保存中间结果文件
                      ):
    
    # Step 1 preprocess
    delete_empty_columns(sheet)
    delete_empty_rows(sheet)

    nrows = sheet.max_row
    ncols = sheet.max_column

    nrow_cells = get_nrow_cells(sheet)
    ncol_cells = get_ncol_cells(sheet)

    if log_file is not None:    # Log
        with open(log_file, 'w') as file:
            file.write(f'{DELIMITER} 进入 process_sheet_vlm() 函数 {DELIMITER}\n')
            file.write(f'表格行数: {nrows} 表格列数: {ncols}\n表格每行最多出现的Cell数: {nrow_cells} 表格每列最多出现的Cell数: {ncol_cells}\n')
    logAuto(f"当前表格 nrows: {nrows} ncols: {ncols}", log)

    # 递归返回条件 1: 匹配到最小结构单元
    res = match_min_table_structure(sheet)
    if res is not None:
        logAuto(f"匹配到最小结构单元: {res}", log)
        if log_file is not None:    # Log
            with open(log_file, 'w') as file:
                file.write(f'{DELIMITER} 递归返回条件: 匹配到最小结构单元 {DELIMITER}\n')
                file.write(f'最小结构单元: {res}\n')
        return res

    # 递归返回条件 2: 表为小表
    if (nrow_cells < SMALL_TABLE_ROWS and ncol_cells < BIG_TABLE_COLUMNS) or (
        nrow_cells < BIG_TABLE_ROWS and ncol_cells < SMALL_TABLE_COLUMNS
    ):
        logAuto(f"表为小表，为避免幻觉直接解析", log)
        res = vlm_get_json(sheet)
        if log_file is not None:    # Log
            with open(log_file, 'w') as file:
                file.write(f'{DELIMITER} 递归返回条件: 表为小表 {DELIMITER}\n')
                file.write(f'VLM直接解析结果: {res}\n')
        return res

    # Step 2 split merged entire
    sheet_dict = rowspan_entire(sheet)  # 获得深度可能不为1的sheet_dict
    if len(sheet_dict) == 1 and list(sheet_dict.keys())[0] == DEFAULT_TABLE_NAME:
        sheet_dict = colspan_entire(sheet)
    logAuto(f"跨表拆解出 {len(sheet_dict)} 个子表", log)
    logAuto(sheet_dict, log)
    if log_file is not None:    # Log
        with open(log_file, 'w') as file:
            file.write(f'{DELIMITER} 整行/整列合并单元格拆分结果, key为table意思是不存在整行/整列合并 {DELIMITER}\n')
            file.write(f'{sheet_dict}\n')

    # Step 3 iterate each element
    for key, sheet in sheet_dict.items():

        if sheet is None:
            continue

        delete_empty_columns(sheet)
        delete_empty_rows(sheet)

        # Step 4 judge the direction of the schema

        # 根据表格合并单元格的粒度变化来判断Schema方向
        # if granularity_decrease_col(sheet):
        #     granularity_directon_col = SCHEMA_TOP
        # else:
        #     granularity_directon_col = SCHEMA_FAIL
        # if granularity_decrease_row(sheet):
        #     granularity_directon_row = SCHEMA_LEFT
        # else:
        #     granularity_directon_row = SCHEMA_FAIL

        # 其次使用VLM辅助判断
        schema_vague = vlm_get_schema(sheet, save=cache)
        pos_list, schema_list, pos2schema = schema_pos_match(
            sheet, schema_vague, enable_embedding=True
        )  # 将 vlm 的输出与表格内容对应，并获取 schema 的位置
        direction = get_schema_direction_by_pos(sheet, pos_list)
        direction = SCHEMA_TOP
        # direction = SCHEMA_TOP
        schema_height = get_schema_height(sheet, direction)
        d = "上部" if direction == SCHEMA_TOP else "左侧"
        logAuto(f"{key} 的 Schema 方向为 {d}", log)
        logAuto(f"{key} 的 Schema 高度为 {schema_height}", log)
        logAuto(f"{key} 的 Schema 为: {schema_list}", log)

        # Step 5 根据 schema 方向拆分表格, Schema在左边, 上下拆分为并列的多个子部分; Schema在上面, 左右拆分为并列的多个子部分
        parallel_sheet = None
        if direction == SCHEMA_TOP:
            parallel_sheet = split_subtable_row(sheet)
            logAuto(
                f"{key} 子表根据粒度变化拆解为了 {len(parallel_sheet)} 个子部分", log
            )
        elif direction == SCHEMA_LEFT:
            # 根据粒度变化判断是一个结构化表格，还是一个半结构化子表
            schema, data_sheet = split_schema_column(sheet)
            if (
                granularity_decrease_row(data_sheet)
                and data_sheet.max_column / data_sheet.max_row > 2
            ):  # 粒度递减并且是宽表
                parallel_sheet = []
                sheet = transpose_sheet(sheet)
            else:
                parallel_sheet = split_subtable_each_row(sheet, schema_height)
                logAuto(
                    f"{key} 子表根据行数拆解为了 {len(parallel_sheet)} 个子部分", log
                )
        else:
            res = vlm_get_json(sheet, save=cache)
            sheet_dict[key] = res
            logAuto(f"{key} 子表未识别Schema方向, 使用 VLM 直接解析, 结果为 {res}", log)
            continue

        # Step 6 如果sheet被分为了多个并列的sheet，则分别递归处理，否则直接处理
        if len(parallel_sheet) > 1:
            if isinstance(parallel_sheet, list):
                json_dict = {}
                subtable_cnt = 1
                for index, subsheet in enumerate(parallel_sheet):
                    res = process_sheet_vlm(subsheet, get_json, log)
                    logAuto(f"subsheet {index} done", log)
                    if DEFAULT_TABLE_NAME in res:
                        res = res[DEFAULT_TABLE_NAME]
                        if isinstance(res, dict):
                            json_dict.update(res)
                        else:
                            json_dict[f"{DEFAULT_SUBTABLE_NAME}{subtable_cnt}"] = res
                            subtable_cnt += 1
                    else:
                        json_dict.update(res)
                sheet_dict[key] = json_dict
            elif isinstance(parallel_sheet, dict):
                json_dict = {}
                for name, sheet in parallel_sheet.items():
                    if isinstance(sheet, (str, int, float)) or sheet is None:
                        json_dict[name] = sheet
                    else:
                        res = process_sheet_vlm(sheet, get_json, log)  # a dict
                        if len(res) == 1 and DEFAULT_TABLE_NAME in res:
                            res = res[DEFAULT_TABLE_NAME]
                        if DEFAULT_SUBTABLE_NAME in name and isinstance(res, dict):
                            json_dict.update(res)
                        else:
                            json_dict[name] = res
                    logAuto(f"subsheet {name} done", log)
                sheet_dict[key] = json_dict
            continue

        # Step 7 直接处理
        # Treat as List
        if get_json:
            try:
                schema_sheet, data_sheet = split_schema_row(sheet)
                res = match_list_column(schema_sheet, data_sheet)
            except Exception as e:
                import traceback
                traceback.print_exc()
                logAuto(f"Process List Exception {e}", log)
                res = vlm_get_json(sheet, save=cache)
            sheet_dict[key] = res

        # Step 7.1 尝试根据匹配到的Schema，再进行拆分为多个并列的表格
        # if direction == SCHEMA_TOP:
        #     try_sheet_list = split_subtable_by_schema(sheet, pos_list)
        #     pass
        # elif direction == SCHEMA_LEFT:
        #     pass

    return sheet_dict
