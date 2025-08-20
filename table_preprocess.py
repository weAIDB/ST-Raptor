import pickle
import os
import glob
import json
import multiprocessing
import time
import traceback

from loguru import logger
from tqdm import tqdm

from clean_cache import clear_cache_folder
from table2tree.feature_tree import *
from embedding import *


import logging
# Ignore warnings of transformers pkg
logging.getLogger("transformers").setLevel(logging.ERROR)
import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")

def excel2tree(
    file,
    pkl_dir=None,
    convert_pkl=True, 
    json_dir : bool = None,
    convert_json=True,  
    str_dir=None,
    convert_str=True, 
    embedding_dir=None,
    convert_embedding=True, 
    structured=False,  
    log=False,  
    vlm_cache=False,
):
    """ Convert the input excel file into the HO-Tree (FeatureTree) object.

    Args:
        file (_type_): the input Excel file path
        pkl_dir (_type_, optional): _description_. 输出pkl文件的保存路径
        convert_pkl (bool, optional): _description_. 是否输出FeatureTree对象的pkl文件
        json_dir (_type_, optional): _description_. 输出json文件的保存路径
        convert_json (bool, optional): _description_. Defaults to True. 是否输出FeatureTree对象序列化后的Json文件
        str_dir (_type_, optional): _description_. 输出str文件的保存路径
        convert_str (bool, optional): _description_. Defaults to True. 是否输出FeatureTree对象序列化后的String文件
        embedding_dir (_type_, optional): _description_. 输出embedding文件的保存路径
        convert_embedding (bool, optional): _description_. Defaults to True. 是否将表格每个单元格的内容embedding后保存，用于后续问题回答
        structured (bool, optional): _description_. Defaults to True. 默认是半结构化表格
        log (bool, optional): _description_. Defaults to True. 是否输出日志
        vlm_cache (bool, optional): _description_. Defaults to True. 在vlm转换的时候是否要进行cache
    
    Returns:
        FeatureTree: The convert HO-Tree of the input excel file.
    """
    
    if not os.path.exists(pkl_dir): os.mkdir(pkl_dir)
    if not os.path.exists(json_dir): os.mkdir(json_dir)
    if not os.path.exists(str_dir): os.mkdir(str_dir)
    if not os.path.exists(embedding_dir): os.mkdir(embedding_dir)

    name = os.path.basename(file)[:-5]

    flag = [False, False, False, False]
    if (
        convert_pkl and os.path.exists(os.path.join(pkl_dir, f"{name}.pkl"))
    ) or not convert_pkl:
        flag[0] = True
    if (
        convert_json and os.path.exists(os.path.join(json_dir, f"{name}.json"))
    ) or not convert_json:
        flag[1] = True
    if (
        convert_str and os.path.exists(os.path.join(str_dir, f"{name}.txt"))
    ) or not convert_str:
        flag[2] = True
    if (
        convert_embedding
        and os.path.exists(os.path.join(embedding_dir, f"{name}.embedding.json"))
    ) or not convert_embedding:
        flag[3] = True
    if flag == [True, True, True, True]:
        return

    try:
        f_tree = get_excel_feature_tree(file, structured=structured, log=log, vlm_cache=vlm_cache)
        tree_json = f_tree.__json__()
        tree_str = f_tree.__str__()
    except Exception as e:
        logger.error(f"File: {name}.xlsx Error: {e}")
        with open("./error.txt", "a") as f:
            f.write(f"process_one_table() error: {name}.xlsx\n")
        traceback.print_exc()
        return

    if convert_pkl:
        with open(os.path.join(pkl_dir, f"{name}.pkl"), "wb") as f:
            pickle.dump(f_tree, f)
    if convert_json:
        with open(os.path.join(json_dir, f"{name}.json"), "w") as f:
            json.dump(tree_json, f, indent=4, ensure_ascii=False)
    if convert_str:
        with open(os.path.join(str_dir, f"{name}.txt"), "w") as f:
            f.write(tree_str)
    if convert_embedding:
        embedding_dict = EmbeddingModelMultilingualE5().get_embedding_dict(
            f_tree.all_value_list()
        )
        EmbeddingModelMultilingualE5().save_embedding_dict(
            embedding_dict, os.path.join(embedding_dir, f"{name}.embedding.json")
        )
    return f_tree

def preprocess_one_pkl(
    file,
    json_dir=None,
    convert_json=True,  
    str_dir=None, 
    convert_str=True,  
    embedding_dir=None,
    convert_embedding=True, 
):
    """_summary_

    Args:
        file (_type_): _description_
        json_dir (_type_, optional): _description_. Defaults to None.
        convert_json (bool, optional): _description_. Defaults to True.
        str_dir (_type_, optional): _description_. Defaults to None.
        convert_str (bool, optional): _description_. Defaults to True.
        embedding_dir (_type_, optional): _description_. Defaults to None.
        convert_embedding (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    
    name = os.path.basename(file)[:-4]
    with open(os.path.join(file), "rb") as f:
        f_tree: FeatureTree = pickle.load(f)

    flag = [False, False, False]
    if (
        convert_json and os.path.exists(os.path.join(json_dir, f"{name}.json"))
    ) or not convert_json:
        flag[0] = True
    if (
        convert_str and os.path.exists(os.path.join(str_dir, f"{name}.txt"))
    ) or not convert_str:
        flag[1] = True
    if (
        convert_embedding
        and os.path.exists(os.path.join(embedding_dir, f"{name}.embedding.json"))
    ) or not convert_embedding:
        flag[2] = True
    if flag == [True, True, True]:
        return

    try:
        tree_json = f_tree.__json__()
        tree_str = f_tree.__str__()
    except Exception as e:
        logger.error(f"File: {name}.xlsx Error: {e}")
        with open("./error.txt", "a") as f:
            f.write(f"process_one_pkl() error: {name}.xlsx\n")
        traceback.print_exc()
        return

    if convert_json:
        with open(os.path.join(json_dir, f"{name}.txt"), "w") as f:
            f.write(tree_str)
    if convert_str:
        with open(os.path.join(json_dir, f"{name}.json"), "w") as f:
            json.dump(tree_json, f, indent=4, ensure_ascii=False)
    if convert_embedding:
        embedding_dict = EmbeddingModelMultilingualE5().get_embedding_dict(
            f_tree.all_value_list()
        )
        EmbeddingModelMultilingualE5().save_embedding_dict(
            embedding_dict, os.path.join(embedding_dir, f"{name}.embedding.json")
        )
    return f_tree

def process_excel_files(
    files,
    pkl_dir=None,
    convert_pkl=True, 
    json_dir : bool = None,
    convert_json=True,  
    str_dir=None,
    convert_str=True, 
    embedding_dir=None,
    convert_embedding=True, 
    structured=False,  
    log=False,  
    vlm_cache=False,  
):
    """ Convert the input excel file list into the HO-Tree (FeatureTree) object.

    Args:
        file (_type_): the input Excel file path
        pkl_dir (_type_, optional): _description_. 输出pkl文件的保存路径
        convert_pkl (bool, optional): _description_. 是否输出FeatureTree对象的pkl文件
        json_dir (_type_, optional): _description_. 输出json文件的保存路径
        convert_json (bool, optional): _description_. Defaults to True. 是否输出FeatureTree对象序列化后的Json文件
        str_dir (_type_, optional): _description_. 输出str文件的保存路径
        convert_str (bool, optional): _description_. Defaults to True. 是否输出FeatureTree对象序列化后的String文件
        embedding_dir (_type_, optional): _description_. 输出embedding文件的保存路径
        convert_embedding (bool, optional): _description_. Defaults to True. 是否将表格每个单元格的内容embedding后保存，用于后续问题回答
        structured (bool, optional): _description_. Defaults to True. 默认是半结构化表格
        log (bool, optional): _description_. Defaults to True. 是否输出日志
        vlm_cache (bool, optional): _description_. Defaults to True. 在vlm转换的时候是否要进行cache
    """
    if convert_pkl:
        os.makedirs(pkl_dir, exist_ok=True)
    if convert_json:
        os.makedirs(json_dir, exist_ok=True)
    if convert_str:
        os.makedirs(str_dir, exist_ok=True)
    if convert_embedding:
        os.makedirs(embedding_dir, exist_ok=True)

    for file in tqdm(files, desc="Processing..."):
        excel2tree(
            file,
            pkl_dir=pkl_dir,
            convert_pkl=convert_pkl,
            json_dir=json_dir,
            convert_json=convert_json,
            str_dir=str_dir,
            convert_str=convert_str,
            embedding_dir=embedding_dir,
            convert_embedding=convert_embedding,
            structured=structured,
            log=log,
            vlm_cache=vlm_cache,
        )


def process_pkl_files(
    files,
    json_dir=None,
    convert_json=True,  # 是否输出FeatureTree对象序列化后的Json文件
    str_dir=None,
    convert_str=True,  # 是否输出FeatureTree对象序列化后的String文件
    embedding_dir=None,
    convert_embedding=True,  # 是否将表格每个单元格的内容embedding后保存，用于后续问题回答
    log=False,  # 是否输出日志
    vlm_cache=False,  # 在vlm转换的时候是否要进行cache
):
    if convert_json:
        os.makedirs(json_dir, exist_ok=True)
    if convert_str:
        os.makedirs(str_dir, exist_ok=True)
    if convert_embedding:
        os.makedirs(embedding_dir, exist_ok=True)

    for file in tqdm(files, desc="Processing..."):
        preprocess_one_pkl(
            file,
            json_dir=json_dir,
            convert_json=convert_json,
            str_dir=str_dir,
            convert_str=convert_str,
            embedding_dir=embedding_dir,
            convert_embedding=convert_embedding,
        )


def multi_process_process_excels(
    files,
    pkl_dir=None,
    convert_pkl=True,  # 是否输出FeatureTree对象的pkl文件
    json_dir=None,
    convert_json=True,  # 是否输出FeatureTree对象序列化后的Json文件
    str_dir=None,
    convert_str=True,  # 是否输出FeatureTree对象序列化后的String文件
    embedding_dir=None,
    convert_embedding=True,  # 是否将表格每个单元格的内容embedding后保存，用于后续问题回答
    structured=False,   # 默认是半结构化表格
    log=False,  # 是否输出日志
    vlm_cache=False,  # 在vlm转换的时候是否要进行cache
    n=6,
):
    param_list = [
        (
            file,
            pkl_dir,
            convert_pkl,
            json_dir,
            convert_json,
            str_dir,
            convert_str,
            embedding_dir,
            convert_embedding,
            log,
            vlm_cache,
        )
        for file in files
    ]

    with multiprocessing.Pool(processes=n) as pool:
        pool.starmap(excel2tree, param_list)
    print("All jobs completed!")


def multi_process_process_pkls(
    files,
    json_dir=None,
    convert_json=True,  # 是否输出FeatureTree对象序列化后的Json文件
    str_dir=None,
    convert_str=True,  # 是否输出FeatureTree对象序列化后的String文件
    embedding_dir=None,
    convert_embedding=True,  # 是否将表格每个单元格的内容embedding后保存，用于后续问题回答
    n=6,
):
    param_list = [
        (
            file, 
            json_dir,
            convert_json,
            str_dir,
            convert_str,
            embedding_dir,
            convert_embedding,
        )
        for file in files
    ]

    with multiprocessing.Pool(processes=n) as pool:
        pool.starmap(preprocess_one_pkl, param_list)
    print("All jobs completed!")


def main():
    clear_cache_folder(CACHE_DIR)
    
    # dataset_dir = '/home/zirui/SemiTableQA/data/temptabqa-st/'
    dataset_dir = '/home/zirui/SemiTableQA/data/wikitq-st-demo/'
    
    table_dir = os.path.join(dataset_dir, 'table')
    pkl_dir = os.path.join(dataset_dir, 'pkl')
    json_dir = os.path.join(dataset_dir, 'json')
    str_dir = os.path.join(dataset_dir, 'str')
    embedding_dir = os.path.join(dataset_dir, 'embedding')
    
    files = glob.glob(table_dir + '/*.xlsx')
    
    for file in tqdm(files):
        excel2tree(
            file,
            pkl_dir=pkl_dir,
            convert_pkl=True, 
            json_dir=json_dir,
            convert_json=True,  
            str_dir=str_dir,
            convert_str=True, 
            embedding_dir=embedding_dir,
            convert_embedding=True, 
            structured=False,  
            log=True,  
            vlm_cache=False,  
        )

if __name__ == '__main__':
    main()
    