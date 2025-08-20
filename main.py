"""
The Query Answering Pipeline
"""

import os
import re
import json
import glob
import pickle

from tqdm import tqdm
from loguru import logger

from query.primitive_pipeline import *
from table2tree.extract_excel import *
from table2tree.feature_tree import *
from utils.constants import DELIMITER


def answer_question(
    f_tree: FeatureTree,
    record,
    enable_emebdding=True,
    embedding_cache_file=None,
    log_dir=None,
):
    query = record["query"]

    # create qa log file
    if log_dir is not None:
        log_file = os.path.join(log_dir, f"{record['id']}.txt")
    else:
        log_file = None

    if log_file is not None:  # Log
        with open(log_file, "a") as f:
            f.write(f"{DELIMITER} Query {DELIMITER}\n")
            f.write(query + "\n")

    try:
        final_answer, qa_pair, reliability = qa_RWP(
            f_tree=f_tree,
            query=query,
            enable_emebdding=enable_emebdding,
            embedding_cache_file=embedding_cache_file,
            log_file=log_file,
        )
        record["reliability"] = reliability
        record["model_output"] = final_answer

    except Exception as e:
        if log_file is not None:  # Log
            with open(log_file, "a") as f:
                f.write(f"{DELIMITER} An Error Occurred {DELIMITER}\n")
                f.write(f"Error: {e}\n")
        print(e)
        import traceback
        traceback.print_exc()
        return None

    if log_file is not None:  # Log
        with open(log_file, "a") as f:
            f.write(f"{DELIMITER} Final Output {DELIMITER}\n")
            f.write(json.dumps(record, ensure_ascii=False, indent=4))

    return record


def benchmark(
    table_dir,
    cache_dir,
    input_jsonl,
    output_jsonl,
    pkl_dir=None,
    enable_emebdding=True,
    embedding_cache_dir=None,
    log_dir=None,
):
    """
    table_dir and cache_dir must be not None simutanously. 
    If pkl_dir is pointed, table_dir and cache_dir is useless.
    When using table_dir, we default embedding_cache_dir to be None.
    """

    input_list = []
    with open(input_jsonl, "r", encoding="utf-8") as file:
        for line in file:
            tmp = json.loads(line)
            input_list.append(tmp)

    output_data = []
    if os.path.exists(output_jsonl):
        with open(output_jsonl, 'r', encoding='utf-8') as file:
            for line in file:
                output_data.append(json.loads(line))

    if pkl_dir is not None:
        pkl_files = sorted(glob.glob(pkl_dir + "/*.pkl"))
        input_list.sort(key=lambda x: x["table_id"])

        for index, pkl_file in enumerate(pkl_files):
            name = os.path.basename(pkl_file)[:-4]

            # Find Embedding Cache File
            embedding_cache_file = None
            if embedding_cache_dir is not None:
                if os.path.exists(
                    os.path.join(embedding_cache_dir, f"{name}.embedding.json")
                ):
                    embedding_cache_file = os.path.join(
                        embedding_cache_dir, f"{name}.embedding.json"
                    )

            # Preprocess QA List
            qa_list = []
            for record in input_list:
                if str(record["table_id"]) == str(name):
                    qa_list.append(record)

            # Load FeatureTree
            with open(pkl_file, "rb") as f:
                f_tree: FeatureTree = pickle.load(f)

            # Answer Queries
            for record in qa_list:
                
                flag = False
                for item in output_data:
                    if item['id'] == record['id']:
                        flag = True
                        break
                if flag:    continue
                
                record = answer_question(
                    f_tree=f_tree,
                    record=record,
                    enable_emebdding=enable_emebdding,
                    embedding_cache_file=embedding_cache_file,
                    log_dir=log_dir,
                )

                if record is None:  continue

                # Output QA Information
                output_data.append(record)
                with open(output_jsonl, "a", encoding="utf-8") as file:
                    file.write(f"{json.dumps(record, ensure_ascii=False)}\n")
    else:
        table_files = glob.glob(table_dir + "/*.xlsx")
        os.makedirs(cache_dir, exist_ok=True)

        for table_file in table_files:
            name = os.path.basename(table_file)[:-5]

            # Preprocess QA List
            qa_list = []
            for record in input_list:
                if str(record["table_id"]) == str(name):
                    qa_list.append(record)

            # Make pkl file
            try:
                f_tree = get_excel_feature_tree(table_file, log=True, vlm_cache=False)
                tree_json = f_tree.__json__()
                tree_str = f_tree.__str__([1])
            except Exception as e:
                import traceback
                traceback.print_exc()
                logger.error(f"File: {name}.xlsx Error: {e}")
                with open("./error.txt", "a") as f:
                    f.write(f"process_one_table() error: {name}.xlsx\n")
                return

            with open(os.path.join(cache_dir, f"{name}.pkl"), "wb") as f:
                pickle.dump(f_tree, f)
            with open(os.path.join(cache_dir, f"{name}.txt"), "w", encoding='utf-8') as f:
                f.write(tree_str)
            with open(os.path.join(cache_dir, f"{name}.json"), "w", encoding='utf-8') as f:
                json.dump(tree_json, f, indent=4, ensure_ascii=False)
            embedding_dict = EmbeddingModelMultilingualE5().get_embedding_dict(
                f_tree.all_value_list()
            )
            EmbeddingModelMultilingualE5().save_embedding_dict(
                embedding_dict, os.path.join(cache_dir, f"{name}.embedding.json")
            )

            # Answer Queries
            for record in qa_list:

                flag = False
                for item in output_data:
                    if item['id'] == record['id']:
                        flag = True
                        break
                if flag:    continue
                
                record = answer_question(
                    f_tree=f_tree,
                    record=record,
                    embedding_cache_file=embedding_cache_file,
                    log_dir=log_dir,
                )
                
                if record is None:  continue
                
                output_data.append(record)
                # Output QA Information
                with open(output_jsonl, "a", encoding="utf-8") as file:
                    file.write(f"{json.dumps(record, ensure_ascii=False)}\n")

def main():
    
    # You need to change this
    input_jsonl = None
    table_dir = None
    pkl_dir = None
    embedding_cache_dir = None
    output_jsonl = None
    log_dir = None

    os.makedirs(log_dir, exist_ok=True)
    benchmark(
        table_dir=table_dir,
        input_jsonl=input_jsonl,
        output_jsonl=output_jsonl,
        pkl_dir=pkl_dir,
        embedding_cache_dir=embedding_cache_dir,
        cache_dir=CACHE_DIR,
        log_dir=log_dir,
    )

if __name__ == "__main__":
    main()
    # run()

