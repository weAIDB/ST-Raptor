import time
import glob
import json
import os

from tqdm import tqdm
from loguru import logger
import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

from utils.api_utils import *
from utils.prompt_template import *
from utils.constants import *


def is_equal(a, b):
    prompt = evaluation_prompt.format(a=a, b=b)

    res = generate_deepseek(prompt, API_KEY, API_URL)

    return res


def calculate_meteor(reference, hypothesis):
    """
    reference: 参考字符串，如 "the cat is on the mat"）
    hypothesis: 候选字符串，如 "a cat sits on the mat"）
    """
    # 分词处理
    ref_tokens = nltk.word_tokenize(reference)
    hyp_tokens = nltk.word_tokenize(hypothesis)

    # 计算METEOR
    return meteor_score([ref_tokens], hyp_tokens)


def calculate_rouge(reference, hypothesis):
    """
    reference: 参考文本 (字符串)
    hypothesis: 生成文本 (字符串)
    返回 ROUGE-1、ROUGE-2、ROUGE-L 的 F1 分数
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)

    return {
        "ROUGE-1": scores["rouge1"].fmeasure,
        "ROUGE-2": scores["rouge2"].fmeasure,
        "ROUGE-L": scores["rougeL"].fmeasure,
    }


def calculate_bleu(reference, hypothesis):
    """
    reference: 参考文本（列表，支持多个参考，如 ["参考句子1", "参考句子2"]）
    hypothesis: 生成文本（字符串）
    """
    # 将文本分词
    if not isinstance(reference, list):
        reference = [reference]
    refs = [nltk.word_tokenize(ref) for ref in reference]
    hyp = nltk.word_tokenize(hypothesis)

    # 计算BLEU-4（默认权重）
    return sentence_bleu(refs, hyp, weights=(0.25, 0.25, 0.25, 0.25))


def evaluate(input_file, output_dir):
    basename = os.path.basename(input_file)
    output_file = os.path.join(output_dir, basename)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    data = []
    with open(input_file, "r") as file:
        for line in file:
            tmp = json.loads(line.strip())
            data.append(tmp)

    res = []
    if os.path.exists(output_file):
        with open(output_file, "r") as file:
            for line in file:
                res.append(json.loads(line.strip()))

    for row in tqdm(data, desc="Processing..."):

        flag = False
        for x in res:
            if x is None:
                continue
            if row["id"] == x["id"]:
                flag = True
                break
        if flag:
            continue

        try:
            if 'tablellama' in basename:
                judge = is_equal(row["label"], row["model_output"][:-4])
            else:
                judge = is_equal(row["label"], row["model_output"])
        except Exception as e:
            import traceback
            traceback.print_exc()
            # judge = 'F'
            # print(row)
            print(e)
            continue

        row["judge"] = judge
        res.append(row)

        with open(output_file, "a") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Calculate Acc
    correct = 0
    total = len(res)
    for row in res:
        if "T" in row["judge"]:
            correct += 1
    accuracy = correct / total
    print(f"{basename} Accuracy: {accuracy}")

    # Calculate METEOR
    meteor_score = 0
    total = len(res)
    for row in res:
        meteor_score += calculate_meteor(str(row["label"]), str(row["model_output"]))
    meteor_score = meteor_score / total
    print(f"{basename} METEOR: {meteor_score}")

    # Calculate ROUGE-1/2/L
    r1 = 0
    r2 = 0
    rl = 0
    total = len(res)
    for row in res:
        score_dict = calculate_rouge(str(row["label"]), str(row["model_output"]))
        r1 += score_dict["ROUGE-1"]
        r2 += score_dict["ROUGE-2"]
        rl += score_dict["ROUGE-L"]
    r1 = r1 / total
    r2 = r2 / total
    rl = rl / total
    print(f"{basename} ROUGE-1: {r1}")
    print(f"{basename} ROUGE-2: {r2}")
    print(f"{basename} ROUGE-L: {rl}")

    # Calculate BLEU
    bleu = 0
    total = len(res)
    for row in res:
        bleu += calculate_bleu(str(row["label"]), str(row["model_output"]))
    bleu = bleu / total
    print(f"{basename} BLEU: {bleu}")
