import pickle
import math

from embedding import *
from table2tree.feature_tree import *
from utils.api_utils import *
from utils.constants import *
from utils.prompt_template import *


class Verifier():
    
    
    def check_answer(self, query, answer):
        """判断 answer 的内容是否可以和query对应上"""
        prompt = check_answer_prompt.format(query=query, answer=answer)
        
        res = generate_deepseek(prompt, API_KEY, API_URL)
        
        if "T" in res:
            return True
        return False

    def get_queryset_by_answer(self, f_tree: FeatureTree, query, answer, n=4):

        prompt = back_verification_prompt.format(table=f_tree.__json__(), query=query, answer=answer, n=n)

        res = generate_deepseek(prompt, API_KEY, API_URL)
        res = res.splitlines()
        res = [x.strip() for x in res]

        return res

    def back_verify(self, f_tree: FeatureTree, query, answer, n=4):

        query_list = self.get_queryset_by_answer(f_tree, query, answer, n)

        similarity = EmbeddingModelMultilingualE5().one_to_many_semilarity(
            query, query_list
        )[0]

        score = sum(similarity) / len(similarity)

        return score, query_list

