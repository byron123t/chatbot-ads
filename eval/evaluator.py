import re
from src.API import OpenAIAPI
from data import prompts


class Evaluator:
    def __init__(self):
        self.api = OpenAIAPI(model='gpt-3.5-turbo')
        self.api_judge = OpenAIAPI(model='gpt-4o')

    def evaluate_qa(self, question:str, true_answer:str, pred_answer:str, n_tries=3):
        cur_tries = 0
        while cur_tries < n_tries:
            message, _ = self.api.handle_response(prompts.SYS_EVAL_COMPARISON, prompts.USER_EVAL_COMPARISON.format(question=question, true_answer=true_answer, pred_answer=pred_answer))
            stripped = message.strip().replace('"', '').replace("'", '').lower()
            if stripped.startswith('correct'):
                return True
            elif stripped.startswith('false'):
                return False
            cur_tries += 1
        return False
    
    def evaluate_judge(self, answers:list, n_tries=3):
        cur_tries = 0
        while cur_tries < n_tries:
            message, _ = self.api_judge.handle_response(prompts.SYS_EVAL_COMPARISON, prompts.USER_EVAL_COMPARISON.format(answer1=answers[0], answer2=answers[1]))
            stripped = message.strip().replace('"', '').replace("'", '').lower()
            if stripped.startswith('a'):
                return 0
            elif stripped.startswith('b'):
                return 1
            elif stripped.startswith('tie'):
                return -1
            cur_tries += 1
        return -1

    def stats_judge(self, question:str, answer:str, n_tries=3):
        cur_tries = 0
        while cur_tries < n_tries:
            message, _ = self.api_judge.handle_response(prompts.SYS_EVAL_STATS, prompts.USER_EVAL_STATS.format(question=question, answer=answer))
            stripped = message.strip().replace('"', '').replace("'", '').lower()
            if stripped.startswith('rating: [['):
                match = re.finditer(r'\[\[\d+\]\]', stripped)
                match.replace('[', '').replace(']', '')
                score = int(match)
                return score, message
            else:
                matches = re.findall(r'\[\[\d+\]\]', stripped)
                if len(matches) > 0:
                    match = matches[0].replace('[', '').replace(']', '')
                    score = int(match)
                    return score, message
            cur_tries += 1
        return -1, None
