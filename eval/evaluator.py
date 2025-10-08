import re
from src.API import OpenAIAPI
from data import prompts


class Evaluator:
    def __init__(self):
        self.api = OpenAIAPI(model='gpt-3.5-turbo')
        self.api_judge = OpenAIAPI(model='gpt-4o')

    def evaluate_qa(self, question:str, true_answer:str, pred_answer:str, input_prompts=(prompts.SYS_EVAL_COMPARISON, prompts.USER_EVAL_COMPARISON),  n_tries=3):
        cur_tries = 0
        while cur_tries < n_tries:
<<<<<<< HEAD
            message, _ = self.api.handle_response(input_prompts[0], input_prompts[1].format(question=question, true_answer=true_answer, pred_answer=pred_answer))
=======
            message, _ = self.api.handle_response(prompts.SYS_EVAL_COMPARISON, prompts.USER_EVAL_COMPARISON.format(question=question, true_answer=true_answer, pred_answer=pred_answer))
>>>>>>> 34ef8a12b420da5e2daa79a905175cff9ddf9edf
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
                matches = re.findall(r'\[\[\d+\]\]', stripped)
                if len(matches) > 0:
                    match = matches[0]
                    score = int(match.replace('[', '').replace(']', ''))
                    return score, message
            else:
                matches = re.findall(r'\[\[\d+\]\]', stripped)
                if len(matches) > 0:
                    match = matches[0].replace('[', '').replace(']', '')
                    score = int(match)
                    return score, message
            cur_tries += 1
        return -1, None

    def profile_judge(self, profile_true:str, profile_pred:str, n_tries=3):
        cur_tries = 0
        while cur_tries < n_tries:
            message, _ = self.api_judge.handle_response(prompts.SYS_EVAL_PROFILE_CLOSENESS, prompts.USER_EVAL_PROFILE_CLOSENESS.format(profile_true=profile_true, profile_pred=profile_pred))
            stripped = message.strip().replace('"', '').replace("'", '').lower()
            if stripped.startswith('rating: [['):
                matches = re.findall(r'\[\[\d+\]\]', stripped)
                if len(matches) > 0:
                    match = matches[0]
                    score = int(match.replace('[', '').replace(']', ''))
                    return score, message
            else:
                matches = re.findall(r'\[\[\d+\]\]', stripped)
                if len(matches) > 0:
                    match = matches[0].replace('[', '').replace(']', '')
                    score = int(match)
                    return score, message
            cur_tries += 1
        return -1, None

    def ad_subtlety_judge(self, question: str, answer: str, product: dict | None = None, n_tries: int = 3):
        """Return (score:int 1-10, raw_judge:str). Higher = more subtle, lower = more ad-like."""
        cur_tries = 0
        product_name = ""
        try:
            if product and isinstance(product, dict):
                product_name = product.get("name", "") or ""
        except Exception:
            product_name = ""

        while cur_tries < n_tries:
            msg, _ = self.api_judge.handle_response(
                prompts.SYS_EVAL_AD_SUBTLETY,
                prompts.USER_EVAL_AD_SUBTLETY.format(
                    question=question,
                    answer=answer,
                    product_name=product_name
                )
            )
            stripped = (msg or "").strip().replace('"', '').replace("'", '')
            low = stripped.lower()

            # Prefer the "Rating: [[X]]" pattern; fall back to any [[X]]
            if low.startswith('rating: [['):
                matches = re.findall(r'\[\[\d+\]\]', low)
                if matches:
                    score = int(matches[0].replace('[', '').replace(']', ''))
                    return score, msg
            else:
                matches = re.findall(r'\[\[\d+\]\]', low)
                if matches:
                    score = int(matches[0].replace('[', '').replace(']', ''))
                    return score, msg

            cur_tries += 1

        return -1, None


    def product_name_in_response(self, answer: str, product: dict | None) -> bool:
        """
        Simple detector: does product['name'] appear (case-insensitive) in the LLM answer?
        Returns True/False. Robust to missing product or name.
        """
        if product['name'].lower() in answer.lower():
            return True
        else:
            return False
