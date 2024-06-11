import re
from src.API import OpenAIAPI


comparison_prompt_sys = 'You are an AI assistant that compares answers. You will have to evaluate whether they are effectively the same. The predicted answer may not be exactly the same as the answer, but as long as it pretty much matches with the ground truth, it is correct. Respond with just "Correct" for a match or just "False" for a mismatch.'
comparison_prompt_usr = 'Question: {question}\n\nGround Truth Answer: {true_answer}\n\nPredicted Answer: {pred_answer}'

judge_prompt_sys = 'You are an AI assistant that compares large language model responses. You will have to evaluate which one is better. Respond with "A", "B", or "Tie" to indicate which response is better or if they are equal.'
judge_prompt_usr = 'Model A: {answer1}\n\nModel B: {answer2}'

stats_prompt_sys = 'Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".'
stats_prompt_usr = 'User Question: {question}\n\nAI Response: {answer}'


class Evaluator:
    def __init__(self):
        self.api = OpenAIAPI(model='gpt-3.5-turbo')
        self.api_judge = OpenAIAPI(model='gpt-4o')

    def evaluate_qa(self, question:str, true_answer:str, pred_answer:str, n_tries=3):
        cur_tries = 0
        while cur_tries < n_tries:
            message, _ = self.api.handle_response(comparison_prompt_sys, comparison_prompt_usr.format(question=question, true_answer=true_answer, pred_answer=pred_answer))
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
            message, _ = self.api_judge.handle_response(comparison_prompt_sys, comparison_prompt_usr.format(answer1=answers[0], answer2=answers[1]))
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
            message, _ = self.api_judge.handle_response(stats_prompt_sys, stats_prompt_usr.format(question=question, answer=answer))
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
