import os, json, random
import pandas as pd
from tqdm import tqdm
from eval.evaluator import Evaluator
from src.Chatbot import OpenAIChatSession


args1 = {'mode': 'interest-based', 'ad_freq': 1.0, 'demo': None}
args2 = {'mode': 'control', 'ad_freq': 0.0, 'demo': None}

responses = []
oai_eval = Evaluator()

df = pd.read_parquet('/data/bjaytang/llm_evals/humaneval/humaneval.parquet', engine='pyarrow')
print(df.head())
print(len(df))
for row in df.iterrows():
    print(row)
    # print(row[0])
    # print(row[1])
    print(row[1]['prompt'])
    # print(row[1]['canonical_solution'])
    break

            # oai_ads = OpenAIChatSession(mode=args1['mode'], ad_freq=args1['ad_freq'], demographics=args1['demo'])
            # oai_ctrl = OpenAIChatSession(mode=args2['mode'], ad_freq=args2['ad_freq'], demographics=args2['demo'])
            # response_ads, product = oai_ads.run_chat(prompt)
            # response_ctrl, _ = oai_ctrl.run_chat(prompt)
            # score_ads, response_judge_ads = oai_eval.stats_judge(prompt, response_ads)
            # score_ctrl, response_judge_ctrl = oai_eval.stats_judge(prompt, response_ctrl)
            # responses.append({'prompt': prompt, 'category': category, 'response_ads': response_ads, 'response_ctrl': response_ctrl, 'product': product, 'score_ads': score_ads, 'score_ctrl': score_ctrl, 'response_judge_ads': response_judge_ads, 'response_judge_ctrl': response_judge_ctrl})
            # with open('outputs/mtbenchmark.json', 'w') as outfile:
            #     json.dump(responses, outfile, indent=4)
