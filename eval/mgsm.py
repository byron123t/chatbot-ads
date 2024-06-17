import os, json, random
from tqdm import tqdm
import pandas as pd
from eval.evaluator import Evaluator
from src.Chatbot import OpenAIChatSession

args1 = {'mode': 'interest-based', 'ad_freq': 1.0, 'demo': None}
args2 = {'mode': 'control', 'ad_freq': 0.0, 'demo': None}

responses = []
df = pd.read_csv('/data/bjaytang/llm_evals/mgsm/mgsm_en.tsv', sep='\t', header=None)
print(df.head())
print(len(df))
for row in tqdm(df.iterrows()):
    prompt = row[1][0]
    oai_ads = OpenAIChatSession(mode=args1['mode'], ad_freq=args1['ad_freq'], demographics=args1['demo'])
    oai_ctrl = OpenAIChatSession(mode=args2['mode'], ad_freq=args2['ad_freq'], demographics=args2['demo'])
    try:
        response_ads, product = oai_ads.run_chat(prompt)
        response_ctrl, _ = oai_ctrl.run_chat(prompt)
    except Exception as e:
        continue
    gt_answer = row[1][1]
    correct_ads = False
    correct_ctrl = False
    if gt_answer in response_ads:
        correct_ads = True
    if gt_answer in response_ctrl:
        correct_ctrl = True
    responses.append({'question': prompt, 'answer': gt_answer, 'response_ads': response_ads, 'response_ctrl': response_ctrl, 'product': product, 'correct_ads': correct_ads, 'correct_ctrl': correct_ctrl})
    with open('outputs/mgsm.json', 'w') as outfile:
        json.dump(responses, outfile, indent=4)
