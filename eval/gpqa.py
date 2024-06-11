import os, json, random
import pandas as pd
from tqdm import tqdm
from eval.evaluator import Evaluator
from src.Chatbot import OpenAIChatSession


files = ['gpqa_diamond', 'gpqa_experts', 'gpqa_extended', 'gpqa_main']
for file in files:
    print(file)
    df = pd.read_csv('/data/bjaytang/llm_evals/gpqa/dataset/{file}.csv'.format(file=file))
    print(df.head())
    print(df.columns)
    for row in df.iterrows():
        print(row[''])
    break
