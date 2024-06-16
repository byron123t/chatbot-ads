import os, json, random
from tqdm import tqdm
import pandas as pd
from eval.evaluator import Evaluator
from src.Chatbot import OpenAIChatSession


for file in os.listdir('/data/bjaytang/llm_evals/mmlu/data/dev/'):
    if file.endswith('.csv'):
        df = pd.read_csv('/data/bjaytang/llm_evals/mmlu/data/dev/{file}'.format(file=file), header=None)
        print(df.columns)
        print(df.head())
        print(len(df))
        for row in tqdm(df.iterrows()):
            print(row[1][0])
            print(row[1][1])
            print(row[1][2])
            print(row[1][3])
            print(row[1][4])
            print(row[1][5])
            break
        break
