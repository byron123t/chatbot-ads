import os, json, random
from tqdm import tqdm
from eval.evaluator import Evaluator
from src.Chatbot import OpenAIChatSession


drop_prompt = 'Can you read this passage and answer the question?\n\nPassage: {passage}\n\nQuestion: {question}'

args1 = {'mode': 'interest-based', 'ad_freq': 1.0, 'demo': None}
args2 = {'mode': 'control', 'ad_freq': 0.0, 'demo': None}
# args3 = {'mode': 'influencer', 'ad_freq': 1.0, 'demo': demographics}

samples = []
with open('/data/bjaytang/llm_evals/drop_dataset/drop_dataset_dev.json', 'r') as infile:
    data = json.load(infile)
    for key, val in tqdm(data.items()):
        for question in val['qa_pairs']:
            samples.append({'key': key, 'passage': val['passage'], 'questions': [{'question': question['question'], 'answer': question['answer']} for question in val['qa_pairs']]})

random.shuffle(samples)

oai_eval = Evaluator()

responses = []
for sample in tqdm(samples[:200]):
    passage = sample['passage']
    question = sample['questions'][0]['question']
    answer = sample['questions'][0]['answer']
    oai_ads = OpenAIChatSession(mode=args1['mode'], ad_freq=args1['ad_freq'], demographics=args1['demo'])
    oai_ctrl = OpenAIChatSession(mode=args2['mode'], ad_freq=args2['ad_freq'], demographics=args2['demo'])
    # oai_influencer = OpenAIChatSession(mode=args3['mode'], ad_freq=args3['ad_freq'], demographics=args3['demo'])
    try:
        response_ads, product = oai_ads.run_chat(drop_prompt.format(passage=passage, question=question))
        response_ctrl, _ = oai_ctrl.run_chat(drop_prompt.format(passage=passage, question=question))
    except Exception as e:
        continue
    correct_ads = False
    correct_ctrl = False
    if len(answer['number']) > 0:
        gt_answer = answer['number']
        if gt_answer in response_ads:
            correct_ads = True
        if gt_answer in response_ctrl:
            correct_ctrl = True
    elif len(answer['spans']) > 0:
        gt_answer = answer['spans']
        for span in gt_answer:
            if span in response_ads:
                correct_ads = True
            if span in response_ctrl:
                correct_ctrl = True
    elif len(answer['date']):
        gt_answer = answer['date']
        for date_type, date in gt_answer.items():
            if date in response_ads:
                correct_ads = True
            if date in response_ctrl:
                correct_ctrl = True

    responses.append({'key': key, 'passage': passage, 'question': question, 'answer': answer, 'response_ads': response_ads, 'response_ctrl': response_ctrl, 'product': product, 'correct_ads': correct_ads, 'correct_ctrl': correct_ctrl})
    with open('outputs/drop.json', 'w') as outfile:
        json.dump(responses, outfile, indent=4)
    # correct = oai_eval.evaluate_qa(question, gt_answer, response_ads)
    # print(correct, simple_check)
    # correct = oai_eval.evaluate_qa(question, gt_answer, response_ctrl)
    # print(correct, simple_check)
