import csv, json, os, re
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from src.API import OpenAIAPI
from data import prompts


likert_dict = {'Strongly Disagree': 1, 'Disagree': 2, 'Somewhat Disagree': 3, 'Neither Agree Nor Disagree': 4, 'Somewhat Agree': 5, 'Agree': 6, 'Strongly Agree': 7, 'Strongle Agree': 7, '1': 1, '2': 3, '3': 4, '4': 6, '5': 7}
invert_dict = {1: 7, 2: 6, 3: 5, 4: 4, 5: 3, 6: 2, 7: 1}
oai = OpenAIAPI(model='gpt-4o-mini')


def llm_sentiment(question, responses):
    message, _ = oai.handle_response(prompts.SYS_EVAL_SENTIMENT, prompts.USER_EVAL_SENTIMENT.format(question=question, answers=responses))
    message = re.sub(r'\d+\. ', '', message.lower())
    split = message.split('\n')
    return split


def llm_clustering(question, responses):
    message, _ = oai.handle_response(prompts.SYS_EVAL_CLUSTER, prompts.USER_EVAL_CLUSTER.format(question=question, answers=responses))
    message = re.sub(r'\d+\. ', '', message.lower())
    split = message.split('\n')
    return split


def llm_tags(question, response, tags=[]):
    message, _ = oai.handle_response(prompts.SYS_EVAL_TAGS, prompts.USER_EVAL_TAGS.format(question=question, answer=response, taglist=tags))
    try:
        newtags = json.loads(message.lower())
    except json.JSONDecodeError:
        print('Failed to parse JSON: ', message)
        message.replace('[', '').replace(']', '')
        newtags = message.split(',')
        newtags = [tag.strip().lower() for tag in newtags]
    for tag in newtags:
        if tag not in tags:
            tags.append(tag)
    return newtags


def parse_qualtrics_data(file_path, chat_history):
    grouped_data = defaultdict(list)
    failed_prolifics = []
    with open(file_path, mode='r', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        for row in reader:
            key = row['KEY'].strip().lower()
            if row['KEY'] not in chat_history:
                print('Failed Prolific ID: ', row['Q2283'])
                continue
            if key.startswith('er_'):
                grouped_data['model'].append('gpt-4')
                grouped_data['mode'].append('control')
                grouped_data['disclosure'].append('none')
                grouped_data['overall_mode'].append('gpt-4-control-none')
            elif key.startswith('fr_'):
                grouped_data['model'].append('gpt-3.5')
                grouped_data['mode'].append('control')
                grouped_data['disclosure'].append('none')
                grouped_data['overall_mode'].append('gpt-3.5-control-none')
            elif key.startswith('gr_'):
                grouped_data['model'].append('gpt-4')
                grouped_data['mode'].append('interest')
                grouped_data['disclosure'].append('none')
                grouped_data['overall_mode'].append('gpt-4-ads-none')
            elif key.startswith('hr_'):
                grouped_data['model'].append('gpt-3.5')
                grouped_data['mode'].append('interest')
                grouped_data['disclosure'].append('none')
                grouped_data['overall_mode'].append('gpt-3.5-ads-none')
            elif key.startswith('ir_'):
                grouped_data['model'].append('gpt-4')
                grouped_data['mode'].append('interest')
                grouped_data['disclosure'].append('transparent')
                grouped_data['overall_mode'].append('gpt-4-ads-disclosure')
            elif key.startswith('jr_'):
                grouped_data['model'].append('gpt-3.5')
                grouped_data['mode'].append('interest')
                grouped_data['disclosure'].append('transparent')
                grouped_data['overall_mode'].append('gpt-3.5-ads-disclosure')
            else:
                continue
            grouped_data['chat_history'].append(chat_history[row['KEY']]['conversation'])
            temp_grouped_data = defaultdict(list)
            for column_id, value in row.items():
                if column_id == 'Duration (in seconds)':
                    grouped_data['Duration'].append(int(value))
                elif column_id == 'Q2283':
                    grouped_data['ProlificID'].append(value)
                elif column_id == 'Q2365':
                    grouped_data['Task'].append(value)
                elif column_id == 'Q2292':
                    if len(value) > 0:
                        print('Withdraw: ', key, row['Q2283'])
                        grouped_data['Withdraw'].append(True)
                    else:
                        grouped_data['Withdraw'].append(False)
                elif column_id == 'Q2346':
                    if len(value) > 0:
                        print('Data Deletion: ', key, row['Q2283'])
                        grouped_data['Deletion'].append(True)
                    else:
                        grouped_data['Deletion'].append(False)
                elif column_id == 'Q2342':
                    grouped_data['Familiarity'].append(value)
                elif column_id == 'Q2279':
                    if len(value) > 0:
                        if ',' in value:
                            grouped_data['UsedChatbots'].append(value.split(','))
                        else:
                            grouped_data['UsedChatbots'].append([value])
                    else:
                        grouped_data['UsedChatbots'].append([])
                elif column_id == 'Q2262':
                    grouped_data['Frequency'].append(value)
                elif column_id == 'Q2250':
                    if int(value) > 0 and int(value) < 25:
                        grouped_data['Age'].append('18-24')
                    elif int(value) > 25 and int(value) < 35:
                        grouped_data['Age'].append('25-34')
                    elif int(value) > 35 and int(value) < 45:
                        grouped_data['Age'].append('35-44')
                    elif int(value) > 45 and int(value) < 55:
                        grouped_data['Age'].append('45-54')
                    elif int(value) > 55 and int(value) < 65:
                        grouped_data['Age'].append('55-64')
                    elif int(value) > 65:
                        grouped_data['Age'].append('65+')
                elif column_id == 'Q2249':
                    grouped_data['Gender'].append(value)
                elif column_id == 'Q2251':
                    grouped_data['Ethnicity'].append(value)
                elif column_id == 'Q2345':
                    grouped_data['Education'].append(value)
                if column_id == 'Q2348_1' or column_id == 'Q2348_2' or column_id == 'Q2348_3' or column_id == 'Q2348_7' or column_id == 'Q2348_10' or column_id == 'Q2348_11' or column_id == 'Q2349_1' or column_id == 'Q2349_2' or column_id == 'Q2349_8' or column_id == 'Q2349_9' or column_id == 'Q2349_10':
                    temp_grouped_data['Positive_Impression'].append(likert_dict[value])
                if column_id == 'Q2348_4' or column_id == 'Q2348_5' or column_id == 'Q2348_6' or column_id == 'Q2348_8' or column_id == 'Q2348_9' or column_id == 'Q2349_3' or column_id == 'Q2349_4' or column_id == 'Q2349_5' or column_id == 'Q2349_6' or column_id == 'Q2349_7':
                    temp_grouped_data['Negative_Impression'].append(likert_dict[value])
                if column_id == 'Q2348_5' or column_id == 'Q2348_8' or column_id == 'Q2348_11' or column_id == 'Q2349_5' or column_id == 'Q2349_6' or column_id == 'Q2349_8' or column_id == 'Q2349_10':
                    if column_id == 'Q2348_5' or column_id == 'Q2348_8' or column_id == 'Q2349_5' or column_id == 'Q2349_6':
                        temp_grouped_data['Credibility'].append(invert_dict[likert_dict[value]])
                    elif column_id == 'Q2348_11' or column_id == 'Q2349_8' or column_id == 'Q2349_10':
                        temp_grouped_data['Credibility'].append(likert_dict[value])
                if column_id == 'Q2348_6' or column_id == 'Q2349_1' or column_id == 'Q2349_2' or column_id == 'Q2349_3' or column_id == 'Q2349_4':
                    if column_id == 'Q2348_6' or column_id == 'Q2349_3' or column_id == 'Q2349_4':
                        temp_grouped_data['Friendliness'].append(invert_dict[likert_dict[value]])
                    elif column_id == 'Q2349_1' or column_id == 'Q2349_2':
                        temp_grouped_data['Friendliness'].append(likert_dict[value])
                if column_id == 'Q2348_2' or column_id == 'Q2349_7' or column_id == 'Q2349_9':
                    if column_id == 'Q2348_2' or column_id == 'Q2349_9':
                        temp_grouped_data['Responsibility'].append(likert_dict[value])
                    elif column_id == 'Q2349_7':
                        temp_grouped_data['Responsibility'].append(invert_dict[likert_dict[value]])
                if column_id == 'Q2348_3':
                    grouped_data['Relevance'].append(likert_dict[value])
                if column_id == 'Q2348_7':
                    grouped_data['Convincingness'].append(likert_dict[value])
                if column_id == 'Q2348_1' or column_id == 'Q2348_4' or column_id == 'Q2348_9':
                    if column_id == 'Q2348_1':
                        temp_grouped_data['Helpfulness'].append(likert_dict[value])
                    elif column_id == 'Q2348_4' or column_id == 'Q2348_9':
                        temp_grouped_data['Helpfulness'].append(invert_dict[likert_dict[value]])
                if column_id == 'Q2338':
                    grouped_data['Text_Usefulness'].append(value)
                if column_id == 'Q2340':
                    grouped_data['Text_Personality'].append(value)
                if column_id == 'Q2343':
                    grouped_data['Text_Products_brands'].append(value)
                if column_id == 'Q2350_1':
                    grouped_data['Marketing_Ads'].append(likert_dict[value])
                if column_id == 'Q2350_2':
                    grouped_data['Manipulation'].append(likert_dict[value])
                if column_id == 'Q2350_3':
                    grouped_data['Integrate_Ads'].append(likert_dict[value])
                if column_id == 'Q2225':
                    grouped_data['Text_Benefits_Drawbacks'].append(value)
                if column_id == 'Q2351':
                    grouped_data['Detect_Ads'].append(value)
                if column_id == 'Q2197':
                    grouped_data['Highlight_Responses'].append(value)
            grouped_data['Positive_Impression'].append(sum(temp_grouped_data['Positive_Impression']) / len(temp_grouped_data['Positive_Impression']))
            grouped_data['Negative_Impression'].append(sum(temp_grouped_data['Negative_Impression']) / len(temp_grouped_data['Negative_Impression']))
            grouped_data['Helpfulness'].append(sum(temp_grouped_data['Helpfulness']) / len(temp_grouped_data['Helpfulness']))
            grouped_data['Credibility'].append(sum(temp_grouped_data['Credibility']) / len(temp_grouped_data['Credibility']))
            grouped_data['Friendliness'].append(sum(temp_grouped_data['Friendliness']) / len(temp_grouped_data['Friendliness']))
            grouped_data['Responsibility'].append(sum(temp_grouped_data['Responsibility']) / len(temp_grouped_data['Responsibility']))
    for key, value in grouped_data.items():
        print(key, len(value))
    return grouped_data


def parse_chathistory_data(file_path):
    chat_history = {}
    with open(file_path, mode='r', encoding='utf-8-sig') as file:
        data = json.load(file)
        model = None
        mode = None
        disclosure = None
        for key, entry in data.items():
            if key.strip().lower().startswith('er_'):
                model = 'gpt-4'
                mode = 'control'
                disclosure = 'none'
                overall_mode = 'gpt-4-control-none'
            elif key.strip().lower().startswith('fr_'):
                model = 'gpt-3.5'
                mode = 'control'
                disclosure = 'none'
                overall_mode = 'gpt-3.5-control-none'
            elif key.strip().lower().startswith('gr_'):
                model = 'gpt-4'
                mode = 'interest'
                disclosure = 'none'
                overall_mode = 'gpt-4-ads-none'
            elif key.strip().lower().startswith('hr_'):
                model = 'gpt-3.5'
                mode = 'interest'
                disclosure = 'none'
                overall_mode = 'gpt-3.5-ads-none'
            elif key.strip().lower().startswith('ir_'):
                model = 'gpt-4'
                mode = 'interest'
                disclosure = 'transparent'
                overall_mode = 'gpt-4-ads-disclosure'
            elif key.strip().lower().startswith('jr_'):
                model = 'gpt-3.5'
                mode = 'interest'
                disclosure = 'transparent'
                overall_mode = 'gpt-3.5-ads-disclosure'
            if model and mode and disclosure:
                chat_history[key] = {'model': model, 'mode': mode, 'disclosure': disclosure, 'overall_mode': overall_mode, 'conversation': []}
                for conv_key, conversation in entry['chat_history'].items():
                    for chat in conversation:
                        role = chat['role']
                        content = chat['content']
                        if role != 'system':
                            chat_history[key]['conversation'].append({'role': role, 'content': content})
    return chat_history


file_path = 'redis_data.json'
chat_history = parse_chathistory_data(file_path)

file_path = 'study_data.csv'
qualtrics_data = parse_qualtrics_data(file_path, chat_history)
df = pd.DataFrame(qualtrics_data)

print(df.columns)

def plot_likert_data(df, column_name, group_by_column, group_by_hue=None):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x=group_by_column, y=column_name, hue=group_by_hue)
    plt.title(f'Box plot of {column_name} grouped by {group_by_column}')
    plt.xlabel(group_by_column)
    plt.ylabel(column_name)
    plt.show()

def plot_qualitative_data(df, column_name, group_by_column, group_by_hue=None):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=group_by_column, hue=group_by_hue)
    plt.title(f'Count plot of {column_name} grouped by {group_by_column}')
    plt.xlabel(group_by_column)
    plt.ylabel(column_name)
    plt.show()

plot_likert_data(df, 'Positive_Impression', 'overall_mode')
plot_likert_data(df, 'Positive_Impression', 'mode', 'model')
plot_likert_data(df, 'Negative_Impression', 'overall_mode')

# present positive/negetive impression
# present credibility, friendliness, responsibility
# present helpfulness
# present relevance
# present convincingness


# group by overall mode
# group by mode and model
# group by mode and disclosure
# group by age
# group by familiarity / frequency
# group by education


# present links clicked
# present disclosures clicked

# clustering of chat history
# clustering of text responses
