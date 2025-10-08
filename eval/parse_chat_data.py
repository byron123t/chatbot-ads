import csv, json, os, re, tiktoken
from collections import defaultdict, OrderedDict
from statistics import mean, median
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
from scipy.stats import shapiro, levene, f_oneway
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pingouin as pg
from src.API import OpenAIAPI
from data import prompts


plt.rcParams.update({'font.size': 18})
likert_dict = {'Strongly Disagree': 1, 'Disagree': 2, 'Somewhat Disagree': 3, 'Neither Agree Nor Disagree': 4, 'Somewhat Agree': 5, 'Agree': 6, 'Strongly Agree': 7, 'Strongle Agree': 7, '1': 1, '2': 3, '3': 4, '4': 6, '5': 7}
invert_dict = {1: 7, 2: 6, 3: 5, 4: 4, 5: 3, 6: 2, 7: 1}
oai = OpenAIAPI(model='gpt-4o-mini')
ENCODING = tiktoken.encoding_for_model('gpt-3.5-turbo')


def parse_prolific_demographics(file_path):
    participant_ids = []
    with open(file_path, mode='r', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['Status'].strip() == 'AWAITING REVIEW':
                time_taken = float(row['Time taken'])
                if time_taken > 900:
                    participant_ids.append({'id': row['Participant id'], 'age': row['Age'], 'sex': row['Sex'], 'race': row['Ethnicity simplified']})
    return participant_ids


def parse_qualtrics_data(file_path, chat_history, participants):
    grouped_data = defaultdict(list)
    failed_prolifics = []
    keylist = []
    cronbachs_data = {'Credibility': {'q2': [], 'q3': []}, 'Helpfulness': {'q1': [], 'q2': [], 'q3': []}, 'Convincingness': {'q2': [], 'q3': []}, 'Relevance': {'q1': [], 'q2': [], 'q3': []}, 'Neutrality': {'q1': [], 'q2': []}, 'Godspeed': {'q1': [], 'q2': [], 'q3': [], 'q4': [], 'q5': [], 'q6': [], 'q7': [], 'q8': [], 'q9': [], 'q10': [], 'q11': [], 'q12': []}, 'Sentiment': {'Q2348_1': [], 'Q2348_2': [], 'Q2348_8': [], 'Q2348_4': [], 'Q2348_15': [], 'Q2348_14': [], 'Q2348_10': [], 'Q2348_6': [], 'Q2348_9': [], 'Q2348_3': [], 'Q2348_7': [], 'Q2348_12': [], 'Q2348_13': [], 'Q2348_5': [], 'Q2348_11': []}}
    with open(file_path, mode='r', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        for row in reader:
            key = row['KEY'].strip().lower()
            if row['KEY'] not in chat_history:
                failed_prolifics.append(row['Q2283'])
                print('Failed Prolific ID (Did not use chatbot): ', row['Q2283'])
                continue
            else:
                for p in participants:
                    if row['Q2283'] == p['id']:
                        keylist.append(key)
                        age = int(p['age'])
                        print(age)
                        if age > 0 and age < 25:
                            grouped_data['age'].append('18-24')
                        elif age >= 25 and age < 35:
                            grouped_data['age'].append('25-34')
                        elif age >= 35 and age < 45:
                            grouped_data['age'].append('35-44')
                        elif age >= 45 and age < 55:
                            grouped_data['age'].append('45-54')
                        elif age >= 55 and age < 65:
                            grouped_data['age'].append('54-64')
                        elif age >= 65:
                            grouped_data['age'].append('65+')
                        grouped_data['sex'].append(p['sex'])
                        grouped_data['race'].append(p['race'])
                        break
                if key not in keylist:
                    failed_prolifics.append(row['Q2283'])
                    print('Failed Prolific ID (Rushed through study): ', row['Q2283'])
                    continue
            if key.startswith('er_'):
                grouped_data['model'].append('gpt-4')
                grouped_data['mode'].append('control')
                grouped_data['disclosure'].append('none')
                grouped_data['overall_mode'].append('C4o')
            elif key.startswith('fr_'):
                grouped_data['model'].append('gpt-3.5')
                grouped_data['mode'].append('control')
                grouped_data['disclosure'].append('none')
                grouped_data['overall_mode'].append('C3.5')
            elif key.startswith('gr_'):
                grouped_data['model'].append('gpt-4')
                grouped_data['mode'].append('interest')
                grouped_data['disclosure'].append('none')
                grouped_data['overall_mode'].append('A4o')
            elif key.startswith('hr_'):
                grouped_data['model'].append('gpt-3.5')
                grouped_data['mode'].append('interest')
                grouped_data['disclosure'].append('none')
                grouped_data['overall_mode'].append('A3.5')
            elif key.startswith('ir_'):
                grouped_data['model'].append('gpt-4')
                grouped_data['mode'].append('interest')
                grouped_data['disclosure'].append('transparent')
                grouped_data['overall_mode'].append('DA4o')
            elif key.startswith('jr_'):
                grouped_data['model'].append('gpt-3.5')
                grouped_data['mode'].append('interest')
                grouped_data['disclosure'].append('transparent')
                grouped_data['overall_mode'].append('DA3.5')
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
                elif column_id == 'Q2345':
                    grouped_data['Education'].append(value)
                
                # Credibility
                if column_id == 'Q2348_1' or column_id == 'Q2348_3' or column_id == 'Q2348_7':
                    if column_id == 'Q2348_3' or column_id == 'Q2348_7':
                        temp_grouped_data['Credibility'].append(invert_dict[likert_dict[value]])
                    # else:
                    #     temp_grouped_data['Credibility'].append(likert_dict[value])
                    # if column_id == 'Q2348_1':
                    #     cronbachs_data['Credibility']['q1'].append(likert_dict[value])
                    if column_id == 'Q2348_3':
                        cronbachs_data['Credibility']['q2'].append(invert_dict[likert_dict[value]])
                    if column_id == 'Q2348_7':
                        cronbachs_data['Credibility']['q3'].append(invert_dict[likert_dict[value]])
                    
                
                # Helpfulness
                if column_id == 'Q2348_2' or column_id == 'Q2348_8' or column_id == 'Q2348_12':
                    if column_id == 'Q2348_12':
                        temp_grouped_data['Helpfulness'].append(invert_dict[likert_dict[value]])
                    else:
                        temp_grouped_data['Helpfulness'].append(likert_dict[value])
                    if column_id == 'Q2348_2':
                        cronbachs_data['Helpfulness']['q1'].append(likert_dict[value])
                    if column_id == 'Q2348_8':
                        cronbachs_data['Helpfulness']['q2'].append(likert_dict[value])
                    if column_id == 'Q2348_12':
                        cronbachs_data['Helpfulness']['q3'].append(invert_dict[likert_dict[value]])
                    
                # Convincingness
                if column_id == 'Q2348_4' or column_id == 'Q2348_13' or column_id == 'Q2348_15':
                    if column_id == 'Q2348_13':
                        temp_grouped_data['Convincingness'].append(invert_dict[likert_dict[value]])
                    elif column_id == 'Q2348_4':
                        pass
                    else:
                        temp_grouped_data['Convincingness'].append(likert_dict[value])
                    # if column_id == 'Q2348_4':
                    #     cronbachs_data['Convincingness']['q1'].append(likert_dict[value])
                    if column_id == 'Q2348_13':
                        cronbachs_data['Convincingness']['q2'].append(invert_dict[likert_dict[value]])
                    if column_id == 'Q2348_15':
                        cronbachs_data['Convincingness']['q3'].append(likert_dict[value])
                
                # Relevance
                if column_id == 'Q2348_5' or column_id == 'Q2348_14' or column_id == 'Q2348_10':
                    if column_id == 'Q2348_5':
                        temp_grouped_data['Relevance'].append(invert_dict[likert_dict[value]])
                    else:
                        temp_grouped_data['Relevance'].append(likert_dict[value])
                    if column_id == 'Q2348_5':
                        cronbachs_data['Relevance']['q1'].append(invert_dict[likert_dict[value]])
                    if column_id == 'Q2348_14':
                        cronbachs_data['Relevance']['q2'].append(likert_dict[value])
                    if column_id == 'Q2348_10':
                        cronbachs_data['Relevance']['q3'].append(likert_dict[value])
                
                # Neutrality
                if column_id == 'Q2348_6' or column_id == 'Q2348_9' or column_id == 'Q2348_11':
                    if column_id != 'Q2348_11':
                    #     temp_grouped_data['Neutrality'].append(invert_dict[likert_dict[value]])
                    # else:
                        temp_grouped_data['Neutrality'].append(likert_dict[value])
                    if column_id == 'Q2348_6':
                        cronbachs_data['Neutrality']['q1'].append(likert_dict[value])
                    if column_id == 'Q2348_9':
                        cronbachs_data['Neutrality']['q2'].append(likert_dict[value])
                    # if column_id == 'Q2348_11':
                    #     cronbachs_data['Neutrality']['q3'].append(invert_dict[likert_dict[value]])
                        
                # Godspeed
                if column_id in ['Q2349_1', 'Q2349_2', 'Q2349_3', 'Q2349_4', 'Q2349_5', 'Q2349_6', 'Q2349_7', 'Q2349_8', 'Q2349_9', 'Q2349_10', 'Q2349_11', 'Q2349_12']:
                    if column_id in ['Q2349_1', 'Q2349_2', 'Q2349_8', 'Q2349_9', 'Q2349_10', 'Q2349_11']:
                        temp_grouped_data['Godspeed'].append(likert_dict[value])
                        if column_id == 'Q2349_1':
                            cronbachs_data['Godspeed']['q1'].append(likert_dict[value])
                        if column_id == 'Q2349_2':
                            cronbachs_data['Godspeed']['q2'].append(likert_dict[value])
                        if column_id == 'Q2349_8':
                            cronbachs_data['Godspeed']['q3'].append(likert_dict[value])
                        if column_id == 'Q2349_9':
                            cronbachs_data['Godspeed']['q4'].append(likert_dict[value])
                        if column_id == 'Q2349_10':
                            cronbachs_data['Godspeed']['q5'].append(likert_dict[value])
                        if column_id == 'Q2349_11':
                            cronbachs_data['Godspeed']['q6'].append(likert_dict[value])
                    else:
                        temp_grouped_data['Godspeed'].append(invert_dict[likert_dict[value]])
                        if column_id == 'Q2349_3':
                            cronbachs_data['Godspeed']['q7'].append(invert_dict[likert_dict[value]])
                        if column_id == 'Q2349_4':
                            cronbachs_data['Godspeed']['q8'].append(invert_dict[likert_dict[value]])
                        if column_id == 'Q2349_5':
                            cronbachs_data['Godspeed']['q9'].append(invert_dict[likert_dict[value]])
                        if column_id == 'Q2349_6':
                            cronbachs_data['Godspeed']['q10'].append(invert_dict[likert_dict[value]])
                        if column_id == 'Q2349_7':
                            cronbachs_data['Godspeed']['q11'].append(invert_dict[likert_dict[value]])
                        if column_id == 'Q2349_12':
                            cronbachs_data['Godspeed']['q12'].append(invert_dict[likert_dict[value]])
                    
                # Ads
                # Felt Advertising
                if column_id == 'Q2350_1':
                    temp_grouped_data['FeltAdvertising'].append(likert_dict[value])

                # Felt Manipulated
                if column_id == 'Q2350_2':
                    temp_grouped_data['FeltManipulated'].append(likert_dict[value])
                
                # Believe Tech Companies Will Integrate
                if column_id == 'Q2350_3':
                    temp_grouped_data['TechIntegrate'].append(likert_dict[value])
                
                # Benefits or Drawbacks With Ads
                if column_id == 'Q2225':
                    temp_grouped_data['TextRawBenefitsDrawbacks'].append(value)
                    # temp_grouped_data['TextSentimentBenefitsDrawbacks'].append(llm_sentiment('Benefits or Drawbacks With Ads', value))
                    # temp_grouped_data['TextTagsBenefitsDrawbacks'].append(llm_tags('Benefits or Drawbacks With Ads', value))
                
                # Detect Ads
                if column_id == 'Q2351':
                    temp_grouped_data['TextRawDetectAds'].append(value)
                    # temp_grouped_data['TextSentimentDetectAds'].append(llm_sentiment('Detect Ads', value))
                    # temp_grouped_data['TextTagsDetectAds'].append(llm_tags('Detect Ads', value))
                
                # Problematic Responses
                if column_id == 'Q2197':
                    temp_grouped_data['TextRawProblematicResponses'].append(value)
                    # temp_grouped_data['TextSentimentProblematicResponses'].append(llm_sentiment('Problematic Responses', value))
                    # temp_grouped_data['TextTagsProblematicResponses'].append(llm_tags('Problematic Responses', value))
                
                # Free Response
                # Personality
                if column_id == 'Q2340':
                    temp_grouped_data['TextRawPersonality'].append(value)
                    # temp_grouped_data['TextSentimentPersonality'].append(llm_sentiment('Personality', value))
                    # temp_grouped_data['TextTagsPersonality'].append(llm_tags('Personality', value))
                    
                # Trust
                if column_id == 'Q2343':
                    temp_grouped_data['TextRawTrust'].append(value)
                    # temp_grouped_data['TextSentimentTrust'].append(llm_sentiment('Trust', value))
                    # temp_grouped_data['TextTagsTrust'].append(llm_tags('Trust', value))
                    
                # Influence/Inception
                if column_id == 'Q2376':
                    temp_grouped_data['TextRawInfluenceInception'].append(value)
                    # temp_grouped_data['TextSentimentInfluenceInception'].append(llm_sentiment('Influence/Inception', value))
                    # temp_grouped_data['TextTagsInfluenceInception'].append(llm_tags('Influence/Inception', value))
                
                # Change mind
                if column_id == 'Q2377':
                    temp_grouped_data['TextRawChangeMind'].append(value)
                    # temp_grouped_data['TextSentimentChangeMind'].append(llm_sentiment('Change Mind', value))
                    # temp_grouped_data['TextTagsChangeMind'].append(llm_tags('Change Mind', value))
                
                # Products/Brands
                if column_id == 'Q2375':
                    temp_grouped_data['TextRawProductsBrands'].append(value)
                    # temp_grouped_data['TextSentimentProductsBrands'].append(llm_sentiment('Products/Brands', value))
                    # temp_grouped_data['TextTagsProductsBrands'].append(llm_tags('Products/Brands', value))
                
                # Sponsored
                if column_id == 'Q2374':
                    temp_grouped_data['TextRawSponsored'].append(value)
                    # temp_grouped_data['TextSentimentSponsored'].append(llm_sentiment('Sponsored', value))
                    # temp_grouped_data['TextTagsSponsored'].append(llm_tags('Sponsored', value))
                
                if column_id in ['Q2348_1', 'Q2348_2', 'Q2348_8', 'Q2348_4', 'Q2348_15', 'Q2348_14', 'Q2348_10', 'Q2348_6', 'Q2348_9']:
                    temp_grouped_data['Sentiment'].append(likert_dict[value])
                    cronbachs_data['Sentiment'][column_id].append(likert_dict[value])
                if column_id in ['Q2348_3', 'Q2348_7', 'Q2348_12', 'Q2348_13', 'Q2348_5', 'Q2348_11']:
                    temp_grouped_data['Sentiment'].append(invert_dict[likert_dict[value]])
                    cronbachs_data['Sentiment'][column_id].append(invert_dict[likert_dict[value]])

            grouped_data['Sentiment'].append(sum(temp_grouped_data['Sentiment']) / len(temp_grouped_data['Sentiment']))
            grouped_data['Credibility'].append(sum(temp_grouped_data['Credibility']) / len(temp_grouped_data['Credibility']))
            grouped_data['Helpfulness'].append(sum(temp_grouped_data['Helpfulness']) / len(temp_grouped_data['Helpfulness']))
            grouped_data['Convincingness'].append(sum(temp_grouped_data['Convincingness']) / len(temp_grouped_data['Convincingness']))
            grouped_data['Relevance'].append(sum(temp_grouped_data['Relevance']) / len(temp_grouped_data['Relevance']))
            grouped_data['Neutrality'].append(sum(temp_grouped_data['Neutrality']) / len(temp_grouped_data['Neutrality']))
            grouped_data['Godspeed'].append(sum(temp_grouped_data['Godspeed']) / len(temp_grouped_data['Godspeed']))
            grouped_data['FeltAdvertising'].append(sum(temp_grouped_data['FeltAdvertising']) / len(temp_grouped_data['FeltAdvertising']))
            grouped_data['FeltManipulated'].append(sum(temp_grouped_data['FeltManipulated']) / len(temp_grouped_data['FeltManipulated']))
            grouped_data['TechIntegrate'].append(sum(temp_grouped_data['TechIntegrate']) / len(temp_grouped_data['TechIntegrate']))
            grouped_data['TextRawBenefitsDrawbacks'].append(temp_grouped_data['TextRawBenefitsDrawbacks'])
            grouped_data['TextRawDetectAds'].append(temp_grouped_data['TextRawDetectAds'])
            grouped_data['TextRawProblematicResponses'].append(temp_grouped_data['TextRawProblematicResponses'])
            grouped_data['TextRawPersonality'].append(temp_grouped_data['TextRawPersonality'])
            grouped_data['TextRawTrust'].append(temp_grouped_data['TextRawTrust'])
            grouped_data['TextRawInfluenceInception'].append(temp_grouped_data['TextRawInfluenceInception'])
            grouped_data['TextRawChangeMind'].append(temp_grouped_data['TextRawChangeMind'])
            grouped_data['TextRawProductsBrands'].append(temp_grouped_data['TextRawProductsBrands'])
            grouped_data['TextRawSponsored'].append(temp_grouped_data['TextRawSponsored'])
        print(grouped_data['UsedChatbots'])
            
    for key, value in grouped_data.items():
        print(key, len(value))

    print(grouped_data['sex'])
    return grouped_data, keylist, cronbachs_data


def parse_chathistory_data(file_path):
    chat_history = {}
    chat_metadata = {}
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
                overall_mode = 'C4o'
            elif key.strip().lower().startswith('fr_'):
                model = 'gpt-3.5'
                mode = 'control'
                disclosure = 'none'
                overall_mode = 'C3.5'
            elif key.strip().lower().startswith('gr_'):
                model = 'gpt-4'
                mode = 'interest'
                disclosure = 'none'
                overall_mode = 'A4o'
            elif key.strip().lower().startswith('hr_'):
                model = 'gpt-3.5'
                mode = 'interest'
                disclosure = 'none'
                overall_mode = 'A3.5'
            elif key.strip().lower().startswith('ir_'):
                model = 'gpt-4'
                mode = 'interest'
                disclosure = 'transparent'
                overall_mode = 'DA4o'
            elif key.strip().lower().startswith('jr_'):
                model = 'gpt-3.5'
                mode = 'interest'
                disclosure = 'transparent'
                overall_mode = 'DA3.5'
            if model and mode and disclosure:
                chat_history[key] = {'model': model, 'mode': mode, 'disclosure': disclosure, 'overall_mode': overall_mode, 'conversation': []}
                if 'chat_history' in entry:
                    for conv_key, conversation in entry['chat_history'].items():
                        for chat in conversation:
                            role = chat['role']
                            content = chat['content']
                            if role != 'system':
                                chat_history[key]['conversation'].append({'role': role, 'content': content})
                chat_metadata[key] = {}
                if 'linkclicks' in entry:
                    chat_metadata[key]['link_clicks'] = entry['linkclicks']
                if 'disclosures' in entry:
                    chat_metadata[key]['disclosures'] = entry['disclosures']
                if 'profiles' in entry:
                    chat_metadata[key]['profiles'] = entry['profiles']
                if 'products' in entry:
                    chat_metadata[key]['products'] = entry['products']
    return chat_history, chat_metadata



