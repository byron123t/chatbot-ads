import csv, json, os, re
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
from data import prompts


plt.rcParams.update({'font.size': 18})
likert_dict = {'Strongly Disagree': 1, 'Disagree': 2, 'Somewhat Disagree': 3, 'Neither Agree Nor Disagree': 4, 'Somewhat Agree': 5, 'Agree': 6, 'Strongly Agree': 7, 'Strongle Agree': 7, '1': 1, '2': 3, '3': 4, '4': 6, '5': 7}
invert_dict = {1: 7, 2: 6, 3: 5, 4: 4, 5: 3, 6: 2, 7: 1}


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
    orig_keylist = []
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
                        orig_keylist.append(row['KEY'])
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
    return grouped_data, keylist, cronbachs_data, orig_keylist


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


def chat_stats(chat_history, chat_metadata, keylist):
    stats = {
        'queries_per_variable': {},
        'length_of_queries_per_variable': {},
        'length_of_responses_per_variable': {},
        'length_of_queries_per_variable_tokens': {},
        'length_of_responses_per_variable_tokens': {},
        'link_clicks_per_variable': {},
        'disclosure_clicks': {},
        'product_distribution': {}
    }
    done = []
    product_list = []
    stats['product_distribution'] = {}

    for key, history in chat_history.items():
        if key.lower().strip() in keylist:
            overall_mode = history['overall_mode']
            if overall_mode not in done:
                done.append(overall_mode)
                stats['queries_per_variable'][overall_mode] = []
                stats['length_of_queries_per_variable'][overall_mode] = []
                stats['length_of_responses_per_variable'][overall_mode] = []
                stats['length_of_queries_per_variable_tokens'][overall_mode] = []
                stats['length_of_responses_per_variable_tokens'][overall_mode] = []
                stats['link_clicks_per_variable'][overall_mode] = 0
                stats['disclosure_clicks'][overall_mode] = 0
            
            query_count = 0
            total_query_length = 0
            total_response_length = 0

            for message in history['conversation']:
                role = message['role']
                content = message['content']
                content_length = len(content)
                # tokens = len(ENCODING.encode(str(content)))

                if role == 'user':
                    query_count += 1
                    total_query_length += content_length
                    stats['length_of_queries_per_variable'][overall_mode].append(content_length)
                    # stats['length_of_queries_per_variable_tokens'][overall_mode].append(tokens)
                
                elif role == 'assistant':
                    total_response_length += content_length
                    stats['length_of_responses_per_variable'][overall_mode].append(content_length)
                    # stats['length_of_responses_per_variable_tokens'][overall_mode].append(tokens)
            
            stats['queries_per_variable'][overall_mode].append(query_count)
            
            if key in chat_metadata:
                if 'link_clicks' in chat_metadata[key]:
                    for key1, value in chat_metadata[key]['link_clicks'].items():
                        stats['link_clicks_per_variable'][overall_mode] += value
                
                if overall_mode in ['DA4o', 'DA3.5']:
                    if 'disclosures' in chat_metadata[key]:
                        stats['disclosure_clicks'][overall_mode] += chat_metadata[key]['disclosures']
                
                if overall_mode in ['A4o', 'A3.5', 'DA4o', 'DA3.5']:
                    if 'products' in chat_metadata[key]:
                        for product in chat_metadata[key]['products']:
                            if product not in product_list:
                                product_list.append(product)
                    for product in product_list:
                        for message in history['conversation']:
                            if product.lower().strip() in message['content'].lower().strip():
                                if product not in stats['product_distribution']:
                                    stats['product_distribution'][product] = 1
                                stats['product_distribution'][product] += 1

    product_dict = {}
    for product, count in stats['product_distribution'].items():
        if product not in ['The Metropolitan Museum of Art']:
            if count > 9:
                product_dict[product] = count

    stats['average_length_of_queries_per_variable'] = {
        key: median(lengths) if lengths else 0
        for key, lengths in stats['length_of_queries_per_variable'].items()
    }
    
    stats['average_length_of_responses_per_variable'] = {
        key: median(lengths) if lengths else 0
        for key, lengths in stats['length_of_responses_per_variable'].items()
    }
    
    stats['average_length_of_responses_per_variable_tokens'] = {
        key: median(lengths) if lengths else 0
        for key, lengths in stats['length_of_responses_per_variable_tokens'].items()
    }
    
    stats['average_length_of_queries_per_variable_tokens'] = {
        key: median(lengths) if lengths else 0
        for key, lengths in stats['length_of_queries_per_variable_tokens'].items()
    }

    stats['average_queries_per_variable'] = {
        key: median(queries) if queries else 0
        for key, queries in stats['queries_per_variable'].items()
    }
    del stats['length_of_queries_per_variable']
    del stats['length_of_responses_per_variable']
    return stats


def export_profiles(chat_history, chat_metadata, keylist):
    profiles = {}
    for key, history in chat_history.items():
        if key.lower().strip() in keylist:
            overall_mode = history['overall_mode']
            if overall_mode not in profiles:
                profiles[overall_mode] = []
            if key in chat_metadata:
                if 'profiles' in chat_metadata[key]:
                    profiles[overall_mode].append({key: chat_metadata[key]['profiles']})
    with open('profiles.json', mode='w', encoding='utf-8') as file:
        json.dump(profiles, file, indent=2)


import json
from pathlib import Path
from typing import Mapping, Sequence, Any

import pandas as pd


def _serialise_cell(value: Any) -> Any:
    """
    Normalise any list / nested-list cells so they fit cleanly in a CSV.
    • Lists  → JSON strings (keeps commas & quotes intact)
    • Other  → returned unchanged
    """
    if isinstance(value, (list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return value


def export_survey_data(
    qualtrics_data: Mapping[str, Sequence[Any]],
    keylist: Sequence[str],
    out_path: str | Path = "survey_data.json",
) -> dict[str, dict]:
    n_rows = len(keylist)

    for col, col_vals in qualtrics_data.items():
        if len(col_vals) != n_rows:
            raise ValueError(
                f"Column length mismatch: '{col}' has {len(col_vals)} "
                f"entries but keylist has {n_rows}"
            )

    out: dict[str, dict] = {}
    for i, rid in enumerate(keylist):
        row_dict = {}
        for col, col_vals in qualtrics_data.items():
            row_dict[col] = col_vals[i]
        out[rid] = row_dict

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"✓ Survey export written to {out_path.resolve()}")
    return out


def wordbubble_responses(df):
    response_columns = [
        'TextRawBenefitsDrawbacks',
        'TextRawDetectAds',
        'TextRawProblematicResponses',
        'TextRawPersonality',
        'TextRawTrust',
        'TextRawInfluenceInception',
        'TextRawChangeMind',
        'TextRawProductsBrands',
        'TextRawSponsored'
    ]
    # for col in response_columns:
    #     print(' || '.join(df['overall_mode'].dropna().astype(str)))
    #     print(" || ".join(df[col].dropna().astype(str)))
    all_responses = " ".join(
        " ".join(df[col].dropna().astype(str)).replace('[\'', '').replace('\']', '') for col in response_columns if col in df.columns
    )
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords).generate(all_responses)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(os.path.join('plots/wordcloud_responses.pdf'))


def wordbubble_chats(chat_history):
    all_responses = ''
    for key, history in chat_history.items():
        for message in history['conversation']:
            role = message['role']
            content = message['content']
            if role == 'assistant':
                all_responses += content + ' '
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords).generate(all_responses)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(os.path.join('plots/wordcloud_assistant.pdf'))
    
    all_responses = ''
    for key, history in chat_history.items():
        for message in history['conversation']:
            role = message['role']
            content = message['content']
            if role == 'user':
                all_responses += content + ' '
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords).generate(all_responses)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(os.path.join('plots/wordcloud_user.pdf'))


def analyze_user_chats(chat_history, chat_metadata):
    keywords = [" ads ", " ad ", " sponsor ", " product ", " brand ", " ad.", " ad.", "ad?", " ad," " ads.", " ads.", " ads,", " ads?", " ad!", " ads!" " sponsor ", " sponsor.", "sponsor ", " product ", " product.", "product ", " brand ", " brand.", "brand ", " brand!", " brand?", "brand," " product?", " product!", "product," " sponsor?", "sponsor!", " sponsored ", "sponsored.", "sponsored,", "sponsored?", "sponsored!", "sponsorship", "sponsorship!", "sponsorship,", "sponsorship.", "sponsorship?", "advertising", "advertisement"]
    user_chat_analysis = {'overall_mode': [], 'assistant_product_mentions': [], 'user_product_mentions': [], 'keyword_mentions': []}

    for key, chat_data in chat_history.items():
        conversation = chat_data['conversation']
        
        if chat_data['overall_mode'] in ['DA4o', 'DA3.5', 'A4o', 'A3.5']:
            user_chat_analysis['overall_mode'].append(chat_data['overall_mode'])
            
            product_mention_count_user = 0
            product_mention_count_assistant = 0
            keyword_mentions_user = 0
            for message in conversation:
                role = message['role']
                content = message['content'].lower()
                if role == 'assistant':
                    if any(product.lower() in content for product in chat_metadata[key].get('products', [])):
                        product_mention_count_assistant += 1
                elif role == 'user':
                    if any(product.lower() in content for product in chat_metadata[key].get('products', [])):
                        print(content)
                        product_mention_count_user += 1
                    elif any(keyword in content for keyword in keywords):
                        print(content)
                        keyword_mentions_user += 1

            user_chat_analysis['assistant_product_mentions'].append(product_mention_count_assistant)
            user_chat_analysis['user_product_mentions'].append(product_mention_count_user)
            user_chat_analysis['keyword_mentions'].append(keyword_mentions_user)
    return user_chat_analysis


def process_user_data(survey_data, chat_history, chat_metadata):
    pass


def process_chat_data(survey_data, chat_history, chat_metadata):
    pass


def generate_theme_matrix(df, response_columns):
    # Initialize a Counter to count co-occurrences
    theme_counter = Counter()
    all_themes = set()

    # Loop through each response in each column to extract themes
    for col in response_columns:
        if col in df.columns:
            for response in df[col].dropna():
                themes = llm_tags(question=col, response=response)
                themes_set = set(themes)
                all_themes.update(themes_set)
                for theme in themes_set:
                    for co_theme in themes_set:
                        if theme != co_theme:
                            theme_counter[(theme, co_theme)] += 1

    # Create a sorted list of unique themes
    all_themes = sorted(all_themes)
    
    # Initialize a co-occurrence matrix
    theme_matrix = np.zeros((len(all_themes), len(all_themes)))

    # Fill the co-occurrence matrix
    for i, theme1 in enumerate(all_themes):
        for j, theme2 in enumerate(all_themes):
            if i != j:
                theme_matrix[i, j] = theme_counter[(theme1, theme2)]

    return theme_matrix, all_themes


def heatmap(df):
    # Define columns containing free responses
    response_columns = [
        'TextRawBenefitsDrawbacks',
        'TextRawDetectAds',
        'TextRawProblematicResponses',
        'TextRawPersonality',
        'TextRawTrust',
        'TextRawInfluenceInception',
        'TextRawChangeMind',
        'TextRawProductsBrands',
        'TextRawSponsored'
    ]

    # Generate the co-occurrence matrix and list of themes
    theme_matrix, themes = generate_theme_matrix(df, response_columns)
    
    # Create a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(theme_matrix, annot=True, xticklabels=themes, yticklabels=themes, cmap='Blues')
    plt.title('Co-occurrence of Themes in Free Responses')
    plt.show()


def classify_responses(df, response_columns):
    classification_results = {}

    for col in response_columns:
        if col in df.columns:
            responses = df[col].dropna().tolist()
            classifications = llm_yesnomaybe(question=col, responses=responses)
            print(classifications)
            classification_results[col] = classifications
    
    return classification_results


def convert_to_ordinal_scale(classification_results):
    ordinal_scale = {'yes': 2, 'neutral': 1, 'no': 0}
    ordinal_data = {}

    for question, responses in classification_results.items():
        ordinal_data[question] = [ordinal_scale.get(response, None) for response in responses]

    return ordinal_data


def plot_ordinal_data(ordinal_data):
    for question, ordinals in ordinal_data.items():
        counts = Counter(ordinals)
        categories = ['No', 'Neutral', 'Yes']
        values = [counts[0], counts[1], counts[2]]

        plt.figure(figsize=(6, 4))
        plt.bar(categories, values, color=['red', 'orange', 'green'])
        plt.title(f"Response Distribution for {question}")
        plt.xlabel("Response")
        plt.ylabel("Count")
        plt.show()


def sentiments(df):
    response_columns = [
        'TextRawBenefitsDrawbacks',
        'TextRawDetectAds',
        'TextRawProblematicResponses',
        'TextRawPersonality',
        'TextRawTrust',
        'TextRawInfluenceInception',
        'TextRawChangeMind',
        'TextRawProductsBrands',
        'TextRawSponsored'
    ]

    classification_results = classify_responses(df, response_columns)

    ordinal_data = convert_to_ordinal_scale(classification_results)

    plot_ordinal_data(ordinal_data)


def yesnomaybe(df):
    pass


def t_test():
    pass


def anova():
    pass



file_path = 'demographics.csv'
participants = parse_prolific_demographics(file_path)

file_path = 'redis_data.json'
chat_history, chat_metadata = parse_chathistory_data(file_path)

file_path = 'study_data.csv'
qualtrics_data, keylist, cronbachs_data, orig_keylist = parse_qualtrics_data(file_path, chat_history, participants)
df = pd.DataFrame(qualtrics_data)

print(keylist)

print(chat_stats(chat_history, chat_metadata, keylist))

print(analyze_user_chats(chat_history, chat_metadata))

print(df.columns)

print(df)
print(chat_metadata)

export_survey_data(qualtrics_data, orig_keylist)
export_profiles(chat_history, chat_metadata, keylist)
wordbubble_responses(df)
wordbubble_chats(chat_history)

# Need to load qualitative data here


# USER
# Ad receptiveness
# Demographic profile
# Generated demographic profile
# Generated interest profile
# Engagement level (have different levels)
# Writing style
# Composite quantitative sentiment with chatbot
# Composite qualitative sentiments with chatbot

# CHATS
# Ad receptiveness
# Response receptiveness
# Query sentiment
# Query length
# Conversation rounds count
# Conversation rounds length
# Ad link clicks

# low, medium, high

# dataset format {'user': {'key': {'ad_receptiveness': 0, 'demographic_profile': {}, 'generated_profile': {}, 'engagement_level': 0, 'writing_style': {}, 'sentiment': 0, 'qualitative_sentiments': {}}}, 'chat': {'key': {'receptiveness': 0, 'ad_presence': T/F, 'sentiment': 0, 'query_length': 0, 'conversation_rounds_count': 0, 'conversation_rounds_length': 0, 'ad_link_clicks': 0}}}
score_map = {'low': {'': 0}, 'medium': {'': 0}, 'high': {'': 0}}
prompt_map = {'low': '', 'medium': '', 'high': ''}


user_data = process_user_data(df, chat_history, chat_metadata)

chat_data = process_chat_data(df, chat_history, chat_metadata)



response_columns = [
    'TextRawBenefitsDrawbacks',
    'TextRawDetectAds',
    'TextRawProblematicResponses',
    'TextRawPersonality',
    'TextRawTrust',
    'TextRawInfluenceInception',
    'TextRawChangeMind',
    'TextRawProductsBrands',
    'TextRawSponsored'
]

# sentiments(df)
# yesnomaybe(df)

# heatmap(df)
# sentiments(df)


# group by overall mode
# group by mode and model
# group by mode and disclosure
# group by age
# group by familiarity / frequency
# group by education


# clustering of chat history
# clustering of text responses
