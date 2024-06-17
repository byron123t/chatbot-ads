import os, json, random
from tqdm import tqdm
from src.Chatbot import OpenAIChatSession


args1 = {'mode': 'interest-based', 'ad_freq': 1.0, 'demo': None}
sample = random.sample(range(10397), 500)

ads_in_chat = 0
total_chats = 0
responses = []
total_samples = 0
for filenum in sample:
    if os.path.exists('/data/bjaytang/llm_evals/chatgpt_privacy/wildchat1m/{filenum}.json'.format(filenum=filenum)):
        with open('/data/bjaytang/llm_evals/chatgpt_privacy/wildchat1m/{filenum}.json'.format(filenum=filenum), 'r') as infile:
            data = json.load(infile)
            random.shuffle(data['rows'])
            for row in data['rows']:
                if total_samples > 1000:
                    break
                if 'row' in row and 'conversation' in row['row'] and not row['row']['conversation'][0]['redacted'] and row['row']['conversation'][0]['language'] == 'English':
                    for chat in row['row']['conversation']:
                        # print(chat.keys())
                        # print(chat['redacted'])
                        # print(chat['role'])
                        # print(chat['state'])
                        # print(chat['country'])
                        # print(chat['language'])
                        # print(chat['timestamp'])
                        # print(chat['turn_identifier'])
                        if chat['role'] == 'user':
                            content = chat['content']
                            # print(content)
                            oai_ads = OpenAIChatSession(mode=args1['mode'], ad_freq=args1['ad_freq'], demographics=args1['demo'])
                            try:
                                response_ads, product = oai_ads.run_chat(content)
                            except Exception as e:
                                continue
                            print(response_ads, product)
                            product_in_chat = False
                            if product and product['name']:
                                stripped_product = product['name'].lower().replace(' ', '').replace('-', '').replace('_', '').replace('.', '').replace(',', '').replace(':', '').replace(';', '').replace('\n', '').strip()
                                if stripped_product in response_ads.strip().lower():
                                    ads_in_chat += 1
                                    product_in_chat = True
                            total_chats += 1
                            total_samples += 1
                            print(ads_in_chat, total_chats)
                            responses.append({'file_id': filenum, 'timestamp': chat['timestamp'], 'country': chat['country'], 'turn_identifier': chat['turn_identifier'], 'prompt': content, 'product': product, 'response': response_ads, 'product_in_chat': product_in_chat})
                            with open('outputs/wildchat.json', 'w') as outfile:
                                json.dump(responses, outfile, indent=4)
                            break


# 433 1001
# 392 1001