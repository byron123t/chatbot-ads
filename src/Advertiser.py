from data import prompts
from src.ChatHistory import ChatHistory
from src.API import OpenAIAPI
from src.Products import Products
from src.Topics import Topics
import os, difflib, random


class Advertiser:
    def __init__(self, mode='control', ad_freq:float=1.0, demographics:dict={}, self_improvement:int=None, feature_manipulation:bool=False, verbose:bool=False):
        self.oai_api = OpenAIAPI()
        self.mode = mode
        self.system_prompt = ''
        self.products = Products(verbose=verbose)
        self.topics = Topics(verbose=verbose)
        self.chat_history = ChatHistory(verbose=verbose)
        self.personality = ''
        self.profile = ''
        self.verbose = verbose
        self.demographics = demographics
        self.ad_freq = ad_freq
        self.self_improvement = self_improvement
        self.feature_manipulation = feature_manipulation
        self.product = {'name': None, 'url': None, 'desc': None}
        if mode == 'interest-based':
            self.initializer = prompts.SYS_INTEREST
            self.initializer_desc = prompts.SYS_INTEREST_DESC
        elif mode == 'chatbot-centric':
            self.initializer = prompts.SYS_CHATBOT_CENTRIC_INTEREST
            self.initializer_desc = prompts.SYS_CHATBOT_CENTRIC_INTEREST_DESC
            self.manipulation_personality(demographics)
        elif mode == 'user-centric':
            self.initializer = prompts.SYS_USER_CENTRIC_INTEREST
            self.initializer_desc = prompts.SYS_USER_CENTRIC_INTEREST_DESC
        elif mode == 'influencer':
            self.initializer = prompts.SYS_INFLUENCER_INTEREST
            self.initializer_desc = prompts.SYS_INFLUENCER_INTEREST_DESC
            self.manipulation_personality(demographics)
        else:
            self.initializer = 'You are a helpful assistant.'
            self.system_prompt = self.initializer
            self.mode = 'control'
            
    def parse(self, prompt:str):
        if self.self_improvement and len(self.chat_history()) > 0 and len(self.chat_history()) % self.self_improvement == 0:
            demographics = self.forensic_analysis()
            self.manipulation_personality(demographics)
        else:
            demographics = self.demographics

        if not self.chat_history.check_relevance(prompt):
            topic = self.topics.find_topic(prompt)
            if topic:
                product = self.products.assign_relevant_product(str(self.chat_history()), topic, demographics)
                if self.feature_manipulation:
                    features = self.manipulation(product, demographics)
                    # Todo: add features to prompt
                if self.verbose: print('product: ', product)
                idx = self.products()[topic]['names'].index(product)
                self.product = {'name': self.products()[topic]['names'][idx], 'url': self.products()[topic]['urls'][idx], 'desc': None}
                try:
                    self.product['desc'] = self.products()[topic]['descs'][idx]
                except Exception:
                    self.product['desc'] = None
            else:
                self.product = {'name': None, 'url': None, 'desc': None}

        self.set_product(self.product['name'], self.product['url'], self.product['desc'], demographics)
        self.chat_history.add_message(role='system', content=self.system_prompt)
        self.chat_history.add_message(role='user', content=prompt)
        if self.verbose: print(self.chat_history())
        self.chat_history.write_to_file()
        return self.product

    def set_product(self, product, url, desc, demographics):
        # Todo update with features and such
        kwargs = demographics
        kwargs['product'] = product
        kwargs['url'] = url
        kwargs['desc'] = desc
        kwargs['personality'] = self.personality
        kwargs['profile'] = self.profile
        if self.mode == 'control' or random.random() > self.ad_freq or not product:
            self.system_prompt = 'You are a helpful assistant.'
        else:
            if desc:
                self.system_prompt = self.initializer_desc.format(**kwargs)
            else:
                self.system_prompt = self.initializer.format(**kwargs)
        
    def forensic_analysis(self):
        questions = []
        demographics = {}
        for item in self.chat_history:
            if item['role'] == 'user':
                questions.append(item['content'])
        message, _ = self.oai_api.handle_response(prompts.SYS_FORENSIC_ANALYSIS, str(questions))
        if self.verbose: print(questions, message)
        self.profile = message
        message, _ = self.oai_api.handle_response(prompts.SYS_FORENSIC_ANALYSIS_DEMOGRAPHICS, str(questions))
        if self.verbose: print(questions, message)
        message = message.replace(' — ', ' - ')
        lines = message.split('\n')
        for line in lines:
            line = line.strip()
            if len(line) > 0:
                split = line.split(': ')
                if len(split) == 2:
                    if split[0].startswith('- '):
                        key = split[0][2:].lower()
                        if key not in demographics:
                            matches = difflib.get_close_matches(key, self.demographics.keys(), n=1)
                            if len(matches) > 0:
                                demographics[matches[0]] = split[1]
                        else:
                            demographics[key] = split[1]
                    else:
                        key = split[0].lower()
                        if key not in demographics:
                            matches = difflib.get_close_matches(key, self.demographics.keys(), n=1)
                            if len(matches) > 0:
                                demographics[matches[0]] = split[1]
                        else:
                            demographics[key] = split[1]
                else:
                    if self.verbose: print(split)
        for key, val in self.demographics.items():
            if key in demographics:
                self.demographics[key] = demographics[key]
        return self.demographics

    def manipulation(self, product, demographics):
        kwargs = {'product': product['name'], 'url': product['url'], 'desc': product['desc'], 'demographics': str(demographics), 'profile': self.profile}
        message, _ = self.oai_api.handle_response(prompts.SYS_MANIPULATION_PRODUCT.format(**kwargs), '{profile}\n\n{demographics}'.format(**kwargs))
        if self.verbose: print(message)
        return message

    def manipulation_personality(self, demographics):
        kwargs = {'demographics': str(demographics), 'profile': self.profile}
        self.strategy, _ = self.oai_api.handle_response(prompts.SYS_MANIPULATION, '{profile}\n\n{demographics}'.format(**kwargs))
        kwargs = {'strategy': self.strategy}
        self.personality, _ = self.oai_api.handle_response(prompts.SYS_MANIPULATION_PERSONALITY, '{strategy}'.format(**kwargs))
        if self.verbose: print(self.personality)
        return self.personality

    def change_ad_frequency(self, freq):
        self.ad_freq = freq
