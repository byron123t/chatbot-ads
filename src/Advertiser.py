from data import prompts
from src.ChatHistory import ChatHistory
from src.API import OpenAIAPI
from src.Products import Products
from src.Topics import Topics
import difflib, random, json, copy
from redis import Redis


r = Redis(host='localhost', port=6379, password='', decode_responses=True)


class Advertiser:
    def __init__(self, mode:str='control', session:str='', ad_freq:float=1.0, demographics:dict={}, conversation_id:str='', self_improvement:int=None, feature_manipulation:bool=False, verbose:bool=False):
        self.oai_api = OpenAIAPI()
        self.mode = mode
        self.system_prompt = ''
        self.products = Products(verbose=verbose)
        self.topics = Topics(verbose=verbose)
        self.chat_history = ChatHistory(verbose=verbose, session=session, conversation_id=conversation_id)
        self.personality = ''
        self.profile = ''
        self.session = session
        self.conversation_id = conversation_id
        self.verbose = verbose
        self.demographics = demographics
        self.ad_freq = ad_freq
        self.self_improvement = self_improvement
        self.feature_manipulation = feature_manipulation
        self.product = {self.conversation_id: {'name': None, 'url': None, 'desc': None}}
        self.topic = {self.conversation_id: None}
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
        if r.exists(self.session):
            if r.hexists(self.session, 'mode'):
                self.mode = json.loads(r.hget(self.session, 'mode'))
                self.demographics = json.loads(r.hget(self.session, 'demographics'))
            else:
                r.hset(self.session, 'mode', json.dumps(self.mode))
                r.hset(self.session, 'demographics', json.dumps(self.demographics))
            if r.hexists(self.session, 'product'):
                self.product = json.loads(r.hget(self.session, 'product'))
                self.topic = json.loads(r.hget(self.session, 'topic'))
                if self.conversation_id not in self.product:
                    self.product[conversation_id] = {'name': None, 'url': None, 'desc': None}
                    self.topic[conversation_id] = None
            else:
                r.hset(self.session, 'product', json.dumps(self.product))
                r.hset(self.session, 'topic', json.dumps(self.topic))
        else:
            r.hset(self.session, 'mode', json.dumps(self.mode))
            r.hset(self.session, 'demographics', json.dumps(self.demographics))
            r.hset(self.session, 'product', json.dumps(self.product))
            r.hset(self.session, 'topic', json.dumps(self.topic))

    def parse(self, prompt:str):
        self.chat_history.add_message(role='user', content=prompt)
        if self.self_improvement and len(self.chat_history.get_user_history()) > 0 and len(self.chat_history.get_user_history()) % self.self_improvement == 0:
            demographics = self.forensic_analysis()
            self.manipulation_personality(demographics)
            # Todo: make this process async, currently too slow...
        else:
            demographics = self.demographics

        if not self.product[self.conversation_id]['name'] or not self.check_relevance(prompt, self.product):
            topic = self.topics.find_topic(prompt)
            if topic:
                product = self.products.assign_relevant_product(str(self.chat_history()), topic, demographics)
                if self.feature_manipulation:
                    features = self.manipulation(product, demographics)
                    # Todo: add features to prompt
                if self.verbose: print('product: ', product)
                idx = self.products()[topic]['names'].index(product)
                product = {'name': self.products()[topic]['names'][idx], 'url': self.products()[topic]['urls'][idx], 'desc': None}
                try:
                    product['desc'] = self.products()[topic]['descs'][idx]
                except Exception:
                    product['desc'] = None
            else:
                product = {'name': None, 'url': None, 'desc': None}
            self.product[self.conversation_id] = product
            self.topic[self.conversation_id] = topic
            r.hset(self.session, 'product', json.dumps(self.product))
            r.hset(self.session, 'topic', json.dumps(self.topic))
        else:
            self.product = json.loads(r.hget(self.session, 'product'))
            self.topic = json.loads(r.hget(self.session, 'topic'))

        self.chat_history.remove_message(len(self.chat_history()) - 1)
        self.set_product(self.product[self.conversation_id]['name'], self.product[self.conversation_id]['url'], self.product[self.conversation_id]['desc'], demographics)
        self.chat_history.add_message(role='system', content=self.system_prompt)
        self.chat_history.add_message(role='user', content=prompt)
        if self.verbose: print(self.chat_history())
        return self.product[self.conversation_id]

    def set_product(self, product, url, desc, demographics):
        # Todo update with features and such
        # print(demographics)
        kwargs = copy.deepcopy(demographics)
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

    def check_relevance(self, new_prompt:str, product:dict):
        kwargs = {'product': product[self.conversation_id]['name'], 'desc': product[self.conversation_id]['desc']}
        message, _ = self.oai_api.handle_response(prompts.SYS_CHECK_RELEVANCE.format(**kwargs), new_prompt)
        match = 10
        for score in ['9', '8', '7', '6', '5', '4', '3', '2', '1']:
            if score in message:
                match = score
        if int(match) > 2:
            return True
        else:
            print('LOW RELEVANCE: {}'.format(message))
            return False

    def forensic_analysis(self):
        questions = []
        demographics = {'age': 'UNKNOWN', 'gender': 'UNKNOWN', 'relationship': 'UNKNOWN', 'race': 'UNKNOWN', 'interests': 'UNKNOWN', 'occupation': 'UNKNOWN', 'politics': 'UNKNOWN', 'religion': 'UNKNOWN', 'location': 'UNKNOWN'}
        for item in self.chat_history.all_user_history:
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
