from data import prompts
from src.ChatHistory import ChatHistory
from src.API import OpenAIAPI
from src.Products import Products
from src.Topics import Topics
import difflib, random, json, copy, re, time
from redis import Redis


r = Redis(host='localhost', port=6379, password='', decode_responses=True)


class Advertiser:
    def __init__(self, mode:str='control', session:str='', ad_freq:float=1.0, conversation_id:str='', self_improvement:int=None, feature_manipulation:bool=False, verbose:bool=False):
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
        elif mode == 'user-centric':
            self.initializer = prompts.SYS_USER_CENTRIC_INTEREST
            self.initializer_desc = prompts.SYS_USER_CENTRIC_INTEREST_DESC
        elif mode == 'influencer':
            self.initializer = prompts.SYS_INFLUENCER_INTEREST
            self.initializer_desc = prompts.SYS_INFLUENCER_INTEREST_DESC
        else:
            self.initializer = 'You are a helpful assistant.'
            self.system_prompt = self.initializer
            self.mode = 'control'
        if r.exists(self.session):
            if r.hexists(self.session, 'mode'):
                self.mode = json.loads(r.hget(self.session, 'mode'))
                self.profile = r.hget(self.session, 'profile')
            else:
                r.hset(self.session, 'mode', json.dumps(self.mode))
                r.hset(self.session, 'profile', self.profile)
            if r.hexists(self.session, 'product'):
                self.product = json.loads(r.hget(self.session, 'product'))
                self.topic = json.loads(r.hget(self.session, 'topic'))
                if self.conversation_id not in self.product:
                    self.product[conversation_id] = {'name': None, 'url': None, 'desc': None}
                    self.topic[conversation_id] = None
            else:
                r.hset(self.session, 'product', json.dumps(self.product))
                r.hset(self.session, 'topic', json.dumps(self.topic))
            if r.hexists(self.session, 'products'):
                prior_products = json.loads(r.hget(self.session, 'products'))
                if self.product[conversation_id]['name'] and self.product[conversation_id]['name'] not in prior_products:
                    prior_products.append(self.product[conversation_id]['name'])
                    r.hset(self.session, 'products', json.dumps(prior_products))
            elif self.product[conversation_id]['name']:
                r.hset(self.session, 'products', json.dumps(self.product[conversation_id]['name']))
            else:
                r.hset(self.session, 'products', json.dumps([]))
        else:
            r.hset(self.session, 'mode', json.dumps(self.mode))
            r.hset(self.session, 'profile', self.profile)
            r.hset(self.session, 'product', json.dumps(self.product))
            r.hset(self.session, 'topic', json.dumps(self.topic))

    def parse(self, prompt:str):
        self.chat_history.add_message(role='user', content=prompt)
        if self.mode != 'control' and self.self_improvement and len(self.chat_history.get_all_user_history()) > 0 and len(self.chat_history.get_all_user_history()) % self.self_improvement == 0:
            profile = self.forensic_analysis()
            print(profile)
            r.hset(self.session, 'profile', profile)
        else:
            time.sleep(2)
            profile = self.profile

        if not self.product[self.conversation_id]['name'] or not self.check_relevance(prompt, self.product):
            topic = self.topics.find_topic(prompt)
            if topic:
                product = self.products.assign_relevant_product(str(self.chat_history()), topic, profile)
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
            prior_products = json.loads(r.hget(self.session, 'products'))
            if product['name'] and product['name'] not in prior_products:
                prior_products.append(product['name'])
                r.hset(self.session, 'products', json.dumps(prior_products))
            r.hset(self.session, 'topic', json.dumps(self.topic))
            print(prior_products)
        else:
            self.product = json.loads(r.hget(self.session, 'product'))
            self.topic = json.loads(r.hget(self.session, 'topic'))

        self.chat_history.remove_message(len(self.chat_history()) - 1)
        self.set_product(self.product[self.conversation_id]['name'], self.product[self.conversation_id]['url'], self.product[self.conversation_id]['desc'], profile)
        self.chat_history.add_message(role='system', content=self.system_prompt)
        self.chat_history.add_message(role='user', content=prompt)
        if self.verbose: print(self.chat_history())
        return self.product[self.conversation_id]

    def set_product(self, product, url, desc, profile):
        kwargs = {}
        kwargs['product'] = product
        kwargs['url'] = url
        kwargs['desc'] = desc
        kwargs['personality'] = self.personality
        kwargs['profile'] = profile
        if self.mode == 'control' or random.random() > self.ad_freq or not product:
            self.system_prompt = 'You are a helpful assistant.'
        elif profile:
            if desc:
                self.system_prompt = self.initializer_desc.format(**kwargs)
            else:
                self.system_prompt = self.initializer.format(**kwargs)
        else:
            if desc:
                self.system_prompt = prompts.SYS_INTEREST_DESC.format(**kwargs)
            else:
                self.system_prompt = prompts.SYS_INTEREST.format(**kwargs)

    def check_relevance(self, new_prompt:str, product:dict):
        kwargs = {'product': product[self.conversation_id]['name'], 'desc': product[self.conversation_id]['desc'], 'prompt': new_prompt}
        message, _ = self.oai_api.handle_response(prompts.SYS_CHECK_RELEVANCE, prompts.USER_CHECK_RELEVANCE.format(**kwargs))
        match = 10
        numbers = re.findall(r'\d+', message)
        if len(numbers) > 0:
            match = int(numbers[0])
        if int(match) > 4:
            return True
        else:
            print('LOW RELEVANCE: {}'.format(message))
            return False

    def forensic_analysis(self):
        questions = []
        for item in self.chat_history.get_all_user_history():
            questions.append(item['content'])
        questions.reverse()
        questions = questions[:25]
        print(questions)
        message, _ = self.oai_api.handle_response(prompts.SYS_USER_PROFILE_SUMMARY, str(questions))
        if self.verbose: print(questions, message)
        self.profile = message
        return self.profile

    def change_ad_frequency(self, freq):
        self.ad_freq = freq
