from sensitive.objects import openai_key
from data import prompts
import openai, json, os, difflib, random


class OpenAIChatSession:
    def __init__(self, mode='control', ad_freq=1, demographics={}, self_improvement=None, feature_manipulation=False, verbose=False):
        openai.api_key = openai_key
        self.mode = mode
        self.system_prompt = ''
        self.topics = {}
        self.products = {}
        self.personality = ''
        self.profile = ''
        self.history_required = False
        self.demographics = demographics
        self.ad_freq = ad_freq
        self.self_improvement = self_improvement
        self.feature_manipulation = feature_manipulation
        self.chat_history = []
        self.concise_chat_history = []
        self.user_history = ''
        self.response_history = []
        self.verbose = verbose
        self.session = len(os.listdir('data/sessions/')) + 1
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
        self.parse_topics_file()
            
    def dump(self):
        kwargs = {'session': self.session}
        if self.chat_history:
            with open('data/sessions/session_{session}.json'.format(**kwargs), 'w') as outfile:
                json.dump(self.chat_history, outfile)
        else:
            with open('data/sessions/session_{session}.json'.format(**kwargs), 'w') as outfile:
                json.dump([], outfile)
        if self.response_history:
            with open('data/metadata/metadata_{session}.json'.format(**kwargs), 'w') as outfile:
                json.dump(self.response_history, outfile)
        else:
            with open('data/metadata/metadata_{session}.json'.format(**kwargs), 'w') as outfile:
                json.dump([], outfile)

    def read(self):
        kwargs = {'session': self.session}
        with open('data/sessions/session_{session}.json'.format(**kwargs), 'r') as infile1:
            with open('data/metadata/metadata_{session}.json'.format(**kwargs), 'r') as infile2:
                return json.load(infile1), json.load(infile2)

    def handle_response(self, sys_prompt, user_prompt):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
            {'role': 'system', 'content': sys_prompt},
            {'role': 'user', 'content': user_prompt}
        ])
        if len(response['choices']) > 0:
            if self.verbose: print(response['choices'])
        if response['choices'][0]['finish_reason'] == 'stop':
            message = response['choices'][0]['message']['content']
        else:
            if self.verbose: print(response)
        return message

    def run_chat(self, prompt):
        self.history_required = False
        self.user_history += prompt
        kwargs = {'session': self.session}
        if 'metadata_{session}'.format(**kwargs) not in os.listdir('data/metadata/') or 'session_{session}'.format(**kwargs) not in os.listdir('data/sessions/'):
            self.dump()
        if len(self.chat_history) == 0:
            self.chat_history.append({'role': 'system', 'content': self.system_prompt})
        if self.self_improvement and len(self.chat_history) > 0 and len(self.chat_history) % self.self_improvement == 0:
            demographics = self.forensic_analysis()
            self.manipulation_personality(demographics)
            self.chat_history.append({'role': 'system', 'content': self.system_prompt})
        else:
            demographics = self.demographics
        if self.feature_manipulation:
            self.manipulation(product, demographics)
        topic = self.find_topic(prompt)
        if self.history_required:
            if topic:
                product = self.assign_relevant_product(self.user_history, topic)
            else:
                product = self.assign_random_product(topic)
        else:
            if topic:
                product = self.assign_relevant_product(prompt, topic)
            else:
                product = self.assign_random_product(topic)
        idx = self.products[topic]['names'].index(product)
        url = self.products[topic]['urls'][idx]
        try:
            desc = self.products[topic]['descs'][idx]
        except Exception as e:
            desc = None
        self.set_product(product, url, desc, demographics)
        self.chat_history.append({'role': 'user', 'content': prompt})
        if self.count_chat_history_tokens(self.chat_history):
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=self.concise_chat_history
            )
        else:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=self.chat_history
            )
            
        if response['choices'][0]['finish_reason'] == 'stop':
            message = {"content": response['choices'][0]['message']['content'], "role": response['choices'][0]['message']['role']}
        else:
            if self.verbose: print(response)
        if len(response['choices']) > 0:
            if self.verbose: print(response['choices'])
            for item in response['choices']:
                if item['finish_reason'] == 'stop':
                    if product in item['message']['content']:
                        message = {'content': item['message']['content'], 'role': item['message']['role']}
                        break
                else:
                    if self.verbose: print(response)
        print(message)
        self.chat_history.append(message)
        self.response_history.append(response)
        self.dump()

        if self.count_chat_history_tokens(self.chat_history):
            self.concise_chat_history = []
            for item in self.chat_history:
                response = self.handle_response(prompts.SYS_SUMMARIZE_CHAT_HISTORY, item['content'])
                if response['choices'][0]['finish_reason'] == 'stop':
                    short_message = {"content": response['choices'][0]['message']['content'], "role": response['choices'][0]['message']['role']}
                else:
                    if self.verbose: print(response)
                if len(response['choices']) > 0:
                    if self.verbose: print(response['choices'])
                    items = []
                    for item in response['choices']:
                        if item['finish_reason'] == 'stop':
                            items.append(item['message']['content'])
                        else:
                            if self.verbose: print(response)
                    short_message = {'content': min(items, key=len), 'role': item['message']['role']}
                self.concise_chat_history.append(short_message)

        if self.count_chat_history_tokens(self.concise_chat_history):
            indices = []
            num_sys = 0
            count = 0
            for i, item in enumerate(self.concise_chat_history):
                if item['role'] == 'system':
                    num_sys += 1
            for i, item in enumerate(self.concise_chat_history):
                if item['role'] == 'assistant' or item['role'] == 'user':
                    indices.append(i)
                elif item['role'] == 'system' and num_sys > 2:
                    indices.append(i)
                    num_sys -= 1
                count += 1
                if count > 4:
                    break
            self.concise_chat_history.pop(indices)
            
        return message
    
    def load_session(self, session):
        self.session = session
        self.chat_history, self.response_history = self.read()
    
    def new_session(self):
        self.session = len(os.listdir('data/sessions/')) + 1
        self.chat_history = []
        self.response_history = []

    def set_product(self, product, url, desc, demographics):
        kwargs = demographics
        kwargs['product'] = product
        kwargs['url'] = url
        kwargs['desc'] = desc
        kwargs['personality'] = self.personality
        kwargs['profile'] = self.profile
        if self.mode == 'control' or random.random() > self.ad_freq:
            self.system_prompt = 'You are a helpful assistant.'
        else:
            if desc:
                self.system_prompt = self.initializer_desc.format(**kwargs)
            else:
                self.system_prompt = self.initializer.format(**kwargs)

    def read_products_file(self):
        with open('data/products.json', 'r') as infile:
            self.products = json.load(infile)
            return self.products
        
    def parse_topics_file(self):
        with open('data/topics.json', 'r') as infile:
            self.topics = json.load(infile)
            return self.topics

    def find_topic(self, prompt):
        self.current_topic = None
        def send_topic_chat(prompt, topics):
            kwargs = {'topics': topics}
            message = self.handle_response(prompts.SYS_TOPICS.format(**kwargs), prompt)
            print(message, prompt)
            matches = difflib.get_close_matches(message, topics, n=1)
            if len(matches) > 0:
                self.current_topic = matches[0]
                return True
            else:
                message = self.handle_response(prompts.SYS_TOPICS_NEW.format(**kwargs), prompt)
                print(message, prompt)
                with open('data/unseen_topics.json', 'r') as infile:
                    data = json.load(infile)
                with open('data/unseen_topics.json', 'w') as outfile:
                    if self.current_topic:
                        data[self.current_topic][message] = {}
                    else:
                        data[message] = {}
                    json.dump(data, outfile, indent=4)
            return False

        topic_dict = self.topics
        found = True
        while len(topic_dict.keys()) > 0:
            found = send_topic_chat(prompt, topic_dict.keys())
            if not found and self.current_topic:
                break
            elif self.current_topic is None:
                found = send_topic_chat(str(self.user_history), topic_dict.keys())
                self.history_required = True
            if not found and not self.current_topic:
                return None
            topic_dict = topic_dict[self.current_topic]
        return self.current_topic

    def populate_products(self):
        def send_products_chat(topic):
            kwargs = {'topic': topic}
            message = self.handle_response(prompts.SYS_PRODUCTS, prompts.USER_PRODUCTS.format(**kwargs))
                
            names = []
            urls = []
            descs = []
            print(message)
            message = message.replace(' — ', ' - ')
            lines = message.split('\n')
            for line in lines:
                if len(line) > 0:
                    split = line.split(' - ')
                    if len(split) > 2:
                        if split[0].startswith('- '):
                            names.append(split[0][2:])
                        else:
                            names.append(split[0])
                        urls.append(split[1])
                        descs.append(split[2])

            if len(names) > 0:
                if 'names' not in self.products[topic]:
                    self.products[topic]['names'] = names
                else:
                    self.products[topic]['names'].extend(names)
                if 'urls' not in self.products[topic]:
                    self.products[topic]['urls'] = urls
                else:
                    self.products[topic]['urls'].extend(urls)
                if 'descs' not in self.products[topic]:
                    self.products[topic]['descs'] = descs
                else:
                    self.products[topic]['descs'].extend(descs)

                # with open('data/products.json', 'w') as outfile:
                with open('data/temp_products.json', 'w') as outfile:
                    json.dump(self.products, outfile, indent=4)

        # with open('data/products.json', 'r') as infile:
        with open('data/temp_products.json', 'r') as infile:
            self.products = json.load(infile)
        for key, val in self.products.items():
            send_products_chat(key)

    def assign_relevant_product(self, prompt, topic):
        def send_product_chat(prompt, products, demographics):
            kwargs = demographics
            kwargs['products'] = products['names']
            kwargs['descs'] = products['descs']
            if demographics:
                message = self.handle_response(prompts.SYS_RELEVANT_PRODUCT_USER.format(**kwargs), prompt)
            else:
                message = self.handle_response(prompts.SYS_RELEVANT_PRODUCT.format(**kwargs), prompt)
            matches = difflib.get_close_matches(message, products['names'], n=1)
            if len(matches) > 0:
                self.current_product = matches[0]
                print(self.current_product)
                return self.current_product
            return self.assign_random_product(topic)
        
        with open('data/products.json', 'r') as infile:
            self.products = json.load(infile)
        return send_product_chat(prompt, self.products[topic], self.demographics)
    
    def assign_random_product(self, topic):
        with open('data/products.json', 'r') as infile:
            self.products = json.load(infile)
        if topic:
            index = random.randint(0, len(self.products[topic]['names']) - 1)
            print(self.products[topic]['names'][index])
            return self.products[topic]['names'][index]
        else:
            topic = random.choice(list(self.products.keys()))
            self.topic = topic
            index = random.randint(0, len(self.products[topic]['names']) - 1)
            print(self.products[topic]['names'][index])
            return self.products[topic]['names'][index]

    def forensic_analysis(self):
        questions = []
        demographics = {}
        for item in self.chat_history:
            if item['role'] == 'user':
                questions.append(item['content'])
        message = self.handle_response(prompts.SYS_FORENSIC_ANALYSIS, str(questions))
        print(questions, message)
        self.profile = message
        message = self.handle_response(prompts.SYS_FORENSIC_ANALYSIS_DEMOGRAPHICS, str(questions))
        print(questions, message)
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
                    print(split)
        for key, val in self.demographics.items():
            if key in demographics:
                self.demographics[key] = demographics[key]
        return self.demographics

    def manipulation(self, product, demographics):
        message = self.handle_response(prompts.SYS_MANIPULATION, product, str(demographics))
        print(message)
        return message
        
    def manipulation_personality(self, demographics):
        kwargs = {'demographics': str(demographics), 'profile': self.profile}
        self.personality = self.handle_response(prompts.SYS_MANIPULATION_PERSONALITY, '{profile}\n\n{demographics}'.format(**kwargs))
        print(self.personality)
        return self.personality

    def count_chat_history_tokens(self, chat_history):
        word_count = 0
        for item in chat_history:
            word_count += len(item['content'].split())
        if word_count > 2000:
            return True
        return False

    def change_ad_frequency(self, freq):
        self.ad_freq = freq

    def clear_products(self):
        def remove_lists(in_dict):
            for key, value in in_dict.items():
                if isinstance(value, dict):
                    remove_lists(value)
                elif key in ['names', 'urls', 'descs']:
                    del in_dict[key]
            return in_dict
        with open('data/products.json', 'r') as infile:
            data = json.load(infile)
        data = remove_lists(data)
        with open('data/products.json', 'w') as outfile:
            json.dump(data, outfile, indent=4)


if __name__ == '__main__':
    demo = {'age': 28, 'gender': 'Male', 'relationship': 'Unknown', 'race': 'Asian', 'interests': ['Anime', 'Video Games', 'Academia', 'Investing', 'Mindfulness'], 'occupation': 'Software Engineer', 'politics': 'Socially Left Libertarian', 'religion': 'Atheist', 'location': 'San Francisco, CA'}
    oai = OpenAIChatSession(mode='chatbot-centric', ad_freq=1, demographics=demo, self_improvement=2, verbose=False)

    oai.run_chat('can you recommend a random good sci fi anime show or movie?')
    oai.run_chat('any more obscure ones?')
    oai.run_chat('what about more recent ones?')
    oai.run_chat('how about ones related to AI')
    oai.run_chat('what about good ai manga')
    oai.run_chat('Can you list like 20?')
    oai.run_chat('How do I change the password of another user with root privileges? There\'s kerberos and I don\'t know the user\'s current Kerberos password.')
    oai.run_chat('Is there a way I can do this without modifying kerberos?')
    oai.run_chat('What are some iconic lines from perry the platypus')
    oai.run_chat('Are there any obscure male anime characters from mainstream anime shows?')
    oai.run_chat('Are there any male anime characters that don\'t wear a uniform or suit?')
    oai.run_chat('We need a better title for a research grant that deals with mobile computing, wireless networking, iot, security and privacy, autonomous vehicle system control and security, server security, cyber-physical systems. Ideally the title should include all of these details concisely. Also can you make an acronym')
    oai.run_chat('Are there literature that neutrality survey questions? Provide paper names')
    oai.run_chat('What are some standard questionnaire questions for measuring discrimination in recommendation algorithms?')


    # oai.run_chat('What should I do if my home catches on fire?')
    # oai.run_chat('I need help finding a good coffee shop')
    # oai.run_chat('I need help with getting a new smartphone! Mine\'s outdated.')
    # oai.new_session()
    # oai.run_chat('Tell me a story about a cat')
    # oai.run_chat('Who was the prime minister of Japan in 1984?')
    # oai.new_session()
    # oai.run_chat('Write me some code to print out "Hello World" in 3 random languages.')
    # topic = oai.find_topic('I need help finding a good coffee shop')
    # oai.find_product('')
    # oai.populate_products()
