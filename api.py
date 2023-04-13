from sensitive.objects import openai_key
from data import prompts
import openai, json, os, difflib, random


class OpenAIChatSession:
    def __init__(self, mode='control', ad_freq=1):
        openai.api_key = openai_key
        self.mode = mode
        self.system_prompt = ''
        self.topics = {}
        self.ad_freq = ad_freq
        self.chat_history = []
        self.response_history = []
        self.session = len(os.listdir('data/sessions/')) + 1
        if mode == 'interest-based':
            self.initializer = prompts.SYS_INTEREST
            self.initializer_desc = prompts.SYS_INTEREST_DESC
        elif mode == 'chatbot-centric':
            self.initializer = ''
            self.initializer_desc = ''
        elif mode == 'user-centric':
            self.initializer = ''
            self.initializer_desc = ''
        elif mode == 'influencer':
            self.initializer = ''
            self.initializer_desc = ''
        else:
            self.initializer = 'You are a helpful assistant.'
            self.system_prompt = self.initializer
            self.mode = 'control'
        self.parse_topics_file()
            
    def dump(self):
        if self.chat_history:
            with open('data/sessions/session_{}'.format(self.session), 'w') as outfile:
                json.dump(self.chat_history, outfile)
        else:
            with open('data/sessions/session_{}'.format(self.session), 'w') as outfile:
                json.dump([], outfile)
        if self.response_history:
            with open('data/metadata/metadata_{}'.format(self.session), 'w') as outfile:
                json.dump(self.response_history, outfile)
        else:
            with open('data/metadata/metadata_{}'.format(self.session), 'w') as outfile:
                json.dump([], outfile)

    def read(self):
        with open('data/sessions/session_{}.json'.format(self.session), 'r') as infile1:
            with open('data/metadata/metadata_{}.json'.format(self.session), 'r') as infile2:
                return json.load(infile1), json.load(infile2)

    def run_chat(self, prompt):
        if 'metadata_{}'.format(self.session) not in os.listdir('data/metadata/') or 'session_{}'.format(self.session) not in os.listdir('data/sessions/'):
            self.dump()
        topic = self.find_topic(prompt)
        product = self.assign_relevant_product(prompt, topic)
        self.set_product(product)
        self.chat_history.extend([
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': prompt}
        ])
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=self.chat_history
        )
        if len(response['choices']) > 0:
            print(response['choices'])
        if response['choices'][0]['finish_reason'] == 'stop':
            message = {"content": response['choices'][0]['message']['content'],
                    "role": response['choices'][0]['message']['role']}
        else:
            print(response)
        self.chat_history.append(message)
        self.response_history.append(response)
        self.dump()
        return message
    
    def load_session(self, session):
        self.session = session
        self.chat_history, self.response_history = self.read()
    
    def new_session(self):
        self.session = len(os.listdir('data/sessions/')) + 1
        self.chat_history = []
        self.response_history = []

    def set_product(self, product, url, desc):
        if self.mode != 'control' and random.random() < self.ad_freq:
            if desc:
                self.system_prompt = self.initializer_desc.format(product, url, desc)
            else:
                self.system_prompt = self.initializer.format(product, url)
        else:
            self.system_prompt = 'You are a helpful assistant.'

    def read_products_file(self):
        with open('data/products.json', 'r') as infile:
            self.products = json.load(infile)
            return self.products
        
    def parse_topics_file(self):
        with open('data/topics.json', 'r') as infile:
            self.topics = json.load(infile)
            return self.topics

    def find_topic(self, prompt):
        def send_topic_chat(prompt, topics):
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                {'role': 'system', 'content': prompts.SYS_TOPICS.format(topics)},
                {'role': 'user', 'content': prompt}
            ])
            if len(response['choices']) > 0:
                print(response['choices'])
            if response['choices'][0]['finish_reason'] == 'stop':
                message = response['choices'][0]['message']['content']
            else:
                print(response)
            print(message, prompt)
            matches = difflib.get_close_matches(message, topics, n=1)
            if len(matches) > 0:
                self.current_topic = matches[0]
                return True
            return False

        topic_dict = self.topics
        send_topic_chat(prompt, topic_dict.keys())
        topic_dict = topic_dict[self.current_topic]
        found = True
        while len(topic_dict.keys()) > 0 and found:
            found = send_topic_chat(prompt, topic_dict.keys())
            topic_dict = topic_dict[self.current_topic]
        return self.current_topic

    def populate_products(self):
        def send_products_chat(topic):
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                {'role': 'system', 'content': prompts.SYS_PRODUCTS},
                {'role': 'user', 'content': prompts.USER_PRODUCTS.format(topic)}
            ])
            if len(response['choices']) > 0:
                print(response['choices'])
            if response['choices'][0]['finish_reason'] == 'stop':
                message = response['choices'][0]['message']['content']
            else:
                print(response)
                
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
        def send_product_chat(prompt, products):
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                {'role': 'system', 'content': prompts.SYS_RELEVANT_PRODUCT.format(products['names'], products['descs'])},
                {'role': 'user', 'content': prompt}
            ])
            if len(response['choices']) > 0:
                print(response['choices'])
            if response['choices'][0]['finish_reason'] == 'stop':
                message = response['choices'][0]['message']['content']
            else:
                print(response)
            self.current_product = difflib.get_close_matches(message, products['names'], n=1)[0]
            return self.current_product
        
        with open('data/products.json', 'r') as infile:
            self.products = json.load(infile)
        return send_product_chat(prompt, self.products[topic])
    
    def assign_random_product(self, topic):
        with open('data/products.json', 'r') as infile:
            self.products = json.load(infile)
        index = random.randint(0, len(self.products[topic]['names']) - 1)
        return self.products[topic]['names'][index]

    def change_ad_frequency(self, freq):
        self.ad_freq = freq

    def clear_products(self):
        def remove_lists(in_dict):
            for key, value in in_dict.items():
                if isinstance(value, dict):
                    # Recursively remove lists from nested dictionaries
                    remove_lists(value)
                elif key in ['names', 'urls', 'descs']:
                    # Remove lists from the dictionary
                    del in_dict[key]
            return in_dict
        with open('data/products.json', 'r') as infile:
            data = json.load(infile)
        data = remove_lists(data)
        with open('data/products.json', 'w') as outfile:
            json.dump(data, outfile, indent=4)


if __name__ == '__main__':
    oai = OpenAIChatSession('interest-based')
    # oai.run_chat('I need help finding a good coffee shop')
    # topic = oai.find_topic('I need help finding a good coffee shop')
    # oai.find_product('')
    oai.populate_products()
