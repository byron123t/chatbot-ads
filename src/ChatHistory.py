from data import prompts
from src.API import OpenAIAPI
import tiktoken, os, json, difflib


absolute_path = os.path.dirname(os.path.abspath(__file__))

class ChatHistory:
    def __init__(self, chat_history:list=[], response_history:list=[], current_conversation:list=[], user_history:list=[], verbose:bool=False):
        self.oai_api = OpenAIAPI(verbose=verbose)
        if not os.path.exists(absolute_path + '/../data/sessions/'):
            os.mkdir(absolute_path + '/../data/sessions/')
        if not os.path.exists(absolute_path + '/../data/metadata/'):
            os.mkdir(absolute_path + '/../data/metadata/')
        self.session = len(os.listdir(absolute_path + '/../data/sessions/')) + 1
        self.chat_history = chat_history
        self.response_history = response_history
        self.current_conversation = current_conversation
        self.user_history = user_history
        self.all_user_history = user_history
        self.encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')

    def __call__(self):
        return self.current_conversation

    def add_message(self, role:str=None, content:str=None, message:dict=None, response:dict=None):
        if message:
            self.chat_history.append(message)
            if message['role'] == 'user':
                self.current_conversation.append(message)
                self.all_user_history.append(message)
            elif message['role'] == 'system':
                for i, item in enumerate(self.current_conversation):
                    if item['role'] == 'system':
                        self.current_conversation.pop(i)
                self.current_conversation.append(message)
        else:
            self.chat_history.append({'role': role, 'content': content})
            if role == 'user':
                self.current_conversation.append({'role': role, 'content': content})
                self.all_user_history.append({'role': role, 'content': content})
            elif role == 'system':
                for i, item in enumerate(self.current_conversation):
                    if item['role'] == 'system':
                        self.current_conversation.pop(i)
                self.current_conversation.append({'role': role, 'content': content})
            if response:
                self.response_history.append(response)

    def remove_message(self, index:int):
        if index < len(self.current_conversation) or index >= len(self.current_conversation):
            self.current_conversation.pop(index)
        else:
            raise IndexError('Index out of range')

    def remove_messages(self, indices:list):
        indices.sort(reverse=True)
        for index in indices:
            self.remove_message(index)
    
    def check_relevance(self, new_prompt:str):
        self.user_history = self.remove_non_user_messages()
        if len(self.user_history) > 0:
            kwargs = {'prompts': str(self.user_history)}
            message, _ = self.oai_api.handle_response(prompts.SYS_CHECK_RELEVANCE.format(**kwargs), new_prompt)
            matches = difflib.get_close_matches(message, ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'], n=1)
            if len(matches) > 0:
                if int(matches[0]) > 4:
                    return True
                else:
                    self.current_conversation = []
                    return False
            else:   
                return True
        else:
            return False

    def remove_non_user_messages(self):
        return [item for item in self.current_conversation if item['role'] == 'user']

    def check_length(self):
        tokens = self.encoding.encode(str(self.current_conversation))
        if len(tokens) > 3500:
            return True
        else:
            return False

    def manage_length(self):
        if self.check_length():
            self.remove_non_user_messages()
        while self.check_length():
            self.current_conversation.pop(0)

    def write_to_file(self):
        kwargs = {'session': self.session}
        if self.chat_history:
            with open(absolute_path + '/../data/sessions/session_{session}.json'.format(**kwargs), 'w') as outfile:
                json.dump(self.chat_history, outfile)
        if self.response_history:
            with open(absolute_path + '/../data/metadata/metadata_{session}.json'.format(**kwargs), 'w') as outfile:
                json.dump(self.response_history, outfile)

    def read_from_file(self):
        kwargs = {'session': self.session}
        with open(absolute_path + '/../data/sessions/session_{session}.json'.format(**kwargs), 'r') as infile1:
            with open(absolute_path + '/../data/metadata/metadata_{session}.json'.format(**kwargs), 'r') as infile2:
                self.chat_history = json.load(infile1)
                self.response_history = json.load(infile2)
                self.current_conversation = self.chat_history

    def load_session(self, session):
        self.session = session
        self.read_from_file()

    def new_session(self):
        self.session = len(os.listdir(absolute_path + '/../data/sessions/')) + 1
        self.chat_history = []
        self.response_history = []
        self.current_conversation = []
