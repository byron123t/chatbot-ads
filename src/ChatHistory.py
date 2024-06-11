from data import prompts
from src.API import OpenAIAPI
from src.Config import ROOT
import tiktoken, os, json, difflib, uuid
from redis import Redis


r = Redis(host='localhost', port=6379, password='', decode_responses=True)


class ChatHistory:
    def __init__(self, session:str='', chat_history:dict={}, response_history:dict={}, current_conversation:dict={}, user_history:dict={}, conversation_id:str='', verbose:bool=False):
        self.oai_api = OpenAIAPI(verbose=verbose)
        self.chat_history = chat_history
        self.response_history = response_history
        self.current_conversation = current_conversation
        self.user_history = user_history
        self.all_user_history = user_history
        if conversation_id:
            conversation_id = conversation_id
        else:
            conversation_id = str(uuid.uuid4())
        if session:
            self.load_session(session, conversation_id)
        else:
            self.new_session(session, conversation_id)
        self.encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')

    def __call__(self):
        return self.current_conversation[self.conversation_id]
    
    def get_user_history(self):
        return self.all_user_history[self.conversation_id]
    
    def get_all_user_history(self):
        all_chats = []
        for conversation_id, chats in self.chat_history.items():
            for chat in chats:
                if chat['role'] == 'user':
                    all_chats.append(chat)
        return all_chats
    
    def add_to_dict(self, dictionary:dict, value:dict):
        if self.conversation_id in dictionary:
            dictionary[self.conversation_id].append(value)
        else:
            dictionary[self.conversation_id] = [value]
    
    def add_message(self, role:str=None, content:str=None, message:dict=None, response:dict=None):
        if message:
            self.add_to_dict(self.chat_history, message)
            if message['role'] == 'user':
                self.add_to_dict(self.current_conversation, message)
                self.add_to_dict(self.all_user_history, message)
            elif message['role'] == 'system':
                for i, item in enumerate(self()):
                    if item['role'] == 'system':
                        self().pop(i)
                self.add_to_dict(self.current_conversation, message)
            elif message['role'] == 'assistant':
                # counter = 0
                # for i, item in enumerate(reversed(self())):
                #     if item['role'] == 'assistant':
                #         counter += 1
                #         if counter >= 2:
                #             self().pop(i)
                self.add_to_dict(self.current_conversation, message)
        else:
            self.add_to_dict(self.chat_history, {'role': role, 'content': content})
            if role == 'user':
                self.add_to_dict(self.current_conversation, {'role': role, 'content': content})
                self.add_to_dict(self.all_user_history, {'role': role, 'content': content})
            elif role == 'system':
                for i, item in enumerate(self()):
                    if item['role'] == 'system':
                        self().pop(i)
                self.add_to_dict(self.current_conversation, {'role': role, 'content': content})
            elif role == 'assistant':
                # counter = 0
                # for i, item in enumerate(reversed(self())):
                #     if item['role'] == 'assistant':
                #         counter += 1
                #         if counter >= 2:
                #             self().pop(i)
                self.add_to_dict(self.current_conversation, {'role': role, 'content': content})
        if response:
            self.add_to_dict(self.response_history, {'role': role, 'content': content})
        self.manage_length()
        self.write()

    def remove_message(self, index:int):
        if index < len(self()) or index >= len(self()):
            self().pop(index)
        else:
            raise IndexError('Index out of range')
        self.write()

    def remove_messages(self, indices:list):
        indices.sort(reverse=True)
        for index in indices:
            self.remove_message(index)
        self.write()

    def check_length(self):
        tokens = self.encoding.encode(str(self()))
        if len(tokens) > 2500:
            return True
        else:
            return False

    def manage_length(self):
        while self.check_length():
            self().pop(0)
        self.write()

    def write(self):
        r.hset(self.session, 'chat_history', json.dumps(self.chat_history))
        r.hset(self.session, 'response_history', json.dumps(self.response_history))
        r.hset(self.session, 'current_conversation', json.dumps(self.current_conversation))
        r.hset(self.session, 'user_history', json.dumps(self.user_history))

    def read(self):
        self.chat_history = json.loads(r.hget(self.session, 'chat_history'))
        self.response_history = json.loads(r.hget(self.session, 'response_history'))
        self.current_conversation = json.loads(r.hget(self.session, 'current_conversation'))
        self.user_history = json.loads(r.hget(self.session, 'user_history'))

    def load_session(self, session:str='', conversation_id:str=''):
        if not session:
            session = str(uuid.uuid4())
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
        self.session = session
        self.conversation_id = conversation_id
        if r.exists(self.session):
            self.read()
        else:
            self.new_session(self.session, self.conversation_id)

    def new_session(self, session:str='', conversation_id:str=''):
        if not session:
            session = str(uuid.uuid4())
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
        self.session = session
        self.conversation_id = conversation_id
        self.chat_history = {conversation_id: []}
        self.response_history = {conversation_id: []}
        self.current_conversation = {conversation_id: []}
        self.all_user_history = {conversation_id: []}
        self.write()
