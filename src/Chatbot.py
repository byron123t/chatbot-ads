from src.API import OpenAIAPI
from src.Advertiser import Advertiser
from src.Config import ROOT
import json, os, argparse, openai
from flask import Response


absolute_path = os.path.dirname(os.path.abspath(__file__))

class OpenAIChatSession:
    def __init__(self, session:str='', mode:str='control', model='gpt-3.5-turbo', ad_freq:float=1.0, conversation_id:str='', self_improvement:int=None, feature_manipulation:bool=False, ad_transparency:str='none', verbose:bool=False):
        self.oai_response_api = OpenAIAPI(verbose=verbose, model=model)
        self.oai_api = OpenAIAPI(verbose=verbose)
        self.advertiser = Advertiser(mode=mode, session=session, ad_freq=ad_freq, self_improvement=self_improvement, feature_manipulation=feature_manipulation, verbose=verbose, conversation_id=conversation_id)
        self.verbose = verbose
        self.ad_transparency = ad_transparency

    def run_chat(self, prompt:str):
        product = self.advertiser.parse(prompt)
        message, response = self.oai_api.handle_response(chat_history=self.advertiser.chat_history(), stream=True)
        new_message = {'role': 'assistant', 'content': ''}
        for chunk in message:
            try:
                if len(chunk.choices) > 0:
                    token = chunk.choices[0].delta.content
                    if token:
                        print(token, end='', flush=True)
                        new_message['content'] += token
            except Exception as e:
                print(e)
        new_response = {'id': chunk.id, 'object': 'chat.completion', 'created': chunk.created, 'model': chunk.model, 'usage': None, 'choices': None, 'finish_reason': None}
        self.advertiser.chat_history.add_message(message=new_message, response=new_response)
        return message

    def run_chat_live(self, prompt:str):
        product = self.advertiser.parse(prompt)
        print(self.advertiser.chat_history())
        print(product)
        message, response = self.oai_response_api.handle_response(chat_history=self.advertiser.chat_history(), stream=True)
        new_message = {'role': 'assistant', 'content': ''}
        token_count = 0
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        finish_reason = None
        for chunk in message:
            token_count += 1
            # yield 'data: {}\n\n'.format(json.dumps(chunk, separators=(',', ':')))
            if self.ad_transparency == 'disclosure':
                sent_disclosure = False
            try:
                if len(chunk.choices) > 0:
                    token = chunk.choices[0].delta.content
                    finish_reason = chunk.choices[0].finish_reason
                    out_data = {'content': token, 'finish_reason': finish_reason}
                    if token:
                        new_message['content'] += token
                    if self.ad_transparency == 'icon' and not sent_disclosure and product['name']:
                        print('REACHED')
                        stripped_product = product['name'].lower().replace(' ', '').replace('-', '').replace('_', '').replace('.', '').replace(',', '').replace(':', '').replace(';', '').replace('\n', '').strip()
                        stripped_message = new_message['content'].lower().replace(' ', '').replace('-', '').replace('_', '').replace('.', '').replace(',', '').replace(':', '').replace(';', '').replace('\n', '').strip()
                        print(stripped_message)
                        print(stripped_product)
                        if stripped_product in stripped_message:
                            sent_disclosure = True
                            yield 'data: {}\n\n'.format(json.dumps({'content': '$^^ad^^$', 'finish_reason': None}, separators=(',', ':')))
                    elif self.ad_transparency == 'disclosure' and not sent_disclosure and product['name']:
                        print('REACHED')
                        stripped_product = product['name'].lower().replace(' ', '').replace('-', '').replace('_', '').replace('.', '').replace(',', '').replace(':', '').replace(';', '').replace('\n', '').strip()
                        stripped_message = new_message['content'].lower().replace(' ', '').replace('-', '').replace('_', '').replace('.', '').replace(',', '').replace(':', '').replace(';', '').replace('\n', '').strip()
                        print(stripped_message)
                        print(stripped_product)
                        if stripped_product in stripped_message:
                            sent_disclosure = True
                            yield 'data: {}\n\n'.format(json.dumps({'content': '$^^ad^^$', 'finish_reason': None}, separators=(',', ':')))
                            yield 'data: {}\n\n'.format(json.dumps({'content': '$^^disclosure^^$', 'finish_reason': None}, separators=(',', ':')))
                    yield 'data: {}\n\n'.format(json.dumps(out_data, separators=(',', ':')))
            except Exception as e:
                print(e)
        usage['completion_tokens'] = token_count
        tokens = self.advertiser.chat_history.encoding.encode(str(prompt))
        usage['prompt_tokens'] = len(tokens)
        usage['total_tokens'] = len(tokens) + token_count
        new_response = {'id': chunk.id, 'object': 'chat.completion', 'created': chunk.created, 'model': chunk.model, 'usage': token_count, 'choices': [new_message], 'finish_reason': finish_reason}
        self.advertiser.chat_history.add_message(message=new_message, response=new_response)
        return 'data: [DONE]'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chatbot Advertising Demo')
    parser.add_argument('--mode', type=str, default='interest-based', choices=['interest-based', 'chatbot-centric', 'user-centric', 'influencer'], help='Chatbot settings: mode (string), choose from [interest-based, chatbot-centric, user-centric, influencer]')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo', choices=['gpt-3.5-turbo', 'gpt-4o'], help='Chatbot settings: model (string), choose from [gpt-3.5-turbo, gpt-4o]')
    parser.add_argument('--ad-freq', type=float, default=1.0, help='Chatbot settings: ad frequency (float), 0.0 - 1.0 (0.0 = no ads, 1.0 = ads every message)')
    parser.add_argument('--self-improvement', type=int, default=None, help='Chatbot settings: self improvement (int), self improvement of demographics and profiling every X messages')
    parser.add_argument('--verbose', action='store_true', help='Chatbot settings: verbose (bool), print details for debugging')
    args = parser.parse_args()
    
    oai = OpenAIChatSession(mode=args.mode, model=args.model, ad_freq=args.ad_freq, self_improvement=args.self_improvement, verbose=args.verbose)
    print('How can I help you today?\nRunning the following parameters:\n\tMode: {}\n\tModel: {}\n\tAd Frequency: {}\n\tSelf Improvement: {}\n\tVerbose: {}'.format(oai.advertiser.mode, args.model, oai.advertiser.ad_freq, oai.advertiser.self_improvement, oai.verbose))

    print('User: ')
    user_input = input()
    while True:
        if user_input == 'new_session':
            print('Session ID: ')
            oai.advertiser.chat_history.new_session(input())
            print('New session started with ID: {}'.format(oai.advertiser.chat_history.session))
            continue
        elif user_input == 'load_session':
            print('Session ID: ')
            oai.advertiser.chat_history.load_session(input())
            print('Loaded session with ID: {}'.format(oai.advertiser.chat_history.session))
            continue
        elif user_input == 'exit':
            print('Exiting...')
            exit()
        oai.run_chat('{}'.format(user_input))
        print('\n\n')
        print('User: ')
        user_input = input()