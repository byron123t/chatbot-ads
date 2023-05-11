from src.API import OpenAIAPI
from src.Advertiser import Advertiser
from src.Config import ROOT
import json, os, argparse, openai
from flask import Response


absolute_path = os.path.dirname(os.path.abspath(__file__))

class OpenAIChatSession:
    def __init__(self, mode:str='control', ad_freq:float=1.0, demographics:dict={}, self_improvement:int=None, feature_manipulation:bool=False, verbose:bool=False):
        self.oai_api = OpenAIAPI(verbose=verbose)
        self.advertiser = Advertiser(mode=mode, ad_freq=ad_freq, demographics=demographics, self_improvement=self_improvement, feature_manipulation=feature_manipulation, verbose=verbose)
        self.verbose = verbose

    def run_chat(self, prompt:str):
        product = self.advertiser.parse(prompt)
        message, response = self.oai_api.handle_response(chat_history=self.advertiser.chat_history(), stream=True)
        new_message = {'role': 'assistant', 'content': ''}
        for chunk in message:
            if 'choices' in chunk and len(chunk['choices']) > 0 and 'delta' in chunk['choices'][0] and 'content' in chunk['choices'][0]['delta']:
                token = chunk['choices'][0]['delta']['content']
                print(token, end='', flush=True)
                new_message['content'] += token
        new_response = {'id': chunk['id'], 'object': 'chat.completion', 'created': chunk['created'], 'model': chunk['model'], 'usage': None, 'choices': None, 'finish_reason': None}
        self.advertiser.chat_history.add_message(message=new_message, response=new_response)
        self.advertiser.chat_history.write_to_file()
        return message

    def run_chat_live(self, prompt:str):
        product = self.advertiser.parse(prompt)
        message, response = self.oai_api.handle_response(chat_history=self.advertiser.chat_history(), stream=True)
        new_message = {'role': 'assistant', 'content': ''}
        for chunk in message:
            print(chunk)
            yield 'data: {}\n\n'.format(json.dumps(chunk, separators=(',', ':')))
            if 'choices' in chunk and len(chunk['choices']) > 0 and 'delta' in chunk['choices'][0] and 'content' in chunk['choices'][0]['delta']:
                token = chunk['choices'][0]['delta']['content']
                new_message['content'] += token
        new_response = {'id': chunk['id'], 'object': 'chat.completion', 'created': chunk['created'], 'model': chunk['model'], 'usage': None, 'choices': None, 'finish_reason': None}
        self.advertiser.chat_history.add_message(message=new_message, response=new_response)
        self.advertiser.chat_history.write_to_file()
        return 'data: [DONE]'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chatbot Advertising Demo')
    parser.add_argument('--demographic-file', type=str, default=absolute_path + '/../data/user_demographics.json', help='Name of the demographics file to process')
    parser.add_argument('--mode', type=str, default='interest-based', choices=['interest-based', 'chatbot-centric', 'user-centric', 'influencer'], help='Chatbot settings: mode (string), choose from [interest-based, chatbot-centric, user-centric, influencer]')
    parser.add_argument('--ad-freq', type=float, default=1.0, help='Chatbot settings: ad frequency (float), 0.0 - 1.0 (0.0 = no ads, 1.0 = ads every message)')
    parser.add_argument('--self-improvement', type=int, default=None, help='Chatbot settings: self improvement (int), self improvement of demographics and profiling every X messages')
    parser.add_argument('--verbose', action='store_true', help='Chatbot settings: verbose (bool), print details for debugging')
    args = parser.parse_args()
    
    with open(os.path.join(ROOT, 'data/user_demographics.json'), 'r') as infile:
        demo = json.load(infile)

    oai = OpenAIChatSession(mode=args.mode, ad_freq=args.ad_freq, demographics=demo, self_improvement=args.self_improvement, verbose=args.verbose)
    print('How can I help you today?\nRunning the following parameters:\n\tMode: {}\n\tAd Frequency: {}\n\tDemographics: {}\n\tSelf Improvement: {}\n\tVerbose: {}'.format(oai.advertiser.mode, oai.advertiser.ad_freq, oai.advertiser.demographics, oai.advertiser.self_improvement, oai.verbose))

    print('User: ')
    user_input = input()
    while True:
        if user_input == 'new_session':
            oai.advertiser.chat_history.new_session()
            print('New session started with ID: {}'.format(oai.advertiser.chat_history.session))
            continue
        elif user_input == 'load_session':
            print('Session ID: ')
            oai.advertiser.chat_history.load_session(int(input()))
            print('Loaded session with ID: {}'.format(oai.advertiser.chat_history.session))
            continue
        elif user_input == 'exit':
            print('Exiting...')
            exit()
        try:
            oai.run_chat('{}'.format(user_input))
        except openai.error.APIConnectionError:
            continue
        print('\n\n')
        print('User: ')
        user_input = input()