from data import prompts
from src.API import OpenAIAPI
from src.Config import ROOT
import json, difflib, os


class Topics:
    def __init__(self, verbose=False):
        self.oai_api = OpenAIAPI(verbose=verbose)
        self.topics = {}
        self.verbose = verbose
        self.parse_topics_file()

    def __call__(self):
        return self.topics

    def parse_topics_file(self):
        with open(os.path.join(ROOT, 'data/topics.json'), 'r') as infile:
            self.topics = json.load(infile)
            return self.topics

    def find_topic(self, prompt:str):
        self.current_topic = None
        def send_topic_chat(prompt, topics):
            kwargs = {'topics': topics}
            message, _ = self.oai_api.handle_response(prompts.SYS_TOPICS.format(**kwargs), prompt)
            if self.verbose: print(message, prompt)
            matches = difflib.get_close_matches(message, topics, n=1)
            if len(matches) > 0:
                self.current_topic = matches[0]
                return True
            else:
                message, _ = self.oai_api.handle_response(prompts.SYS_TOPICS_NEW.format(**kwargs), prompt)
                if self.verbose: print(message, prompt)
                with open(os.path.join(ROOT, 'data/unseen_topics.json'), 'r') as infile:
                    data = json.load(infile)
                with open(os.path.join(ROOT, 'data/unseen_topics.json'), 'w') as outfile:
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
            if not found and not self.current_topic:
                return None
            topic_dict = topic_dict[self.current_topic]
        return self.current_topic
