"""
This script will use ChatGPT to provide various labels for response prompts.
"""
from src.Chatbot import OpenAIChatSession

score_ranges = [(1,10), (1,5), (6,10)]

metrics = ['relevance', 'misinformation', 'disinformation', 'neutrality', 'helpfulness']

prompt_answer = 'hi'
prompt_reply = 'hi'

oai = OpenAIChatSession()

for score_range in score_ranges:
    print('Score range: {}'.format(score_range))
    for metric in metrics:
        oai.advertiser.chat_history.new_session()
        print('Assessing metric: {}'.format(metric))
        oai.run_chat('I am going to give you a block of text. The block of text answers the prompt "{}". ' \
            'Give me a score of the {} of the block of text that answers the prompt on a scale from {} as ' \
            'the lowest to {} as the highest. ' \
            'Do not provide me an explanation. Only give me the number. Here is the block of text: {}' \
            .format(prompt_answer, metric, score_range[0], score_range[1], prompt_reply))
        print('')