"""
This script will use ChatGPT to provide various labels for response prompts.
"""
from src.Chatbot import OpenAIChatSession

LOW_SCORE = 0
HIGH_SCORE = 10

metrics = ['relevance', 'misinformation', 'disinformation', 'neutrality', 'helpfulness']

prompt_answer = 'hi'
prompt_reply = 'hi'

for metric in metrics:
    print('Assessing metric: {}'.format(metric))
    oai = OpenAIChatSession()
    oai.run_chat('I am going to give you a block of text. The block of text answers the prompt "{}". ' \
        'Give me a score of the {} of the block of text that answers the prompt on a scale from {} to {}. ' \
        'Only give me the score without an explanation. Here is the block of text: {}' \
        .format(prompt_answer, metric, LOW_SCORE, HIGH_SCORE, prompt_reply))
    print('')