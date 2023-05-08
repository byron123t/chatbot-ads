from src.Chatbot import OpenAIChatSession
# from src.API import OpenAIAPI
import json, os, argparse, openai

LOW_SCORE = 0
HIGH_SCORE = 10

prompt_answer = 'hi'
prompt_reply = 'hi'

oai = OpenAIChatSession()
oai.run_chat('I am going to give you a block of text. The block of text answers the prompt "{}". ' \
    'Give me a score of how relevant the block of text answers the prompt on a scale from {} to {}. ' \
    'Only give me the score without an explanation. Here is the block of text: {}' \
    .format(prompt_answer, LOW_SCORE, HIGH_SCORE, prompt_reply))