from flask import Flask, request, redirect, url_for, render_template, session, json, send_from_directory, Response
import os
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from zipfile import ZipFile
from redis import Redis
import string
import random
import time
import threading
from datetime import datetime, timedelta
import logging
from logging.handlers import RotatingFileHandler
from src.Chatbot import OpenAIChatSession


root = os.path.abspath('../')
with open(os.path.join(root, 'data/user_demographics.json'), 'r') as infile:
    demo = json.load(infile)
args = {'mode': 'interest-based', 'ad_freq': 1.0, 'demo': demo}
# args = {'mode': 'control', 'ad_freq': 0.0, 'demo': demo}
ROOT = os.path.abspath('.')
UPLOAD_FOLDER = os.path.join(ROOT, 'static', 'temp')
EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)


class RunThread(threading.Thread):
    def __init__(self):
        self.oai = OpenAIChatSession(mode=args['mode'], ad_freq=args['ad_freq'], demographics=args['demo'])
        super().__init__()

    def run(self, prompt):
        yield self.oai.run_chat_live(prompt)


@app.route('/api', methods=['GET', 'POST'])
def main():
    print(request)
    if 'error' in session:
        err = session['error']
        session.pop('error')
        return err
    if request.method == 'POST':
        prompts = json.loads(request.data)
        print(prompts)
        prompt = prompts['message']
        oai = OpenAIChatSession(mode=args['mode'], ad_freq=args['ad_freq'], demographics=args['demo'], session=prompts['session_key'], conversation_id=prompts['conversation_id'])
        return oai.run_chat_live(prompt['content'])
        
        # run_thread = RunThread()
        # return run_thread.run(prompt)
    else:
        return [{ 'id': 'gpt-3.5-turbo', 'name': 'GPT-3.5' },{ 'id': 'gpt-4', 'name': 'GPT-4' }]
        # prompt = request.form
        # return oai.run_chat_live(prompt['prompt'])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4444)
