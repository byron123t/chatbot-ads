from flask import Flask, request, redirect, url_for, render_template, session, json, send_from_directory, Response
from flask_cors import CORS
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
from redis import Redis


root = os.path.abspath('../')
ROOT = os.path.abspath('.')
UPLOAD_FOLDER = os.path.join(ROOT, 'static', 'temp')
EXTENSIONS = {'png', 'jpg', 'jpeg'}
r = Redis(host='localhost', port=6379, password='', decode_responses=True)
SESSIONKEYMODEMAP = {'interestads_gpt4_transparent': {'mode': 'user-centric', 'model': 'gpt-4o', 'ad_freq': 1.0, 'ad_transparency': 'disclosure', 'self_improvement': 3, 'feature_manipulation': True}, 'interestads_gpt4_none': {'mode': 'user-centric', 'model': 'gpt-4o', 'ad_freq': 1.0, 'self_improvement': 3, 'feature_manipulation': True}, 'interestads_gpt35_transparent': {'mode': 'user-centric', 'model': 'gpt-3.5-turbo', 'ad_freq': 1.0, 'ad_transparency': 'disclosure', 'self_improvement': 3, 'feature_manipulation': True}, 'interestads_gpt35_none': {'mode': 'user-centric', 'model': 'gpt-3.5-turbo', 'ad_freq': 1.0, 'self_improvement': 3, 'feature_manipulation': True}, 'control_gpt4': {'mode': 'control', 'model': 'gpt-4o', 'ad_freq': 0.0}, 'control_gpt35': {'mode': 'control', 'model': 'gpt-3.5-turbo', 'ad_freq': 0.0}}

app = Flask(__name__)
CORS(app)


@app.route('/api', methods=['GET', 'POST'])
def api():
    print(request)
    if 'error' in session:
        err = session['error']
        session.pop('error')
        return err
    if request.method == 'POST':
        prompts = json.loads(request.data)
        print(prompts)
        if not r.exists('SESSIONKEY_VARIABLEMODE_MAPPER'):
            r.set('SESSIONKEY_VARIABLEMODE_MAPPER', json.dumps({'interestads_gpt4_transparent': [], 'interestads_gpt4_none': [], 'interestads_gpt35_transparent': [], 'interestads_gpt35_none': [], 'control_gpt4': [], 'control_gpt35': []}))
        data = json.loads(r.get('SESSIONKEY_VARIABLEMODE_MAPPER'))
        found = False
        for mode, session_keys in data.items():
            if prompts['session_key'] in session_keys:
                found = True
                break
        if found:
            kwargs = SESSIONKEYMODEMAP[mode].copy()
        else:
            lengths = {k: len(v) for k, v in data.items()}
            mode = min(lengths, key=lengths.get)
            kwargs = SESSIONKEYMODEMAP[mode].copy()
            data[mode].append(prompts['session_key'])
            r.set('SESSIONKEY_VARIABLEMODE_MAPPER', json.dumps(data))
        print(mode)
        print(SESSIONKEYMODEMAP)
        prompt = prompts['message']
        kwargs['session'] = prompts['session_key']
        kwargs['conversation_id'] = prompts['conversation_id']
        oai = OpenAIChatSession(**kwargs)
        return oai.run_chat_live(prompt['content'])
    else:
        return [{ 'id': 'gpt-3.5-turbo', 'name': 'GPT-3.5' },{ 'id': 'gpt-4', 'name': 'GPT-4' }]


@app.route('/disclosure', methods=['GET', 'POST'])
def disclosure():
    print(request)
    if 'error' in session:
        err = session['error']
        session.pop('error')
        return err
    if request.method == 'POST':
        prompts = json.loads(request.data)
        if prompts['mode'] == 'disclosuretracking':
            if r.exists(prompts['session_key']):
                disclosures_clicked = 0
                if r.exists(prompts['session_key'], 'disclosures'):
                    disclosures_clicked = r.hget(prompts['session_key'], 'disclosures')
                if not disclosures_clicked:
                    disclosures_clicked = 0
                else:
                    disclosures_clicked = int(disclosures_clicked)
                disclosures_clicked += 1
                r.hset(prompts['session_key'], 'disclosures', disclosures_clicked)
            print(disclosures_clicked)
            return ['DisclosureDone']
        elif prompts['mode'] == 'products':
            products = json.loads(r.hget(prompts['session_key'], 'product'))
            conversation_data = json.loads(prompts['conversation_id'])
            for product_id, product in products.items():
                if product_id == conversation_data['id']:
                    print(product)
                    return product
        elif prompts['mode'] == 'profile':
            try:
                profile = json.loads(r.hget(prompts['session_key'], 'profile'))
            except Exception:
                profile = ['No Profile Generated Yet']
            return profile
    else:
        return ['Test for disclosure']


@app.route('/linkclick', methods=['GET', 'POST'])
def linkclick():
    print(request)
    if 'error' in session:
        err = session['error']
        session.pop('error')
        return err
    if request.method == 'POST':
        prompts = json.loads(request.data)
        if r.exists(prompts['session_key']):
            if r.exists(prompts['session_key'], 'linkclicks'):
                linkclicks = r.hget(prompts['session_key'], 'linkclicks')
                if not linkclicks:
                    linkclicks = {}
                else:
                    linkclicks = json.loads(linkclicks)
            else:
                linkclicks = {}
            if prompts['message'] in linkclicks:
                linkclicks[prompts['message']] += 1
            else:
                linkclicks[prompts['message']] = 1
            r.hset(prompts['session_key'], 'linkclicks', json.dumps(linkclicks))
        print(linkclicks)
        return ['LinkClickDone']
    else:
        return ['Test for linkclick']


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4444)
