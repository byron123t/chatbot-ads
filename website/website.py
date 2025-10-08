from flask import Flask, request, redirect, url_for, render_template, session, json, send_from_directory, Response, g
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
import uuid
import base64


# Rate limit config
RATE_WINDOW_SEC = 50000         # rolling window length
RATE_LIMIT = 20             # max requests allowed per IP in the window
RATE_BURST = None            # optional hard cap for instantaneous bursts (None disables)
TRUST_X_FORWARDED_FOR = True # set True if behind a proxy / load balancer
RL_PREFIX = "rl"             # redis key prefix

root = os.path.abspath('../')
ROOT = os.path.abspath('.')
UPLOAD_FOLDER = os.path.join(ROOT, 'static', 'temp')
EXTENSIONS = {'png', 'jpg', 'jpeg'}
r = Redis(host='localhost', port=6379, password='', decode_responses=True)
SESSIONKEYMODEMAP = {'interestads_gpt4omini_transparent': {'mode': 'adaptive', 'model': 'o4-mini', 'ad_freq': 1.0, 'ad_transparency': 'disclosure', 'self_improvement': 1, 'feature_manipulation': True}, 'interestads_gpt4omini_none': {'mode': 'adaptive', 'model': 'o4-mini', 'ad_freq': 1.0, 'ad_transparency': 'none', 'self_improvement': 1, 'feature_manipulation': True}}

app = Flask(__name__)
CORS(app)


def new_uuid(short: bool = False) -> str:
    """
    Generate a random UUID (v4).
    - short=False -> standard 36-char string (e.g., '550e8400-e29b-41d4-a716-446655440000')
    - short=True  -> URL-safe 22-char base64 (no padding), good for URLs/ids
    """
    u = uuid.uuid4()
    if short:
        return base64.urlsafe_b64encode(u.bytes).rstrip(b'=').decode('ascii')
    return str(u)


def get_client_ip():
    if TRUST_X_FORWARDED_FOR:
        # The left-most IP is the original client
        xff = request.headers.get('X-Forwarded-For', '')
        if xff:
            return xff.split(',')[0].strip()
    return request.remote_addr or 'unknown'



def rate_limit_ip(ip: str, limit: int = RATE_LIMIT, window: int = RATE_WINDOW_SEC):
    """
    Sliding window rate limiter using a Redis ZSET per IP.
    Keeps only timestamps in the last `window` seconds.
    Returns (allowed: bool, remaining: int, reset_in: int, count: int)
    """
    now_ms = int(time.time() * 1000)
    window_ms = window * 1000
    key = f"{RL_PREFIX}:ip:{ip}"

    pipe = r.pipeline()
    # 1) drop old entries
    pipe.zremrangebyscore(key, 0, now_ms - window_ms)
    # 2) add current hit
    pipe.zadd(key, {str(now_ms): now_ms})
    # 3) get current count
    pipe.zcard(key)
    # 4) set TTL so idle keys go away
    pipe.expire(key, window + 5)
    _, _, count, _ = pipe.execute()

    remaining = max(0, limit - count)
    allowed = count <= limit

    # Compute reset_in (seconds until we’re safely back under limit)
    # Peek at the oldest timestamp (min score) to know when it falls out of window
    reset_in = 0
    if not allowed:
        oldest = r.zrange(key, 0, 0, withscores=True)
        if oldest:
            oldest_ts = int(oldest[0][1])  # score is the ms timestamp
            reset_in = max(0, int((oldest_ts + window_ms - now_ms) / 1000))

    return allowed, remaining, reset_in, count


def rate_limit_burst(ip: str, max_per_sec: int):
    if not max_per_sec:
        return True, max_per_sec
    sec_key = f"{RL_PREFIX}:burst:{ip}:{int(time.time())}"
    # atomic increment; first time this second -> expire in 2s
    new_val = r.incr(sec_key)
    if new_val == 1:
        r.expire(sec_key, 2)
    return new_val <= max_per_sec, max_per_sec - new_val


@app.before_request
def _global_rate_limiter():
    # Only count POST /api calls
    if request.method != "POST":
        return
    # Accept /api (and ignore other endpoints incl. /static, /healthz, etc.)
    if (request.path or "").rstrip("/") != "/api":
        return

    ip = get_client_ip()

    # Sliding window (primary)
    allowed, remaining, reset_in, count = rate_limit_ip(ip)

    # Optional burst (extra protection)
    if allowed and RATE_BURST:
        allowed_burst, _ = rate_limit_burst(ip, RATE_BURST)
        allowed = allowed and allowed_burst
        if not allowed and reset_in == 0:
            reset_in = 1

    # Headers only for /api
    headers = {
        "X-RateLimit-Limit": str(RATE_LIMIT),
        "X-RateLimit-Remaining": str(max(0, remaining)),
        "X-RateLimit-Window": f"{RATE_WINDOW_SEC}s",
    }
    if not allowed:
        headers["Retry-After"] = str(max(1, reset_in))
        return Response(["Too Many Requests"], status=429, headers=headers)

    # Attach headers so after_request can add them
    g.rate_limit_headers = headers


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
            r.set('SESSIONKEY_VARIABLEMODE_MAPPER', json.dumps({'interestads_gpt4omini_transparent': [], 'interestads_gpt4omini_none': [], 'control_gpt4omini': [], 'incorrect_session_key': []}))
        data = json.loads(r.get('SESSIONKEY_VARIABLEMODE_MAPPER'))
        found = False
        for mode, session_keys in data.items():
            if prompts['session_key'] in session_keys:
                found = True
                break
        if found:
            kwargs = SESSIONKEYMODEMAP[mode].copy()
        else:
            if prompts['session_key'].strip().lower().startswith('chatbotrtcl'):
                mode = 'interestads_gpt4omini_transparent'
            elif prompts['session_key'].strip().lower().startswith('nosponsor'):
                mode = 'interestads_gpt4omini_none'
            else:
                mode = 'incorrect_session_key'
                return None
            kwargs = SESSIONKEYMODEMAP[mode].copy()
            if mode not in data:
                data[mode] = []
            data[mode].append(prompts['session_key'])
            r.set('SESSIONKEY_VARIABLEMODE_MAPPER', json.dumps(data))
        print(mode)
        print(SESSIONKEYMODEMAP)
        prompt = prompts['message']
        kwargs['conversation_id'] = prompts['conversation_id']
        kwargs['session'] = prompts['session_key']
        oai = OpenAIChatSession(**kwargs)
        return oai.run_chat_live(prompt['content'])
    else:
        return [{ 'id': 'gpt-3.5-turbo', 'name': 'GPT-3.5' },{ 'id': 'gpt-4', 'name': 'GPT-4' }]


@app.route('/disclosure', methods=['GET', 'POST'])
def disclosure():
    print(request)
    disclosures_clicked = 0
    if 'error' in session:
        err = session['error']
        session.pop('error')
        return err
    if request.method == 'POST':
        prompts = json.loads(request.data)
        if prompts['mode'] == 'disclosuretracking':
            if r.exists(prompts['session_key']):
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
            print(prompts['session_key'])
            print(r.hget(prompts['session_key'], 'products'))
            products = json.loads(r.hget(prompts['session_key'], 'products'))
            products.reverse()
            print(products)
            return products
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
