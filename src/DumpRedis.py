import redis
import json

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, db=0)

# Fetch all keys
keys = r.keys('*')

# Prepare a dictionary to hold your Redis data
redis_data = {}

# Loop through the keys to fetch values
variables = r.get('SESSIONKEY_VARIABLEMODE_MAPPER')
for key in keys:
    # Assuming all values are strings; adjust as necessary for other types
    print(key)
    if 'SESSIONKEY_VARIABLEMODE_MAPPER' in str(key):
        continue
    chat_history = r.hget(key, 'chat_history')
    response_history = r.hget(key, 'response_history')
    current_conversation = r.hget(key, 'current_conversation')
    user_history = r.hget(key, 'user_history')
    mode = r.hget(key, 'mode')
    demographics = r.hget(key, 'demographics')
    product = r.hget(key, 'product')
    disclosures = r.hget(key, 'disclosures')
    linkclicks = r.hget(key, 'linkclicks')
    profiles = r.hget(key, 'profile')
    topic = r.hget(key, 'topic')
    products = r.hget(key, 'products')
    str_key = key.decode("utf-8")
    redis_data[str_key] = {}
    if chat_history:
        redis_data[str_key]['chat_history'] = json.loads(chat_history)
    if response_history:
        redis_data[str_key]['response_history'] = json.loads(response_history)
    if current_conversation:
        redis_data[str_key]['current_conversation'] = json.loads(current_conversation)
    if user_history:
        redis_data[str_key]['user_history'] = json.loads(user_history)
    if mode:
        redis_data[str_key]['mode'] = json.loads(mode)
    if demographics:
        redis_data[str_key]['demographics'] = json.loads(demographics)
    if product:
        redis_data[str_key]['product'] = json.loads(product)
    if topic:
        redis_data[str_key]['topic'] = json.loads(topic)
    if disclosures:
        redis_data[str_key]['disclosures'] = json.loads(disclosures)
    if linkclicks:
        redis_data[str_key]['linkclicks'] = json.loads(linkclicks)
    if profiles:
        redis_data[str_key]['profiles'] = json.loads(profiles)
    if products:
        redis_data[str_key]['products'] = json.loads(products)
    redis_data['SESSIONKEY_VARIABLEMODE_MAPPER'] = json.loads(variables)
    

# Dump data to a JSON file
with open('redis_data.json', 'w') as jsonfile:
    json.dump(redis_data, jsonfile, indent=4)

print("Data dumped to redis_data.json")