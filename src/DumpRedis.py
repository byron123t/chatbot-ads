import redis
import json

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, db=0)

# Fetch all keys
keys = r.keys('*')

# Prepare a dictionary to hold your Redis data
redis_data = {}

# Loop through the keys to fetch values
for key in keys:
    # Assuming all values are strings; adjust as necessary for other types
    chat_history = r.hget(key, 'chat_history')
    response_history = r.hget(key, 'response_history')
    current_conversation = r.hget(key, 'current_conversation')
    user_history = r.hget(key, 'user_history')
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

# Dump data to a JSON file
with open('redis_data.json', 'w') as jsonfile:
    json.dump(redis_data, jsonfile, indent=4)

print("Data dumped to redis_data.json")