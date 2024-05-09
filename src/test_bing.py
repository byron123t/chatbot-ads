
import json
import os 
from pprint import pprint
import requests
from sensitive.objects import azure_api_key

'''
This sample makes a call to the Bing Web Search API with a query and returns relevant web search.
Documentation: https://docs.microsoft.com/en-us/bing/search-apis/bing-web-search/overview
'''

# Add your Bing Search V7 subscription key and endpoint to your environment variables.
subscription_key = azure_api_key
endpoint = "https://api.bing.microsoft.com/v7.0/images/search"

# Query term(s) to search for. 
query = "wallpaper"

# Construct a request
mkt = 'en-US'
params = { 'q': query, 'mkt': mkt }
headers = { 'Ocp-Apim-Subscription-Key': subscription_key }

# Call the API
try:
    response = requests.get(endpoint, headers=headers, params=params)
    response.raise_for_status()

    print("Headers:")
    print(response.headers)

    print("JSON Response:")
    pprint(response.json())
    for image in response.json()['queryExpansions']:
        print(image['text'], image['thumbnail']['thumbnailUrl'])
except Exception as ex:
    raise ex