import base64
import json

import requests
from requests import post
import os


client_id = "113b4d69bfd245cd80c330f6ca1825d3"
client_secret = "58a6b3b91a6f4baf92c796a967be2e5d"

def get_token():
    auth_string = client_id + ":" + client_secret
    auth_bytes = auth_string.encode("utf-8")
    auth_base64 = str(base64.b64encode(auth_bytes), encoding="utf-8")

    url = "https://accounts.spotify.com/api/token"
    headers = {
        "Authorization": "Basic " + auth_base64,
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = { "grant_type": "client_credentials"}
    result = post(url, headers=headers, data=data)
    json_result = json.loads(result.content)
    token = json_result["access_token"]
    return token

# token = get_token()
# print(token)


url = "https://api.spotify.com/v1/tracks/2TpxZ7JUBn3uw46aR7qd6V"

headers = {
    "Authorization": "Bearer " + get_token(), }

response = requests.get(url, headers=headers)

print(response.json())