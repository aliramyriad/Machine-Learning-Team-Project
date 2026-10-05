import base64
import json

from requests import post, get
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()  

# Before, running, ensure CLIENT_ID and CLIENT_SECRET are in .env! 
client_id = os.getenv('CLIENT_ID')
client_secret = os.getenv("CLIENT_SECRET")

# Generate spotify access token from user credentials. Expires in 1 hour.
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

token = get_token()
print(token)


# Getting a track (working)
url = "https://api.spotify.com/v1/tracks/2TpxZ7JUBn3uw46aR7qd6V"
headers = {
    "Authorization": "Bearer " + get_token(), }
response = get(url, headers=headers)
print(response.json())


# Getting track features (deprecated, not working :[ )
url = "https://api.spotify.com/v1/audio-features/2TpxZ7JUBn3uw46aR7qd6V"
headers = {
    "Authorization": "Bearer " + get_token(), }
response = get(url, headers=headers)
print(response.json())
