import os
import requests
from dotenv import load_dotenv

# Load your ElevenLabs key from the .env file
load_dotenv()
api_key = os.getenv("ELEVENLABS_API_KEY")

url = "https://api.elevenlabs.io/v1/voices"
headers = {"xi-api-key": api_key}

print("📡 Fetching your authorized voices from ElevenLabs...\n")
response = requests.get(url, headers=headers)

if response.status_code == 200:
    voices = response.json().get("voices", [])
    for voice in voices:
        print(f"👤 Name: {voice['name']}")
        print(f"🔑 ID: {voice['voice_id']}")
        print("-" * 30)
else:
    print(f"❌ API Blocked Us! Error: {response.text}")