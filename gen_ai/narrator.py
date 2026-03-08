# gen_ai/narrator.py
import os
import requests
from groq import Groq
from dotenv import load_dotenv
from prompts import MitraPrompts
import base64

load_dotenv()

class MitraNarrator:
    def __init__(self):
        # 1. The Brain: Free Groq API (Llama 3)
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.prompts = MitraPrompts()
        
        # 2. TTS Configuration
        self.active_provider = os.getenv("ACTIVE_TTS_PROVIDER", "deepgram").lower()
        self.elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")
        self.deepgram_key = os.getenv("DEEPGRAM_API_KEY")
        self.sarvam_key = os.getenv("SARVAM_API_KEY") 
        
        # Voice IDs
        self.elevenlabs_voice_id = "IKne3meq5aSn9XLyUdCD" # For male:- RpiHVNPKGBg7UmgmrKrN(Aashish), For Female:- 2zRM7PkgwBPiau2jvVXc(Monika Sogam)
        self.deepgram_voice_model = "aura-arcas-en"       # Deepgram's male professional voice (English only). Female:- "aura-asteria-en" or "aura-luna-en"

    def generate_text(self, context_payload, target_language="English"):
        """Generates the text using free cloud Llama 3 via Groq, supporting multiple languages."""
        
        # 1. Handle Human-in-the-loop Fallback
        if context_payload["status"] == "needs_confirmation":
            return self.prompts.get_confirmation_text(context_payload["name_guess"], target_language)
        
        # 2. Handle Unknown Person
        if context_payload["status"] == "unknown":
            if target_language.lower() == "hindi":
                return "मैं इस व्यक्ति को नहीं पहचान पा रहा हूँ। क्या आप चाहेंगे कि मैं इनका चेहरा सेव कर लूँ?"
            return "I don't recognize this person. Would you like me to save their face?"

        # 3. Generate normal contextual narration
        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant", 
                messages=[
                    {"role": "system", "content": self.prompts.get_system_persona(target_language)},
                    {"role": "user", "content": self.prompts.get_narration_prompt(context_payload["context"])}
                ],
                temperature=0.4,
                max_tokens=80 # Slightly higher to accommodate longer Hindi translations
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Groq API Error: {e}")
            return f"This is {context_payload['context']['name']}."

    def generate_audio(self, text, output_filename):
        """Routes the text to the active TTS provider."""
        if self.active_provider == "sarvam":
            return self._generate_audio_sarvam(text, output_filename)
        elif self.active_provider == "elevenlabs":
            return self._generate_elevenlabs(text, output_filename)
        elif self.active_provider == "deepgram":
            return self._generate_deepgram(text, output_filename)
        else:
            print(f"Error: Unknown TTS provider '{self.active_provider}'")
            return None

    def _generate_elevenlabs(self, text, output_filename):
        """Internal method for ElevenLabs TTS"""
        if not self.elevenlabs_key:
            print("No ElevenLabs key found.")
            return None

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.elevenlabs_voice_id}"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.elevenlabs_key
        }
        data = {
            "text": text,
            "model_id": "eleven_multilingual_v2", # Must use V2 for Hindi/Regional support
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}
        }

        try:
            response = requests.post(url, json=data, headers=headers)
            if response.status_code == 200:
                with open(output_filename, 'wb') as f:
                    f.write(response.content)
                return output_filename
            print(f"ElevenLabs Error: {response.text}")
        except Exception as e:
            print(f"ElevenLabs Crash: {e}")
        return None

    def _generate_deepgram(self, text, output_filename):
        """Internal method for Deepgram Aura TTS"""
        if not self.deepgram_key:
            print("No Deepgram key found.")
            return None

        url = f"https://api.deepgram.com/v1/speak?model={self.deepgram_voice_model}"
        headers = {
            "Authorization": f"Token {self.deepgram_key}",
            "Content-Type": "application/json"
        }
        data = {"text": text}

        try:
            response = requests.post(url, json=data, headers=headers)
            if response.status_code == 200:
                with open(output_filename, 'wb') as f:
                    f.write(response.content)
                return output_filename
            print(f"Deepgram Error: {response.text}")
        except Exception as e:
            print(f"Deepgram Crash: {e}")
        return None
    
    def _generate_audio_sarvam(self, text, output_filename):
        """Internal method for Sarvam AI TTS"""
        print("🎙️ Routing to Sarvam AI (Hindi)...")
        
        if not self.sarvam_key:
            print("❌ Sarvam API key missing in .env!")
            return None

        url = "https://api.sarvam.ai/text-to-speech"
        payload = {
            "inputs": [text], 
            "target_language_code": "hi-IN",
            "speaker": "shubh", # Sarvam
            "pace": 1.0,
            "speech_sample_rate": 8000,
            "enable_preprocessing": True,
            "model": "bulbul:v3"
        }
        headers = {
            "api-subscription-key": self.sarvam_key, 
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            
            # Sarvam returns the audio as a Base64 encoded string
            audio_base64 = response.json().get("audios", [])[0]
            audio_bytes = base64.b64decode(audio_base64)
            
            # We save it directly to the filename api.py gives us
            # Note: We replace .mp3 with .wav because Sarvam natively outputs wav data
            final_filename = output_filename.replace(".mp3", ".wav")
            
            with open(final_filename, "wb") as f:
                f.write(audio_bytes)
                
            return final_filename
            
        except Exception as e:
            print(f"❌ Sarvam Error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(e.response.text)
            return None

# --- Quick Test ---
if __name__ == "__main__":
    from utils import ContextAssembler
    from mock_data import MOCK_HIGH_CONFIDENCE
    
    assembler = ContextAssembler()
    narrator = MitraNarrator()
    
    payload = assembler.prepare_llm_payload(MOCK_HIGH_CONFIDENCE)
    
    # Let's test it in Hindi!
    test_language = "Hindi" 
    
    print(f"🧠 Generating Narration Text in {test_language}...")
    text = narrator.generate_text(payload, target_language=test_language)
    print(f"🗣️ Text: {text}")
    
    audio_path = narrator.generate_audio(text)
    
    if audio_path:
        print(f"✅ Success! Audio saved to: {audio_path}")