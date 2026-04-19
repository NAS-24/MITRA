# gen_ai/api.py
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import os

# Import your custom modules
from utils import ContextAssembler
from narrator import MitraNarrator

app = FastAPI(
    title="MITRA GenAI & Interaction API",
    description="Transforms face recognition data into empathetic, human-like voice interactions.",
    version="1.0.0"
)

# Allow frontend HTML file to call this API from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Mount the 'static' directory so the Frontend can access the .mp3 files
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# 2. Initialize our core classes
assembler = ContextAssembler()
narrator = MitraNarrator()

# 3. Define the exact JSON structure we expect from the Backend (The Contract)
class RecognitionEvent(BaseModel):
    person_id: str
    name: str
    relationship: str
    last_met_timestamp: str
    notes: str
    confidence_score: float
    is_unknown: bool
    preferred_language: Optional[str] = Field(default="English", description="e.g., 'English', 'Hindi'")

# 4. The main POST endpoint
@app.post("/generate_interaction")
async def generate_interaction(event: RecognitionEvent):
    try:
        # A. Convert Pydantic model to a standard dictionary
        backend_data = event.model_dump()

        # B. Assemble the context and calculate the "time ago" logic
        payload = assembler.prepare_llm_payload(backend_data)

        # C. Extract the preferred language
        target_lang = backend_data.get("preferred_language", "English")

        # D. Generate the empathetic text (Groq / Llama 3)
        narration_text = narrator.generate_text(payload, target_language=target_lang)

        # E. Generate the Audio (Deepgram / ElevenLabs)
        audio_filename = f"static/audio_{event.person_id}.mp3"
        audio_path = narrator.generate_audio(narration_text, output_filename=audio_filename)

        # F. Format the context clue safely for the UI Card
        context_clue = "Recently"
        if payload.get("context") and "last_met" in payload["context"]:
            context_clue = f"Last met: {payload['context']['last_met']}"

        # G. Return the perfectly structured JSON to the Frontend
        return {
            "status": "success",
            "narration_text": narration_text,
            "display_card": {
                "title": event.name,
                "subtitle": event.relationship,
                "context_clue": context_clue
            },
            "voice_engine": narrator.active_provider.capitalize(),
            "audio_url": f"http://localhost:5000/{audio_path}" if audio_path else None
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the server locally
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting MITRA GenAI Module on port 5000...")
    uvicorn.run("api:app", host="0.0.0.0", port=5000, reload=True)