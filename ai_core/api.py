from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor
import asyncio
from pipeline import process_image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=2)

@app.post("/upload-image/")
async def upload_image(
    file: UploadFile = File(...),
    name: str = Form(None),
    relationship: str = Form(None),
    notes: str = Form(None)
):
    frame = await file.read()

    name         = (name or "").strip() or None
    relationship = (relationship or "").strip() or None
    notes        = (notes or "").strip() or None

    print("DEBUG INPUT:", name, relationship, notes, flush=True)

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        executor,
        lambda: process_image(frame=frame, name=name, relationship=relationship, notes=notes)
    )

    print("PIPELINE RAW RESULT:", result, flush=True)

    if result is None:
        return JSONResponse(
            content={"status": "error", "message": "Pipeline returned None"},
            status_code=500
        )

    # Unknown face — frontend needs is_unknown: true to show register form
    if "error" in result and "unknown" in str(result.get("error", "")).lower():
        return JSONResponse(content={
            "is_unknown": True,
            "person_id": "",
            "name": "",
            "relationship": "",
            "last_met_timestamp": "",
            "notes": "",
            "confidence_score": 0.0,
            "preferred_language": "English"
        }, status_code=200)

    # Known face — normalise whatever pipeline returns into the
    # exact schema that generate_interaction (port 5000) expects
    if result.get("name") or result.get("person_id"):
        return JSONResponse(content={
            "is_unknown":          False,
            "person_id":           result.get("person_id")          or result.get("id", ""),
            "name":                result.get("name",               ""),
            "relationship":        result.get("relationship",       ""),
            "last_met_timestamp":  result.get("last_met_timestamp") or result.get("last_met", ""),
            "notes":               result.get("notes",              ""),
            "confidence_score":    float(result.get("confidence_score") or result.get("confidence", 0.0)),
            "preferred_language":  result.get("preferred_language", "English")
        }, status_code=200)

    # Any other error from pipeline
    return JSONResponse(content=result, status_code=400)