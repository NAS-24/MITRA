from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pipeline import process_image

app = FastAPI()

@app.post("/upload-image/")
async def upload_image(
    file: UploadFile = File(...),
    name: str = Form(None),
    relationship: str = Form(None),
    notes: str = Form(None)
):
    frame = await file.read()

    
    name = name.strip() if name else None
    relationship = relationship.strip() if relationship else None
    notes = notes.strip() if notes else None

    print("DEBUG INPUT:", name, relationship, notes, flush=True)

    result = process_image(
        frame=frame,   
        name=name,
        relationship=relationship,
        notes=notes
    )

    
    if result is None:
        return JSONResponse(
            content={"status": "error", "message": "Pipeline returned None"},
            status_code=500
        )

    if result.get("status") == "error":
        return JSONResponse(content=result, status_code=400)

    return JSONResponse(content=result, status_code=200)