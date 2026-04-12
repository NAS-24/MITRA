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
    image_bytes = await file.read()

    result = process_image(
        image_bytes,
        name=name,
        relationship=relationship,
        notes=notes
    )

    if "error" in result:
        return JSONResponse(content=result, status_code=400)

    return JSONResponse(content=result)