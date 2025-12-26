from fastapi import APIRouter, UploadFile, File, HTTPException, Request
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

router = APIRouter()

BASE_PUBLIC_URL = os.getenv("BASE_PUBLIC_URL", "http://localhost:8000")  # set to your ngrok or domain

@router.post("/upload_syllabus")
async def upload_syllabus(request: Request, file: UploadFile = File(...), course_key: str = None):
    if not course_key:
        raise HTTPException(status_code=400, detail="course_key required")

    # validate filename + extension
    filename = Path(file.filename).name
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    # ensure storage path
    static_dir = Path(__file__).resolve().parents[1] / "static" / "syllabus"
    static_dir.mkdir(parents=True, exist_ok=True)

    save_path = static_dir / filename
    # Avoid overwriting unintended files; you can add unique suffix if needed
    with open(save_path, "wb") as f:
        contents = await file.read()
        f.write(contents)

    public_url = f"{BASE_PUBLIC_URL}/static/syllabus/{filename}"
    return {"url": public_url, "filename": filename, "course_key": course_key}
