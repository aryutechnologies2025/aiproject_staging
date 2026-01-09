from fastapi import APIRouter
from app.schemas.ats_schema import (
    ATSScanRequest,
    ATSScanResponse
)
from app.services.ats_service import scan_resume_with_ai

router = APIRouter()


@router.post("/scan", response_model=ATSScanResponse)
async def ats_scan(payload: ATSScanRequest):
    return await scan_resume_with_ai(payload)
