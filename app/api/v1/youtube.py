from fastapi import APIRouter
router = APIRouter()

@router.get("/")
async def youtube_root():
    return {"message": "YouTube service OK"}