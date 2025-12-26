# main.py
from fastapi import FastAPI
from app.models import hrms_alert
from app.core.database import Base, engine
import os
from fastapi.staticfiles import StaticFiles
from app.api.v1 import whatsapp, youtube, admin, health, resume_builder, prompt, suggestion_api, hrms, yura_chat_api
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv


load_dotenv()

app = FastAPI(title="Aryu Academy AI Bot", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],          # Allow GET, POST, PUT, DELETE
    allow_headers=["*"],          # Allow all headers
)

@app.on_event("startup")
async def startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


app.include_router(whatsapp.router, prefix="/api/v1/whatsapp", tags=["WhatsApp"])
app.include_router(youtube.router, prefix="/api/v1/youtube", tags=["YouTube"])
app.include_router(admin.router, prefix="/api/v1/admin", tags=["Admin"])
app.include_router(health.router, prefix="/api/v1/health", tags=["Health"])
app.include_router(prompt.router, prefix="/api/v1/prompts", tags=["Prompts"])
app.include_router(resume_builder.router, prefix="/api/v1/resume", tags=["Resume Builder"])
app.include_router(suggestion_api.router, prefix="/api/v1/suggest", tags=["Suggestions AI"])
app.include_router(hrms.router)
app.include_router(yura_chat_api.router)


# Serve static files from app/static
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


