import os
import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from app.api.v1.resume_builder import resume_builder
from app.modules.ats_scanner import router as ats_routes
from app.modules.voice_agent.router import router as voice_agent_router
from app.modules.voice_agent.scheduler import setup_scheduler
from app.modules.voice_agent import database as db
from app.core.database import Base, engine
from app.api.v1 import (
    whatsapp, youtube, admin, health,
    prompt, suggestion_api, hrms, yura_chat_api,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ],
    force=True
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Aryu Academy AI Bot", version="1.0.0")

# app = FastAPI(
#     docs_url=None,
#     redoc_url=None,
#     openapi_url=None
# )

# ─────────────────────────────────────────────────────────────────────────────
# CORS
# ─────────────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://passats.aryuacademy.com",
        "https://ai.aryuacademy.com",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=[
        "Authorization",
        "Content-Type",
        "Accept",
        "Origin",
        "X-Requested-With",
    ],
)

# ─────────────────────────────────────────────────────────────────────────────
# EXCEPTION HANDLERS
# ─────────────────────────────────────────────────────────────────────────────

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Custom 422 handler — replaces FastAPI's default.

    Problem:  FastAPI's default encoder calls bytes.decode() with no encoding
              argument (assumes UTF-8). When a request body contains non-UTF-8
              bytes — e.g. Latin-1 ö (0xf6) pasted from Word or Outlook —
              the encoder crashes with UnicodeDecodeError and returns a 500
              instead of the intended 422 validation error.

    Fix:      Iterate the error list ourselves and safely decode any raw bytes
              using errors="replace" so no byte sequence can crash the handler.
    """
    safe_errors = []
    for error in exc.errors():
        safe_error = {}
        for key, value in error.items():
            if isinstance(value, bytes):
                safe_error[key] = value.decode("utf-8", errors="replace")
            elif isinstance(value, str):
                # Re-encode/decode to strip any stray surrogate characters
                safe_error[key] = value.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
            else:
                safe_error[key] = value
        safe_errors.append(safe_error)

    logger.warning(
        f"Request validation error on {request.method} {request.url.path}: "
        f"{len(safe_errors)} error(s)"
    )

    return JSONResponse(
        status_code=422,
        content={"detail": safe_errors},
    )


@app.exception_handler(UnicodeDecodeError)
async def unicode_decode_error_handler(request: Request, exc: UnicodeDecodeError):
    """
    Catch any UnicodeDecodeError that bubbles up from body reading or
    middleware before it reaches our validation handler.
    Typically caused by non-UTF-8 encoded resume/JD text from Word or Outlook.
    """
    logger.warning(
        f"UnicodeDecodeError on {request.method} {request.url.path}: {exc}"
    )
    return JSONResponse(
        status_code=400,
        content={
            "detail": (
                "Request body contains non-UTF-8 characters. "
                "Please ensure your resume text is UTF-8 encoded. "
                "If pasting from Microsoft Word or Outlook, try pasting "
                "into Notepad first to strip special characters."
            )
        },
    )

@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init_db()
 
    scheduler = setup_scheduler()
    scheduler.start()
 
    yield
 
    scheduler.shutdown(wait=False)

# ─────────────────────────────────────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# ─────────────────────────────────────────────────────────────────────────────
# ROUTERS
# ─────────────────────────────────────────────────────────────────────────────

app.include_router(whatsapp.router,       prefix="/api/v1/whatsapp",  tags=["WhatsApp"])
app.include_router(youtube.router,        prefix="/api/v1/youtube",   tags=["YouTube"])
app.include_router(admin.router,          prefix="/api/v1/admin",     tags=["Admin"])
app.include_router(health.router,         prefix="/api/v1/health",    tags=["Health"])
app.include_router(prompt.router,         prefix="/api/v1/prompts",   tags=["Prompts"])
app.include_router(resume_builder.router, prefix="/api/v1/resume",    tags=["Resume Builder"])
app.include_router(voice_agent_router,   prefix="/api/v1/voice",     tags=["Voice Agent"])
app.include_router(suggestion_api.router, prefix="/api/v1/suggest",   tags=["Suggestions AI"])
app.include_router(hrms.router)
app.include_router(yura_chat_api.router)
app.include_router(ats_routes.router,     prefix="/api/v1/ats",       tags=["ATS"])

# ─────────────────────────────────────────────────────────────────────────────
# STATIC FILES
# ─────────────────────────────────────────────────────────────────────────────

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")