import os
import logging
import traceback

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from dotenv import load_dotenv

from app.api.v1.resume_builder import resume_builder
from app.modules.ats_scanner import router as ats_routes
from app.modules.voice_agent.router import router as voice_agent_router
from app.modules.voice_agent.scheduler import setup_scheduler
from app.modules.voice_agent import database as db
from app.core.database import Base, engine
from app.api.v1 import (
    youtube, admin, health,
    prompt, suggestion_api, hrms, yura_chat_api,
)
from app.modules.whatsapp_bot.router import router as whatsapp_router

load_dotenv()

# Determine environment (default to production for safety)
ENVIRONMENT = os.getenv("ENVIRONMENT", "production").lower()
IS_PRODUCTION = ENVIRONMENT == "production"

logging.basicConfig(
    level=logging.INFO if IS_PRODUCTION else logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ],
    force=True
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# LIFESPAN MANAGEMENT (Replaces @app.on_event("startup"))
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Initialize Database Tables safely
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        
    # 2. Voice Agent DB & Scheduler
    await db.init_db()
    scheduler = setup_scheduler()
    scheduler.start()
 
    yield # App runs here
 
    # 3. Graceful Shutdown
    scheduler.shutdown(wait=False)

# ─────────────────────────────────────────────────────────────────────────────
# APP INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────
# Only expose Swagger/Redoc docs if we are explicitly NOT in production
app = FastAPI(
    title="Aryu Academy AI Bot", 
    version="1.0.0",
    docs_url=None if IS_PRODUCTION else "/docs",
    redoc_url=None if IS_PRODUCTION else "/redoc",
    openapi_url=None if IS_PRODUCTION else "/openapi.json",
    lifespan=lifespan
)

# ─────────────────────────────────────────────────────────────────────────────
# SECURITY MIDDLEWARE
# ─────────────────────────────────────────────────────────────────────────────

# 1. Trusted Host Middleware (Prevents HTTP Host Header attacks)
ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "localhost,127.0.0.1,passats.aryuacademy.com,ai.aryuacademy.com").split(",")
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=[host.strip() for host in ALLOWED_HOSTS]
)

# 2. CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://passats.aryuacademy.com",
        "https://ai.aryuacademy.com",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE","OPTIONS"],
    allow_headers=[
        "Authorization",
        "Content-Type",
        "Accept",
        "Origin",
        "X-Requested-With",
    ],
)

# 3. Custom Security Headers Middleware
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        # Prevents browsers from sniffing MIME types (Stops XSS via image uploads)
        response.headers["X-Content-Type-Options"] = "nosniff"
        # Prevents Clickjacking by disallowing iframes from external sites
        response.headers["X-Frame-Options"] = "DENY"
        # Enables Cross-Site Scripting filter built into browsers
        response.headers["X-XSS-Protection"] = "1; mode=block"
        # Forces browsers to use HTTPS exclusively
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        # Restricts how much referral info is passed when routing away
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response

app.add_middleware(SecurityHeadersMiddleware)

# ─────────────────────────────────────────────────────────────────────────────
# EXCEPTION HANDLERS
# ─────────────────────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    CRITICAL: Catch-all for unhandled 500 errors. 
    Ensures Python tracebacks NEVER leak to the frontend in production.
    """
    logger.error(f"Unhandled Exception on {request.method} {request.url.path}: {exc}")
    if not IS_PRODUCTION:
        traceback.print_exc() # Print to console for local debugging
        
    return JSONResponse(
        status_code=500,
        content={"success": False, "message": "An internal server error occurred. Please try again later."}
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Safely decodes raw bytes (e.g. from MS Word pastes) to prevent 500 crashes."""
    safe_errors = []
    for error in exc.errors():
        safe_error = {}
        for key, value in error.items():
            if isinstance(value, bytes):
                safe_error[key] = value.decode("utf-8", errors="replace")
            elif isinstance(value, str):
                safe_error[key] = value.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
            else:
                safe_error[key] = value
        safe_errors.append(safe_error)

    logger.warning(f"Validation error on {request.method} {request.url.path}: {len(safe_errors)} error(s)")
    return JSONResponse(status_code=422, content={"detail": safe_errors})

@app.exception_handler(UnicodeDecodeError)
async def unicode_decode_error_handler(request: Request, exc: UnicodeDecodeError):
    logger.warning(f"UnicodeDecodeError on {request.method} {request.url.path}: {exc}")
    return JSONResponse(
        status_code=400,
        content={"detail": "Request body contains non-UTF-8 characters. Please ensure your resume text is UTF-8 encoded."}
    )

# ─────────────────────────────────────────────────────────────────────────────
# ROUTERS
# ─────────────────────────────────────────────────────────────────────────────

app.include_router(whatsapp_router,       prefix="/api/v1/whatsapp",  tags=["WhatsApp"])
app.include_router(youtube.router,        prefix="/api/v1/youtube",   tags=["YouTube"])
app.include_router(admin.router,          prefix="/api/v1/admin",     tags=["Admin"])
app.include_router(health.router,         prefix="/api/v1/health",    tags=["Health"])
app.include_router(prompt.router,         prefix="/api/v1/prompts",   tags=["Prompts"])
app.include_router(resume_builder.router, prefix="/api/v1/resume",    tags=["Resume Builder"])
app.include_router(voice_agent_router,    prefix="/api/v1/voice",     tags=["Voice Agent"])
app.include_router(suggestion_api.router, prefix="/api/v1/suggest",   tags=["Suggestions AI"])
app.include_router(hrms.router)
app.include_router(yura_chat_api.router)
app.include_router(ats_routes.router,     prefix="/api/v1/ats",       tags=["ATS"])

# ─────────────────────────────────────────────────────────────────────────────
# STATIC FILES
# ─────────────────────────────────────────────────────────────────────────────

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
# Warning: Ensure no sensitive dotfiles (.env, .git) are ever placed in the static folder.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
