"""
ATS Scanner Router v3
─────────────────────────────────────────────────────────────────────────────
FastAPI router for all ATS scanning endpoints.

Endpoints:
  POST /ats/scan            Full AI-powered ATS scan (JSON resume + optional JD)
  POST /ats/scan-file       Upload PDF/DOCX + scan (with optional JD)
  POST /ats/scan-quick      Rules-only scan (no AI, fast)
  GET  /ats/score/:score    Explain a specific ATS score
  GET  /ats/help            API documentation
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, validator
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.modules.ats_scanner.service import ATSScannerService, create_ats_scan
from app.modules.ats_scanner.utils.text_extraction import extract_text
from app.services.resume_builder_services.resume_parser_service import parse_resume_to_schema

logger = logging.getLogger(__name__)
router = APIRouter()

_scanner = ATSScannerService()   # singleton — avoids rebuilding skill DB per request


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST MODELS
# ─────────────────────────────────────────────────────────────────────────────

class ATSScanRequest(BaseModel):
    """
    Typed request model for /scan and /scan-quick.
    Using a Pydantic model instead of raw dict gives us:
      1. Clear 422 errors when the body is wrong (not a crash)
      2. A place to sanitise text fields before they hit the engine
    """
    resume_data:     Dict[str, Any]
    job_description: Optional[str] = None
    include_ai:      bool = True

    @validator("resume_data")
    def resume_must_not_be_empty(cls, v):
        if not v:
            raise ValueError("resume_data cannot be empty")
        has_content = any([
            v.get("name"), v.get("summary"),
            v.get("experience"), v.get("skills"),
            v.get("education"),
        ])
        if not has_content:
            raise ValueError(
                "resume_data appears empty. Provide at least: "
                "name, summary, experience, skills, or education."
            )
        return v

    @validator("job_description", pre=True, always=True)
    def sanitise_jd(cls, v):
        return _sanitise_text(v)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _sanitise_text(text: Optional[str]) -> Optional[str]:
    """
    Safely normalise any string that may contain non-UTF-8 bytes.
    Handles resumes/JDs pasted from Microsoft Word, Outlook, or
    non-English sources (e.g. ö, ü, é in Latin-1 encoding).

    Uses errors="replace" so nothing can crash the handler —
    unrecognised bytes become the Unicode replacement character (U+FFFD).
    """
    if not text:
        return text
    if isinstance(text, bytes):
        return text.decode("utf-8", errors="replace")
    return text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")


def _parsed_resume_to_dict(parsed) -> dict:
    """Convert parsed resume schema object to plain dict for the scanner."""
    if hasattr(parsed, "dict"):
        return parsed.dict()
    if hasattr(parsed, "__dict__"):
        return parsed.__dict__
    return dict(parsed)


def _validate_resume(data: dict) -> None:
    """Basic validation — at least one meaningful section must be present."""
    has_content = any([
        data.get("name"), data.get("summary"),
        data.get("experience"), data.get("skills"),
        data.get("education"),
    ])
    if not has_content:
        raise HTTPException(
            status_code=400,
            detail=(
                "resume_data appears empty. Please provide at least: "
                "name, summary, experience, skills, or education."
            ),
        )


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 1: Full AI-Powered Scan (JSON)
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/scan")
async def ats_scan(
    payload: ATSScanRequest,          # ← typed model; gives clean 422 on bad input
    db:      AsyncSession = Depends(get_db),
):
    """
    Full AI-powered ATS scan with Groq AI + rule-based analysis.

    Request body:
    {
        "resume_data": {
            "name":         "Jane Smith",
            "email":        "jane@example.com",
            "phone":        "+1 555-123-4567",
            "location":     "Boston, MA",
            "summary":      "...",
            "experience":   [{"title": "...", "company": "...", "bullets": ["..."]}],
            "education":    [{"degree": "...", "institution": "...", "year": "2020"}],
            "skills":       ["Python", "SQL", "Tableau"],
            "projects":     [...],
            "certifications": [...],
            "languages":    [...],
            "volunteer":    [...],
            "awards":       [...],
            "publications": [...],
            "hobbies":      [...]
        },
        "job_description": "Optional: paste the full job posting here",
        "include_ai": true
    }

    Returns:
        Comprehensive ATS analysis with score, section feedback,
        keyword gaps, AI rewrites, and improvement roadmap.
    """
    resume_data     = payload.resume_data
    job_description = payload.job_description     # already sanitised by validator
    include_ai      = payload.include_ai

    logger.info(
        f"ATS scan request — name={resume_data.get('name', 'Unknown')}, "
        f"has_jd={bool(job_description)}, include_ai={include_ai}"
    )

    try:
        result = await _scanner.scan(
            resume=resume_data,
            job_description=job_description or None,
            db=db,
            include_ai=bool(include_ai),
        )
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ATS scan error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"ATS scan failed: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 2: File Upload Scan (PDF / DOCX)
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/scan-file")
async def ats_scan_from_file(
    file:            UploadFile     = File(...),
    job_description: Optional[str] = Form(default=None),
    include_ai:      bool           = Form(default=True),
    db:              AsyncSession   = Depends(get_db),
):
    """
    Upload a PDF or DOCX resume and run a full ATS scan.

    Form fields:
      • file            — PDF or DOCX resume file
      • job_description — Optional: paste job posting text
      • include_ai      — Whether to run Groq AI analysis (default true)

    Returns: Same response format as POST /scan
    """
    filename = (file.filename or "").lower()

    if not filename.endswith((".pdf", ".docx")):
        raise HTTPException(
            status_code=400,
            detail="Only PDF and DOCX files are supported. Please convert your resume.",
        )

    # Sanitise job description — may contain Latin-1 bytes from Word / Outlook
    job_description = _sanitise_text(job_description)

    logger.info(f"ATS file scan: {file.filename}, has_jd={bool(job_description)}")

    # Step 1: Extract text
    try:
        text = await extract_text(file)
        if not text or len(text.strip()) < 50:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not extract readable text from the file. "
                    "Ensure the resume is not a scanned image-only PDF."
                ),
            )
        # Sanitise extracted text — PDFs from Windows may use Latin-1
        text = _sanitise_text(text)
        logger.info(f"Text extracted: {len(text)} characters")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Text extraction error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")

    # Step 2: Parse resume to schema
    try:
        file_type     = filename.split(".")[-1]
        parsed_resume = parse_resume_to_schema(text=text, file_type=file_type)
        resume_dict   = _parsed_resume_to_dict(parsed_resume)

        # ─── ADD THIS LINE ────────────────────────────────────────────────────
        # raw_text must be passed so _enrich_resume() in service.py can run
        # _recover_education_from_text() on Canva / two-column PDF resumes
        # where the structured parser returns education: []
        resume_dict["raw_text"] = text
        # ─────────────────────────────────────────────────────────────────────

        logger.info(f"Resume parsed: name='{resume_dict.get('name')}' | "
                    f"education_entries={len(resume_dict.get('education') or [])}")
    except Exception as e:
        logger.error(f"Resume parsing error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Failed to parse resume: {str(e)}")

    # Step 3: Run ATS scan
    try:
        result = await _scanner.scan(
            resume=resume_dict,
            job_description=job_description or None,
            db=db,
            include_ai=include_ai,
        )
        result["meta"] = {
            "source":      "file_upload",
            "source_file": file.filename,
            "parsed_name": resume_dict.get("name"),
        }
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ATS scan from file error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"ATS scan failed: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 3: Quick Rules-Only Scan (no AI, fast)
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/scan-quick")
async def ats_scan_quick(
    payload: ATSScanRequest,          # ← same typed model, include_ai ignored
    db:      AsyncSession = Depends(get_db),
):
    """
    Fast rules-only ATS scan — no AI call, responds in <1 second.
    Ideal for: real-time feedback while users edit their resume.

    Same request format as /scan, but include_ai is always false.
    """
    try:
        result = await _scanner.scan(
            resume=payload.resume_data,
            job_description=payload.job_description,
            db=None,
            include_ai=False,
        )
        result["scan_type"] = "rules_only"
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Quick scan error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Quick scan failed: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 4: Score Explanation
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/score/{score}")
async def explain_score(score: int):
    """
    Get a plain-language explanation of an ATS score.

    Example: GET /ats/score/68
    """
    if not 0 <= score <= 100:
        raise HTTPException(status_code=400, detail="Score must be between 0 and 100")

    if score >= 90:
        status = "Excellent"
        message = "Your resume is highly optimised for ATS systems."
        verdict = "Apply with confidence. Focus on tailoring keywords per role."
        colour  = "green"
    elif score >= 80:
        status = "Very Good"
        message = "Your resume will pass most ATS systems with strong results."
        verdict = "You're competitive. Minor refinements will push you into the top tier."
        colour  = "green"
    elif score >= 72:
        status = "Good"
        message = "Your resume passes ATS screening."
        verdict = "Solid foundation. Address remaining issues to strengthen your position."
        colour  = "yellow"
    elif score >= 60:
        status = "Needs Improvement"
        message = "Your resume has ATS weaknesses that may cause it to be filtered out."
        verdict = "Fix high-priority issues before applying to competitive roles."
        colour  = "orange"
    elif score >= 45:
        status = "Poor"
        message = "Your resume is likely to fail ATS screening for most roles."
        verdict = "Significant revision required. Follow the improvement roadmap."
        colour  = "red"
    else:
        status = "Critical"
        message = "Your resume will be rejected by most ATS systems."
        verdict = "Start with the Critical issues immediately before applying anywhere."
        colour  = "red"

    return {
        "score":       score,
        "status":      status,
        "colour":      colour,
        "message":     message,
        "verdict":     verdict,
        "grade":       _score_to_grade(score),
        "percentile":  _score_to_percentile(score),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 5: Help / API Docs
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/help")
async def ats_help():
    """Full API documentation for the ATS scanner."""
    return {
        "service":  "Universal ATS Scanner v3",
        "version":  "3.0.0",
        "powered_by": "Rule-based engine + Groq AI (llama-3.1-8b-instant)",

        "endpoints": {
            "POST /ats/scan": {
                "description": "Full AI-powered ATS scan (JSON resume + optional JD)",
                "ai_powered":  True,
                "response_time": "2-5 seconds",
            },
            "POST /ats/scan-file": {
                "description": "Upload PDF or DOCX resume for ATS scan",
                "accepts":     "multipart/form-data",
                "formats":     ["PDF", "DOCX"],
                "ai_powered":  True,
            },
            "POST /ats/scan-quick": {
                "description": "Fast rules-only scan (no AI, <1 second)",
                "ai_powered":  False,
                "response_time": "< 500ms",
                "use_case": "Real-time feedback while user is editing",
            },
            "GET /ats/score/{score}": {
                "description": "Explain a specific ATS score in plain language",
            },
        },

        "industries_supported": [
            "Technology & Software",
            "Healthcare & Nursing",
            "Finance & Accounting",
            "Marketing & Digital",
            "Sales",
            "Human Resources",
            "Legal",
            "Education & Teaching",
            "Design & Creative",
            "Engineering (all disciplines)",
            "Project Management",
            "Supply Chain & Logistics",
            "Construction & Trades",
            "Hospitality & Tourism",
            "Science & Research",
            "Government & Non-profit",
            "Customer Service",
            "And all other professions",
        ],

        "sections_analysed": [
            "Contact Information",
            "Professional Summary",
            "Work Experience",
            "Education",
            "Skills",
            "Projects",
            "Certifications & Licenses",
            "Languages (spoken)",
            "Volunteer Work",
            "Publications & Research",
            "Awards & Achievements",
            "Hobbies & Interests",
            "References",
        ],

        "score_dimensions": {
            "format_compliance":  "10% — file type, fonts, layout",
            "structure_quality":  "20% — all required sections present",
            "content_quality":    "30% — bullets, metrics, action verbs",
            "keyword_alignment":  "25% — JD keyword matching (if JD provided)",
            "ats_compliance":     "15% — parsing safety, section headings",
        },

        "ai_features": [
            "Industry auto-detection",
            "Section-by-section AI scoring and verdict",
            "Before/after bullet rewrite examples",
            "Summary rewrite suggestions",
            "Keyword gap analysis with placement advice",
            "Top 5 ATS-passing tactics specific to this resume",
            "Priority action plan with estimated score gains",
        ],

        "response_fields": {
            "ats_score":          "Final weighted score 0-100",
            "grade":              "Letter grade A+ to F",
            "score_breakdown":    "Scores per dimension",
            "section_analysis":   "Detailed per-section analysis (all 13 sections)",
            "keyword_analysis":   "JD keyword matching (if JD provided)",
            "ai_analysis":        "Groq AI insights and rewrites",
            "recommendations":    "Roadmap, quick wins, ATS tactics",
            "issues":             "All issues sorted by severity",
            "summary":            "Executive summary with next steps",
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _score_to_grade(score: int) -> str:
    if score >= 90: return "A+"
    if score >= 85: return "A"
    if score >= 80: return "A-"
    if score >= 75: return "B+"
    if score >= 70: return "B"
    if score >= 65: return "B-"
    if score >= 60: return "C+"
    if score >= 55: return "C"
    if score >= 50: return "C-"
    if score >= 40: return "D"
    return "F"


def _score_to_percentile(score: int) -> str:
    if score >= 90: return "Top 5% of applicants"
    if score >= 80: return "Top 15% of applicants"
    if score >= 70: return "Top 30% of applicants"
    if score >= 60: return "Top 50% of applicants"
    if score >= 50: return "Bottom 50%"
    return "High rejection risk"