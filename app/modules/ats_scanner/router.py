"""
ATS Scanner Router v4.0 — LlamaParse markdown pipeline for file scanning.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, validator
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.modules.ats_scanner.service import ATSScannerService, create_ats_scan
from app.modules.ats_scanner.utils.ats_extractor import extract_resume_markdown
from app.modules.ats_scanner.utils.ats_markdown_parser import parse_resume_markdown

logger = logging.getLogger(__name__)
router = APIRouter()

_scanner = ATSScannerService()


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST MODELS
# ─────────────────────────────────────────────────────────────────────────────

class ATSScanRequest(BaseModel):
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
    if not text:
        return text
    if isinstance(text, bytes):
        return text.decode("utf-8", errors="replace")
    return text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")


def _validate_resume_dict(data: dict) -> None:
    has_content = any([
        data.get("name"), data.get("summary"),
        data.get("experience"), data.get("skills"),
        data.get("education"),
    ])
    if not has_content:
        raise HTTPException(
            status_code=400,
            detail="resume_data appears empty. Provide name, summary, experience, skills, or education.",
        )


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 1: Full AI-Powered Scan (JSON payload)
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/scan")
async def ats_scan(
    payload: ATSScanRequest,
    db:      AsyncSession = Depends(get_db),
):
    """Full AI-powered ATS scan — accepts pre-parsed JSON resume."""
    logger.info(
        f"ATS scan — name={payload.resume_data.get('name', 'Unknown')}, "
        f"has_jd={bool(payload.job_description)}"
    )
    try:
        result = await _scanner.scan(
            resume=payload.resume_data,
            job_description=payload.job_description or None,
            db=db,
            include_ai=bool(payload.include_ai),
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ATS scan error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"ATS scan failed: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 2: File Upload Scan — LlamaParse → Markdown → ATS
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/scan-file")
async def ats_scan_from_file(
    file:            UploadFile     = File(...),
    job_description: Optional[str] = Form(default=None),
    include_ai:      bool           = Form(default=True),
    db:              AsyncSession   = Depends(get_db),
):
    """
    Upload PDF or DOCX → LlamaParse extracts markdown →
    regex parser builds ATS dict → full ATS scan.
    """
    filename = (file.filename or "").lower()

    if not filename.endswith((".pdf", ".docx")):
        raise HTTPException(
            status_code=400,
            detail="Only PDF and DOCX files are supported.",
        )

    job_description = _sanitise_text(job_description)
    logger.info(f"ATS file scan start: {file.filename}, has_jd={bool(job_description)}")

    # ── Step 1: Extract markdown via LlamaParse (or local fallback) ──────────
    try:
        markdown = await extract_resume_markdown(file)
        if not markdown or len(markdown.strip()) < 50:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not extract readable text from the file. "
                    "Ensure the resume is not a scanned image-only PDF."
                ),
            )
        logger.info(f"Markdown extracted: {len(markdown)} chars")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Markdown extraction error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Failed to extract file content: {str(e)}")

    # ── Step 2: Parse markdown → ATS-ready dict ──────────────────────────────
    try:
        resume_dict = parse_resume_markdown(markdown)

        logger.info(
            f"Markdown parsed — name='{resume_dict.get('name')}' | "
            f"exp={len(resume_dict.get('experience') or [])} | "
            f"edu={len(resume_dict.get('education') or [])} | "
            f"skills={len(resume_dict.get('skills') or [])}"
        )

        # Guard: reject if nothing useful was parsed
        if not any([
            resume_dict.get("name"),
            resume_dict.get("experience"),
            resume_dict.get("education"),
            resume_dict.get("skills"),
            resume_dict.get("summary"),
        ]):
            raise HTTPException(
                status_code=400,
                detail="Resume content could not be parsed. Please check the file format.",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Markdown parse error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Failed to parse resume: {str(e)}")

    # ── Step 3: ATS Scan ─────────────────────────────────────────────────────
    try:
        result = await _scanner.scan(
            resume=resume_dict,
            job_description=job_description or None,
            db=db,
            include_ai=include_ai,
        )
        result["meta"] = {
            "source":       "file_upload",
            "source_file":  file.filename,
            "pipeline":     "llamaparse_markdown",
            "parsed_name":  resume_dict.get("name"),
            "markdown_len": len(markdown),
            "exp_count":    len(resume_dict.get("experience") or []),
            "edu_count":    len(resume_dict.get("education") or []),
            "skills_count": len(resume_dict.get("skills") or []),
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
    payload: ATSScanRequest,
    db:      AsyncSession = Depends(get_db),
):
    """Fast rules-only ATS scan — no AI call."""
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
    if not 0 <= score <= 100:
        raise HTTPException(status_code=400, detail="Score must be between 0 and 100")

    if score >= 90:
        status, message, colour = "Excellent", "Your resume is highly optimised for ATS systems.", "green"
        verdict = "Apply with confidence. Focus on tailoring keywords per role."
    elif score >= 80:
        status, message, colour = "Very Good", "Your resume will pass most ATS systems with strong results.", "green"
        verdict = "You're competitive. Minor refinements will push you into the top tier."
    elif score >= 72:
        status, message, colour = "Good", "Your resume passes ATS screening.", "yellow"
        verdict = "Solid foundation. Address remaining issues to strengthen your position."
    elif score >= 60:
        status, message, colour = "Needs Improvement", "Your resume has ATS weaknesses that may cause filtering.", "orange"
        verdict = "Fix high-priority issues before applying to competitive roles."
    elif score >= 45:
        status, message, colour = "Poor", "Your resume is likely to fail ATS screening for most roles.", "red"
        verdict = "Significant revision required. Follow the improvement roadmap."
    else:
        status, message, colour = "Critical", "Your resume will be rejected by most ATS systems.", "red"
        verdict = "Start with the Critical issues immediately before applying anywhere."

    return {
        "score": score, "status": status, "colour": colour,
        "message": message, "verdict": verdict,
        "grade": _score_to_grade(score),
        "percentile": _score_to_percentile(score),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 5: Help
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/help")
async def ats_help():
    return {
        "service": "Universal ATS Scanner v4.0",
        "version": "4.0.0",
        "powered_by": "LlamaParse (markdown) + Rule-based engine + Groq AI",
        "pipeline": "PDF/DOCX → LlamaParse Markdown → Regex Parser → ATS Engine",
        "endpoints": {
            "POST /ats/scan":       "Full AI-powered ATS scan (JSON resume + optional JD)",
            "POST /ats/scan-file":  "Upload PDF/DOCX — LlamaParse markdown pipeline",
            "POST /ats/scan-quick": "Fast rules-only scan (no AI, <1 second)",
            "GET  /ats/score/{n}":  "Explain a specific ATS score",
        },
        "why_markdown_pipeline": (
            "LlamaParse markdown preserves section structure (headings, bullets, tables) "
            "far better than raw text extraction, giving the regex parser reliable signals "
            "for education, experience, and skills detection."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SCORE HELPERS
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
