"""
ATS Scanner Router v5.0 — Unified parsing pipeline (Resume Builder → ATS).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, validator
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import validate_file_security
from app.core.database import get_db
from app.modules.ats_scanner.service import ATSScannerService, create_ats_scan
from app.modules.resume_builder.extractor import extract_with_llamaparse
from app.modules.resume_builder.parser_service import parse_resume_with_ai

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
        header = v.get("header") or {}
        has_content = any([
            header.get("name") if isinstance(header, dict) else v.get("name"),
            v.get("summary"),
            v.get("experience"),
            v.get("skills"),
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


def _normalise_resume_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalise both flat and nested (header-wrapped) resume schemas
    into a single canonical dict consumed by ATSScannerService.
    """
    header = data.get("header") or {}
    if isinstance(header, dict) and header:
        flat = dict(data)
        flat["name"]     = header.get("name")     or data.get("name", "")
        flat["email"]    = header.get("email")    or data.get("email", "")
        flat["phone"]    = header.get("phone")    or data.get("phone", "")
        flat["location"] = header.get("location") or data.get("location", "")
        flat["linkedin"] = header.get("link")     or data.get("linkedin", "")
        flat["title"]    = header.get("title")    or data.get("title", "")
    else:
        flat = data

    # Unwrap nested summary dict: {"summary": {"summary": "..."}} → {"summary": "..."}
    summary = flat.get("summary")
    if isinstance(summary, dict):
        flat["summary"] = summary.get("summary", "")

    return flat


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
        f"ATS scan — has_jd={bool(payload.job_description)}"
    )
    try:
        normalised = _normalise_resume_dict(payload.resume_data)
        result = await _scanner.scan(
            resume=normalised,
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
# ENDPOINT 2: File Upload Scan — LlamaParse → Resume Builder → ATS
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/scan-file")
async def ats_scan_from_file(
    file:            UploadFile     = Depends(validate_file_security),
    job_description: Optional[str] = Form(default=None),
    include_ai:      bool           = Form(default=True),
    db:              AsyncSession   = Depends(get_db),
):
    """
    Upload PDF or DOCX →
    LlamaParse extracts structured items →
    Resume Builder parser builds canonical JSON →
    ATS Scanner scores the canonical JSON.

    Single unified pipeline — no duplicate parsing.
    """
    job_description = _sanitise_text(job_description)
    logger.info(f"ATS file scan: {file.filename}, has_jd={bool(job_description)}")

    # ── Step 1: Read file bytes ──────────────────────────────────────────────
    await file.seek(0)
    file_bytes = await file.read()
    await file.seek(0)

    content_type = file.content_type or (
        "application/pdf"
        if (file.filename or "").lower().endswith(".pdf")
        else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )

    # ── Step 2: LlamaParse extraction ────────────────────────────────────────
    try:
        extractor_output = await extract_with_llamaparse(
            file_bytes, file.filename or "resume.pdf", content_type
        )
        if not extractor_output.get("success") or not extractor_output.get("raw_items"):
            raise HTTPException(
                status_code=400,
                detail="Could not extract readable content from the file. "
                       "Ensure the resume is not a scanned image-only PDF.",
            )
        logger.info(f"LlamaParse extracted {len(extractor_output['raw_items'])} blocks")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Extraction error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Failed to extract file content: {str(e)}")

    # ── Step 3: Resume Builder parser → canonical JSON ───────────────────────
    try:
        parse_result = await parse_resume_with_ai(
            extractor_output=extractor_output,
            file_bytes=file_bytes,
            filename=file.filename,
            content_type=content_type,
        )
        if not parse_result.get("success") or not parse_result.get("parsed"):
            raise HTTPException(
                status_code=400,
                detail="Resume content could not be parsed. Please check the file format.",
            )
        parsed = parse_result["parsed"]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Resume Builder parse error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Failed to parse resume: {str(e)}")

    # ── Step 4: Normalise schema ─────────────────────────────────────────────
    resume_dict = _normalise_resume_dict(parsed)

    logger.info(
        f"Parsed — name='{resume_dict.get('name')}' | "
        f"exp={len(resume_dict.get('experience') or [])} | "
        f"edu={len(resume_dict.get('education') or [])} | "
        f"skills={len(resume_dict.get('skills') or [])}"
    )

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

    # ── Step 5: ATS Scan on canonical JSON ───────────────────────────────────
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
            "pipeline":     "llamaparse_resume_builder",
            "parsed_name":  resume_dict.get("name"),
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
        normalised = _normalise_resume_dict(payload.resume_data)
        result = await _scanner.scan(
            resume=normalised,
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
        "service": "Universal ATS Scanner v5.0",
        "version": "5.0.0",
        "powered_by": "LlamaParse + Resume Builder Parser + Rule Engine + Groq AI",
        "pipeline": "PDF/DOCX → LlamaParse → Resume Builder Parser → Canonical JSON → ATS Engine",
        "endpoints": {
            "POST /ats/scan":       "Full AI-powered ATS scan (JSON resume + optional JD)",
            "POST /ats/scan-file":  "Upload PDF/DOCX — unified Resume Builder pipeline",
            "POST /ats/scan-quick": "Fast rules-only scan (no AI, <1 second)",
            "GET  /ats/score/{n}":  "Explain a specific ATS score",
        },
        "why_unified_pipeline": (
            "The ATS Scanner reuses the Resume Builder's LlamaParse + LLM parsing pipeline. "
            "This eliminates duplicate extraction logic, improves section detection accuracy, "
            "and ensures education, skills, projects, and certifications are never incorrectly "
            "reported as missing."
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