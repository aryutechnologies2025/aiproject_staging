# /home/aryu_user/Arun/aiproject_staging/app/api/v1/ats_api_v2.py
"""
Production-Grade ATS Scanner API
Supports:
- Form-data with file upload (PDF/DOCX)
- JSON payload (internal resume builder)
- Comprehensive error handling and validation
- ATS scoring and detailed feedback
"""

from fastapi import Request, UploadFile, APIRouter, Depends, HTTPException, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
import logging
from typing import Optional
import json

from app.core.database import get_db
from app.schemas.ats_schema import ATSScanRequest, ATSScanResponse
from app.services.resume_builder_services.resume_parser_service import parse_resume_to_schema
from app.utils.ats_scanner.text_extraction import extract_text
from app.services.llm_client import call_llm
from app.services.resume_builder_services.ats_scanner_service import ATSScannerService

logger = logging.getLogger(__name__)
router = APIRouter()


# =====================================================
# DATA VALIDATION
# =====================================================

class ValidationError(Exception):
    """Custom validation error"""
    pass


class ATSValidation:
    """Validation utilities for ATS data"""
    
    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
    ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc"}
    MIN_RESUME_WORDS = 50
    
    @staticmethod
    def validate_file(file: UploadFile) -> None:
        """Validate uploaded file"""
        
        if not file:
            raise ValidationError("No file provided")
        
        # Check extension
        filename_lower = file.filename.lower() if file.filename else ""
        if not any(filename_lower.endswith(ext) for ext in ATSValidation.ALLOWED_EXTENSIONS):
            raise ValidationError(
                f"Invalid file type. Allowed: {', '.join(ATSValidation.ALLOWED_EXTENSIONS)}"
            )
        
        # Check size
        if file.size and file.size > ATSValidation.MAX_FILE_SIZE:
            raise ValidationError(f"File too large. Max size: 5 MB")
    
    @staticmethod
    def validate_resume_data(resume: dict) -> None:
        """Validate parsed resume data"""
        
        if not resume:
            raise ValidationError("Resume data is empty")
        
        # Check minimum content
        total_words = 0
        
        for field in ["summary", "skills", "experience", "education"]:
            if isinstance(resume.get(field), list):
                total_words += len(resume[field])
            elif isinstance(resume.get(field), str):
                total_words += len(resume[field].split())
            elif isinstance(resume.get(field), dict):
                text = " ".join(str(v) for v in resume[field].values())
                total_words += len(text.split())
        
        if total_words < ATSValidation.MIN_RESUME_WORDS:
            raise ValidationError(
                f"Resume too short. Minimum {ATSValidation.MIN_RESUME_WORDS} words required"
            )
        
        # Check required sections
        if not resume.get("experience"):
            raise ValidationError("Resume must include work experience")


# =====================================================
# REQUEST/RESPONSE MODELS
# =====================================================

class ATSScanRequestV2:
    """Request model for ATS scan (flexible input handling)"""
    
    def __init__(self, resume_data: dict, job_description: Optional[str] = None):
        self.resume_data = resume_data
        self.job_description = job_description


class ATSScanResponseV2:
    """Response model for ATS scan"""
    
    def __init__(self, scan_results: dict):
        self.ats_score = scan_results.get("ats_score", 0)
        self.score_status = scan_results.get("score_status")
        self.critical_issues_count = scan_results.get("critical_issues_count", 0)
        self.section_analysis = scan_results.get("section_analysis", [])
        self.keyword_analysis = scan_results.get("keyword_analysis", {})
        self.recommendations = scan_results.get("recommendations", {})
        self.issues = scan_results.get("issues", {})
        self.summary = scan_results.get("summary", {})
        self.ai_analysis = scan_results.get("ai_analysis")
        self.score_breakdown = scan_results.get("score_breakdown", {})


# =====================================================
# ENDPOINTS
# =====================================================

@router.post("/scan")
async def ats_scan_unified(
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    Unified ATS scan endpoint
    
    Supports multiple input methods:
    1. JSON body with resume data
    2. Form-data with PDF/DOCX file upload
    3. Form-data with JSON payload
    
    Query Parameters:
    - include_ai_analysis: bool (default: true) - Enable AI analysis if JD provided
    
    Returns:
        Complete ATS scan results with scores and feedback
    """
    
    try:
        content_type = request.headers.get("content-type", "")
        include_ai = request.query_params.get("include_ai_analysis", "true").lower() == "true"
        
        resume_dict = None
        job_description = None
        
        logger.info(f"ATS scan request - Content-Type: {content_type}, Include AI: {include_ai}")
        
        # =====================
        # CASE 1: JSON BODY
        # =====================
        if "application/json" in content_type:
            logger.info("Processing JSON body request")
            
            body = await request.json()
            
            # Validate JSON structure
            if "resume" not in body:
                raise ValidationError("Missing 'resume' field in JSON body")
            
            resume_dict = body.get("resume")
            job_description = body.get("job_description")
            
            # Validate data
            ATSValidation.validate_resume_data(resume_dict)
        
        # =====================
        # CASE 2: FORM-DATA
        # =====================
        elif "multipart/form-data" in content_type:
            logger.info("Processing form-data request")
            
            form = await request.form()
            
            file: UploadFile = form.get("file")
            payload_json = form.get("payload")
            job_description = form.get("job_description")
            
            # Sub-case 2a: JSON payload in form
            if payload_json:
                logger.info("Parsing JSON from form payload")
                
                try:
                    payload_data = json.loads(payload_json)
                except json.JSONDecodeError as e:
                    raise ValidationError(f"Invalid JSON in payload: {str(e)}")
                
                resume_dict = payload_data.get("resume", payload_data)
                ATSValidation.validate_resume_data(resume_dict)
            
            # Sub-case 2b: File upload
            elif file:
                logger.info(f"Processing file upload: {file.filename}")
                
                # Validate file
                ATSValidation.validate_file(file)
                
                # Extract text
                try:
                    text = await extract_text(file)
                except Exception as e:
                    logger.error(f"Text extraction failed: {str(e)}")
                    raise ValidationError(f"Failed to extract text from file: {str(e)}")
                
                # Parse resume
                try:
                    parsed_resume = parse_resume_to_schema(
                        text=text,
                        file_type=file.filename.split(".")[-1] if file.filename else "pdf"
                    )
                    resume_dict = parsed_resume.dict()
                except Exception as e:
                    logger.error(f"Resume parsing failed: {str(e)}")
                    raise ValidationError(f"Failed to parse resume: {str(e)}")
                
                # Add job description if provided
                if job_description:
                    resume_dict["job_description"] = job_description
            
            else:
                raise ValidationError("Form-data must include 'payload' or 'file'")
        
        else:
            raise ValidationError(
                f"Unsupported content type: {content_type}. "
                "Use 'application/json' or 'multipart/form-data'"
            )
        
        # =====================
        # VALIDATION
        # =====================
        if not resume_dict:
            raise ValidationError("Could not parse resume data")
        
        logger.info("Resume data validated successfully")
        
        # =====================
        # ATS SCAN
        # =====================
        logger.info("Starting ATS scan")
        
        scanner = ATSScannerService()
        
        # Prepare LLM client if needed
        llm_client = None
        if include_ai and job_description:
            llm_client = lambda prompt: call_llm(
                user_message=prompt,
                agent_name="ats_evaluator",
                db=db,
            )
        
        # Run scan
        scan_results = await scanner.scan(
            resume=resume_dict,
            job_description=job_description,
            llm_client=llm_client,
            db=db
        )
        
        logger.info(f"ATS scan completed. Score: {scan_results.get('ats_score')}")
        
        # =====================
        # RESPONSE
        # =====================
        response = ATSScanResponseV2(scan_results)
        
        return {
            "success": True,
            "data": {
                "ats_score": response.ats_score,
                "score_status": response.score_status,
                "critical_issues_count": response.critical_issues_count,
                "score_breakdown": response.score_breakdown,
                "section_analysis": response.section_analysis,
                "keyword_analysis": response.keyword_analysis,
                "issues": response.issues,
                "recommendations": response.recommendations,
                "summary": response.summary,
                "ai_analysis": response.ai_analysis
            }
        }
    
    except ValidationError as e:
        logger.warning(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"ATS scan error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error during ATS scan"
        )


@router.post("/quick-score")
async def ats_quick_score(
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    Quick ATS score without detailed analysis
    Fast endpoint for real-time feedback
    
    Accepts:
    - JSON body with resume data
    - Form-data with file
    
    Returns:
        Quick score (0-100) and critical issues only
    """
    
    try:
        content_type = request.headers.get("content-type", "")
        
        resume_dict = None
        
        # Parse input (simplified)
        if "application/json" in content_type:
            body = await request.json()
            resume_dict = body.get("resume", body)
        
        elif "multipart/form-data" in content_type:
            form = await request.form()
            file = form.get("file")
            
            if file:
                text = await extract_text(file)
                parsed = parse_resume_to_schema(
                    text=text,
                    file_type=file.filename.split(".")[-1]
                )
                resume_dict = parsed.dict()
        
        if not resume_dict:
            raise ValidationError("Could not parse input")
        
        # Quick score (rules only, no AI)
        scanner = ATSScannerService()
        scan_results = await scanner.scan(resume=resume_dict)
        
        return {
            "success": True,
            "ats_score": scan_results.get("ats_score"),
            "critical_issues": scan_results.get("critical_issues_count"),
            "ready_to_apply": scan_results.get("summary", {}).get("ready_to_apply"),
            "main_issues": [
                issue["message"] 
                for issue in scan_results.get("issues", {}).get("critical", [])[:3]
            ]
        }
    
    except Exception as e:
        logger.error(f"Quick score error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/score-guide")
async def score_guide():
    """
    Get explanation of ATS scores
    
    Returns:
        Score ranges and what they mean
    """
    
    return {
        "score_ranges": {
            "85-100": {
                "status": "Excellent",
                "meaning": "Ready to apply - resume is highly optimized",
                "action": "Submit your application"
            },
            "70-84": {
                "status": "Good",
                "meaning": "Ready to apply with minor improvements",
                "action": "Consider implementing quick wins"
            },
            "55-69": {
                "status": "Needs Improvement",
                "meaning": "Address major issues before applying",
                "action": "Follow high-priority recommendations"
            },
            "0-54": {
                "status": "Critical Issues",
                "meaning": "Significant improvements required",
                "action": "Complete overhaul needed"
            }
        },
        "what_affects_score": {
            "40%": "ATS rule compliance (format, structure, fonts)",
            "60%": "Keyword matching and job relevance"
        }
    }


# =====================================================
# ERROR HANDLERS
# =====================================================

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "ats_scanner"}