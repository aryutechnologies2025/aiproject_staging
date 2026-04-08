import logging
from typing import Dict, Any
from fastapi import HTTPException, Request

from .security_manager import SecurityManager
from .input_sanitizer import InputSanitizer
from .request_validator import RequestValidator
from .security_audit_logger import SecurityAuditLogger
from .ai_parser import ImprovedUniversalResumeParser

logger = logging.getLogger(__name__)


async def parse_resume_with_ai(
    extractor_output: Dict[str, Any],
    file_bytes: bytes = None,
    filename: str = None,
    content_type: str = None,
    request: Request = None
) -> Dict[str, Any]:

    try:
        # Extract client info for logging
        user_id = "anonymous"
        client_ip = "unknown"
        
        if request:
            user_id = request.headers.get("X-User-ID", "anonymous")
            client_ip = RequestValidator.get_client_ip(request)
        
        # Security: Validate request
        if request:
            logger.info(f"Validating request from {client_ip}")
            is_valid, error = RequestValidator.validate_request(request)
            if not is_valid:
                logger.warning(f"Request validation failed: {error}")
                SecurityAuditLogger.log_security_violation(
                    violation_type="invalid_request",
                    client_ip=client_ip,
                    user_id=user_id,
                    details=error
                )
                raise HTTPException(status_code=400, detail=error)
        
        # Security: Validate file
        if file_bytes and filename and content_type:
            logger.info(f"Validating file: {filename}")
            is_valid, error = SecurityManager.validate_file(file_bytes, filename, content_type)
            if not is_valid:
                logger.warning(f"File validation failed: {error}")
                SecurityAuditLogger.log_security_violation(
                    violation_type="invalid_file",
                    client_ip=client_ip,
                    user_id=user_id,
                    details=error
                )
                raise HTTPException(status_code=400, detail=error)
        
        # Security: Check rate limit
        if request:
            is_allowed, error = SecurityManager.check_rate_limit(user_id, client_ip)
            if not is_allowed:
                logger.warning(f"Rate limit exceeded: {error}")
                SecurityAuditLogger.log_security_violation(
                    violation_type="rate_limit_exceeded",
                    client_ip=client_ip,
                    user_id=user_id,
                    details=error
                )
                raise HTTPException(status_code=429, detail=error)
        
        # Validate extractor output
        if not extractor_output:
            raise HTTPException(status_code=400, detail="Empty extractor output")
        
        raw_items = extractor_output.get("raw_items")
        if not raw_items:
            raise HTTPException(status_code=400, detail="No raw_items")

        # --- OPTIMIZATION: Payload Pre-processing ---
        # Prevent HTTP 413 by truncating massive individual blocks before AI processing
        for item in raw_items:
            if isinstance(item.get("text"), str) and len(item["text"]) > 10000:
                logger.warning(f"Truncating oversized block in {filename}")
                item["text"] = item["text"][:10000]
        
        logger.info(f"Starting optimized universal resume parsing...")
        
        # Parse with AI (UniversalResumeParser handles internal section chunking)
        result = await ImprovedUniversalResumeParser.parse(extractor_output)
        
        if not result.get("success", False):
            logger.error("Universal parsing failed")
            return {
                "success": False,
                "message": "Failed to parse resume",
                "parsed": None
            }
        
        parsed = result.get("parsed", {})
        
        # Security: Validate parsed JSON data
        logger.info("Validating parsed data...")
        is_valid, error = SecurityManager.validate_json_data(parsed)
        if not is_valid:
            logger.warning(f"Parsed data validation failed: {error}")
            raise HTTPException(status_code=500, detail="Error validating parsed data")
        
        # Security: Sanitize output
        logger.info("Sanitizing output...")
        sanitized_result = InputSanitizer.sanitize_resume_data(parsed)
        
        # Security: Log successful file upload
        if file_bytes and filename:
            SecurityAuditLogger.log_file_upload(
                filename=filename,
                file_size=len(file_bytes),
                client_ip=client_ip,
                user_id=user_id,
                status="success",
                details=f"Exp: {len(sanitized_result.get('experience', []))}, Edu: {len(sanitized_result.get('education', []))}"
            )
        
        logger.info(f"✓ Resume parsing successful")
        return {
            "success": True,
            "message": "Resume parsed successfully",
            "parsed": sanitized_result
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Resume parsing error: {str(e)}", exc_info=True)
        if request:
            client_ip = RequestValidator.get_client_ip(request)
            user_id = request.headers.get("X-User-ID", "anonymous")
            SecurityAuditLogger.log_security_violation(
                violation_type="parsing_error",
                client_ip=client_ip,
                user_id=user_id,
                details=str(e)
            )
        return {
            "success": False,
            "message": f"Resume parsing failed: {str(e)}",
            "parsed": None
        }