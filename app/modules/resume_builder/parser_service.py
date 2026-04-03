import logging
from typing import Dict, Any
from fastapi import HTTPException

from .ai_parser import UniversalResumeParser

logger = logging.getLogger(__name__)


async def parse_resume_with_ai(extractor_output: Dict[str, Any]) -> Dict[str, Any]:
    """Service endpoint for resume parsing"""
    try:
        if not extractor_output:
            raise HTTPException(status_code=400, detail="Empty extractor output")
        
        raw_items = extractor_output.get("raw_items")
        if not raw_items:
            raise HTTPException(status_code=400, detail="No raw_items")
        
        logger.info(f"Starting universal resume parsing...")
        
        result = await UniversalResumeParser.parse(extractor_output)
        
        if not result.get("success", False):
            logger.error("Universal parsing failed")
            return {
                "success": False,
                "message": "Failed to parse resume",
                "parsed": None
            }
        
        parsed = result.get("parsed", {})
        
        logger.info(f"✓ Resume parsing successful")
        logger.info(f"  - Experience entries: {len(parsed.get('experience', []))}")
        logger.info(f"  - Education entries: {len(parsed.get('education', []))}")
        logger.info(f"  - Projects: {len(parsed.get('projects', []))}")
        logger.info(f"  - Skills: {len(parsed.get('skills', []))}")
        
        return {
            "success": True,
            "message": "Resume parsed successfully",
            "parsed": parsed
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Resume parsing error: {str(e)}", exc_info=True)
        return {
            "success": False,
            "message": f"Resume parsing failed: {str(e)}",
            "parsed": None
        }
    