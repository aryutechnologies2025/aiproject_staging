import json
import logging
import asyncio
from typing import Dict, List, Any

from .ai_client import call_ai, RateLimitError
from .universal_extractor import UniversalExtractor

logger = logging.getLogger(__name__)

import time

_last_call_time = 0
_lock = asyncio.Lock()

async def rate_limited_call(func, *args, **kwargs):
    global _last_call_time
    async with _lock:
        now = time.time()
        elapsed = now - _last_call_time
        if elapsed < 4.0:
            await asyncio.sleep(4.0 - elapsed)
        result = await func(*args, **kwargs)
        _last_call_time = time.time()
        return result


class LLMSectionIdentifier:
    """
    Use LLM to intelligently identify and extract sections from ANY resume format.
    This is the core of universal parsing - let LLM understand structure, not regex.
    """
    
    @staticmethod
    async def identify_and_extract_sections(raw_items: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        LLM analyzes the resume and returns extracted sections.
        Works for any format, any structure, any content.
        """
        
        # Get all content
        all_items = UniversalExtractor.get_all_items_flat(raw_items)
        all_text = "\n".join(all_items[:200])  # Take first 200 items for context
        
        prompt = f"""You are a universal resume analyzer. Analyze this resume content and extract ALL distinct sections.

RESUME CONTENT:
{all_text}

TASK:
Identify and extract each section of the resume. Return as JSON where:
- Keys are section names (standardized)
- Values are the COMPLETE, UNTRUNCATED content of that section

SECTION TYPES TO IDENTIFY:
1. HEADER/PERSONAL - name, title, contact info
2. SUMMARY/OBJECTIVE - professional summary, career objective, profile
3. EXPERIENCE - work history, jobs, employment, roles, internships
4. EDUCATION - degrees, qualifications, schools, universities
5. SKILLS - technical skills, tools, languages, competencies
6. PROJECTS - portfolio, case studies, work examples
7. CERTIFICATIONS - licenses, certifications, credentials, awards
8. LANGUAGES - language proficiencies
9. OTHER - volunteer, publications, research, patents, hobbies, associations

CRITICAL RULES:
- DO NOT skip any section present in resume
- DO NOT truncate content - include EVERYTHING from each section
- DO NOT merge different sections
- DO NOT summarize - extract exact content
- If section doesn't exist, use empty string ""
- Preserve all details, numbers, dates, achievements
- Include all bullet points, achievements, descriptions

Return ONLY valid JSON with no markdown, no code blocks, no explanations:

{{
  "header": "...",
  "summary": "...",
  "experience": "...",
  "education": "...",
  "skills": "...",
  "projects": "...",
  "certifications": "...",
  "languages": "...",
  "other": "..."
}}"""
        
        try:
            logger.info("🔍 STAGE 1: Identifying sections with LLM...")
            
            response = await rate_limited_call(
                call_ai,
                prompt=prompt,
                max_output_tokens=5000,
                use_gemini_first=False
            )
            
            # Parse JSON
            text = response.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            
            sections = json.loads(text)
            
            # Validate we got all sections
            expected_sections = ["header", "summary", "experience", "education", "skills", 
                               "projects", "certifications", "languages", "other"]
            
            for section in expected_sections:
                if section not in sections:
                    sections[section] = ""
            
            logger.info(f"✓ Identified sections: {[k for k, v in sections.items() if v.strip()]}")
            return sections
            
        except Exception as e:
            logger.error(f"Failed to identify sections: {str(e)}")
            return {
                "header": "", "summary": "", "experience": "", "education": "",
                "skills": "", "projects": "", "certifications": "", "languages": "", "other": ""
            }