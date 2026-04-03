import json
import logging
import asyncio
from typing import Dict, List, Any

from .ai_client import call_ai
from .universal_extractor import UniversalExtractor
from .llm_section_identifier import LLMSectionIdentifier
from .llm_section_parser import LLMSectionParser

logger = logging.getLogger(__name__)


class UniversalResumeParser:
    """
    Production-grade universal resume parser.
    
    ARCHITECTURE:
    1. Universal Extraction - Get all content without assumptions
    2. LLM Section Identification - Let LLM identify sections from ANY format
    3. LLM Section Parsing - Parse each section with specialized prompts
    4. Contact Info Extraction - Regex-based fallback for contact details
    """
    
    @staticmethod
    async def parse(extractor_output: Dict[str, Any]) -> Dict[str, Any]:
        """Main parsing pipeline"""
        try:
            raw_items = extractor_output.get("raw_items", [])
            if not raw_items:
                logger.error("No raw_items provided")
                return UniversalResumeParser._empty_result()
            
            logger.info(f"=" * 60)
            logger.info(f"UNIVERSAL RESUME PARSING - {len(raw_items)} items")
            logger.info(f"=" * 60)
            
            # STAGE 1: Universal extraction
            logger.info("\n📥 STAGE 1: Extracting all content...")
            all_items = UniversalExtractor.get_all_items_flat(raw_items)
            logger.info(f"✓ Got {len(all_items)} content blocks")
            
            # STAGE 2: LLM section identification
            logger.info("\n🔍 STAGE 2: Identifying sections with LLM...")
            identified_sections = await LLMSectionIdentifier.identify_and_extract_sections(raw_items)
            
            active_sections = {k: v for k, v in identified_sections.items() if v.strip()}
            logger.info(f"✓ Found {len(active_sections)} active sections")
            
            # STAGE 3: LLM section parsing
            logger.info("\n⚙️  STAGE 3: Parsing sections with specialized prompts...")
            parsed_sections = await LLMSectionParser.parse_sections(identified_sections)
            
            # STAGE 4: Contact info extraction (fallback)
            logger.info("\n📞 STAGE 4: Extracting contact information...")
            contact_info = UniversalExtractor.extract_contact_info_raw(raw_items)
            
            if parsed_sections.get("header"):
                parsed_sections["header"]["email"] = contact_info.get("email", "")
                parsed_sections["header"]["phone"] = contact_info.get("phone", "")
                
                # Use best available link
                link = contact_info.get("linkedin") or contact_info.get("github") or contact_info.get("portfolio")
                if link:
                    parsed_sections["header"]["link"] = link
            
            logger.info(f"\n✅ PARSING COMPLETE")
            logger.info(f"=" * 60)
            
            return {
                "success": True,
                "parsed": parsed_sections
            }
        
        except Exception as e:
            logger.error(f"\n❌ PARSING FAILED: {str(e)}", exc_info=True)
            return UniversalResumeParser._empty_result()
    
    @staticmethod
    def _empty_result() -> Dict[str, Any]:
        """Return empty result"""
        return {
            "success": False,
            "message": "Failed to parse resume",
            "parsed": {
                "header": {"name": "", "email": "", "phone": "", "location": "", "link": "", "title": ""},
                "summary": "",
                "education": [],
                "experience": [],
                "skills": [],
                "projects": [],
                "certifications": [],
                "languages": [],
            }
        }