"""
ai_parser.py — Single-call structured resume parser with automatic deterministic fallback.

Architecture:
1. Primary: High-accuracy single-call Google Gemini structured extraction (PDF/text).
2. Fallback: Ultra-fast offline Deterministic AST Parser (PyMuPDF / regex AST).

Guarantees 100% service uptime even if external AI APIs experience rate limits (429) or outages.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from app.modules.resume_builder.gemini_client import get_gemini_client
from app.modules.resume_builder.deterministic_parser import DeterministicResumeParser
from app.modules.resume_builder.schemas import (
    CanonicalResume,
    map_to_legacy_parse_dict,
)
from app.modules.resume_builder.universal_extractor import UniversalExtractor

logger = logging.getLogger("resume_builder.ai_parser")

EXTRACTION_SYSTEM_PROMPT = """You are an expert AI Resume Parser and Career Data Extraction Engine.
Analyze the provided resume document or text thoroughly and extract all information into the requested JSON schema.

Extraction Guidelines:
1. "personal_information":
   - "name": Full legal or professional candidate name.
   - "title": Current or target designation (e.g. "Senior Software Engineer").
   - "email": Valid email address.
   - "phone": Phone number with country code if available.
   - "location": City, State, Country.
   - "link": Comma-separated URLs (LinkedIn, GitHub, Portfolio, Website).

2. "summary":
   - The professional summary, executive summary, objective, or profile statement.

3. "experience":
   - Extract ALL employment history and internships.
   - Extract "position", "company", "location", "fromYear", "toYear", "isOngoing" (true if current).
   - "bullets": List of all bullet points, accomplishments, metrics, and responsibilities. Do not skip any bullets.

4. "education":
   - Extract ALL degrees, diplomas, academic programs.
   - "degree", "institution", "location", "fromYear", "toYear".

5. "skills":
   - Extract all technical, programming, domain, and tool skills mentioned across the entire resume.
   - Deduplicate and normalize.

6. "projects":
   - Extract key projects: "title", "description", "technologies" (list), "fromYear", "toYear", "bullets".

7. "certifications":
   - Extract all licenses and certifications: "title", "issuer", "year".

8. "languages":
   - List all spoken/written languages.

9. "other":
   - Awards, honors, publications, or volunteer work.

Accuracy is paramount. Never invent or hallucinate dates, names, or metrics not present in the document.
Return ONLY the structured JSON according to the schema.
"""


class ImprovedUniversalResumeParser:
    """
    High-performance resume parser leveraging single-call Gemini structured extraction
    with transparent deterministic AST fallback.
    """

    @staticmethod
    async def parse_document_bytes(
        file_bytes: bytes,
        content_type: str = "application/pdf",
        filename: str = "resume.pdf",
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Parse raw resume document bytes directly using Gemini document understanding,
        with local PyMuPDF deterministic fallback on any error.
        """
        try:
            client = get_gemini_client()
            if client.is_configured:
                prompt = (
                    f"Extract the complete structured information from this resume document ({filename}). "
                    "Ensure all jobs, dates, skills, and contact details are accurately extracted."
                )

                raw_json = await client.generate(
                    prompt=prompt,
                    system_instruction=EXTRACTION_SYSTEM_PROMPT,
                    response_schema=CanonicalResume,
                    document_bytes=file_bytes,
                    document_mime_type=content_type,
                    operation="resume_parsing_document",
                    user_id=user_id,
                    request_id=request_id,
                )

                canonical = CanonicalResume.model_validate_json(raw_json)
                legacy_dict = map_to_legacy_parse_dict(canonical)

                logger.info(f"✓ Successfully parsed resume document '{filename}' with Gemini")
                return {
                    "success": True,
                    "parsed": legacy_dict,
                    "canonical": canonical.model_dump(),
                    "source": "gemini",
                }

        except Exception as e:
            logger.warning(f"[AIParser] Gemini document parsing failed ({e}). Activating deterministic AST fallback...")

        # ── Deterministic Fallback ──
        try:
            raw_text = ""
            if filename.lower().endswith(".pdf") or "pdf" in content_type.lower():
                import fitz
                doc = fitz.open(stream=file_bytes, filetype="pdf")
                raw_text = "\n".join(page.get_text() for page in doc)
                doc.close()
            else:
                raw_text = file_bytes.decode("utf-8", errors="ignore")

            canonical = DeterministicResumeParser.parse_text(raw_text)
            legacy_dict = map_to_legacy_parse_dict(canonical)
            logger.info(f"✓ Successfully parsed '{filename}' via Deterministic AST Fallback")

            return {
                "success": True,
                "parsed": legacy_dict,
                "canonical": canonical.model_dump(),
                "source": "deterministic_fallback",
            }
        except Exception as fallback_err:
            logger.error(f"[AIParser] Fallback parsing also failed: {fallback_err}", exc_info=True)
            return ImprovedUniversalResumeParser._empty_result()

    @staticmethod
    async def parse_text(
        text_content: str,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Parse extracted text content using Gemini structured output with deterministic fallback.
        """
        if not text_content or len(text_content.strip()) < 20:
            logger.warning("[AIParser] Empty or insufficient text for parsing.")
            return ImprovedUniversalResumeParser._empty_result()

        try:
            client = get_gemini_client()
            if client.is_configured:
                prompt = (
                    "Extract the complete structured information from the following resume text:\n\n"
                    f"{text_content}"
                )

                raw_json = await client.generate(
                    prompt=prompt,
                    system_instruction=EXTRACTION_SYSTEM_PROMPT,
                    response_schema=CanonicalResume,
                    operation="resume_parsing_text",
                    user_id=user_id,
                    request_id=request_id,
                )

                canonical = CanonicalResume.model_validate_json(raw_json)
                legacy_dict = map_to_legacy_parse_dict(canonical)

                return {
                    "success": True,
                    "parsed": legacy_dict,
                    "canonical": canonical.model_dump(),
                    "source": "gemini",
                }

        except Exception as e:
            logger.warning(f"[AIParser] Gemini text parsing failed ({e}). Activating deterministic AST fallback...")

        # ── Deterministic Fallback ──
        try:
            canonical = DeterministicResumeParser.parse_text(text_content)
            legacy_dict = map_to_legacy_parse_dict(canonical)
            return {
                "success": True,
                "parsed": legacy_dict,
                "canonical": canonical.model_dump(),
                "source": "deterministic_fallback",
            }
        except Exception as fallback_err:
            logger.error(f"[AIParser] Fallback parsing failed: {fallback_err}")
            return ImprovedUniversalResumeParser._empty_result()

    @staticmethod
    async def parse(extractor_output: Any) -> Dict[str, Any]:
        """
        Backward-compatible parse method supporting raw_items lists, text keys, or raw strings.
        """
        if not extractor_output:
            return ImprovedUniversalResumeParser._empty_result()

        raw_items = []
        text_content = ""

        if isinstance(extractor_output, str):
            text_content = extractor_output
        elif isinstance(extractor_output, dict):
            raw_items = extractor_output.get("raw_items", [])
            if raw_items:
                text_content = UniversalExtractor.extract_all_content(raw_items)
            elif "text" in extractor_output:
                text_content = str(extractor_output["text"])
            elif "raw_text" in extractor_output:
                text_content = str(extractor_output["raw_text"])
            elif "content" in extractor_output:
                text_content = str(extractor_output["content"])
            elif "parsed" in extractor_output and isinstance(extractor_output["parsed"], dict):
                return {"success": True, "parsed": extractor_output["parsed"], "source": "pre_parsed"}

        if not text_content or len(text_content.strip()) < 10:
            return ImprovedUniversalResumeParser._empty_result()

        result = await ImprovedUniversalResumeParser.parse_text(text_content)

        # Regex-based contact safety net to ensure zero contact data is missed
        if raw_items:
            contact_info = UniversalExtractor.extract_contact_info_raw(raw_items)
            if result.get("success") and result.get("parsed"):
                header = result["parsed"].get("header", {})
                for field in ("name", "email", "phone", "location"):
                    if not header.get(field) and contact_info.get(field):
                        header[field] = contact_info[field]
                result["parsed"]["header"] = header

        return result

    @staticmethod
    def _empty_result() -> Dict[str, Any]:
        return {
            "success": False,
            "parsed": {
                "header": {"name": "", "email": "", "phone": "", "location": "", "link": "", "title": ""},
                "summary": {"summary": ""},
                "education": [],
                "experience": [],
                "skills": [],
                "projects": [],
                "certifications": [],
                "languages": [],
                "other": [],
            },
            "token_report": {},
            "failed_sections": [],
        }
