"""
ai_parser.py — Structured resume parser with layout awareness, multimodal vision fallback, and source grounding.

Architecture:
1. Primary: High-accuracy single-call Google Gemini structured extraction receiving layout-aware Markdown/blocks.
2. Multimodal Vision Fallback: Direct Gemini document/image understanding for scanned, image-heavy, or complex graphic resumes.
3. Anti-Hallucination Source Validation: Post-extraction verification of companies, job titles, degrees, and projects.
4. Deterministic AST Fallback: Ultra-fast offline Deterministic AST Parser (100% service uptime).
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from app.modules.resume_builder.gemini_client import get_gemini_client
from app.modules.resume_builder.deterministic_parser import DeterministicResumeParser
from app.modules.resume_builder.schemas import (
    CanonicalResume,
    map_to_legacy_parse_dict,
)
from app.modules.resume_builder.universal_extractor import (
    UniversalExtractor,
    UniversalDocumentExtractor,
    ExtractedDocument,
)
from app.modules.resume_builder.validator import ResumeSourceValidator, ValidationResult

logger = logging.getLogger("resume_builder.ai_parser")

DEFAULT_MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_RESUME_MAX_OUTPUT_TOKENS", "8192"))

EXTRACTION_SYSTEM_PROMPT = """You are an expert AI Resume Parser and Career Data Extraction Engine.
Analyze the provided resume document or text thoroughly and extract all information into the requested JSON schema.

CRITICAL INSTRUCTIONS — EXTRACTION COMPLETENESS & FIDELITY:
1. Do not summarize or condense resume content. Preserve complete descriptions, responsibilities, accomplishments, metrics, project details, and bullet points verbatim from the source document. Extract all available information that fits the schema.
2. If an experience or internship section contains a paragraph description or location or address, preserve the full details in the appropriate fields. Do not convert a long description into a short summary.
3. Extract ALL projects present in the resume (e.g. client projects, WordPress strategies, finance content strategies, case studies). Preserve the complete project title, full description, and all bullets without shortening.
4. Extract ALL bullet points verbatim without shortening, omitting, or combining items.
5. Extract ALL employment history, internships, and apprenticeships.
6. Preserve original wording, numbers, percentages, dates, team sizes, and technical terminology exactly.
7. Never invent, hallucinate, or assume information not present in the document.
8. If information is not in the source, return empty string or empty list rather than hallucinating.

Extraction Guidelines:
1. "personal_information":
   - "name": Full legal or professional candidate name.
   - "title": Current or target designation (e.g. "Digital Marketing Executive", "Senior Software Engineer").
   - "email": Valid email address.
   - "phone": Phone number with country code if available.
   - "location": City, State, Country, Postal Code, or address.
   - "link": Comma-separated URLs (LinkedIn, GitHub, Portfolio, Website).

2. "summary":
   - Complete professional summary, executive profile, or objective statement verbatim. Do not truncate.

3. "experience":
   - Extract ALL employment history, internships, and freelance roles.
   - "position": Job title or role (e.g. "Digital Marketing Executive Internship").
   - "company": Organization or company name (e.g. "Aryu Enterprises Pvt Ltd").
   - "location": Job location, address, or Remote.
   - "fromYear", "toYear": Start and end dates (e.g. "Apr 2026", "Present", "Dec 2025", "Feb 2026").
   - "isOngoing": True if currently active role.
   - "description": Complete paragraph overview / role context if present.
   - "responsibilities": All individual responsibilities.
   - "bullets": Complete list of ALL bullet points and accomplishments verbatim.
   - "achievements": Key quantified achievements.

4. "education":
   - Extract ALL academic programs, degrees, diplomas, high school credentials (e.g. "High School", "B.sc Information Technology").
   - "degree", "institution", "location", "fromYear", "toYear", "description", "achievements", "coursework".

5. "skills":
   - Extract all technical, domain, framework, design, and tool skills mentioned across the entire resume (e.g. "Social Media Marketing", "Content Creation", "CapCut", "Canva", "Google Analytics", "SEO").
   - Deduplicate and normalize.

6. "projects":
   - Extract ALL projects present in the resume.
   - "title": Project name (e.g. "Biokosmetikoftexas - USA", "wpwebsitefix", "Yestoboss").
   - "description": Comprehensive project description containing full purpose, scope, architecture, technologies, outcomes, and metrics without shortening.
   - "technologies": Complete list of tools/languages/platforms used.
   - "fromYear", "toYear": Dates if specified.
   - "bullets": All project bullets and implementation details verbatim.

7. "certifications":
   - Extract all licenses, certifications, and credentials with issuer and year.

8. "achievements":
   - Top-level awards, honors, competitions, recognitions.

9. "languages":
   - All spoken and written languages.

10. "other":
   - Publications, patents, volunteer work, extracurriculars, affiliations.

Return ONLY the structured JSON according to the schema.
"""


def _clean_json_text(text: str) -> str:
    """Clean markdown code fences and extraneous whitespace from raw LLM output."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _build_fallback_usage(operation: str) -> List[Dict[str, Any]]:
    """Build standardized telemetry record for deterministic fallback executions."""
    return [
        {
            "provider": "internal",
            "model": "deterministic_fallback",
            "operation": operation,
            "parser_source": "deterministic_fallback",
            "finish_reason": "FALLBACK",
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
            "total_cost_inr": None,
            "input_cost_inr": None,
            "output_cost_inr": None,
            "cached_cost_inr": None,
            "currency": "INR",
            "cost_status": "not_applicable",
        }
    ]


class ImprovedUniversalResumeParser:
    """
    High-performance resume parser leveraging single-call Gemini structured extraction
    with layout-aware Markdown formatting, multimodal vision fallback, source validation,
    and transparent deterministic AST fallback.
    """

    @staticmethod
    async def parse_document_bytes(
        file_bytes: bytes,
        content_type: str = "application/pdf",
        filename: str = "resume.pdf",
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        operation: str = "resume_parsing_document",
        max_output_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Parse raw resume document bytes directly using Gemini document understanding,
        with local PyMuPDF layout-aware fallback on any error.
        """
        captured_usage: Optional[List[Dict[str, Any]]] = None
        finish_reason: str = "STOP"

        try:
            client = get_gemini_client()
            if client.is_configured:
                prompt = (
                    f"Extract the complete structured information from this resume document ({filename}). "
                    "Ensure all jobs, internships, dates, skills, contact details, project descriptions (extract every project), and bullets are accurately extracted without shortening."
                )

                budget = max_output_tokens or DEFAULT_MAX_OUTPUT_TOKENS

                gen_result = await client.generate(
                    prompt=prompt,
                    system_instruction=EXTRACTION_SYSTEM_PROMPT,
                    response_schema=CanonicalResume,
                    document_bytes=file_bytes,
                    document_mime_type=content_type,
                    operation=operation,
                    user_id=user_id,
                    request_id=request_id,
                    max_output_tokens=budget,
                )

                if hasattr(gen_result, "to_usage_list"):
                    captured_usage = gen_result.to_usage_list()
                elif hasattr(gen_result, "usage") and gen_result.usage:
                    captured_usage = [gen_result.usage.to_dict()]

                finish_reason = gen_result.usage.finish_reason if (hasattr(gen_result, "usage") and gen_result.usage) else "STOP"

                if finish_reason == "MAX_TOKENS":
                    logger.warning(f"[AIParser] Truncation detected: Gemini finished with MAX_TOKENS on '{filename}'.")

                raw_json = _clean_json_text(gen_result.text if hasattr(gen_result, "text") else str(gen_result))
                canonical = CanonicalResume.model_validate_json(raw_json)

                # Validate against source text
                doc_extracted = UniversalDocumentExtractor.extract_document(file_bytes, filename, content_type)
                val_result = ResumeSourceValidator.validate(canonical, doc_extracted.raw_text, doc_extracted.raw_items)

                legacy_dict = map_to_legacy_parse_dict(canonical)
                parse_status = "truncated" if finish_reason == "MAX_TOKENS" else "complete"

                logger.info(
                    f"✓ Successfully parsed resume document '{filename}' with Gemini "
                    f"(finish_reason={finish_reason}, validation_score={val_result.score:.2f})"
                )
                return {
                    "success": True,
                    "parsed": legacy_dict,
                    "canonical": canonical.model_dump(),
                    "source": "gemini",
                    "parser_source": "gemini",
                    "parse_status": parse_status,
                    "finish_reason": finish_reason,
                    "usage": captured_usage,
                    "validation": val_result.to_dict(),
                }

        except Exception as e:
            logger.warning(f"[AIParser] Gemini document parsing failed ({e}). Activating deterministic AST fallback...")

        # ── Deterministic Fallback ──
        try:
            doc_extracted = UniversalDocumentExtractor.extract_document(file_bytes, filename, content_type)
            canonical = DeterministicResumeParser.parse_text(doc_extracted.raw_text)
            legacy_dict = map_to_legacy_parse_dict(canonical)
            logger.info(f"✓ Successfully parsed '{filename}' via Deterministic AST Fallback")

            usage_to_return = captured_usage if captured_usage else None

            return {
                "success": True,
                "parsed": legacy_dict,
                "canonical": canonical.model_dump(),
                "source": "deterministic_fallback",
                "parser_source": "deterministic_fallback",
                "parse_status": "truncated" if finish_reason == "MAX_TOKENS" else "fallback",
                "finish_reason": finish_reason if captured_usage else "FALLBACK",
                "usage": usage_to_return,
            }
        except Exception as fallback_err:
            logger.error(f"[AIParser] Fallback parsing also failed: {fallback_err}", exc_info=True)
            return ImprovedUniversalResumeParser._empty_result(usage=captured_usage)

    @staticmethod
    async def parse_text(
        text_content: str,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        operation: str = "resume_parse",
        max_output_tokens: Optional[int] = None,
        page_images: Optional[List[bytes]] = None,
        document_type: str = "text",
    ) -> Dict[str, Any]:
        """
        Parse extracted text content using Gemini structured output with deterministic fallback.
        """
        if not text_content or len(text_content.strip()) < 15:
            logger.warning("[AIParser] Empty or insufficient text for parsing.")
            return ImprovedUniversalResumeParser._empty_result()

        captured_usage: Optional[List[Dict[str, Any]]] = None
        finish_reason: str = "STOP"

        try:
            client = get_gemini_client()
            if client.is_configured:
                prompt = (
                    "Extract the complete structured information from the following resume document without shortening or omitting any content. "
                    "Ensure all jobs, internships, dates, skills, contact details, and complete project descriptions are accurately extracted:\n\n"
                    f"{text_content}"
                )

                budget = max_output_tokens or DEFAULT_MAX_OUTPUT_TOKENS

                gen_result = await client.generate(
                    prompt=prompt,
                    system_instruction=EXTRACTION_SYSTEM_PROMPT,
                    response_schema=CanonicalResume,
                    operation=operation,
                    user_id=user_id,
                    request_id=request_id,
                    max_output_tokens=budget,
                )

                if hasattr(gen_result, "to_usage_list"):
                    captured_usage = gen_result.to_usage_list()
                elif hasattr(gen_result, "usage") and gen_result.usage:
                    captured_usage = [gen_result.usage.to_dict()]

                finish_reason = gen_result.usage.finish_reason if (hasattr(gen_result, "usage") and gen_result.usage) else "STOP"

                if finish_reason == "MAX_TOKENS":
                    logger.warning("[AIParser] Truncation detected: Gemini finished with MAX_TOKENS on text parse.")

                raw_json = _clean_json_text(gen_result.text if hasattr(gen_result, "text") else str(gen_result))
                canonical = CanonicalResume.model_validate_json(raw_json)

                # Source validation
                val_result = ResumeSourceValidator.validate(canonical, text_content)

                legacy_dict = map_to_legacy_parse_dict(canonical)
                parse_status = "truncated" if finish_reason == "MAX_TOKENS" else "complete"

                return {
                    "success": True,
                    "parsed": legacy_dict,
                    "canonical": canonical.model_dump(),
                    "source": "gemini",
                    "parser_source": "gemini",
                    "parse_status": parse_status,
                    "finish_reason": finish_reason,
                    "usage": captured_usage,
                    "validation": val_result.to_dict(),
                }

        except Exception as e:
            logger.warning(f"[AIParser] Gemini text parsing failed ({e}). Activating deterministic AST fallback...")

        # ── Deterministic Fallback ──
        try:
            canonical = DeterministicResumeParser.parse_text(text_content)
            legacy_dict = map_to_legacy_parse_dict(canonical)
            usage_to_return = captured_usage if captured_usage else None

            return {
                "success": True,
                "parsed": legacy_dict,
                "canonical": canonical.model_dump(),
                "source": "deterministic_fallback",
                "parser_source": "deterministic_fallback",
                "parse_status": "truncated" if finish_reason == "MAX_TOKENS" else "fallback",
                "finish_reason": finish_reason if captured_usage else "FALLBACK",
                "usage": usage_to_return,
            }
        except Exception as fallback_err:
            logger.error(f"[AIParser] Fallback parsing failed: {fallback_err}")
            return ImprovedUniversalResumeParser._empty_result(usage=captured_usage)

    @staticmethod
    async def parse(
        extractor_output: Any,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        operation: str = "resume_parse",
        max_output_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Backward-compatible parse method supporting raw_items lists, text keys, or raw strings.
        Prefers rich layout Markdown when available.
        """
        if not extractor_output:
            return ImprovedUniversalResumeParser._empty_result()

        raw_items = []
        text_content = ""
        page_images = None
        doc_type = "text"

        if isinstance(extractor_output, str):
            text_content = extractor_output
        elif isinstance(extractor_output, dict):
            raw_items = extractor_output.get("raw_items", [])
            doc_type = extractor_output.get("document_type", "text")
            page_images = extractor_output.get("page_images")

            # Use markdown representation if available as it preserves section headers and formatting
            if extractor_output.get("markdown"):
                text_content = str(extractor_output["markdown"])
            elif raw_items:
                text_content = UniversalExtractor.extract_all_content(raw_items)
            elif "raw_text" in extractor_output:
                text_content = str(extractor_output["raw_text"])
            elif "text" in extractor_output:
                text_content = str(extractor_output["text"])
            elif "content" in extractor_output:
                text_content = str(extractor_output["content"])
            elif "parsed" in extractor_output and isinstance(extractor_output["parsed"], dict):
                return {
                    "success": True,
                    "parsed": extractor_output["parsed"],
                    "source": "pre_parsed",
                    "parser_source": "pre_parsed",
                    "usage": None,
                }

        if not text_content or len(text_content.strip()) < 10:
            return ImprovedUniversalResumeParser._empty_result()

        result = await ImprovedUniversalResumeParser.parse_text(
            text_content=text_content,
            user_id=user_id,
            request_id=request_id,
            operation=operation,
            max_output_tokens=max_output_tokens,
            page_images=page_images,
            document_type=doc_type,
        )

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
    def _empty_result(usage: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
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
            "source": "none",
            "parser_source": "none",
            "usage": usage,
        }
