import logging
from typing import Dict, Any

from .universal_extractor import UniversalExtractor
from .llm_section_identifier import LLMSectionIdentifier
from .llm_section_parser import LLMSectionParser
from .batch_processor import BatchProcessor
from .adaptive_token_manager import AdaptiveTokenManager

logger = logging.getLogger(__name__)


class ImprovedUniversalResumeParser:

    @staticmethod
    async def parse(extractor_output: Dict[str, Any]) -> Dict[str, Any]:
        try:
            raw_items = extractor_output.get("raw_items", [])
            if not raw_items:
                return ImprovedUniversalResumeParser._empty_result()

            # --- Step 1: Identify sections via LLM ---------------------------
            identified_sections = await LLMSectionIdentifier.identify_and_extract_sections(raw_items)

            # --- Step 2: Prepare batch processor & token manager -------------
            batch_processor = BatchProcessor(max_concurrent=2)
            token_manager = AdaptiveTokenManager(total_tokens=20000)

            for section_name, content in identified_sections.items():
                if not str(content).strip():
                    continue
                will_overflow, _ = token_manager.check_overflow(section_name, content)
                if will_overflow:
                    max_chars = token_manager.get_max_input_chars(section_name)
                    content = token_manager.compress_content(section_name, content, max_chars)
                token_manager.record_usage(section_name, len(content))
                await batch_processor.add_section(section_name, content)

            # --- Step 3: Parse each section via LLM --------------------------
            parsed_sections = await batch_processor.process_all(LLMSectionParser.parse_section)

            # Fill in empty defaults for any failed sections
            for task in batch_processor.get_failed_tasks():
                parsed_sections[task.section_name] = ImprovedUniversalResumeParser._empty_section(
                    task.section_name
                )

            # --- Step 4: Merge into final result -----------------------------
            final = ImprovedUniversalResumeParser._empty_result()["parsed"]
            final.update(parsed_sections)

            # --- Step 5: Regex-based contact patch (universal safety net) ----
            # This runs regardless of LLM success to ensure no contact field is lost.
            contact_info = UniversalExtractor.extract_contact_info_raw(raw_items)
            header = final.get("header", {})
            if not isinstance(header, dict):
                header = {}

            # Patch each header field: use LLM value if present, else regex fallback
            for field in ("name", "email", "phone", "location"):
                if not header.get(field) and contact_info.get(field):
                    header[field] = contact_info[field]

            # Merge link field: combine LLM links + regex-found links, deduplicate
            llm_links = set(
                lnk.strip()
                for lnk in re.split(r"[,\s]+", header.get("link", ""))
                if lnk.strip()
            ) if header.get("link") else set()

            regex_links = set()
            for key in ("linkedin", "github", "portfolio"):
                val = contact_info.get(key, "").strip()
                if val:
                    regex_links.add(val)

            all_links = llm_links | regex_links
            if all_links:
                header["link"] = ", ".join(sorted(all_links))

            final["header"] = header

            # --- Step 6: Token reporting -------------------------------------
            token_report = token_manager.get_token_report()
            logger.info(
                f"Token usage: {token_report['total_used']}/{token_report['total_budget']} "
                f"({token_report['utilization_percent']}%)"
            )

            return {
                "success": True,
                "parsed": final,
                "token_report": token_report,
                "failed_sections": [t.section_name for t in batch_processor.get_failed_tasks()],
            }

        except Exception as e:
            logger.error(f"Parsing failed: {str(e)}", exc_info=True)
            return ImprovedUniversalResumeParser._empty_result()

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _empty_result() -> Dict[str, Any]:
        return {
            "success": False,
            "parsed": {
                "header": {
                    "name": "", "email": "", "phone": "",
                    "location": "", "link": "", "title": "",
                },
                "summary": {"summary": ""},
                "education": [],
                "experience": [],
                "skills": [],
                "projects": [],
                "certifications": [],
                "languages": [],
            },
            "token_report": {},
            "failed_sections": [],
        }

    @staticmethod
    def _empty_section(section_name: str) -> Any:
        empties = {
            "header": {"name": "", "title": "", "email": "", "phone": "", "location": "", "link": ""},
            "summary": {"summary": ""},
        }
        return empties.get(section_name, [])


# Re-export so existing imports continue to work
import re  # noqa: E402 — needed by the link-merge logic above
