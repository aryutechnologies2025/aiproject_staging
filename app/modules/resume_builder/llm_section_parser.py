import json
import logging
import asyncio
from typing import Dict, List, Any
import re
from .ai_client import call_ai
from .llm_section_identifier import rate_limited_call

logger = logging.getLogger(__name__)


class LLMSectionParser:
    """
    Parse each section independently with structured output.
    Uses LLM to understand context and extract data intelligently.
    """
    
    @staticmethod
    async def parse_sections(sections: Dict[str, str]) -> Dict[str, Any]:
        """Parse all sections in parallel"""
        
        tasks = []
        section_names = []
        
        # Priority order to optimize rate limits
        priority = ["experience", "projects", "education", "skills", "summary", 
                   "certifications", "languages", "header"]
        
        for section_name in priority:
            section_text = sections.get(section_name, "").strip()
            if section_text:
                tasks.append(LLMSectionParser.parse_section(section_name, section_text))
                section_names.append(section_name)
        
        # Parse other sections
        for section_name, section_text in sections.items():
            if section_name not in priority and section_text.strip():
                tasks.append(LLMSectionParser.parse_section(section_name, section_text))
                section_names.append(section_name)
        
        if not tasks:
            return LLMSectionParser._empty_parsed_sections()
        
        try:
            logger.info(f"🔄 STAGE 2: Parsing {len(tasks)} sections in parallel...")
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            parsed_sections = {}
            for section_name, result in zip(section_names, results):
                if isinstance(result, Exception):
                    logger.warning(f"⚠ Failed to parse {section_name}: {str(result)[:100]}")
                    parsed_sections[section_name] = LLMSectionParser._empty_section(section_name)
                else:
                    parsed_sections[section_name] = result
                    logger.info(f"✓ Parsed {section_name}")
            
            # Fill missing sections
            for section_name in ["header", "summary", "education", "experience", 
                                "skills", "projects", "certifications", "languages"]:
                if section_name not in parsed_sections:
                    parsed_sections[section_name] = LLMSectionParser._empty_section(section_name)
            
            return parsed_sections
        
        except Exception as e:
            logger.error(f"Parallel parsing failed: {str(e)}")
            return LLMSectionParser._empty_parsed_sections()
    
    @staticmethod
    async def parse_section(section_name: str, section_text: str) -> Any:
        """Parse individual section with intelligent LLM prompt"""
        
        prompts = LLMSectionParser._build_prompts()
        prompt = prompts.get(section_name)
        
        if not prompt:
            logger.warning(f"No prompt for section: {section_name}")
            return LLMSectionParser._empty_section(section_name)
        
        if section_name == "education":
            section_text = LLMSectionParser._preprocess_education_content(section_text)

        prompt_with_content = prompt.format(content=section_text[:5000])
        
        token_limits = {
            "experience": 3000,
            "projects": 3000,
            "education": 2000,
            "skills": 1500,
            "summary": 1500,
            "certifications": 1500,
            "languages": 1000,
            "header": 1000,
        }
        
        max_tokens = token_limits.get(section_name, 1500)
        
        for attempt in range(2):
            try:
                response = await rate_limited_call(
                    call_ai,
                    prompt=prompt_with_content,
                    max_output_tokens=max_tokens,
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
                
                parsed = json.loads(text)
                
                # Validate non-empty
                if LLMSectionParser._is_valid_parsed(section_name, parsed):
                    return parsed
                elif attempt < 1:
                    await asyncio.sleep(1)
            
            except json.JSONDecodeError:
                logger.warning(f"JSON error for {section_name}, attempt {attempt + 1}")
                if attempt < 1:
                    await asyncio.sleep(1)
            except Exception as e:
                logger.warning(f"Error parsing {section_name}: {str(e)[:80]}")
                if attempt < 1:
                    await asyncio.sleep(1)
        
        return LLMSectionParser._empty_section(section_name)
    
    @staticmethod
    def _build_prompts() -> Dict[str, str]:
        """Build specialized prompts for each section"""
        
        return {
            "header": """Extract personal/header information from:
{content}

Return JSON:
{{"name": "full name", "title": "job title/profession", "location": "city, country", "email": "", "phone": "", "link": "portfolio/linkedin url"}}

Rules:
- Extract exact name as written
- Extract professional title/role
- Extract location (city, state/country)
- Leave email, phone, link empty here (will be extracted separately)
- Return only the JSON structure""",
            
            "summary": """Extract professional summary/objective from:
{content}

Return JSON:
{{"summary": "complete summary text as single paragraph"}}

Rules:
- Extract COMPLETE summary text
- Join all bullet points into ONE paragraph
- Include ALL information
- Keep exact wording
- Result should be full text, not truncated""",
            
            "experience": """Extract ALL work experience/jobs from:
{content}

Return JSON array (INCLUDE EVERY JOB):
[{{"position": "job title", "company": "company name", "location": "location", "fromYear": "YYYY", "toYear": "YYYY or Present", "isOngoing": true/false, "description": ["one sentence summary"], "bullets": ["achievement 1", "achievement 2"]}}]

Rules:
- Extract EVERY job mentioned - DO NOT skip any
- Include full job title
- Include company name exactly as written
- Extract years only (e.g. "2024", not full dates)
- isOngoing: true only if "Present" or "Currently"
- bullets: Array of ALL achievements/points listed
- DO NOT merge jobs
- Include internships, part-time, contract work""",
            
            "education": """Extract ALL education from:
{content}

Return JSON array (INCLUDE EVERY ENTRY):
[{{"degree": "degree name", "institution": "school/university name", "location": "city, state/country", "fromYear": "YYYY", "toYear": "YYYY"}}]

Rules:
- Extract EVERY degree/qualification/certification mentioned
- If no clear structure, parse carefully: look for patterns like "Degree Name — University Name — Location — Years"
- degree: Full name of degree (e.g., "Bachelor of Science in Computer Science")
- institution: Name of school/university/college exactly as written
- location: City and state/country (e.g., "Chennai, Tamil Nadu, India")
- fromYear: Start year as 4 digits (e.g., "2021")
- toYear: End/graduation year as 4 digits (e.g., "2024")
- Include all types: bachelors, masters, diplomas, certifications
- If year format is "09/2021 – 04/2024", extract as fromYear: "2021", toYear: "2024"
- Empty string "" for missing fields
- Return as valid JSON array even if single entry""",
            
            "skills": """Extract ALL skills from:
{content}

Return JSON array:
["skill1", "skill2", "skill3", "skill4", "skill5"]

Rules:
- Extract ALL skills mentioned: languages, frameworks, tools, databases, libraries
- Extract from all subsections
- Return as flat array of strings
- Remove exact duplicates but keep all unique skills
- Include everything mentioned""",
            
            "projects": """Extract ALL projects from:
{content}

Return JSON array (INCLUDE EVERY PROJECT):
[{{"title": "project name/title", "description": "1-2 sentence description", "technologies": ["tech1", "tech2"], "fromYear": "YYYY", "toYear": "YYYY", "bullets": ["achievement 1", "achievement 2"]}}]

Rules:
- Extract EVERY project mentioned - DO NOT skip any
- Include full project title
- Extract technologies as array
- Include all bullet points/achievements for each project
- Extract years if available
- DO NOT merge projects
- Include portfolio, case studies, work examples""",
            
            "certifications": """Extract ALL certifications from:
{content}

Return JSON array (INCLUDE EVERY CERTIFICATION):
[{{"title": "certification name", "issuer": "issuing organization"}}]

Rules:
- Extract EVERY certification/license/award mentioned
- Include exact title
- Include issuing organization
- Extract all entries""",
            
            "languages": """Extract ALL languages from:
{content}

Return JSON array:
["language1", "language2", "language3"]

Rules:
- Extract all languages mentioned
- Include proficiency level in the language if mentioned
- Return as array of strings""",
        }
    
    @staticmethod
    def _preprocess_education_content(section_text: str) -> str:
        """Preprocess education section to improve parsing"""
        # Clean up common education section formats
        
        # Handle table format education
        if "|" in section_text and "degree" in section_text.lower():
            # Convert table to readable format
            lines = section_text.split("\n")
            cleaned_lines = []
            for line in lines:
                if "|" in line and "---" not in line:
                    # Remove table separators, keep content
                    parts = [p.strip() for p in line.split("|")]
                    cleaned_lines.append(" ".join([p for p in parts if p]))
            section_text = "\n".join(cleaned_lines)
        
        # Normalize date formats
        # "09/2021 – 04/2024" -> clearly mark years
        section_text = re.sub(r'(\d{2})/(\d{4})\s*–\s*(\d{2})/(\d{4})', 
                             r'\2 to \4', section_text)
        section_text = re.sub(r'(\d{4})\s*–\s*(\d{4})', r'\1 to \2', section_text)
        
        # Normalize separators (—, -, to, at)
        section_text = re.sub(r'\s*[-–—]\s*', ' | ', section_text)
        
        return section_text
    
    @staticmethod
    def _is_valid_parsed(section_name: str, parsed: Any) -> bool:
        """Check if parsed section has content"""
        if not parsed:
            return False
        
        if section_name == "header":
            return isinstance(parsed, dict) and any(parsed.get(k) for k in ["name", "title"])
        elif section_name == "summary":
            summary = parsed.get("summary", "") if isinstance(parsed, dict) else str(parsed)
            return isinstance(summary, str) and len(summary.strip()) > 20
        elif section_name == "education":
            # Education can be empty or array
            if isinstance(parsed, list):
                return len(parsed) >= 0  # Accept even empty array
            return isinstance(parsed, dict)
        elif section_name in ["skills", "languages"]:
            return isinstance(parsed, list) and len(parsed) > 0
        elif section_name in ["experience", "projects", "certifications"]:
            return isinstance(parsed, list) and len(parsed) > 0
        else:
            return bool(parsed)
    
    @staticmethod
    def _empty_section(section_name: str) -> Any:
        """Return empty structure"""
        if section_name == "header":
            return {"name": "", "title": "", "location": "", "email": "", "phone": "", "link": ""}
        elif section_name == "summary":
            return {"summary": ""}
        elif section_name in ["skills", "languages"]:
            return []
        else:
            return []
    
    @staticmethod
    def _empty_parsed_sections() -> Dict[str, Any]:
        """Return all empty sections"""
        return {
            "header": {"name": "", "title": "", "location": "", "email": "", "phone": "", "link": ""},
            "summary": {"summary": ""},
            "education": [],
            "experience": [],
            "skills": [],
            "projects": [],
            "certifications": [],
            "languages": [],
        }