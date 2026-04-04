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
    Optimized for token usage and deduplication.
    """
    
    @staticmethod
    async def parse_sections(sections: Dict[str, str]) -> Dict[str, Any]:
        """Parse all sections in parallel with deduplication"""
        
        tasks = []
        section_names = []
        
        # Priority order to optimize rate limits
        priority = ["experience", "projects", "education", "skills", "summary", 
                   "certifications", "languages", "header"]
        
        for section_name in priority:
            section_text = sections.get(section_name, "").strip()
            if section_text:
                # Optimize section text before parsing
                optimized_text = LLMSectionParser._optimize_section_text(section_name, section_text)
                tasks.append(LLMSectionParser.parse_section(section_name, optimized_text))
                section_names.append(section_name)
        
        # Parse other sections
        for section_name, section_text in sections.items():
            if section_name not in priority and section_text.strip():
                optimized_text = LLMSectionParser._optimize_section_text(section_name, section_text)
                tasks.append(LLMSectionParser.parse_section(section_name, optimized_text))
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
                    # Deduplicate result
                    parsed_sections[section_name] = LLMSectionParser._deduplicate_result(section_name, result)
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
        
        # Preprocess education to handle duplicates
        if section_name == "education":
            section_text = LLMSectionParser._preprocess_education_content(section_text)

        # Reduce input size (optimize tokens)
        prompt_with_content = prompt.format(content=section_text[:3500])
        
        # Optimized token limits (reduced by 40-50%)
        token_limits = {
            "experience": 1500,      # Reduced from 3000
            "projects": 1500,        # Reduced from 3000
            "education": 800,        # Reduced from 2000
            "skills": 600,           # Reduced from 1500
            "summary": 800,          # Reduced from 1500
            "certifications": 700,   # Reduced from 1500
            "languages": 500,        # Reduced from 1000
            "header": 600,           # Reduced from 1000
        }
        
        max_tokens = token_limits.get(section_name, 800)
        
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
    def _optimize_section_text(section_name: str, section_text: str) -> str:
        """
        Optimize section text before LLM processing
        Reduces token usage by removing noise while preserving content
        """
        
        if section_name == "education":
            return LLMSectionParser._optimize_education(section_text)
        elif section_name == "experience":
            return LLMSectionParser._optimize_experience(section_text)
        elif section_name == "projects":
            return LLMSectionParser._optimize_projects(section_text)
        elif section_name == "skills":
            return LLMSectionParser._optimize_skills(section_text)
        elif section_name == "summary":
            return LLMSectionParser._optimize_summary(section_text)
        else:
            return section_text[:3500]
    
    @staticmethod
    def _optimize_education(section_text: str) -> str:
        """Optimize education section - keep only relevant content"""
        lines = section_text.split("\n")
        optimized = []
        
        for line in lines:
            line = line.strip()
            if not line or (line.startswith("|") and "---" in line):
                continue
            
            keywords = ["degree", "bachelor", "master", "phd", "diploma", 
                       "university", "college", "institute", "school", "gpa", "cgpa"]
            
            if any(kw in line.lower() for kw in keywords):
                optimized.append(line)
            elif any(year in line for year in ["20" + str(i) for i in range(10, 30)]):
                if line not in optimized:
                    optimized.append(line)
        
        return "\n".join(optimized)
    
    @staticmethod
    def _optimize_experience(section_text: str) -> str:
        """Optimize experience section"""
        lines = section_text.split("\n")
        optimized = []
        
        for line in lines:
            line = line.strip()
            if not line or (line.startswith("|") and "---" in line):
                continue
            
            # Keep bullets, job titles, dates
            if line.startswith("•") or line.startswith("*") or line.startswith("-"):
                optimized.append(line)
            elif any(year in line for year in ["20" + str(i) for i in range(10, 30)]):
                optimized.append(line)
            elif len(line) > 10 and line[0].isupper() and any(c.isupper() for c in line[1:]):
                optimized.append(line)
        
        return "\n".join(optimized)
    
    @staticmethod
    def _optimize_projects(section_text: str) -> str:
        """Optimize projects section"""
        lines = section_text.split("\n")
        optimized = []
        
        for line in lines:
            line = line.strip()
            if not line or (line.startswith("|") and "---" in line):
                continue
            
            # Keep project titles and achievements
            if any(kw in line.lower() for kw in ["platform", "system", "app", "website", "application"]):
                optimized.append(line)
            elif line.startswith("•") or line.startswith("*") or line.startswith("-"):
                optimized.append(line)
            elif "," in line and any(tech in line.lower() for tech in 
                                    ["react", "node", "python", "java", "javascript"]):
                optimized.append(line)
        
        return "\n".join(optimized)
    
    @staticmethod
    def _optimize_skills(section_text: str) -> str:
        """Optimize skills section - remove redundancy"""
        lines = section_text.split("\n")
        skills = []
        seen = set()
        
        for line in lines:
            line = line.strip()
            if not line or (line.startswith("|") and "---" in line):
                continue
            
            line = line.lstrip("•*- ").strip()
            
            if "," in line:
                for skill in line.split(","):
                    skill = skill.strip().lower()
                    if skill and len(skill) > 2 and skill not in seen:
                        seen.add(skill)
                        skills.append(skill)
            else:
                skill = line.lower()
                if line and len(line) > 2 and skill not in seen:
                    seen.add(skill)
                    skills.append(line)
        
        return "\n".join(skills)
    
    @staticmethod
    def _optimize_summary(section_text: str) -> str:
        """Optimize summary - remove duplicates and preserve content"""
        lines = section_text.split("\n")
        unique_lines = []
        seen = set()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            line = line.lstrip("•*- ").strip()
            
            if line not in seen:
                seen.add(line)
                unique_lines.append(line)
        
        return " ".join(unique_lines)
    
    @staticmethod
    def _deduplicate_result(section_name: str, parsed: Any) -> Any:
        """
        Deduplicate parsed results to prevent duplicates
        Fixes education and other sections with duplicate entries
        """
        
        if section_name == "education" and isinstance(parsed, list):
            return LLMSectionParser._deduplicate_education(parsed)
        
        elif section_name == "experience" and isinstance(parsed, list):
            return LLMSectionParser._deduplicate_experience(parsed)
        
        elif section_name == "projects" and isinstance(parsed, list):
            return LLMSectionParser._deduplicate_projects(parsed)
        
        elif section_name == "skills" and isinstance(parsed, list):
            return LLMSectionParser._deduplicate_skills(parsed)
        
        return parsed
    
    @staticmethod
    def _deduplicate_education(education_list: List[Dict]) -> List[Dict]:
        """Remove duplicate education entries"""
        if not education_list:
            return []
        
        unique = []
        seen = set()
        
        for edu in education_list:
            # Create key from normalized fields
            key = (
                (edu.get("degree") or "").lower().strip(),
                (edu.get("institution") or "").lower().strip(),
                (edu.get("fromYear") or "").strip(),
                (edu.get("toYear") or "").strip(),
            )
            
            if key not in seen and key[0]:  # Ensure degree exists
                seen.add(key)
                unique.append(edu)
        
        if len(education_list) != len(unique):
            logger.info(f"✓ Removed {len(education_list) - len(unique)} duplicate education entries")
        
        return unique
    
    @staticmethod
    def _deduplicate_experience(experience_list: List[Dict]) -> List[Dict]:
        """Remove duplicate experience entries"""
        if not experience_list:
            return []
        
        unique = []
        seen = set()
        
        for exp in experience_list:
            key = (
                (exp.get("position") or "").lower().strip(),
                (exp.get("company") or "").lower().strip(),
                (exp.get("fromYear") or "").strip(),
                (exp.get("toYear") or "").strip(),
            )
            
            if key not in seen and key[1]:  # Ensure company exists
                seen.add(key)
                unique.append(exp)
        
        if len(experience_list) != len(unique):
            logger.info(f"✓ Removed {len(experience_list) - len(unique)} duplicate experience entries")
        
        return unique
    
    @staticmethod
    def _deduplicate_projects(projects_list: List[Dict]) -> List[Dict]:
        """Remove duplicate project entries"""
        if not projects_list:
            return []
        
        unique = []
        seen = set()
        
        for proj in projects_list:
            key = (proj.get("title") or "").lower().strip()
            
            if key and key not in seen:
                seen.add(key)
                unique.append(proj)
        
        if len(projects_list) != len(unique):
            logger.info(f"✓ Removed {len(projects_list) - len(unique)} duplicate projects")
        
        return unique
    
    @staticmethod
    def _deduplicate_skills(skills_list: List[str]) -> List[str]:
        """Remove duplicate skills (case-insensitive)"""
        if not skills_list:
            return []
        
        unique = []
        seen = set()
        
        for skill in skills_list:
            normalized = skill.lower().strip()
            # Handle variations like React.js vs ReactJS
            normalized = normalized.replace(".js", "").replace(".py", "")
            
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique.append(skill)
        
        if len(skills_list) != len(unique):
            logger.info(f"✓ Removed {len(skills_list) - len(unique)} duplicate skills")
        
        return unique
    
    @staticmethod
    def _build_prompts() -> Dict[str, str]:
        """Build optimized prompts (shorter, focused)"""
        
        return {
            "header": """Extract name, title, location from:
{content}

Return JSON:
{{"name": "", "title": "", "location": "", "email": "", "phone": "", "link": ""}}

Rules: Extract only present data. Empty strings for missing fields.""",
            
            "summary": """Extract professional summary from:
{content}

Return JSON:
{{"summary": "complete summary as single paragraph"}}

Rules: Join all bullets into ONE paragraph. Include all information.""",
            
            "experience": """Extract ALL jobs from:
{content}

Return JSON array:
[{{"position": "", "company": "", "location": "", "fromYear": "", "toYear": "", "isOngoing": false, "description": [], "bullets": []}}]

Rules:
- Extract EVERY job
- isOngoing: true if "Present"
- bullets: array of ALL achievements
- Return valid JSON array""",
            
            "education": """Extract ALL education from:
{content}

Return JSON array:
[{{"degree": "", "institution": "", "location": "", "fromYear": "", "toYear": ""}}]

Rules:
- Extract EVERY entry (no duplicates)
- degree, institution, location, years (YYYY)
- Empty strings for missing
- Return valid JSON array""",
            
            "skills": """Extract ALL unique skills from:
{content}

Return JSON array:
["skill1", "skill2", "skill3"]

Rules:
- Extract ALL unique skills (no duplicates)
- Return flat array
- Include all mentioned""",
            
            "projects": """Extract ALL projects from:
{content}

Return JSON array:
[{{"title": "", "description": "", "technologies": [], "bullets": []}}]

Rules:
- Extract EVERY project
- technologies as array
- bullets: ALL achievements
- Return valid JSON array""",
            
            "certifications": """Extract ALL certifications from:
{content}

Return JSON array:
[{{"title": "", "issuer": ""}}]

Rules: Extract EVERY certification. title, issuer. Return valid JSON array.""",
            
            "languages": """Extract ALL languages from:
{content}

Return JSON array:
["language1", "language2"]

Rules: Extract ALL languages mentioned. Return valid JSON array.""",
        }
    
    @staticmethod
    def _preprocess_education_content(section_text: str) -> str:
        """Preprocess education section to improve parsing and prevent duplicates"""
        # Clean up common education section formats
        
        # Handle table format education
        if "|" in section_text and "degree" in section_text.lower():
            lines = section_text.split("\n")
            cleaned_lines = []
            for line in lines:
                if "|" in line and "---" not in line:
                    parts = [p.strip() for p in line.split("|")]
                    cleaned_lines.append(" ".join([p for p in parts if p]))
            section_text = "\n".join(cleaned_lines)
        
        # Normalize date formats
        section_text = re.sub(r'(\d{2})/(\d{4})\s*–\s*(\d{2})/(\d{4})', 
                             r'\2 to \4', section_text)
        section_text = re.sub(r'(\d{4})\s*–\s*(\d{4})', r'\1 to \2', section_text)
        
        # Normalize separators
        section_text = re.sub(r'\s*[-–—]\s*', ' | ', section_text)
        
        # Remove duplicate lines (common in some resumes)
        lines = section_text.split("\n")
        unique_lines = []
        seen = set()
        for line in lines:
            if line.strip() and line.strip() not in seen:
                unique_lines.append(line)
                seen.add(line.strip())
        
        return "\n".join(unique_lines)
    
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
            if isinstance(parsed, list):
                return len(parsed) >= 0
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