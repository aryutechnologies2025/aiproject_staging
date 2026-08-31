"""
resume_parser_service.py — Deterministic & NLP Resume Parser Service.

Provides:
- Contact extraction (Email, Phone, Name with lazy spaCy + regex fallback).
- Intelligent fuzzy section extraction supporting multi-word headings ("Professional Experience", "Academic Background", "Core Competencies").
- Fallback to raw text extraction when section blocks are empty or not provided.
- ATSScanRequest schema parsing for downstream CV and ATS workflows.
"""

import logging
import re
from typing import Any, Dict, List, Optional
from app.schemas.ats_schema import Experience, Education, ATSScanRequest

logger = logging.getLogger("resume_builder.resume_parser_service")

_nlp = None


def get_spacy_nlp():
    """Lazy load spaCy model on first use only."""
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            logger.warning(f"spaCy en_core_web_sm not available: {e}. Using regex fallback.")
            _nlp = False
    return _nlp if _nlp is not False else None


EMAIL_REGEX = r"\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b"
PHONE_REGEX = r"(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,5}\)?[\s.-]?)?\d{3,5}[\s.-]?\d{3,5}[\s.-]?\d{0,5}"


def clean_lines(text: str) -> List[str]:
    """Clean and filter non-empty lines from text."""
    if not text:
        return []
    lines = [l.strip() for l in text.split("\n")]
    return [l for l in lines if len(l) > 1]


def extract_email(text: str) -> str:
    """Extract first valid email address from text."""
    match = re.search(EMAIL_REGEX, text)
    return match.group(0).strip() if match else ""


def extract_phone(text: str) -> str:
    """Extract phone number with international format support."""
    match = re.search(PHONE_REGEX, text)
    if match and len(re.findall(r"\d", match.group(0))) >= 7:
        return match.group(0).strip()
    return ""


def extract_name(text: str) -> str:
    """Extract candidate name using spaCy NER with deterministic fallback."""
    nlp_model = get_spacy_nlp()
    if nlp_model:
        try:
            doc = nlp_model(text[:500])
            for ent in doc.ents:
                if ent.label_ == "PERSON" and len(ent.text.split()) <= 4:
                    return ent.text.strip()
        except Exception:
            pass

    # Deterministic fallback: first non-empty line without digits, @, or URLs
    for line in clean_lines(text[:400]):
        if not any(c.isdigit() for c in line) and "@" not in line and "http" not in line.lower() and len(line.split()) <= 4:
            return line.strip()
    return ""


def extract_skills(text: str) -> List[str]:
    """Extract technical and domain skills from text."""
    skills = []

    # 1. Match lines labeled with skill headers
    matches = re.findall(
        r"(?:Tech|Skills|Stack|Technologies|Tools|Proficiencies)\s*[:\-]\s*(.+)",
        text,
        re.IGNORECASE,
    )
    for match in matches:
        parts = re.split(r"[,|/•;\n]", match)
        skills.extend([p.strip() for p in parts if len(p.strip()) > 1])

    # 2. General comma-separated extraction if line contains known keywords
    if not skills:
        for line in clean_lines(text):
            if any(k in line.lower() for k in ("python", "java", "react", "sql", "aws", "docker", "fastapi")):
                parts = re.split(r"[,|/•;\n]", line)
                skills.extend([p.strip() for p in parts if len(p.strip()) > 1])

    # Deduplicate preserving order
    return list(dict.fromkeys(skills))


def detect_section(line: str) -> Optional[str]:
    """Identify if a line is a section heading."""
    l = line.lower().strip()

    if any(k in l for k in ("experience", "employment", "work history")):
        return "experience"
    if any(k in l for k in ("education", "academic", "qualification")):
        return "education"
    if any(k in l for k in ("skills", "technical stack", "tech stack", "technologies")):
        return "skills"
    if any(k in l for k in ("summary", "profile", "about me", "objective")):
        return "summary"
    if any(k in l for k in ("project",)):
        return "projects"
    if any(k in l for k in ("certification", "certificate", "license")):
        return "certifications"

    return None


def extract_experience(lines: List[str]) -> List[Experience]:
    """Extract structured Experience objects from lines of text."""
    experiences = []
    current = None

    for line in lines:
        clean_l = line.strip()
        if not clean_l:
            continue

        # 1. Bullet point check (MUST precede title matching to avoid matching 'Architected...', 'Managed...', etc.)
        if clean_l.startswith(("•", "-", "*", "–", ">")):
            bullet_text = clean_l.lstrip("•-*–> ").strip()
            if bullet_text and current:
                current["bullets"].append(bullet_text)
            continue

        # 2. Detect job title / role line
        if re.search(r"\b(developer|engineer|intern|manager|analyst|architect|consultant|lead|specialist|designer|administrator)\b", clean_l.lower()):
            if current and current.get("title"):
                experiences.append(current)

            current = {
                "title": clean_l,
                "company": "",
                "bullets": [],
            }
            continue

        # 3. Detect company name (short line right after title)
        if current and not current["company"] and len(clean_l.split()) <= 6:
            current["company"] = clean_l
            continue

        # 4. Continuation bullet
        if current and len(clean_l) > 15:
            current["bullets"].append(clean_l)

    if current and current.get("title"):
        experiences.append(current)

    return [
        Experience(**e)
        for e in experiences
        if e.get("title")
    ]


def extract_education(lines: List[str]) -> List[Education]:
    """Extract structured Education objects grouping degree and institution."""
    education_list = []
    current_degree = None
    current_institution = None

    degree_pattern = r"\b(bachelor|master|b\.tech|m\.tech|phd|bsc|msc|b\.e\.|m\.e\.|diploma|degree|associate)\b"
    inst_pattern = r"\b(university|college|institute|school|academy)\b"

    for line in lines:
        clean_l = line.strip()
        if not clean_l:
            continue

        l_lower = clean_l.lower()
        is_deg = bool(re.search(degree_pattern, l_lower))
        is_inst = bool(re.search(inst_pattern, l_lower))

        if is_deg:
            if current_degree:
                education_list.append(
                    Education(
                        degree=current_degree,
                        educationDescription=[current_institution] if current_institution else [current_degree],
                    )
                )
                current_institution = None
            current_degree = clean_l
        elif is_inst:
            if current_degree and not current_institution:
                current_institution = clean_l
            elif not current_degree:
                current_degree = clean_l
        elif current_degree and not current_institution and len(clean_l.split()) <= 6:
            current_institution = clean_l

    if current_degree:
        education_list.append(
            Education(
                degree=current_degree,
                educationDescription=[current_institution] if current_institution else [current_degree],
            )
        )

    return education_list


def extract_summary(lines: List[str]) -> str:
    """Extract summary statement from lines."""
    summary_lines = []
    for line in lines[:10]:
        if len(line.split()) > 4 and "@" not in line and not line.startswith(("•", "-", "*")):
            summary_lines.append(line)
    return " ".join(summary_lines[:3])


def _find_section_content(sections: Dict[str, Any], keywords: List[str]) -> str:
    """
    Fuzzy match a section heading from the sections dictionary.
    Handles headings like 'Professional Experience', 'Work History', 'Technical Skills', etc.
    """
    if not sections:
        return ""
    for heading, content in sections.items():
        if not isinstance(content, str):
            continue
        h_lower = heading.lower()
        if any(kw in h_lower for kw in keywords):
            return content
    return ""


def split_into_sections(text: str) -> Dict[str, str]:
    """Split raw text into standard section buckets."""
    lines = clean_lines(text)
    sections = {
        "experience": "",
        "education": "",
        "skills": "",
        "summary": "",
    }

    current_section = None

    for line in lines:
        section = detect_section(line)
        if section:
            current_section = section
            continue

        if current_section:
            sections[current_section] += line + "\n"

    return sections


def parse_resume_to_schema(
    text: str,
    file_type: str = "pdf",
    sections: Optional[Dict[str, Any]] = None,
) -> ATSScanRequest:
    """
    Parses resume text and optional section dictionary into standard ATSScanRequest schema.
    Guarantees fuzzy section matching with automatic fallback to full document text.
    """
    sec_dict = sections or {}

    # 1. Experience extraction (fuzzy section match with text fallback)
    exp_text = _find_section_content(sec_dict, ["experience", "employment", "work history", "work experience", "professional experience"])
    if not exp_text.strip():
        # Fallback: extract from full text or split sections
        sec_split = split_into_sections(text)
        exp_text = sec_split.get("experience", "") or text
    experience = extract_experience(clean_lines(exp_text))

    # 2. Education extraction (fuzzy section match with text fallback)
    edu_text = _find_section_content(sec_dict, ["education", "academic", "qualification", "degrees"])
    if not edu_text.strip():
        sec_split = split_into_sections(text)
        edu_text = sec_split.get("education", "") or text
    education = extract_education(clean_lines(edu_text))

    # 3. Summary extraction
    summ_text = _find_section_content(sec_dict, ["summary", "profile", "about", "objective", "executive summary"])
    summary = summ_text.strip() if summ_text.strip() else extract_summary(clean_lines(text))

    # 4. Contact & Skills extraction
    name = extract_name(text)
    email = extract_email(text)
    phone = extract_phone(text)

    skills_text = _find_section_content(sec_dict, ["skill", "tech stack", "technolog", "proficienc", "competenc"])
    skills = extract_skills(skills_text if skills_text.strip() else text)

    return ATSScanRequest(
        name=name,
        email=email,
        phone=phone,
        summary=summary,
        skills=skills,
        experience=experience,
        education=education,
        font="calibri",
        uses_table=False,
        uses_columns=False,
        file_type=file_type,
    )
