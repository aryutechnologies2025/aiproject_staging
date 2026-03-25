# /home/aryu_user/Arun/aiproject_staging/app/modules/resume_builder/resume_parser_service.py
import re
import spacy
from typing import List, Dict
from app.schemas.ats_schema import Experience, Education, ATSScanRequest

nlp = spacy.load("en_core_web_sm")



EMAIL_REGEX = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
PHONE_REGEX = r"\+?\d[\d\s\-]{8,15}\d"


def clean_lines(text: str) -> List[str]:
    lines = [l.strip() for l in text.split("\n")]
    return [l for l in lines if len(l) > 2]


def extract_email(text: str) -> str:
    match = re.search(EMAIL_REGEX, text)
    return match.group(0) if match else ""


def extract_phone(text: str) -> str:
    match = re.search(PHONE_REGEX, text)
    return match.group(0) if match else ""


def extract_name(text: str) -> str:
    doc = nlp(text[:500])
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return ""


def extract_skills(text: str) -> List[str]:
    import re

    skills = []

    # Capture "Tech: ..." lines
    matches = re.findall(
        r"(?:Tech|Skills|Stack|Technologies)\s*[:\-]\s*(.+)",
        text,
        re.IGNORECASE
    )

    for match in matches:
        parts = re.split(r",|\||/", match)
        skills.extend([p.strip() for p in parts if len(p.strip()) > 1])

    return list(set(skills))


def detect_section(line: str):

    l = line.lower()

    if "experience" in l:
        return "experience"

    if "education" in l:
        return "education"

    if "skills" in l:
        return "skills"

    if "summary" in l or "profile" in l:
        return "summary"

    return None


def extract_experience(lines: List[str]) -> List[Experience]:
    experiences = []
    current = None

    for line in lines:

        # Detect job title (strong signal)
        if re.search(r"(developer|engineer|intern|manager|analyst)", line.lower()):
            if current:
                experiences.append(current)

            current = {
                "title": line,
                "company": "",
                "bullets": []
            }
            continue

        # Detect company
        if current and not current["company"] and len(line.split()) <= 6:
            current["company"] = line
            continue

        # Bullets
        if line.startswith(("•", "-", "*")) and current:
            current["bullets"].append(line[1:].strip())

    if current:
        experiences.append(current)

    return [
        Experience(**e)
        for e in experiences if e["title"]
    ]


def extract_education(lines: List[str]) -> List[Education]:

    education_list = []

    for line in lines:

        if re.search(r"(bachelor|master|b\.tech|m\.tech|phd|bsc|msc)", line.lower()):

            education_list.append(
                Education(
                    degree=line,
                    educationDescription=[line]
                )
            )

    return education_list


def extract_summary(lines: List[str]) -> str:

    summary_lines = []

    for line in lines[:10]:

        if len(line.split()) > 6:
            summary_lines.append(line)

    return " ".join(summary_lines[:3])


def parse_resume_to_schema(
    text: str,
    file_type: str,
    sections: dict
) -> ATSScanRequest:

    experience = extract_experience(
        clean_lines(sections.get("experience", ""))
    )

    education = extract_education(
        clean_lines(sections.get("education", ""))
    )

    summary = sections.get("summary") or extract_summary(clean_lines(text))
    
    name = extract_name(text)
    email = extract_email(text)
    phone = extract_phone(text)

    skills = extract_skills(text)

    

    summary = sections.get("summary") or extract_summary(clean_lines(text))

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
        file_type=file_type
    )