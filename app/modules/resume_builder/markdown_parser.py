# /app/modules/resume_builder/markdown_parser.py

import re
from typing import Dict, List, Any


# -----------------------------
# Helpers
# -----------------------------

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_personal_info(lines: List[str]) -> Dict[str, Any]:
    info = {
        "name": "",
        "title": "",
        "phone": "",
        "email": "",
        "location": "",
        "links": {}
    }

    if not lines:
        return info

    # Name (first heading)
    info["name"] = lines[0].replace("#", "").strip()

    # Title (second line)
    if len(lines) > 1:
        info["title"] = clean_text(lines[1].replace("###", ""))

    full_text = " ".join(lines)

    # Email
    email_match = re.search(r'[\w\.-]+@[\w\.-]+', full_text)
    if email_match:
        info["email"] = email_match.group()

    # Phone
    phone_match = re.search(r'(\+?\d[\d\s]{8,})', full_text)
    if phone_match:
        info["phone"] = phone_match.group().strip()

    # Location (simple heuristic)
    if "|" in full_text:
        parts = full_text.split("|")
        if len(parts) >= 3:
            info["location"] = clean_text(parts[2])

    return info


def extract_bullets(section_lines: List[str]) -> List[str]:
    bullets = []
    for line in section_lines:
        if line.strip().startswith("*"):
            bullets.append(clean_text(line.replace("*", "")))
    return bullets


def split_sections(markdown: str) -> Dict[str, List[str]]:
    """
    Dynamically split markdown into sections
    """
    sections = {}
    current_section = "header"
    sections[current_section] = []

    for line in markdown.split("\n"):
        line = line.strip()

        # Section heading
        if line.startswith("## "):
            current_section = line.replace("##", "").strip().lower().replace(" ", "_")
            sections[current_section] = []
        else:
            sections[current_section].append(line)

    return sections


# -----------------------------
# Main Parser
# -----------------------------

def parse_markdown_resume(markdown: str) -> Dict[str, Any]:
    sections = split_sections(markdown)

    result = {
        "personal_info": {},
        "summary": [],
        "experience": [],
        "education": [],
        "languages": [],
        "technical_skills": {},
        "projects": [],
        "extra_sections": {}
    }

    # -------------------------
    # Personal Info
    # -------------------------
    header_lines = sections.get("header", [])
    result["personal_info"] = extract_personal_info(header_lines)

    # -------------------------
    # Summary
    # -------------------------
    if "summary" in sections:
        result["summary"] = extract_bullets(sections["summary"])

    # -------------------------
    # Experience
    # -------------------------
    if "professional_experience" in sections:
        exp_lines = sections["professional_experience"]

        current = {}
        responsibilities = []

        for line in exp_lines:
            if line.startswith("###"):
                if current:
                    current["responsibilities"] = responsibilities
                    result["experience"].append(current)

                current = {
                    "role": clean_text(line.replace("###", "")),
                    "company": "",
                    "location": "",
                    "start_date": "",
                    "end_date": ""
                }
                responsibilities = []

            elif "**" in line:
                company_line = re.sub(r"\*\*", "", line)

                parts = company_line.split("|")
                if len(parts) >= 2:
                    current["company"] = clean_text(parts[0])
                    current["start_date"] = clean_text(parts[1].split("–")[0])
                    current["end_date"] = clean_text(parts[1].split("–")[-1])

                if len(parts) >= 3:
                    current["location"] = clean_text(parts[2])

            elif line.startswith("*"):
                responsibilities.append(clean_text(line.replace("*", "")))

        if current:
            current["responsibilities"] = responsibilities
            result["experience"].append(current)

    # -------------------------
    # Education
    # -------------------------
    if "education" in sections:
        edu_lines = sections["education"]

        current = {}

        for line in edu_lines:
            if line.startswith("###"):
                current["degree"] = clean_text(line.replace("###", ""))

            elif "**" in line:
                current["institution"] = clean_text(line.replace("**", ""))

            elif "CGPA" in line:
                current["cgpa"] = clean_text(line.split(":")[-1])

        if current:
            result["education"].append(current)

    # -------------------------
    # Skills (dynamic)
    # -------------------------
    if "technical_stack" in sections:
        skill_lines = sections["technical_stack"]

        current_category = None

        for line in skill_lines:
            if line.startswith("####"):
                current_category = line.replace("####", "").strip().lower().replace(" ", "_")
                result["technical_skills"][current_category] = []

            elif line.startswith("*") and current_category:
                skills = line.replace("*", "").split(",")
                result["technical_skills"][current_category].extend(
                    [clean_text(s) for s in skills if s.strip()]
                )

    # -------------------------
    # Projects
    # -------------------------
    if "projects" in sections:
        proj_lines = sections["projects"]

        current = {}
        desc = []

        for line in proj_lines:
            if line.startswith("####"):
                if current:
                    current["description"] = desc
                    result["projects"].append(current)

                current = {"name": clean_text(line.replace("####", ""))}
                desc = []

            elif line.startswith("*"):
                text = clean_text(line.replace("*", ""))

                if text.lower().startswith("tech"):
                    current["tech_stack"] = [s.strip() for s in text.split(":")[-1].split(",")]
                else:
                    desc.append(text)

        if current:
            current["description"] = desc
            result["projects"].append(current)

    # -------------------------
    # Extra Sections (dynamic)
    # -------------------------
    known = ["summary", "professional_experience", "education", "technical_stack", "projects"]

    for key, value in sections.items():
        if key not in known and key != "header":
            result["extra_sections"][key] = value

    return result
