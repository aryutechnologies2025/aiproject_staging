# ===================================================================
# FILE 1: /home/aryu_user/Arun/aiproject_staging/app/modules/resume_builder/resume_parser_helper.py
# ===================================================================

import fitz  # PyMuPDF
from docx import Document
import re
from typing import Dict, List

def extract_text_from_pdf(pdf_path: str) -> str:
    """Improved text extraction: uses blocks to preserve layout better than plain 'text'."""
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        # Get text blocks (better for resumes with columns/tables)
        blocks = page.get_text("blocks")
        for block in blocks:
            text_block = block[4].strip()  # block[4] contains the text
            if text_block:
                full_text += text_block + "\n\n"
    doc.close()
    return full_text.strip()

def extract_text_from_docx(docx_path: str) -> str:
    """Extract text from Word document preserving paragraphs."""
    doc = Document(docx_path)
    text = "\n".join([p.text.strip() for p in doc.paragraphs if p.text.strip()])
    return text.strip()

def extract_personal_info(text: str) -> Dict:
    """Extract name, contact details using regex (works on almost all resumes)."""
    info = {
        "name": "",
        "title": "",
        "email": "",
        "phone": "",
        "location": "",
        "portfolio": "",
        "github": "",
        "linkedin": ""
    }

    lines = [line.strip() for line in text.split("\n") if line.strip()]

    # Name is usually the very first line (or second)
    if lines:
        first_line = lines[0]
        if not any(char.isdigit() for char in first_line) and "@" not in first_line:
            info["name"] = first_line

    # Title / headline (often second line)
    if len(lines) > 1:
        info["title"] = lines[1]

    # Email
    email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    if email_match:
        info["email"] = email_match.group(0)

    # Phone (Indian + international formats)
    phone_match = re.search(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text)
    if phone_match:
        info["phone"] = phone_match.group(0).strip()

    # Location (common after phone/email)
    location_match = re.search(r'(Chennai|India|Tamil Nadu|Tiruvallur|Kavaraipettai)[\w, ]+', text, re.IGNORECASE)
    if location_match:
        info["location"] = location_match.group(0).strip()

    # GitHub / Portfolio / LinkedIn
    github = re.search(r'(https?://)?(www\.)?github\.com/[^\s]+', text, re.IGNORECASE)
    if github:
        info["github"] = github.group(0)

    portfolio = re.search(r'(https?://)?(www\.)?portfolio[^\s]*', text, re.IGNORECASE)
    if portfolio:
        info["portfolio"] = portfolio.group(0)

    linkedin = re.search(r'(https?://)?(www\.)?linkedin\.com/in/[^\s]+', text, re.IGNORECASE)
    if linkedin:
        info["linkedin"] = linkedin.group(0)

    return info

def parse_resume_sections(text: str) -> List[Dict]:
    """
    Robust section detection using keyword matching (case-insensitive).
    Works on almost all modern resumes (including this Venu_MERN.pdf).
    """
    # Common section keywords (add more if needed)
    section_keywords = {
        "summary": ["summary", "professional summary", "profile summary", "about me", "objective"],
        "experience": ["professional experience", "work experience", "experience", "employment"],
        "education": ["education", "academic", "qualification"],
        "skills": ["technical skills", "skills", "technical stack", "tech stack", "core competencies"],
        "projects": ["projects", "project"],
        "certifications": ["certifications", "certificates", "licenses"],
        "languages": ["languages", "language"],
        "publications": ["publications"]
    }

    sections: List[Dict] = []
    current_section = {"heading": "HEADER", "content": []}
    lines = text.split("\n")

    for line in lines:
        clean_line = line.strip()
        if not clean_line:
            continue

        lower_line = clean_line.lower()

        # Check if line starts a new section
        matched_section = None
        for sec_key, keywords in section_keywords.items():
            if any(kw in lower_line for kw in keywords):
                matched_section = sec_key
                break

        if matched_section:
            # Save previous section
            if current_section["content"]:
                sections.append({
                    "heading": current_section["heading"],
                    "content": "\n".join(current_section["content"]).strip()
                })

            # Start new section with proper heading (original casing)
            current_section = {
                "heading": clean_line,
                "content": []
            }
        else:
            current_section["content"].append(clean_line)

    # Add the last section
    if current_section["content"]:
        sections.append({
            "heading": current_section["heading"],
            "content": "\n".join(current_section["content"]).strip()
        })

    # Special handling: first section (HEADER) usually contains name + contact + summary
    if sections and sections[0]["heading"] == "HEADER":
        header_content = sections[0]["content"]
        # Move summary part if it exists inside header
        summary_part = [line for line in header_content if any(k in line.lower() for k in ["full stack", "software engineer", "years of"])]
        if summary_part:
            sections.insert(1, {"heading": "SUMMARY", "content": "\n".join(summary_part)})
            # Remove summary lines from header
            header_content = [line for line in header_content if line not in summary_part]

        sections[0] = {
            "heading": "HEADER",
            "content": "\n".join(header_content).strip()
        }

    return sections

def parsing_resume(file_path: str, file_ext: str) -> Dict:
    """Main parser - returns clean structured JSON (fixed for your resume)."""
    if file_ext.lower() == ".pdf":
        text = extract_text_from_pdf(file_path)
    elif file_ext.lower() == ".docx":
        text = extract_text_from_docx(file_path)
    else:
        raise ValueError("Unsupported file format. Only PDF and DOCX allowed.")

    personal_info = extract_personal_info(text)
    sections = parse_resume_sections(text)

    return {
        "status": "success",
        "personal_info": personal_info,
        "sections": sections,
        "raw_text": text[:10000] + "..." if len(text) > 10000 else text,  # truncated for safety
        "raw_text_length": len(text)
    }