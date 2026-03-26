import re
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from .schemas import (
    PersonalInfo, ContactInfo, LinkEntry, ExperienceEntry, EducationEntry,
    TechnicalStack, DSADetail, ProjectEntry, LanguageEntry, CertificationEntry
)

# Known top‑level headers (lowercase) and their canonical keys
NORM_HEADERS = {
    "summary": ["summary", "professional summary", "profile", "about me"],
    "technical_stack": ["technical stack", "skills", "tech stack", "technical skills", "core competencies"],
    "professional_experience": ["professional experience", "work experience", "experience", "employment"],
    "education": ["education", "academic background", "qualifications"],
    "projects": ["projects", "personal projects", "side projects"],
    "languages": ["languages", "language"],
    "certifications": ["certifications", "certificates", "licenses"],
    "personal_info": ["personal info", "contact", "contact information"],
}

# Inverse mapping: header -> canonical key
HEADER_TO_CANONICAL = {}
for canon, variants in NORM_HEADERS.items():
    for v in variants:
        HEADER_TO_CANONICAL[v] = canon

# Known headers set for quick lookup
KNOWN_HEADERS = set(HEADER_TO_CANONICAL.keys())

# Pre‑processing: split combined headers like "SUMMARY TECHNICAL STACK"
def preprocess_text(text: str) -> str:
    """Split combined headers and ensure each known header is on its own line."""
    # Replace "SUMMARY TECHNICAL STACK" with "SUMMARY\nTECHNICAL STACK"
    text = re.sub(r"(?i)\bSUMMARY\s+TECHNICAL\s+STACK\b", "SUMMARY\nTECHNICAL STACK", text)
    # Add more combined patterns as needed

    # For each known header, ensure it appears as a whole word on its own line
    lines = text.splitlines()
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            new_lines.append(line)
            continue
        lower = stripped.lower()
        # If the line contains a known header at the start (possibly with colon), split it
        for header in sorted(KNOWN_HEADERS, key=len, reverse=True):
            if lower.startswith(header) and (len(stripped) == len(header) or stripped[len(header)] in ": "):
                # Split into header and the rest
                header_part = stripped[:len(header)].strip()
                rest = stripped[len(header):].lstrip()
                new_lines.append(header_part)
                if rest:
                    new_lines.append(rest)
                break
        else:
            new_lines.append(line)
    return "\n".join(new_lines)


def segment_sections(text: str) -> List[Tuple[Optional[str], str]]:
    """
    Split text into sections using known top‑level headers.
    Returns list of (header, content_text). Header is None for preamble.
    """
    lines = text.splitlines()
    sections = []
    current_header = None
    current_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            current_lines.append(line)   # keep blank lines in content
            continue

        # Remove bold markers and trailing colon
        clean = stripped
        if clean.startswith("**") and clean.endswith("**"):
            clean = clean[2:-2].strip()
        if clean.endswith(":"):
            clean = clean[:-1].strip()

        lower = clean.lower()
        if lower in KNOWN_HEADERS:
            # Save previous section
            if current_lines:
                sections.append((current_header, "\n".join(current_lines).strip()))
            current_header = clean
            current_lines = []
        else:
            current_lines.append(line)

    # Append last section
    if current_lines:
        sections.append((current_header, "\n".join(current_lines).strip()))
    return sections


def extract_personal_info(preamble: str, full_text: str) -> PersonalInfo:
    """Parse name, title, contact, links from the preamble."""
    text = preamble or full_text[:2000]
    name = ""
    title = ""
    phone = ""
    email = ""
    location = ""
    links = []

    # Extract email
    m = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    if m:
        email = m.group()

    # Extract phone (simple pattern: digits with optional separators)
    m = re.search(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text)
    if m:
        phone = m.group().replace(' ', '').replace('-', '')

    # Extract location (e.g., "Chennai, Tamil Nadu, India")
    m = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),\s*([A-Z]{2}|[A-Z][a-z]+)\s*,?\s*[A-Z][a-z]+', text)
    if m:
        location = m.group()

    # Links: look for common names (Portfolio, GitHub, LinkedIn) and URLs
    link_names = ["PortFolio", "Github", "GitHub", "LinkedIn"]
    for name in link_names:
        if name in text:
            links.append(LinkEntry(name=name, url=None))
    # Also extract raw URLs
    urls = re.findall(r'https?://[^\s]+|www\.[^\s]+', text)
    for url in urls:
        if "linkedin" in url.lower():
            links.append(LinkEntry(name="LinkedIn", url=url))
        elif "github" in url.lower():
            links.append(LinkEntry(name="GitHub", url=url))
        else:
            links.append(LinkEntry(name="Website", url=url))

    # Name and title: first non‑empty line that is not a contact line
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for line in lines:
        # Skip lines with @, phone, or common social links
        if '@' in line or re.search(r'\d{10}', line) or any(s in line for s in ['linkedin', 'github']):
            continue
        # If line contains '|', it might be name|title
        if '|' in line:
            parts = line.split('|')
            name = parts[0].strip()
            title = '|'.join(parts[1:]).strip()
        else:
            name = line
        break
    # If we got a name but not title, try next line for title
    if name and not title and len(lines) > 1:
        # find the line after name
        for i, ln in enumerate(lines):
            if ln == name and i+1 < len(lines):
                title = lines[i+1]
                break

    return PersonalInfo(
        name=name,
        title=title,
        contact=ContactInfo(phone=phone, email=email, location=location),
        links=links
    )


def parse_summary(text: str) -> List[str]:
    """Extract bullet points/sentences from summary section."""
    if not text.strip():
        return []
    # Split by bullet markers or newlines
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # Remove bullet markers
    lines = [re.sub(r'^[•·▪▸►\-\*]\s*', '', ln) for ln in lines]
    # If only one line, split into sentences
    if len(lines) == 1 and len(lines[0]) > 200:
        # split by '. '
        sentences = [s.strip() for s in lines[0].split('. ') if s.strip()]
        return sentences
    return lines


def parse_experience(text: str) -> List[ExperienceEntry]:
    if not text.strip():
        return []
    
    date_pattern = r'(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\s*[-–—]+\s*(?:\d{4}|[Pp]resent)|' \
                   r'\d{4}\s*[-–—]\s*\d{4}|\d{2}/\d{4}\s*[-–—]\s*(?:\d{2}/\d{4}|[Pp]resent))'
    lines = text.splitlines()
    entries = []
    current_block = []
    
    for line in lines:
        if re.search(date_pattern, line, re.I):
            if current_block:
                entries.append(_parse_exp_block(current_block))
                current_block = []
        current_block.append(line)
    if current_block:
        entries.append(_parse_exp_block(current_block))
    
    return entries

def _parse_exp_block(block_lines: List[str]) -> ExperienceEntry:
    block = "\n".join(block_lines).strip()
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    if not lines:
        return ExperienceEntry()
    
    date_pattern = r'(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\s*[-–—]+\s*(?:\d{4}|[Pp]resent)|' \
                   r'\d{4}\s*[-–—]\s*\d{4}|\d{2}/\d{4}\s*[-–—]\s*(?:\d{2}/\d{4}|[Pp]resent))'
    date_idx = None
    for i, line in enumerate(lines):
        if re.search(date_pattern, line, re.I):
            date_idx = i
            break
    if date_idx is None:
        return ExperienceEntry(details=lines)
    
    # Title: lines before the date line
    title_lines = lines[:date_idx]
    title = ' '.join(title_lines).strip()
    
    # Parse the date line
    date_line = lines[date_idx]
    company = ''
    duration = ''
    location = ''
    
    if '|' in date_line:
        parts = [p.strip() for p in date_line.split('|')]
        if len(parts) >= 1:
            company = parts[0]
        if len(parts) >= 2:
            if re.search(date_pattern, parts[1], re.I):
                duration = parts[1]
            else:
                location = parts[1]
        if len(parts) >= 3:
            if not location:
                location = parts[2]
            else:
                if re.search(date_pattern, parts[2], re.I):
                    duration = parts[2]
        # Also look for date in any part
        for part in parts:
            dm = re.search(date_pattern, part, re.I)
            if dm:
                duration = dm.group()
                break
    else:
        dm = re.search(date_pattern, date_line, re.I)
        if dm:
            duration = dm.group()
        rest = date_line.replace(duration, '').strip()
        if rest:
            if ',' in rest:
                company = rest.split(',')[0].strip()
            else:
                company = rest
    
    # Details: lines after the date line
    details = []
    for line in lines[date_idx+1:]:
        cleaned = re.sub(r'^[•·▪▸►\-\*]\s*', '', line).strip()
        if cleaned:
            details.append(cleaned)
    
    return ExperienceEntry(
        title=title,
        company=company,
        duration=duration,
        location=location,
        details=details
    )


def parse_education(text: str) -> List[EducationEntry]:
    if not text.strip():
        return []
    # Split by blank lines
    blocks = re.split(r'\n\s*\n', text)
    entries = []
    
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        
        degree = ''
        institution = ''
        duration = ''
        location = ''
        cgpa = ''
        
        date_pattern = r'(?:\d{4}\s*[-–—]\s*\d{4}|\d{2}/\d{4}\s*[-–—]\s*\d{2}/\d{4}|[Pp]resent)'
        date_idx = None
        for i, line in enumerate(lines):
            if re.search(date_pattern, line, re.I):
                date_idx = i
                break
        
        if date_idx is not None:
            # Lines before date: degree (first) and institution (second)
            pre_lines = lines[:date_idx]
            if pre_lines:
                degree = pre_lines[0]
                if len(pre_lines) > 1:
                    institution = pre_lines[1]
            # Date line
            date_line = lines[date_idx]
            dm = re.search(date_pattern, date_line, re.I)
            if dm:
                duration = dm.group()
            rest = date_line.replace(duration, '').strip()
            if rest:
                location = rest
            # CGPA search in whole block
            cgpa_match = re.search(r'(?:CGPA|GPA)[:\s]*(\d+\.?\d*)/\d+', block, re.I)
            if cgpa_match:
                cgpa = cgpa_match.group(1)
        else:
            # No date line: first line degree, second institution
            degree = lines[0] if lines else ''
            institution = lines[1] if len(lines) > 1 else ''
            cgpa_match = re.search(r'(?:CGPA|GPA)[:\s]*(\d+\.?\d*)/\d+', block, re.I)
            if cgpa_match:
                cgpa = cgpa_match.group(1)
        
        entries.append(EducationEntry(
            degree=degree,
            institution=institution,
            duration=duration,
            location=location,
            cgpa=cgpa,
            description=''
        ))
    
    return entries


def parse_technical_stack(text: str, full_text: str) -> TechnicalStack:
    if text.strip():
        return _parse_skills_block(text)
    # Fallback: scan full text for known skills
    return _scan_full_text_for_skills(full_text)


def _parse_skills_block(text: str) -> TechnicalStack:
    stack = TechnicalStack()
    current_category = None
    lines = text.splitlines()
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Detect category header (ends with colon, all caps, or short)
        if line.endswith(':') or (line.isupper() and len(line.split()) <= 3):
            cat = line.rstrip(':').lower()
            if 'frontend' in cat or 'front-end' in cat:
                current_category = 'frontend'
            elif 'backend' in cat or 'api' in cat:
                current_category = 'backend_apis'
            elif 'tools' in cat or 'devops' in cat:
                current_category = 'tools_devops'
            elif 'engineering' in cat or 'practices' in cat:
                current_category = 'engineering_practices'
            elif 'database' in cat:
                current_category = 'databases'
            elif 'data structures' in cat or 'algorithms' in cat:
                current_category = 'data_structures_algorithms'
            else:
                current_category = None
            continue
        
        # Extract items from the line
        # Split by comma, semicolon, bullet, etc.
        items = re.split(r'[;,•·▪▸►\n]+', line)
        for item in items:
            item = item.strip()
            if not item or len(item) < 2:
                continue
            # Remove leading bullets
            item = re.sub(r'^[•·▪▸►\-\*]\s*', '', item)
            if current_category == 'frontend':
                stack.frontend.append(item)
            elif current_category == 'backend_apis':
                stack.backend_apis.append(item)
            elif current_category == 'tools_devops':
                stack.tools_devops.append(item)
            elif current_category == 'engineering_practices':
                stack.engineering_practices.append(item)
            elif current_category == 'databases':
                stack.databases.append(item)
            elif current_category == 'data_structures_algorithms':
                if 'linear' in item.lower():
                    stack.data_structures_algorithms.linear.append(item)
                elif 'non-linear' in item.lower():
                    stack.data_structures_algorithms.non_linear.append(item)
                else:
                    stack.data_structures_algorithms.linear.append(item)
            else:
                # Unknown category
                if current_category:
                    stack.other.setdefault(current_category, []).append(item)
                else:
                    stack.other.setdefault('other', []).append(item)
    
    return stack


def _scan_full_text_for_skills(full_text: str) -> TechnicalStack:
    """Fallback: extract skills from full text using a keyword list."""
    # This is a simplified version; you can maintain a larger list.
    common_tech = ['python', 'django', 'fastapi', 'react', 'javascript', 'typescript',
                   'node.js', 'express', 'mongodb', 'postgresql', 'mysql', 'docker',
                   'git', 'redis', 'aws', 'jenkins', 'kubernetes', 'java', 'c++', 'c#',
                   'sql', 'html', 'css', 'tailwind', 'bootstrap', 'flask', 'spring',
                   'jwt', 'oauth', 'rest api', 'graphql', 'pandas', 'numpy']
    found = []
    for tech in common_tech:
        if re.search(r'\b' + re.escape(tech) + r'\b', full_text, re.I):
            found.append(tech)
    # Put all in frontend bucket as a simple list
    stack = TechnicalStack()
    stack.frontend = sorted(set(found))
    return stack


def parse_projects(text: str) -> List[ProjectEntry]:
    if not text.strip():
        return []
    
    # First, split by blank lines
    blocks = re.split(r'\n\s*\n', text)
    projects = []
    
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        
        # Check if this block contains multiple projects (e.g., a list without blank lines)
        # Look for project name lines: short, all caps, or ending with colon, and not a bullet
        lines = block.splitlines()
        current_project = None
        project_text = []
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                if current_project:
                    # finish current project
                    projects.append(_parse_single_project(project_text))
                    current_project = None
                    project_text = []
                continue
            
            # Check if line looks like a project name
            is_project_name = (stripped.isupper() and len(stripped.split()) <= 5) or \
                              (stripped.endswith(':') and len(stripped.split()) <= 6) or \
                              (len(stripped.split()) <= 5 and not stripped.startswith('•'))
            
            if is_project_name and not stripped.startswith('•'):
                if current_project:
                    projects.append(_parse_single_project(project_text))
                current_project = stripped
                project_text = []
            else:
                project_text.append(line)
        
        if current_project:
            projects.append(_parse_single_project(project_text))
    
    # If no projects were detected, treat the whole text as one project
    if not projects:
        projects.append(_parse_single_project(block.splitlines()))
    
    return projects

def _parse_single_project(lines: List[str]) -> ProjectEntry:
    name = ''
    details = []
    tech_stack = []
    in_tech = False
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if 'tech' in stripped.lower() or 'stack' in stripped.lower():
            in_tech = True
            # Split by colon or comma
            parts = re.split(r'[:;,]\s*', stripped.split(':')[-1])
            for p in parts:
                p = p.strip()
                if p and len(p) > 1:
                    tech_stack.append(p)
            continue
        if in_tech:
            # Additional tech lines
            parts = re.split(r'[;,]\s*', stripped)
            for p in parts:
                p = p.strip()
                if p and len(p) > 1:
                    tech_stack.append(p)
            continue
        
        # Remove bullet markers
        cleaned = re.sub(r'^[•·▪▸►\-\*]\s*', '', stripped).strip()
        if not name and len(cleaned) < 60:
            name = cleaned
        elif cleaned:
            details.append(cleaned)
    
    return ProjectEntry(name=name, details=details, tech_stack=tech_stack)


def parse_languages(text: str) -> List[LanguageEntry]:
    if not text.strip():
        return []
    
    entries = []
    # Language proficiency words
    prof_words = ['native', 'advanced', 'intermediate', 'beginner', 'fluent', 'professional']
    
    # First, try to split by newlines
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Remove bullet markers
        line = re.sub(r'^[•·▪▸►\-\*]\s*', '', line)
        # Look for "Language: Proficiency" or "Language – Proficiency"
        if ':' in line:
            parts = line.split(':', 1)
            lang = parts[0].strip()
            prof = parts[1].strip().lower()
        elif '–' in line:
            parts = line.split('–', 1)
            lang = parts[0].strip()
            prof = parts[1].strip().lower()
        else:
            # Try to find a proficiency word in the line
            words = line.split()
            prof = ''
            for i, w in enumerate(words):
                if w.lower() in prof_words:
                    prof = w
                    lang = ' '.join(words[:i] + words[i+1:])
                    break
            else:
                lang = line
                prof = ''
        
        # Clean up lang and prof
        lang = lang.strip().strip('.,:;')
        prof = prof.capitalize()
        if lang and not any(kw in lang.lower() for kw in ['tech', 'built', 'integrated', 'debugging']):
            entries.append(LanguageEntry(language=lang, proficiency=prof))
    
    return entries


def parse_certifications(text: str) -> List[CertificationEntry]:
    """Parse certifications section."""
    if not text.strip():
        return []
    entries = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Remove bullet markers
        line = re.sub(r'^[•·▪▸►\-\*]\s*', '', line)
        # Try to split by comma or dash
        parts = re.split(r'[–—,]', line)
        name = parts[0].strip()
        issuer = parts[1].strip() if len(parts) > 1 else ''
        year = ''
        # Look for year in the string
        ym = re.search(r'\b(19|20)\d{2}\b', line)
        if ym:
            year = ym.group()
        entries.append(CertificationEntry(name=name, issuer=issuer, year=year))
    return entries


def parse_resume(text: str) -> Dict[str, Any]:
    """Main entry point: preprocess, segment, and parse."""
    # Step 1: preprocess text
    text = preprocess_text(text)

    # Step 2: segment into sections
    sections = segment_sections(text)

    # Separate preamble (first section with header None) and others
    preamble = ""
    section_contents = {}
    for header, content in sections:
        if header is None:
            preamble = content
        else:
            section_contents[header.lower()] = content

    # Step 3: extract personal info from preamble
    personal_info = extract_personal_info(preamble, text)

    # Step 4: parse summary
    summary_text = section_contents.get("summary", "")
    summary = parse_summary(summary_text)

    # Step 5: parse experience
    exp_text = section_contents.get("professional experience", "")
    professional_experience = parse_experience(exp_text)

    # Step 6: parse education
    edu_text = section_contents.get("education", "")
    education = parse_education(edu_text)

    # Step 7: parse technical stack (use dedicated section or fallback)
    skills_text = section_contents.get("technical stack", "")
    technical_stack = parse_technical_stack(skills_text, text)

    # Step 8: parse projects
    proj_text = section_contents.get("projects", "")
    projects = parse_projects(proj_text)

    # Step 9: parse languages
    lang_text = section_contents.get("languages", "")
    languages = parse_languages(lang_text)

    # Step 10: parse certifications
    cert_text = section_contents.get("certifications", "")
    certifications = parse_certifications(cert_text)

    # Step 11: collect custom sections (any header not in our canonical list)
    custom_sections = {header: content for header, content in sections if header and header.lower() not in KNOWN_HEADERS}
    # raw sections: all sections (including known) with original header
    raw_sections = {header: content for header, content in sections if header}

    return {
        "personal_info": personal_info,
        "summary": summary,
        "professional_experience": professional_experience,
        "education": education,
        "technical_stack": technical_stack,
        "projects": projects,
        "languages": languages,
        "certifications": certifications,
        "custom_sections": custom_sections,
        "raw_sections": raw_sections
    }