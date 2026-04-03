def build_section_parsing_prompt(section_name: str, section_text: str) -> str:
    """Build specialized prompts for each section"""
    
    schemas = {
        "header": '{"name": "", "email": "", "phone": "", "location": "", "link": "", "title": ""}',
        "summary": '{"summary": "complete text here"}',
        "education": '[{"degree": "", "institution": "", "location": "", "fromYear": "", "toYear": ""}]',
        "experience": '[{"position": "", "company": "", "location": "", "fromYear": "", "toYear": "", "isOngoing": false, "description": [], "bullets": []}]',
        "skills": '["skill1", "skill2"]',
        "projects": '[{"title": "", "description": "", "technologies": [], "fromYear": "", "toYear": "", "bullets": []}]',
        "certifications": '[{"title": "", "issuer": ""}]',
        "languages": '["language1", "language2"]'
    }
    
    schema = schemas.get(section_name, '{}')
    
    base_rules = f"""Parse this {section_name} section and return ONLY valid JSON:

{section_text}

Schema:
{schema}

CRITICAL RULES:
- Return ONLY valid JSON
- No markdown, explanations, or extra text
- Extract ALL content, do NOT skip or merge
- For arrays: include EVERY item mentioned
- For dates: extract years as strings (e.g. "2024")
- Empty strings "" for missing required fields
- Empty arrays [] for missing list fields
"""

    specific_rules = {
        "experience": """
EXPERIENCE SPECIFIC:
- Extract EVERY job mentioned
- Include: position/job title, company name, location, start and end dates
- isOngoing: true only if explicitly "Present" or "Currently"
- description: 1-2 sentence summary of role
- bullets: ARRAY of ALL achievement points listed
- Do NOT merge jobs
- Do NOT skip internships or part-time roles
""",
        "projects": """
PROJECT SPECIFIC:
- Extract EVERY project listed
- Include: full project title, description, technologies used
- technologies: ARRAY of all tech names mentioned
- bullets: ARRAY of ALL features/achievements
- Include links/URLs if mentioned
- Do NOT merge projects
- Do NOT skip any project
""",
        "education": """
EDUCATION SPECIFIC:
- Extract EVERY degree/qualification
- Include: degree name, institution, location, admission and graduation years
- Include GPA/CGPA if mentioned
- Return as array - one entry per degree
""",
        "skills": """
SKILL SPECIFIC:
- Extract ALL skills mentioned in ANY subsection
- Include: programming languages, frameworks, tools, databases, libraries
- Flatten into single array of strings
- Remove duplicates but preserve all unique skills
- Example: ["React", "Node.js", "MongoDB", "Docker"]
""",
        "summary": """
SUMMARY SPECIFIC:
- Extract complete professional summary text
- Join all bullet points into ONE continuous paragraph
- Remove bullet markers (• or *)
- Keep all information without truncation
- Return as single string value, not array
""",
    }
    
    rules = specific_rules.get(section_name, "")
    
    return base_rules + rules