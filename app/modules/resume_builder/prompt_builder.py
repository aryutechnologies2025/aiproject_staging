# prompt_builder.py
def build_section_parsing_prompt(section_name: str, section_text: str) -> str:
    """Optimized compact prompts for token efficiency."""
    
    schemas = {
        "header": '{"name":"","email":"","phone":"","location":"","link":"","title":""}',
        "summary": '{"summary":""}',
        "education": '[{"degree":"","institution":"","location":"","fromYear":"","toYear":""}]',
        "experience": '[{"position":"","company":"","location":"","fromYear":"","toYear":"","isOngoing":false,"description":[],"bullets":[]}]',
        "skills": '[""]',
        "projects": '[{"title":"","description":"","technologies":[],"fromYear":"","toYear":"","bullets":[]}]',
        "certifications": '[{"title":"","issuer":""}]',
        "languages": '[""]'
    }
    
    # Minified instructions to save tokens
    base_rules = f"""JSON only. Parse {section_name}.
Schema: {schemas.get(section_name, '[]')}
Rules:
- No prose. 
- Extract all.
- Dates as "YYYY".
- Missing = "" or [].

Content:
{section_text}
"""
    
    # Section specific overrides (Compact)
    spec = {
        "experience": "EXPERIENCE: Extract EVERY job. 'isOngoing' if 'Present'. 'bullets' is array of achievements.",
        "projects": "PROJECTS: Extract ALL. 'technologies' is array. Include links.",
        "skills": "SKILLS: Flatten all categories into single string array. Unique only.",
        "summary": "SUMMARY: Single string. Remove bullet markers.",
        "education": "EDUCATION: Every degree. Include GPA if present."
    }
    
    return base_rules + spec.get(section_name, "")