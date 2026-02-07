import re

ATS_SAFE_FONTS = {
    "arial", "calibri", "times new roman", "helvetica"
}


def run_ats_rules(resume):
    score = 100
    issues = {}

    def add(section: str, message: str):
        issues.setdefault(section, []).append(message)

    if resume.font.lower() not in ATS_SAFE_FONTS:
        score -= 10
        add("summary", "Non ATS-friendly font detected")

    if resume.uses_table:
        score -= 15
        add("experience", "Tables detected which ATS may not parse correctly")

    if resume.uses_columns:
        score -= 15
        add("summary", "Multiple columns can confuse ATS systems")

    if resume.file_type not in {"pdf", "docx"}:
        score -= 20
        add("summary", "Unsupported file type")

    if not resume.skills:
        score -= 15
        add("skills", "Skills section missing")

    if not resume.experience:
        score -= 20
        add("experience", "Experience section missing")

    return max(score, 0), issues


def extract_keywords(text: str) -> set[str]:
    return set(re.findall(r"\b[a-zA-Z]{3,}\b", text.lower()))

def build_sections_array(section_issues: dict):
    sections = []

    for section, issues in section_issues.items():
        sections.append({
            section: {
                "issues_count": len(issues),
                "issues": issues
            }
        })

    return sections


def extract_missing_skills_simple(resume, job_description: str):
    if not job_description:
        return []

    jd_keywords = extract_keywords(job_description)
    resume_keywords = extract_keywords(
        " ".join(resume.skills)
    )

    missing = jd_keywords - resume_keywords
    return list(sorted(missing))[:5]


def keyword_match(resume, jd: str):
    if not jd:
        return 0

    jd_keywords = extract_keywords(jd)

    resume_text = " ".join(
        resume.skills +
        [b for exp in resume.experience for b in exp.bullets]
    ).lower()

    matched = sum(1 for kw in jd_keywords if kw in resume_text)
    return int((matched / max(len(jd_keywords), 1)) * 100)

def calculate_final_score_non_ai(rule_score: int, keyword_score: int) -> int:
    # ATS rules > keyword match
    final = (rule_score * 0.7) + (keyword_score * 0.3)
    return min(int(final), 100)

def calculate_final_score(
    rule_score: int,
    keyword_score: int,
    ai_quality_score: int
) -> int:
    final = (
        rule_score * 0.6 +
        keyword_score * 0.25 +
        ai_quality_score * 0.15
    )
    return min(int(final), 100)
