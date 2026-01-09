import re

ATS_SAFE_FONTS = {
    "arial", "calibri", "times new roman", "helvetica"
}


def run_ats_rules(resume):
    score = 100
    issues = []

    if resume.font.lower() not in ATS_SAFE_FONTS:
        score -= 10
        issues.append("Non ATS-friendly font detected")

    if resume.uses_table:
        score -= 15
        issues.append("Tables detected (ATS cannot parse tables)")

    if resume.uses_columns:
        score -= 15
        issues.append("Multiple columns detected")

    if resume.file_type not in {"pdf", "docx"}:
        score -= 20
        issues.append("Unsupported file type")

    if not resume.skills:
        score -= 15
        issues.append("Skills section missing")

    if not resume.experience:
        score -= 20
        issues.append("Experience section missing")

    return max(score, 0), issues

def extract_keywords(text: str) -> set[str]:
    return set(re.findall(r"\b[a-zA-Z]{3,}\b", text.lower()))


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
