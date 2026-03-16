# /home/aryu_user/Arun/aiproject_staging/app/utils/ats_rules.py
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ATS_SAFE_FONTS = {
    "arial", "calibri", "times new roman", "helvetica"
}


def run_ats_rules(resume: dict):
    score = 100
    issues = {}

    def add(section: str, message: str):
        issues.setdefault(section, []).append(message)

    font = resume.get("font")
    uses_table = resume.get("uses_table")
    uses_columns = resume.get("uses_columns")
    file_type = resume.get("file_type")
    skills = resume.get("skills")
    experience = resume.get("experience")

    if font and font.lower() not in ATS_SAFE_FONTS:
        score -= 10
        add("summary", "Non ATS-friendly font detected")

    if uses_table:
        score -= 15
        add("experience", "Tables detected which ATS may not parse correctly")

    if uses_columns:
        score -= 15
        add("summary", "Multiple columns can confuse ATS systems")

    if file_type and file_type.lower() not in {"pdf", "docx"}:
        score -= 20
        add("summary", "Unsupported file type")

    if not skills:
        score -= 15
        add("skills", "Skills section missing")

    if not experience:
        score -= 20
        add("experience", "Experience section missing")

    return max(score, 0), issues


def extract_keywords(text: str) -> set[str]:
    return set(re.findall(r"\b[a-zA-Z]{3,}\b", text.lower()))

def semantic_similarity(resume_text: str, job_description: str) -> int:

    if not resume_text or not job_description:
        return 0

    vectorizer = TfidfVectorizer(stop_words="english")

    vectors = vectorizer.fit_transform([resume_text, job_description])

    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]

    return int(similarity * 100)

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


def keyword_match(resume: dict, jd: str):

    if not jd:
        return 0

    jd_keywords = extract_keywords(jd)

    skills = resume.get("skills", [])
    experience = resume.get("experience", [])

    bullets = [
        b
        for exp in experience
        for b in exp.get("bullets", [])
    ]

    resume_text = " ".join(skills + bullets).lower()

    matched = [kw for kw in jd_keywords if kw in resume_text]

    if not jd_keywords:
        return 0

    return int(len(matched) / len(jd_keywords) * 100)

METRIC_REGEX = r"\d+%|\d+x|\$\d+|\d+\+|\d+\s?(users|clients|customers)"

def experience_strength_score(experience):

    strong = 0
    metrics = 0

    for exp in experience:

        for bullet in exp.get("bullets", []):

            if len(bullet.split()) >= 10:
                strong += 1

            if re.search(METRIC_REGEX, bullet.lower()):
                metrics += 1

    total = strong + metrics

    if total == 0:
        return 0

    return min(int((strong + metrics) / (len(experience) * 3) * 100), 100)

def section_completeness_score(resume):

    score = 0

    if resume.get("summary"):
        score += 25

    if resume.get("skills"):
        score += 25

    if resume.get("experience"):
        score += 30

    if resume.get("education"):
        score += 20

    return score

def calculate_final_score_advanced(
    rule_score,
    keyword_score,
    semantic_score,
    experience_score,
    section_score
):

    final = (
        rule_score * 0.15 +
        keyword_score * 0.25 +
        semantic_score * 0.25 +
        experience_score * 0.20 +
        section_score * 0.15
    )

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
