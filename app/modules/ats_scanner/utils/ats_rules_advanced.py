"""
ATS Rules Engine v3 — validates canonical JSON produced by the Resume Builder parser.
All section presence checks operate on structured list/string fields.
Raw text is NEVER used to determine whether a section exists.
"""

import re
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# ENUMS & CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

class SeverityLevel(Enum):
    CRITICAL = "critical"
    HIGH     = "high"
    MEDIUM   = "medium"
    LOW      = "low"


class IssueCategory(Enum):
    FORMAT     = "format"
    STRUCTURE  = "structure"
    CONTENT    = "content"
    KEYWORDS   = "keywords"
    QUALITY    = "quality"
    COMPLIANCE = "compliance"


ATS_SAFE_FONTS = {
    "arial", "calibri", "times new roman", "helvetica", "georgia",
    "verdana", "courier", "courier new", "tahoma", "trebuchet ms"
}

ATS_UNSAFE_FONTS = {
    "wingdings", "symbol", "dingbats", "impact", "comic sans",
    "brush script", "papyrus", "comic sans ms"
}

STRONG_ACTION_VERBS = {
    "led", "managed", "directed", "orchestrated", "spearheaded",
    "supervised", "coordinated", "oversaw", "mentored", "delegated",
    "guided", "coached",
    "achieved", "delivered", "accomplished", "completed", "exceeded",
    "surpassed", "attained", "earned", "won", "captured",
    "developed", "built", "created", "designed", "architected",
    "engineered", "constructed", "launched", "initiated", "innovated",
    "pioneered", "founded",
    "optimized", "improved", "enhanced", "streamlined", "refined",
    "accelerated", "elevated", "boosted", "increased", "reduced",
    "lowered", "minimized", "maximized", "scaled", "expanded",
    "analyzed", "evaluated", "assessed", "examined", "identified",
    "discovered", "uncovered", "determined", "diagnosed", "investigated",
    "implemented", "executed", "deployed", "installed", "integrated",
    "established", "instituted", "released",
    "transformed", "revolutionized", "modernized", "reformed",
    "repositioned", "restructured", "reimagined", "converted"
}

WEAK_ACTION_VERBS = {
    "responsible for", "involved in", "helped with", "assisted",
    "worked on", "was part of", "participated in", "contributed to",
    "was responsible", "helped", "assisted with", "worked with"
}

METRICS_PATTERNS = [
    r"\$[\d,]+\.?\d*\s*[KMB]?",
    r"\d+%",
    r"\d+[xX]",
    r"\d+\+",
    r"(increased|reduced|grew|improved|decreased|expanded)\s+by\s+\d+%",
    r"\d+\s*(users|clients|customers|transactions|records|downloads|requests)",
    r"(top\s+)?\d+\s*(in|out of|of)",
    r"\d+\s*(hours?|days?|weeks?|months?|years?)\s+(saved|reduced|cut)",
    r"(from|to)\s+\d+(\.\d+)?\s*(to|%)",
]

MIN_SKILLS = 5
IDEAL_SKILLS = 10
MAX_SKILLS = 50


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Issue:
    severity:          SeverityLevel
    category:          IssueCategory
    section:           str
    message:           str
    suggestion:        str
    impact_score:      int
    specific_example:  Optional[str] = None
    improvement_example: Optional[str] = None


@dataclass
class SectionIssue:
    section_name:       str
    current_status:     List[str] = field(default_factory=list)
    current_score:      int = 0
    missing_fields:     List[str] = field(default_factory=list)
    incomplete_fields:  List[str] = field(default_factory=list)
    quality_issues:     List[str] = field(default_factory=list)
    improvements:       List[str] = field(default_factory=list)
    specific_suggestions: List[Dict] = field(default_factory=list)


@dataclass
class ORSScore:
    total_score:            int
    format_score:           int
    structure_score:        int
    content_score:          int
    keyword_score:          int
    ats_compliance_score:   int
    critical_issues_count:  int
    all_issues:             List[Issue]
    section_issues:         Dict[str, SectionIssue] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _safe_str(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _is_present(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return len(value.strip()) > 0
    if isinstance(value, (list, dict)):
        return len(value) > 0
    return bool(value)


def _get_summary_str(resume: Dict) -> str:
    """Return summary as plain string from either nested or flat schema."""
    raw = resume.get("summary")
    if isinstance(raw, dict):
        return _safe_str(raw.get("summary", ""))
    return _safe_str(raw)


def _get_education_list(resume: Dict) -> List[Dict]:
    """Return education entries that have at least degree or institution."""
    edu = resume.get("education") or []
    if not isinstance(edu, list):
        return []
    result = []
    for e in edu:
        if isinstance(e, dict) and (
            e.get("degree") or e.get("institution") or e.get("college")
        ):
            result.append(e)
        elif isinstance(e, str) and e.strip():
            result.append({"raw_text": e})
    return result


def _get_skills_list(resume: Dict) -> List[str]:
    skills = resume.get("skills") or []
    if not isinstance(skills, list):
        return []
    return [_safe_str(s) for s in skills if _safe_str(s)]


def _get_experience_list(resume: Dict) -> List[Dict]:
    exp = resume.get("experience") or []
    if not isinstance(exp, list):
        return []
    return [e for e in exp if isinstance(e, dict)]


# ─────────────────────────────────────────────────────────────────────────────
# ATS RULES ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class ATSRulesEngine:
    """Validates canonical Resume Builder JSON — never reads raw text for section detection."""

    def __init__(self):
        self.issues:         List[Issue] = []
        self.section_issues: Dict[str, SectionIssue] = {}
        self.strengths:      List[str] = []

    def analyze(self, resume: Dict) -> ORSScore:
        self.issues         = []
        self.section_issues = {}
        self.strengths      = []

        logger.info("Starting ATS analysis v3 (JSON-based)")

        format_score     = self._analyze_format(resume)
        structure_score  = self._analyze_structure(resume)
        content_score    = self._analyze_content(resume)
        keyword_score    = self._analyze_keywords(resume)
        compliance_score = self._analyze_ats_compliance(resume)

        self._analyze_summary_section(resume)
        self._analyze_skills_section(resume)
        self._analyze_experience_section(resume)
        self._analyze_education_section(resume)

        total_score = self._calculate_weighted_score(
            format_score=format_score,
            structure_score=structure_score,
            content_score=content_score,
            keyword_score=keyword_score,
            compliance_score=compliance_score,
        )

        critical_count = sum(1 for i in self.issues if i.severity == SeverityLevel.CRITICAL)
        logger.info(f"ATS analysis complete. Score: {total_score}, Critical: {critical_count}")

        return ORSScore(
            total_score=max(total_score, 0),
            format_score=format_score,
            structure_score=structure_score,
            content_score=content_score,
            keyword_score=keyword_score,
            ats_compliance_score=compliance_score,
            critical_issues_count=critical_count,
            all_issues=self.issues,
            section_issues=self.section_issues,
        )

    # ── SECTION ANALYSIS — all based on structured JSON ────────────────────

    def _analyze_summary_section(self, resume: Dict) -> None:
        si = SectionIssue(section_name="summary")
        summary = _get_summary_str(resume)

        if not summary:
            si.missing_fields   = ["Professional Summary"]
            si.current_status   = ["critical"]
            si.current_score    = 0
            si.improvements     = [
                "Add 3-5 line professional summary",
                "Include: job title, years of experience, key strengths",
                "Show unique value proposition",
            ]
            si.specific_suggestions = [{
                "issue":      "Missing professional summary",
                "current":    "[NONE]",
                "suggestion": (
                    "Results-driven professional with X+ years of experience. "
                    "Expert in [skill 1], [skill 2], and [skill 3]. "
                    "Proven track record of delivering measurable results."
                ),
            }]
            self._add_issue(
                severity=SeverityLevel.HIGH,
                category=IssueCategory.STRUCTURE,
                section="summary",
                message="Professional summary missing",
                suggestion="Add 3-5 line professional summary at the top",
                impact=15,
                specific_example="[NONE]",
                improvement_example="Results-driven professional with X+ years...",
            )
        else:
            words = len(summary.split())
            if words < 20:
                si.incomplete_fields = ["Summary too brief"]
                si.quality_issues.append(f"Summary is only {words} words (ideal: 30-50)")
                si.current_status    = ["needs_improvement"]
                si.current_score     = 40
                si.improvements      = [
                    f"Expand from {words} to 35-50 words",
                    "Add specific achievements or metrics",
                    "Include target role or specialization",
                ]
                self._add_issue(
                    severity=SeverityLevel.MEDIUM,
                    category=IssueCategory.CONTENT,
                    section="summary",
                    message=f"Summary too brief ({words} words)",
                    suggestion="Expand to 35-50 words with achievements",
                    impact=8,
                    specific_example=summary[:120],
                )
            elif words > 150:
                si.quality_issues.append(f"Summary too long ({words} words)")
                si.current_status    = ["needs_improvement"]
                si.current_score     = 60
                si.improvements      = [
                    f"Reduce from {words} to under 100 words",
                    "Focus on most important achievements",
                ]
                self._add_issue(
                    severity=SeverityLevel.LOW,
                    category=IssueCategory.CONTENT,
                    section="summary",
                    message=f"Summary too long ({words} words)",
                    suggestion="Keep professional summary under 100 words",
                    impact=3,
                )
            else:
                si.current_status = ["good"]
                si.current_score  = 90
                self.strengths.append("Well-written professional summary")

            has_metrics = any(re.search(p, summary) for p in METRICS_PATTERNS)
            if not has_metrics:
                si.quality_issues.append("No metrics or numbers in summary")
                si.improvements.append("Add quantifiable achievements (years, %, scale)")

        self.section_issues["summary"] = si

    def _analyze_skills_section(self, resume: Dict) -> None:
        si = SectionIssue(section_name="skills")
        skills = _get_skills_list(resume)

        if not skills:
            si.missing_fields    = ["Skills list"]
            si.current_status    = ["critical"]
            si.current_score     = 0
            si.improvements      = [
                "Add 8-12 relevant hard skills",
                "Include: languages, frameworks, tools, platforms",
                "Match job description keywords",
            ]
            si.specific_suggestions = [{
                "issue":      "No skills listed",
                "current":    "[NONE]",
                "suggestion": "Python • SQL • Excel • Tableau • Project Management • Communication",
            }]
            self._add_issue(
                severity=SeverityLevel.CRITICAL,
                category=IssueCategory.STRUCTURE,
                section="skills",
                message="Skills section missing or empty",
                suggestion="Add 8-12 hard skills",
                impact=20,
            )
        else:
            skill_count = len(skills)
            if skill_count < MIN_SKILLS:
                si.incomplete_fields = [f"Only {skill_count} skills (minimum {MIN_SKILLS})"]
                si.current_status    = ["needs_improvement"]
                si.current_score     = 50
                si.improvements      = [
                    f"Add {MIN_SKILLS - skill_count} more skills",
                    "Include languages, frameworks, tools, platforms",
                ]
                self._add_issue(
                    severity=SeverityLevel.HIGH,
                    category=IssueCategory.STRUCTURE,
                    section="skills",
                    message=f"Insufficient skills ({skill_count} listed)",
                    suggestion=f"Add to {IDEAL_SKILLS} skills minimum",
                    impact=15,
                    specific_example=f"Current: {', '.join(skills[:5])}",
                )
            elif skill_count > MAX_SKILLS:
                si.quality_issues.append(f"Too many skills ({skill_count} > {MAX_SKILLS})")
                si.current_status = ["needs_improvement"]
                si.current_score  = 60
                si.improvements   = ["Remove outdated or less relevant skills", "Keep only 8-15 most relevant"]
                self._add_issue(
                    severity=SeverityLevel.LOW,
                    category=IssueCategory.CONTENT,
                    section="skills",
                    message=f"Too many skills ({skill_count})",
                    suggestion="Limit to 8-15 most relevant skills",
                    impact=3,
                )
            else:
                si.current_status = ["good"]
                si.current_score  = 85
                self.strengths.append(f"Well-balanced skills section ({skill_count} skills)")

        self.section_issues["skills"] = si

    def _analyze_experience_section(self, resume: Dict) -> None:
        si = SectionIssue(section_name="experience")
        experience = _get_experience_list(resume)

        if not experience:
            si.missing_fields    = ["Work experience entries"]
            si.current_status    = ["critical"]
            si.current_score     = 0
            si.improvements      = [
                "Add at least 2-3 work experience entries",
                "For each: Title, Company, Duration, 3-5 achievement bullets",
            ]
            si.specific_suggestions = [{
                "issue":   "No work experience listed",
                "current": "[NONE]",
                "suggestion": (
                    "Senior Engineer | Tech Company | 2020-Present\n"
                    "• Led team of 5 engineers, delivering 3 major projects\n"
                    "• Optimized system performance by 60%, saving $50K annually"
                ),
            }]
            self._add_issue(
                severity=SeverityLevel.CRITICAL,
                category=IssueCategory.STRUCTURE,
                section="experience",
                message="Work experience section missing",
                suggestion="Add 2-5 work experience entries with achievements",
                impact=25,
            )
        else:
            total_bullets = 0
            weak_bullets  = 0
            metric_bullets = 0
            jobs_issues: List[str] = []

            for idx, job in enumerate(experience, 1):
                title   = _safe_str(job.get("title") or job.get("position", ""))
                company = _safe_str(job.get("company", ""))
                bullets = job.get("bullets") or []
                if not isinstance(bullets, list):
                    bullets = []

                if not title:
                    jobs_issues.append(f"Job {idx}: Missing job title")
                if not company:
                    jobs_issues.append(f"Job {idx}: Missing company name")
                if not bullets:
                    jobs_issues.append(f"Job {idx} ({title or 'Unknown'}): No achievement bullets")
                else:
                    for bullet in bullets:
                        if not bullet:
                            continue
                        total_bullets += 1
                        has_action  = any(v in _safe_str(bullet).lower() for v in STRONG_ACTION_VERBS)
                        has_metrics = any(re.search(p, _safe_str(bullet)) for p in METRICS_PATTERNS)
                        words       = len(_safe_str(bullet).split())
                        if not has_action or words < 8:
                            weak_bullets += 1
                        if has_metrics:
                            metric_bullets += 1

            if total_bullets == 0:
                si.current_score = 20
                si.quality_issues.append("No achievement bullets found")
            else:
                weak_pct   = int((weak_bullets   / total_bullets) * 100)
                metric_pct = int((metric_bullets / total_bullets) * 100)

                if weak_pct > 50:
                    si.current_status = ["needs_improvement"]
                    si.current_score  = 40
                    si.quality_issues.append(f"{weak_pct}% of bullets are weak or lack metrics")
                    si.improvements   = [
                        "Use strong action verbs at start of each bullet",
                        "Add quantifiable results (%, $, numbers)",
                        "Make bullets 15-25 words each",
                    ]
                    self._add_issue(
                        severity=SeverityLevel.HIGH,
                        category=IssueCategory.CONTENT,
                        section="experience",
                        message=f"{weak_pct}% of bullets lack metrics or impact",
                        suggestion="Rewrite bullets with action verbs and specific results",
                        impact=15,
                        specific_example="Worked on database optimization",
                        improvement_example="Optimized database queries reducing API latency by 60%",
                    )
                else:
                    si.current_status = ["good"]
                    si.current_score  = 80
                    self.strengths.append(
                        f"Strong experience bullets with {metric_pct}% containing metrics"
                    )

            if jobs_issues:
                si.incomplete_fields.extend(jobs_issues)

        self.section_issues["experience"] = si

    def _analyze_education_section(self, resume: Dict) -> None:
        """
        Validates education from the structured list produced by the Resume Builder.
        Never reads raw_text — presence is determined solely by len(education) > 0.
        """
        si = SectionIssue(section_name="education")
        education = _get_education_list(resume)

        if not education:
            si.missing_fields = ["Degree", "Institution", "Graduation year"]
            si.current_status = ["high"]
            si.current_score  = 20
            si.improvements   = [
                "Add degree (BS, MS, PhD, Bootcamp, etc.)",
                "Add institution/college name",
                "Add graduation year",
                "Include GPA if 3.7 or higher",
            ]
            si.specific_suggestions = [{
                "issue":   "Education section missing",
                "current": "[NONE]",
                "suggestion": "B.S. Computer Science | State University | Graduated 2020",
            }]
            self._add_issue(
                severity=SeverityLevel.HIGH,
                category=IssueCategory.STRUCTURE,
                section="education",
                message="Education section missing or empty",
                suggestion="Add degree, institution, and graduation year",
                impact=12,
                specific_example="[NONE]",
                improvement_example="B.S. Computer Science | Stanford University | 2020",
            )
        else:
            complete_entries  = 0
            incomplete_entries: List[str] = []

            for idx, edu in enumerate(education, 1):
                if not isinstance(edu, dict):
                    continue
                degree      = _safe_str(edu.get("degree", ""))
                institution = _safe_str(edu.get("institution") or edu.get("college", ""))
                year        = _safe_str(edu.get("year", ""))

                if degree and institution and year:
                    complete_entries += 1
                else:
                    missing_parts = []
                    if not degree:      missing_parts.append("degree")
                    if not institution: missing_parts.append("institution")
                    if not year:        missing_parts.append("year")
                    if missing_parts:
                        incomplete_entries.append(f"Entry {idx}: Missing {', '.join(missing_parts)}")

            if incomplete_entries:
                si.incomplete_fields = incomplete_entries
                si.current_status    = ["needs_improvement"]
                si.current_score     = 60
                si.improvements      = [
                    "Complete missing education details",
                    "Ensure format: Degree | Institution | Year",
                    "Add GPA if 3.7 or higher",
                ]
            else:
                si.current_status = ["good"]
                si.current_score  = 90
                self.strengths.append(f"Complete education section ({len(education)} entries)")

        self.section_issues["education"] = si

    # ── STANDARD ANALYSIS ──────────────────────────────────────────────────

    def _analyze_format(self, resume: Dict) -> int:
        score = 100

        file_type = _safe_str(resume.get("file_type")).lower()
        if file_type and file_type not in {"pdf", "docx", "doc"}:
            self._add_issue(
                severity=SeverityLevel.CRITICAL, category=IssueCategory.FORMAT,
                section="summary", message=f"Unsupported file type: {file_type}",
                suggestion="Convert to PDF or DOCX format", impact=15,
            )
            score -= 15

        font = _safe_str(resume.get("font")).lower()
        if font:
            if font in ATS_UNSAFE_FONTS:
                self._add_issue(
                    severity=SeverityLevel.CRITICAL, category=IssueCategory.FORMAT,
                    section="summary", message=f"ATS-unsafe font: {font}",
                    suggestion="Use Arial, Calibri, or Times New Roman", impact=15,
                )
                score -= 15
            elif font not in ATS_SAFE_FONTS:
                self._add_issue(
                    severity=SeverityLevel.HIGH, category=IssueCategory.FORMAT,
                    section="summary", message=f"Non-standard font: {font}",
                    suggestion="Use standard ATS-safe fonts", impact=8,
                )
                score -= 8
            else:
                self.strengths.append(f"ATS-safe font: {font}")

        if resume.get("uses_table") or resume.get("has_tables"):
            self._add_issue(
                severity=SeverityLevel.HIGH, category=IssueCategory.FORMAT,
                section="experience", message="Tables detected",
                suggestion="Remove tables, use bullet points instead", impact=12,
            )
            score -= 12

        if resume.get("uses_columns") or resume.get("multi_column"):
            self._add_issue(
                severity=SeverityLevel.HIGH, category=IssueCategory.FORMAT,
                section="summary", message="Multi-column layout detected",
                suggestion="Use single-column layout", impact=12,
            )
            score -= 12

        return max(score, 0)

    def _analyze_structure(self, resume: Dict) -> int:
        score = 100

        # Summary
        if not _get_summary_str(resume):
            self._add_issue(
                severity=SeverityLevel.MEDIUM, category=IssueCategory.STRUCTURE,
                section="summary", message="Summary missing or empty",
                suggestion="Add professional summary", impact=7,
            )
            score -= 7

        # Skills
        if not _get_skills_list(resume):
            self._add_issue(
                severity=SeverityLevel.HIGH, category=IssueCategory.STRUCTURE,
                section="skills", message="Skills section missing or empty",
                suggestion="Add skills section", impact=10,
            )
            score -= 10

        # Experience
        if not _get_experience_list(resume):
            self._add_issue(
                severity=SeverityLevel.HIGH, category=IssueCategory.STRUCTURE,
                section="experience", message="Experience section missing",
                suggestion="Add work experience", impact=15,
            )
            score -= 15

        # Education
        if not _get_education_list(resume):
            self._add_issue(
                severity=SeverityLevel.HIGH, category=IssueCategory.STRUCTURE,
                section="education", message="Education section missing",
                suggestion="Add education details", impact=10,
            )
            score -= 10

        return max(score, 0)

    def _analyze_content(self, resume: Dict) -> int:
        score      = 100
        experience = _get_experience_list(resume)

        if experience:
            bullet_analysis = self._analyze_bullets(experience)
            if bullet_analysis["weak_percentage"] > 50:
                self._add_issue(
                    severity=SeverityLevel.HIGH, category=IssueCategory.CONTENT,
                    section="experience",
                    message=f"{bullet_analysis['weak_percentage']}% weak bullets",
                    suggestion="Strengthen with action verbs and metrics", impact=15,
                )
                score -= 15
            elif bullet_analysis["weak_percentage"] > 30:
                self._add_issue(
                    severity=SeverityLevel.MEDIUM, category=IssueCategory.CONTENT,
                    section="experience",
                    message=f"{bullet_analysis['weak_percentage']}% lack metrics",
                    suggestion="Add quantifiable results", impact=8,
                )
                score -= 8

        return max(score, 0)

    def _analyze_bullets(self, experience: List[Dict]) -> Dict:
        results = {"total": 0, "strong_count": 0, "weak_count": 0, "with_metrics": 0, "weak_percentage": 0}
        all_bullets = []
        for exp in experience:
            if isinstance(exp, dict):
                bullets = exp.get("bullets") or []
                if isinstance(bullets, list):
                    all_bullets.extend(bullets)

        if not all_bullets:
            return results

        results["total"] = len(all_bullets)
        for bullet in all_bullets:
            if not bullet:
                continue
            bullet_lower = _safe_str(bullet).lower()
            has_action  = any(v in bullet_lower for v in STRONG_ACTION_VERBS)
            has_weak    = any(v in bullet_lower for v in WEAK_ACTION_VERBS)
            has_metrics = any(re.search(p, _safe_str(bullet)) for p in METRICS_PATTERNS)
            words       = len(_safe_str(bullet).split())

            if has_action and not has_weak and words >= 8:
                results["strong_count"] += 1
            else:
                results["weak_count"] += 1
            if has_metrics:
                results["with_metrics"] += 1

        if results["total"] > 0:
            results["weak_percentage"] = int((results["weak_count"] / results["total"]) * 100)

        return results

    def _analyze_keywords(self, resume: Dict) -> int:
        return 100

    def _analyze_ats_compliance(self, resume: Dict) -> int:
        score    = 100
        summary  = _get_summary_str(resume)
        skills   = " ".join(_get_skills_list(resume))
        bullets  = " ".join(
            _safe_str(b)
            for exp in _get_experience_list(resume)
            for b in (exp.get("bullets") or [])
        )
        resume_text = f"{summary} {skills} {bullets}"

        for char, desc in [("®", "Registered trademark"), ("™", "Trademark"), ("©", "Copyright")]:
            if char in resume_text:
                self._add_issue(
                    severity=SeverityLevel.LOW, category=IssueCategory.COMPLIANCE,
                    section="summary", message=f"Special character: {desc}",
                    suggestion="Remove or use text equivalent", impact=2,
                )
                score -= 2

        return max(score, 0)

    # ── HELPERS ─────────────────────────────────────────────────────────────

    def _add_issue(
        self,
        severity:            SeverityLevel,
        category:            IssueCategory,
        section:             str,
        message:             str,
        suggestion:          str,
        impact:              int,
        specific_example:    Optional[str] = None,
        improvement_example: Optional[str] = None,
    ):
        self.issues.append(Issue(
            severity=severity,
            category=category,
            section=section,
            message=message,
            suggestion=suggestion,
            impact_score=impact,
            specific_example=specific_example,
            improvement_example=improvement_example,
        ))

    def _calculate_weighted_score(
        self,
        format_score:     int,
        structure_score:  int,
        content_score:    int,
        keyword_score:    int,
        compliance_score: int,
    ) -> int:
        total = (
            format_score     * 0.10 +
            structure_score  * 0.15 +
            content_score    * 0.30 +
            keyword_score    * 0.30 +
            compliance_score * 0.15
        )
        return int(total)