# /home/aryu_user/Arun/aiproject_staging/app/utils/ats_scanner/ats_rules_advanced.py
"""
Production-Grade ATS Rules Engine v2
Enhanced with:
- Fixed education detection (handles various formats)
- Advanced section-by-section analysis
- Specific improvement suggestions per section
- Detailed missing/incomplete field detection
- None-safe summary handling (FIX APPLIED)
- Enterprise-level ATS simulation
"""

import re
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# =====================================================
# ENUMS & CONSTANTS
# =====================================================

class SeverityLevel(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class IssueCategory(Enum):
    FORMAT = "format"
    STRUCTURE = "structure"
    CONTENT = "content"
    KEYWORDS = "keywords"
    QUALITY = "quality"
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
    # Leadership
    "led", "managed", "directed", "orchestrated", "spearheaded",
    "supervised", "coordinated", "oversaw", "mentored", "delegated",
    "guided", "coached",
    # Achievement
    "achieved", "delivered", "accomplished", "completed", "exceeded",
    "surpassed", "attained", "earned", "won", "captured",
    # Creation
    "developed", "built", "created", "designed", "architected",
    "engineered", "constructed", "launched", "initiated", "innovated",
    "pioneered", "founded",
    # Improvement
    "optimized", "improved", "enhanced", "streamlined", "refined",
    "accelerated", "elevated", "boosted", "increased", "reduced",
    "lowered", "minimized", "maximized", "scaled", "expanded",
    # Analysis
    "analyzed", "evaluated", "assessed", "examined", "identified",
    "discovered", "uncovered", "determined", "diagnosed", "investigated",
    # Implementation
    "implemented", "executed", "deployed", "installed", "integrated",
    "established", "instituted", "released",
    # Transformation
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

MIN_BULLETS_PER_ROLE = 2
MAX_BULLETS_PER_ROLE = 8
MIN_BULLET_LENGTH = 12
IDEAL_BULLET_LENGTH = 18
MAX_BULLET_LENGTH = 120
MIN_SKILLS = 5
IDEAL_SKILLS = 10
MAX_SKILLS = 50
MIN_RESUME_LENGTH = 200
IDEAL_RESUME_LENGTH = 500
MAX_RESUME_LENGTH = 2000


# =====================================================
# DATA CLASSES
# =====================================================

@dataclass
class Issue:
    severity: SeverityLevel
    category: IssueCategory
    section: str
    message: str
    suggestion: str
    impact_score: int
    specific_example: Optional[str] = None
    improvement_example: Optional[str] = None


@dataclass
class SectionIssue:
    section_name: str
    current_status: List[str] = field(default_factory=list)
    current_score: int = 0
    missing_fields: List[str] = field(default_factory=list)
    incomplete_fields: List[str] = field(default_factory=list)
    quality_issues: List[str] = field(default_factory=list)
    improvements: List[str] = field(default_factory=list)
    specific_suggestions: List[Dict] = field(default_factory=list)


@dataclass
class ORSScore:
    total_score: int
    format_score: int
    structure_score: int
    content_score: int
    keyword_score: int
    ats_compliance_score: int
    critical_issues_count: int
    all_issues: List[Issue]
    section_issues: Dict[str, SectionIssue] = field(default_factory=dict)


# =====================================================
# HELPERS
# =====================================================

def _safe_str(value) -> str:
    """Safely convert any value to a stripped string"""
    if value is None:
        return ""
    return str(value).strip()


def _is_present(value) -> bool:
    """Robust check for meaningful content"""
    if value is None:
        return False
    if isinstance(value, str):
        return len(value.strip()) > 0
    if isinstance(value, (list, dict)):
        return len(value) > 0
    return bool(value)


# =====================================================
# ATS RULES ENGINE
# =====================================================

class ATSRulesEngine:
    """Production-grade ATS compliance validator v2"""

    def __init__(self):
        self.issues: List[Issue] = []
        self.section_issues: Dict[str, SectionIssue] = {}
        self.strengths: List[str] = []

    def analyze(self, resume: Dict) -> ORSScore:
        """Comprehensive ATS analysis of resume"""
        self.issues = []
        self.section_issues = {}
        self.strengths = []

        logger.info("Starting ATS analysis v2")

        format_score = self._analyze_format(resume)
        structure_score = self._analyze_structure(resume)
        content_score = self._analyze_content(resume)
        keyword_score = self._analyze_keywords(resume)
        compliance_score = self._analyze_ats_compliance(resume)

        # Detailed section analysis
        self._analyze_summary_section(resume)
        self._analyze_skills_section(resume)
        self._analyze_experience_section(resume)
        self._analyze_education_section(resume)

        total_score = self._calculate_weighted_score(
            format_score=format_score,
            structure_score=structure_score,
            content_score=content_score,
            keyword_score=keyword_score,
            compliance_score=compliance_score
        )

        critical_count = sum(1 for i in self.issues if i.severity == SeverityLevel.CRITICAL)

        logger.info(f"ATS analysis complete. Score: {total_score}, Critical issues: {critical_count}")

        return ORSScore(
            total_score=max(total_score, 0),
            format_score=format_score,
            structure_score=structure_score,
            content_score=content_score,
            keyword_score=keyword_score,
            ats_compliance_score=compliance_score,
            critical_issues_count=critical_count,
            all_issues=self.issues,
            section_issues=self.section_issues
        )

    # =========== SECTION ANALYSIS ===========

    def _analyze_summary_section(self, resume: Dict) -> None:
        """Detailed analysis of professional summary"""

        section_issue = SectionIssue(section_name="summary")

        # FIX: Use _safe_str to handle None, missing keys, non-string values
        summary = _safe_str(resume.get("summary"))

        if not summary:
            # Summary is genuinely missing
            section_issue.missing_fields = ["Professional Summary"]
            section_issue.current_status = ["critical"]
            section_issue.current_score = 0
            section_issue.improvements = [
                "Add 3-5 line professional summary",
                "Include: job title, years of experience, key strengths",
                "Show unique value proposition"
            ]
            section_issue.specific_suggestions = [
                {
                    "issue": "Missing professional summary",
                    "current": "[NONE]",
                    "suggestion": (
                        "Results-driven Senior Engineer with 8+ years building scalable solutions. "
                        "Expertise in Python, AWS, and microservices. "
                        "Proven track record delivering 200+ projects on time."
                    )
                }
            ]

            self._add_issue(
                severity=SeverityLevel.HIGH,
                category=IssueCategory.STRUCTURE,
                section="summary",
                message="Professional summary missing",
                suggestion="Add 3-5 line professional summary at the top",
                impact=15,
                specific_example="[NONE]",
                improvement_example="Results-driven Senior Engineer with 8+ years..."
            )
            section_issue.current_score = 0

        else:
            words = len(summary.split())

            if words < 20:
                section_issue.incomplete_fields = ["Summary too brief"]
                section_issue.quality_issues.append(f"Summary is only {words} words (ideal: 30-50)")
                section_issue.current_status = ["needs_improvement"]
                section_issue.current_score = 40
                section_issue.improvements = [
                    f"Expand from {words} to 35-50 words",
                    "Add specific achievements or metrics",
                    "Include target role or specialization"
                ]

                self._add_issue(
                    severity=SeverityLevel.MEDIUM,
                    category=IssueCategory.CONTENT,
                    section="summary",
                    message=f"Summary too brief ({words} words)",
                    suggestion="Expand to 35-50 words with achievements",
                    impact=8,
                    specific_example=summary,
                    improvement_example=f"{summary} Delivered 50+ projects with 99.9% uptime."
                )

            elif words > 150:
                section_issue.quality_issues.append(f"Summary too long ({words} words)")
                section_issue.current_status = ["needs_improvement"]
                section_issue.current_score = 60
                section_issue.improvements = [
                    f"Reduce from {words} to under 100 words",
                    "Focus on most important achievements",
                    "Keep punchy and scannable"
                ]

                self._add_issue(
                    severity=SeverityLevel.LOW,
                    category=IssueCategory.CONTENT,
                    section="summary",
                    message=f"Summary too long ({words} words)",
                    suggestion="Keep professional summary under 100 words",
                    impact=3
                )

            else:
                section_issue.current_status = ["good"]
                section_issue.current_score = 90
                self.strengths.append("Well-written professional summary")

            # Check for metrics in summary
            has_metrics = any(re.search(p, summary) for p in METRICS_PATTERNS)
            if not has_metrics:
                section_issue.quality_issues.append("No metrics or numbers in summary")
                section_issue.improvements.append("Add quantifiable achievements (years, %, scale)")

        self.section_issues["summary"] = section_issue

    def _analyze_skills_section(self, resume: Dict) -> None:
        """Detailed analysis of skills section"""

        section_issue = SectionIssue(section_name="skills")
        skills = resume.get("skills", [])

        # Normalize: handle None or non-list
        if not isinstance(skills, list):
            skills = []

        if not _is_present(skills):
            section_issue.missing_fields = ["Skills list"]
            section_issue.current_status = ["critical"]
            section_issue.current_score = 0
            section_issue.improvements = [
                "Add 8-12 relevant hard skills",
                "Include: languages, frameworks, tools, platforms",
                "Match job description keywords"
            ]
            section_issue.specific_suggestions = [
                {
                    "issue": "No skills listed",
                    "current": "[NONE]",
                    "suggestion": "Python • Django • PostgreSQL • AWS • Docker • React • GraphQL • Redis"
                }
            ]

            self._add_issue(
                severity=SeverityLevel.CRITICAL,
                category=IssueCategory.STRUCTURE,
                section="skills",
                message="Skills section missing or empty",
                suggestion="Add 8-12 hard technical skills",
                impact=20
            )
        else:
            skill_count = len(skills)

            if skill_count < MIN_SKILLS:
                section_issue.incomplete_fields = [f"Only {skill_count} skills (minimum {MIN_SKILLS})"]
                section_issue.current_status = ["needs_improvement"]
                section_issue.current_score = 50
                section_issue.improvements = [
                    f"Add {MIN_SKILLS - skill_count} more skills",
                    "Include languages, frameworks, tools, platforms",
                    "Focus on in-demand skills for target role"
                ]

                self._add_issue(
                    severity=SeverityLevel.HIGH,
                    category=IssueCategory.STRUCTURE,
                    section="skills",
                    message=f"Insufficient skills ({skill_count} listed)",
                    suggestion=f"Add to {IDEAL_SKILLS} skills minimum",
                    impact=15,
                    specific_example=f"Current: {', '.join(skills)}",
                    improvement_example=f"{', '.join(skills)} • NewSkill1 • NewSkill2 • NewSkill3"
                )

            elif skill_count > MAX_SKILLS:
                section_issue.quality_issues.append(f"Too many skills ({skill_count} > {MAX_SKILLS})")
                section_issue.current_status = ["needs_improvement"]
                section_issue.current_score = 60
                section_issue.improvements = [
                    "Remove outdated or less relevant skills",
                    "Keep only 8-15 most relevant skills",
                    "Prioritize by relevance to target role"
                ]

                self._add_issue(
                    severity=SeverityLevel.LOW,
                    category=IssueCategory.CONTENT,
                    section="skills",
                    message=f"Too many skills ({skill_count})",
                    suggestion="Limit to 8-15 most relevant skills",
                    impact=3
                )

            else:
                section_issue.current_status = ["good"]
                section_issue.current_score = 85
                self.strengths.append(f"Well-balanced skills section ({skill_count} skills)")

        self.section_issues["skills"] = section_issue

    def _analyze_experience_section(self, resume: Dict) -> None:
        """Detailed analysis of work experience"""

        section_issue = SectionIssue(section_name="experience")
        experience = resume.get("experience", [])

        if not isinstance(experience, list):
            experience = []

        if not _is_present(experience):
            section_issue.missing_fields = ["Work experience entries"]
            section_issue.current_status = ["critical"]
            section_issue.current_score = 0
            section_issue.improvements = [
                "Add at least 2-3 work experience entries",
                "For each: Title, Company, Duration, 3-5 achievement bullets"
            ]
            section_issue.specific_suggestions = [
                {
                    "issue": "No work experience listed",
                    "current": "[NONE]",
                    "suggestion": (
                        "Senior Engineer | Tech Company | 2020-Present\n"
                        "• Led team of 5 engineers, delivering 3 major projects\n"
                        "• Optimized system performance by 60%, saving $50K annually"
                    )
                }
            ]

            self._add_issue(
                severity=SeverityLevel.CRITICAL,
                category=IssueCategory.STRUCTURE,
                section="experience",
                message="Work experience section missing",
                suggestion="Add 2-5 work experience entries with achievements",
                impact=25
            )
        else:
            total_bullets = 0
            weak_bullets = 0
            bullets_with_metrics = 0
            jobs_issues = []

            for idx, job in enumerate(experience, 1):
                if not isinstance(job, dict):
                    continue

                title = _safe_str(job.get("title"))
                company = _safe_str(job.get("company"))
                bullets = job.get("bullets", [])

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

                        has_action = any(v in _safe_str(bullet).lower() for v in STRONG_ACTION_VERBS)
                        has_metrics = any(re.search(p, _safe_str(bullet)) for p in METRICS_PATTERNS)
                        words = len(_safe_str(bullet).split())

                        if not has_action or words < MIN_BULLET_LENGTH:
                            weak_bullets += 1

                        if has_metrics:
                            bullets_with_metrics += 1

            if total_bullets == 0:
                section_issue.current_score = 20
                section_issue.quality_issues.append("No achievement bullets found")
            else:
                weak_pct = int((weak_bullets / total_bullets) * 100)
                metrics_pct = int((bullets_with_metrics / total_bullets) * 100)

                if weak_pct > 50:
                    section_issue.current_status = ["needs_improvement"]
                    section_issue.current_score = 40
                    section_issue.quality_issues.append(
                        f"{weak_pct}% of bullets are weak or lack metrics"
                    )
                    section_issue.improvements = [
                        "Use strong action verbs at start of each bullet",
                        "Add quantifiable results (%, $, numbers)",
                        "Make bullets 15-25 words each",
                        "Show impact and value delivered"
                    ]

                    self._add_issue(
                        severity=SeverityLevel.HIGH,
                        category=IssueCategory.CONTENT,
                        section="experience",
                        message=f"{weak_pct}% of bullets lack metrics or impact",
                        suggestion="Rewrite bullets with action verbs and specific results",
                        impact=15,
                        specific_example="Worked on database optimization",
                        improvement_example=(
                            "Optimized database queries reducing API latency by 60%, "
                            "serving 2M+ requests daily"
                        )
                    )
                else:
                    section_issue.current_status = ["good"]
                    section_issue.current_score = 80
                    self.strengths.append(
                        f"Strong experience bullets with {metrics_pct}% containing metrics"
                    )

            if jobs_issues:
                section_issue.incomplete_fields.extend(jobs_issues)

        self.section_issues["experience"] = section_issue

    def _analyze_education_section(self, resume: Dict) -> None:
        """Fixed: Detailed analysis of education section"""

        section_issue = SectionIssue(section_name="education")
        education = resume.get("education", [])

        # Normalize
        if not isinstance(education, list):
            education = []

        has_education = False
        education_items = []

        if education and len(education) > 0:
            for edu in education:
                if isinstance(edu, dict):
                    degree = _safe_str(edu.get("degree"))
                    institution = _safe_str(edu.get("institution") or edu.get("college", ""))
                    if degree or institution:
                        has_education = True
                        education_items.append(edu)
                elif isinstance(edu, str) and len(edu.strip()) > 3:
                    has_education = True
                    education_items.append({"raw_text": edu})

        if not has_education:
            section_issue.missing_fields = ["Degree", "Institution", "Graduation year"]
            section_issue.current_status = ["high"]
            section_issue.current_score = 20
            section_issue.improvements = [
                "Add degree (BS, MS, PhD, Bootcamp, etc.)",
                "Add institution/college name",
                "Add graduation year",
                "Include GPA if 3.7 or higher",
                "Add relevant coursework if space permits"
            ]
            section_issue.specific_suggestions = [
                {
                    "issue": "Education section missing",
                    "current": "[NONE]",
                    "suggestion": (
                        "B.S. Computer Science | State University | Graduated 2020\n"
                        "Relevant coursework: Data Structures, Algorithms, Machine Learning"
                    )
                }
            ]

            self._add_issue(
                severity=SeverityLevel.HIGH,
                category=IssueCategory.STRUCTURE,
                section="education",
                message="Education section missing or incomplete",
                suggestion="Add degree, institution, and graduation year",
                impact=12,
                specific_example="[NONE]",
                improvement_example="B.S. Computer Science | Stanford University | 2020"
            )
        else:
            complete_entries = 0
            incomplete_entries = []

            for idx, edu in enumerate(education_items, 1):
                if not isinstance(edu, dict):
                    continue

                degree = _safe_str(edu.get("degree"))
                institution = _safe_str(edu.get("institution") or edu.get("college", ""))
                year = _safe_str(edu.get("year"))

                if degree and institution and year:
                    complete_entries += 1
                else:
                    missing = []
                    if not degree:
                        missing.append("degree")
                    if not institution:
                        missing.append("institution")
                    if not year:
                        missing.append("year")

                    incomplete_entries.append(f"Entry {idx}: Missing {', '.join(missing)}")

            if incomplete_entries:
                section_issue.incomplete_fields = incomplete_entries
                section_issue.current_status = ["needs_improvement"]
                section_issue.current_score = 60
                section_issue.improvements = [
                    "Complete missing education details",
                    "Ensure format: Degree | Institution | Year",
                    "Add GPA if 3.7 or higher"
                ]
            else:
                section_issue.current_status = ["good"]
                section_issue.current_score = 90
                self.strengths.append(
                    f"Complete education section ({len(education_items)} entries)"
                )

        self.section_issues["education"] = section_issue

    # =========== STANDARD ANALYSIS METHODS ===========

    def _analyze_format(self, resume: Dict) -> int:
        """Check file format, fonts, and layout safety"""
        score = 100

        file_type = _safe_str(resume.get("file_type")).lower()
        if file_type and file_type not in {"pdf", "docx", "doc"}:
            self._add_issue(
                severity=SeverityLevel.CRITICAL,
                category=IssueCategory.FORMAT,
                section="summary",
                message=f"Unsupported file type: {file_type}",
                suggestion="Convert to PDF or DOCX format",
                impact=15
            )
            score -= 15

        font = _safe_str(resume.get("font")).lower()
        if font:
            if font in ATS_UNSAFE_FONTS:
                self._add_issue(
                    severity=SeverityLevel.CRITICAL,
                    category=IssueCategory.FORMAT,
                    section="summary",
                    message=f"ATS-unsafe font: {font}",
                    suggestion="Use Arial, Calibri, or Times New Roman",
                    impact=15
                )
                score -= 15
            elif font not in ATS_SAFE_FONTS:
                self._add_issue(
                    severity=SeverityLevel.HIGH,
                    category=IssueCategory.FORMAT,
                    section="summary",
                    message=f"Non-standard font: {font}",
                    suggestion="Use standard ATS-safe fonts",
                    impact=8
                )
                score -= 8
            else:
                self.strengths.append(f"ATS-safe font: {font}")

        if resume.get("uses_table"):
            self._add_issue(
                severity=SeverityLevel.HIGH,
                category=IssueCategory.FORMAT,
                section="experience",
                message="Tables detected",
                suggestion="Remove tables, use bullet points instead",
                impact=12
            )
            score -= 12

        if resume.get("uses_columns"):
            self._add_issue(
                severity=SeverityLevel.HIGH,
                category=IssueCategory.FORMAT,
                section="summary",
                message="Multi-column layout detected",
                suggestion="Use single-column layout",
                impact=12
            )
            score -= 12

        return max(score, 0)

    def _analyze_structure(self, resume: Dict) -> int:
        """Check section presence and overall structure"""
        score = 100

        sections_check = {
            "summary": (15, "Professional summary"),
            "skills": (20, "Skills section"),
            "experience": (30, "Work experience"),
            "education": (20, "Education")
        }

        for section, (points, desc) in sections_check.items():
            content = resume.get(section)

            is_empty = not _is_present(content)

            if is_empty and section != "education":  # education handled separately
                self._add_issue(
                    severity=SeverityLevel.HIGH if section in ["experience", "education"] else SeverityLevel.MEDIUM,
                    category=IssueCategory.STRUCTURE,
                    section=section,
                    message=f"{section.title()} missing or empty",
                    suggestion=f"Add {desc}",
                    impact=points // 2
                )
                score -= points // 2

        return max(score, 0)

    def _analyze_content(self, resume: Dict) -> int:
        """Check bullet quality and metrics"""
        score = 100

        experience = resume.get("experience", [])
        if isinstance(experience, list) and experience:
            bullet_analysis = self._analyze_bullets(experience)

            if bullet_analysis["weak_percentage"] > 50:
                self._add_issue(
                    severity=SeverityLevel.HIGH,
                    category=IssueCategory.CONTENT,
                    section="experience",
                    message=f"{bullet_analysis['weak_percentage']}% weak bullets",
                    suggestion="Strengthen with action verbs and metrics",
                    impact=15
                )
                score -= 15
            elif bullet_analysis["weak_percentage"] > 30:
                self._add_issue(
                    severity=SeverityLevel.MEDIUM,
                    category=IssueCategory.CONTENT,
                    section="experience",
                    message=f"{bullet_analysis['weak_percentage']}% lack metrics",
                    suggestion="Add quantifiable results",
                    impact=8
                )
                score -= 8

        return max(score, 0)

    def _analyze_bullets(self, experience: List[Dict]) -> Dict:
        """Analyze strength of all experience bullets"""
        results = {
            "total": 0,
            "strong_count": 0,
            "weak_count": 0,
            "with_metrics": 0,
            "weak_percentage": 0
        }

        all_bullets = []
        for exp in experience:
            if isinstance(exp, dict):
                bullets = exp.get("bullets", [])
                if isinstance(bullets, list):
                    all_bullets.extend(bullets)

        if not all_bullets:
            return results

        results["total"] = len(all_bullets)

        for bullet in all_bullets:
            if not bullet:
                continue

            bullet_lower = _safe_str(bullet).lower()
            has_action = any(v in bullet_lower for v in STRONG_ACTION_VERBS)
            has_weak = any(v in bullet_lower for v in WEAK_ACTION_VERBS)
            has_metrics = any(re.search(p, _safe_str(bullet)) for p in METRICS_PATTERNS)
            words = len(_safe_str(bullet).split())

            if has_action and not has_weak and words >= MIN_BULLET_LENGTH:
                results["strong_count"] += 1
            else:
                results["weak_count"] += 1

            if has_metrics:
                results["with_metrics"] += 1

        if results["total"] > 0:
            results["weak_percentage"] = int((results["weak_count"] / results["total"]) * 100)

        return results

    def _analyze_keywords(self, resume: Dict) -> int:
        """Keyword analysis — handled separately by KeywordEngine"""
        return 100

    def _analyze_ats_compliance(self, resume: Dict) -> int:
        """Check for ATS parsing safety issues"""
        score = 100

        resume_text = " ".join([
            _safe_str(resume.get("summary")),
            " ".join(_safe_str(s) for s in (resume.get("skills") or [])),
            " ".join(
                _safe_str(b)
                for exp in (resume.get("experience") or [])
                if isinstance(exp, dict)
                for b in (exp.get("bullets") or [])
            )
        ])

        problematic_chars = {
            "®": "Registered trademark",
            "™": "Trademark",
            "©": "Copyright"
        }

        for char, desc in problematic_chars.items():
            if char in resume_text:
                self._add_issue(
                    severity=SeverityLevel.LOW,
                    category=IssueCategory.COMPLIANCE,
                    section="summary",
                    message=f"Special character: {desc}",
                    suggestion="Remove or use text equivalent",
                    impact=2
                )
                score -= 2

        return max(score, 0)

    # =========== HELPERS ===========

    def _add_issue(
        self,
        severity: SeverityLevel,
        category: IssueCategory,
        section: str,
        message: str,
        suggestion: str,
        impact: int,
        specific_example: Optional[str] = None,
        improvement_example: Optional[str] = None
    ):
        """Add an issue to the issues list"""
        self.issues.append(Issue(
            severity=severity,
            category=category,
            section=section,
            message=message,
            suggestion=suggestion,
            impact_score=impact,
            specific_example=specific_example,
            improvement_example=improvement_example
        ))

    def _calculate_weighted_score(
        self,
        format_score: int,
        structure_score: int,
        content_score: int,
        keyword_score: int,
        compliance_score: int
    ) -> int:
        """Calculate final weighted ATS score"""
        total = (
            format_score * 0.10 +
            structure_score * 0.15 +
            content_score * 0.30 +
            keyword_score * 0.30 +
            compliance_score * 0.15
        )
        return int(total)