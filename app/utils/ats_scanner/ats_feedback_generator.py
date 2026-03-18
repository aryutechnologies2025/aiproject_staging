# /home/aryu_user/Arun/aiproject_staging/app/utils/ats_scanner/ats_feedback_generator.py
"""
Production-Grade ATS Feedback Generator v2
Generates highly detailed, section-by-section feedback with:
- What's missing
- What to add
- What to remove
- Specific examples
- Impact scores

FIX APPLIED:
- is_present now correctly handles empty strings, None, empty lists
- Summary fallback from raw_text if sections["summary"] is blank
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# =====================================================
# DATA MODELS
# =====================================================

@dataclass
class SectionFeedback:
    """Detailed feedback for a resume section"""
    section_name: str
    current_score: int
    target_score: int
    status: str  # "excellent", "good", "needs_improvement", "critical"
    impact_potential: int  # Points that can be gained

    # Analysis
    is_present: bool
    is_complete: bool
    quality_level: str

    # Issues
    missing_elements: List[Dict] = field(default_factory=list)
    incomplete_elements: List[Dict] = field(default_factory=list)
    quality_issues: List[Dict] = field(default_factory=list)
    excessive_elements: List[Dict] = field(default_factory=list)

    # Recommendations
    top_priority_fixes: List[Dict] = field(default_factory=list)
    quick_wins: List[Dict] = field(default_factory=list)
    detailed_suggestions: List[Dict] = field(default_factory=list)

    # Examples
    example_current: Optional[str] = None
    example_improved: Optional[str] = None

    strengths: List[str] = field(default_factory=list)


@dataclass
class ComprehensiveFeedback:
    """Complete feedback report"""
    overall_score: int
    overall_status: str
    ready_to_apply: bool
    estimated_improvement_potential: int

    section_feedback: Dict[str, SectionFeedback] = field(default_factory=dict)

    # Overall insights
    top_3_priorities: List[str] = field(default_factory=list)
    quick_wins_summary: List[str] = field(default_factory=list)
    strengths_summary: List[str] = field(default_factory=list)

    # Roadmap
    improvement_roadmap: List[Dict] = field(default_factory=list)


# =====================================================
# HELPERS
# =====================================================

def _is_section_present(value: Any) -> bool:
    """
    Robust presence check — correctly handles:
    - None
    - "" (empty string)
    - []  (empty list)
    - {}  (empty dict)
    - "   " (whitespace-only string)
    """
    if value is None:
        return False
    if isinstance(value, str):
        return len(value.strip()) > 0
    if isinstance(value, (list, dict)):
        return len(value) > 0
    return bool(value)


# =====================================================
# DETAILED FEEDBACK GENERATOR
# =====================================================

class DetailedFeedbackGenerator:
    """Generate comprehensive, section-by-section feedback"""

    # Score thresholds
    SCORE_THRESHOLDS = {
        "excellent": (85, 100),
        "good": (70, 84),
        "needs_improvement": (55, 69),
        "critical": (0, 54)
    }

    # Section-specific feedback templates
    SECTION_TEMPLATES = {
        "education": {
            "missing": [
                {"element": "degree_type", "why": "ATS needs to identify qualification level", "impact": 10},
                {"element": "institution_name", "why": "Employers filter by university prestige", "impact": 8},
                {"element": "graduation_year", "why": "Determines experience level and recency", "impact": 7},
                {"element": "gpa_if_strong", "why": "Strong GPA (3.7+) is a differentiator", "impact": 3},
                {"element": "relevant_coursework", "why": "Bridges gap for career changers", "impact": 5},
            ],
            "excessive": [
                {"element": "incomplete_coursework_list", "why": "Takes space, ATS can't parse well", "action": "Keep only top 5-7"},
                {"element": "high_school_education", "why": "Irrelevant after college", "action": "Remove if you have degree"},
                {"element": "honors_awards_per_class", "why": "Too granular, reduces readability", "action": "List only major awards"},
            ],
            "quality_issues": [
                {"issue": "Year listed without degree", "example": "2020", "fix": "B.S. Computer Science | MIT | 2020"},
                {"issue": "Institution name truncated or misspelled", "example": "Univ of California", "fix": "University of California, Berkeley"},
                {"issue": "No graduation date", "example": "B.S. in Progress", "fix": "B.S. Computer Science | Expected May 2025"},
                {"issue": "GPA listed but below 3.5", "example": "GPA: 3.2", "fix": "Remove GPA if below 3.5"},
            ]
        },

        "experience": {
            "missing": [
                {"element": "company_names", "why": "ATS filters by company size and industry", "impact": 8},
                {"element": "job_titles", "why": "ATS matches against job level", "impact": 10},
                {"element": "employment_dates", "why": "Shows experience duration and recency", "impact": 7},
                {"element": "achievement_metrics", "why": "Quantifies impact, crucial for ATS", "impact": 15},
                {"element": "action_verbs_start", "why": "Signals leadership and achievement", "impact": 10},
            ],
            "excessive": [
                {"element": "too_many_old_jobs", "why": "20+ year history is harder to parse", "action": "Keep last 10-15 years"},
                {"element": "outdated_companies", "why": "Distracts from recent achievements", "action": "Summarize as '5+ years at various companies'"},
                {"element": "excessive_bullets", "why": "ATS struggles with 10+ bullets per role", "action": "Keep 4-6 strongest bullets"},
            ],
            "quality_issues": [
                {"issue": "Weak action verbs", "example": "Worked on database optimization", "fix": "Optimized database queries reducing latency by 60%"},
                {"issue": "No numbers/metrics", "example": "Managed customer support team", "fix": "Led team of 12, improving satisfaction from 78% to 94%"},
                {"issue": "Responsibility-focused", "example": "Responsible for testing", "fix": "Designed test framework covering 95% of codebase"},
                {"issue": "Vague achievements", "example": "Contributed to project success", "fix": "Delivered microservices reducing deployment time from 4h to 30min"},
            ]
        },

        "skills": {
            "missing": [
                {"element": "programming_languages", "why": "Core filter for tech roles", "impact": 15},
                {"element": "frameworks_tools", "why": "Specific tech stack matching", "impact": 12},
                {"element": "databases", "why": "Backend skill requirement", "impact": 8},
                {"element": "cloud_platforms", "why": "Modern infrastructure skill", "impact": 10},
                {"element": "soft_skills", "why": "Leadership and communication valued", "impact": 5},
            ],
            "excessive": [
                {"element": "obscure_tools", "why": "ATS can't match, wastes space", "action": "Remove tools with <5% job posting mention"},
                {"element": "15+ year old technologies", "why": "Signals outdated knowledge", "action": "Remove unless explicitly required"},
                {"element": "50+ skill list", "why": "ATS truncates, looks unfocused", "action": "Prioritize top 12-15"},
            ],
            "quality_issues": [
                {"issue": "Inconsistent skill naming", "example": "Python 3 / python / Py3", "fix": "Use standard: Python (specify version if critical)"},
                {"issue": "Vague skills", "example": "Microsoft Office", "fix": "Excel (pivot tables, VBA) / Word / PowerPoint"},
                {"issue": "Missing proficiency levels", "example": "JavaScript", "fix": "JavaScript (Expert) / React (Advanced) / Node.js (Intermediate)"},
                {"issue": "Grouped skills", "example": "Backend: Node, Express, Python", "fix": "Node.js • Express • Python • FastAPI • PostgreSQL"},
            ]
        },

        "summary": {
            "missing": [
                {"element": "professional_summary", "why": "First thing ATS/recruiters see", "impact": 15},
                {"element": "years_experience", "why": "Signals seniority level", "impact": 5},
                {"element": "key_strengths", "why": "Highlights differentiators", "impact": 8},
                {"element": "career_focus", "why": "Shows intentionality and direction", "impact": 5},
            ],
            "excessive": [
                {"element": "personal_pronouns", "why": "Takes space, implied in resume", "action": "Remove 'I' or 'Me'"},
                {"element": "generic_statements", "why": "Applies to everyone", "action": "Remove 'Hard worker' type phrases"},
                {"element": "irrelevant_hobbies", "why": "Wastes valuable space", "action": "Only include if highly relevant"},
            ],
            "quality_issues": [
                {"issue": "Too short (under 20 words)", "example": "Software engineer", "fix": "Results-driven Senior Engineer with 8+ years building scalable solutions. Expertise in Python, AWS, and microservices."},
                {"issue": "Too long (over 150 words)", "example": "Long paragraph...", "fix": "Trim to 40-50 words, use bullets for detail"},
                {"issue": "Lacks quantification", "example": "Improved system performance", "fix": "Improved system performance by 60%, serving 2M+ requests daily"},
                {"issue": "Doesn't show value", "example": "Experienced in many languages", "fix": "Expert in Python/AWS; delivered 50+ projects with 99.9% uptime"},
            ]
        },
    }

    def generate_detailed_feedback(
        self,
        ats_score: int,
        section_scores: Dict,
        resume: Dict,
        ats_issues: List
    ) -> ComprehensiveFeedback:
        """Generate comprehensive, detailed feedback"""

        logger.info(f"Generating detailed feedback for score {ats_score}")

        # Analyze each section
        section_feedback = {}

        for section in ["summary", "skills", "experience", "education"]:
            feedback = self._analyze_section(
                section=section,
                resume=resume,
                current_score=section_scores.get(section, 0),
                ats_issues=ats_issues
            )
            section_feedback[section] = feedback

        # Calculate overall insights
        top_3 = self._identify_top_priorities(section_feedback)
        quick_wins = self._identify_quick_wins(section_feedback)
        strengths = self._identify_strengths(section_feedback)
        roadmap = self._build_improvement_roadmap(section_feedback)

        # Estimate improvement potential
        total_potential = sum(s.impact_potential for s in section_feedback.values())

        ready_to_apply = ats_score >= 75
        overall_status = self._get_status(ats_score)

        return ComprehensiveFeedback(
            overall_score=ats_score,
            overall_status=overall_status,
            ready_to_apply=ready_to_apply,
            estimated_improvement_potential=total_potential,
            section_feedback=section_feedback,
            top_3_priorities=top_3,
            quick_wins_summary=quick_wins,
            strengths_summary=strengths,
            improvement_roadmap=roadmap
        )

    # =========== SECTION ANALYSIS ===========

    def _analyze_section(
        self,
        section: str,
        resume: Dict,
        current_score: int,
        ats_issues: List
    ) -> SectionFeedback:
        """Detailed analysis of a single resume section"""

        template = self.SECTION_TEMPLATES.get(section, {})

        # =====================================================
        # FIX: Use robust presence check — handles None, "", [], {}
        # =====================================================
        section_data = resume.get(section)
        is_present = _is_section_present(section_data)

        is_complete = self._is_section_complete(section, section_data)

        # Get template info
        missing_template = template.get("missing", [])
        excessive_template = template.get("excessive", [])
        quality_template = template.get("quality_issues", [])

        # Analyze what's missing
        missing_elements = self._detect_missing_elements(section, section_data, missing_template)

        # Analyze excessive / removable elements
        excessive_elements = self._detect_excessive_elements(section, section_data, excessive_template)

        # Analyze quality issues
        quality_issues = self._detect_quality_issues(section, section_data, quality_template)

        # Calculate impact
        missing_impact = sum(e.get("impact", 0) for e in missing_elements)
        excessive_impact = sum(e.get("impact", 0) for e in excessive_elements)
        quality_impact = sum(e.get("impact", 3) for e in quality_issues)

        impact_potential = missing_impact + excessive_impact + quality_impact
        target_score = min(current_score + impact_potential, 100)
        status = self._get_status(current_score)

        # Generate recommendations
        top_priority = self._generate_priorities(section, missing_elements, quality_issues)
        quick_wins = self._generate_quick_wins(section, quality_issues, excessive_elements)
        detailed_suggestions = self._generate_suggestions(section, missing_elements, quality_issues, excessive_elements)

        # Get before/after examples
        example_current, example_improved = self._get_examples(section)

        # Identify strengths
        strengths = self._identify_section_strengths(section, resume)

        # Quality level
        if current_score >= 80:
            quality_level = "high"
        elif current_score >= 60:
            quality_level = "medium"
        else:
            quality_level = "low"

        return SectionFeedback(
            section_name=section,
            current_score=current_score,
            target_score=target_score,
            status=status,
            impact_potential=impact_potential,
            is_present=is_present,
            is_complete=is_complete,
            quality_level=quality_level,
            missing_elements=missing_elements,
            excessive_elements=excessive_elements,
            quality_issues=quality_issues,
            top_priority_fixes=top_priority,
            quick_wins=quick_wins,
            detailed_suggestions=detailed_suggestions,
            example_current=example_current,
            example_improved=example_improved,
            strengths=strengths
        )

    def _is_section_complete(self, section: str, data: Any) -> bool:
        """Check if section has all required elements"""

        if not _is_section_present(data):
            return False

        required_fields = {
            "education": ["degree", "institution", "year"],
            "experience": ["title", "company", "bullets"],
            "skills": [],   # Any non-empty list is complete
            "summary": []   # Any non-empty string is complete
        }

        if section not in required_fields or not required_fields[section]:
            return True

        # Check if data has required fields
        if isinstance(data, list) and len(data) > 0:
            first_item = data[0]
            if isinstance(first_item, dict):
                return any(f in first_item for f in required_fields[section])
            return len(str(data).split()) > 5

        if isinstance(data, str):
            return len(data.split()) > 5

        return False

    def _detect_missing_elements(
        self,
        section: str,
        data: Any,
        template: List[Dict]
    ) -> List[Dict]:
        """Detect missing elements from template"""

        missing = []

        if not _is_section_present(data):
            # Entire section missing — all template items are missing
            missing.extend(template)
        else:
            data_str = str(data).lower()
            for item in template:
                element = item.get("element", "").lower().replace("_", " ")
                # Heuristic: if the element keyword isn't mentioned, flag as missing
                if len(element) > 4 and element not in data_str:
                    missing.append(item)

        return missing

    def _detect_excessive_elements(
        self,
        section: str,
        data: Any,
        template: List[Dict]
    ) -> List[Dict]:
        """Detect elements that should be removed"""

        excessive = []

        if not _is_section_present(data):
            return excessive

        data_str = str(data).lower()

        for item in template:
            element = item.get("element", "").lower().replace("_", " ")
            if element and element in data_str:
                excessive.append(item)

        return excessive

    def _detect_quality_issues(
        self,
        section: str,
        data: Any,
        template: List[Dict]
    ) -> List[Dict]:
        """Detect quality issues via example-pattern matching"""

        quality_issues = []

        if not _is_section_present(data):
            return quality_issues

        data_str = str(data).lower()

        for issue in template:
            example = (issue.get("example") or "").lower()
            if example and len(example) > 3 and example in data_str:
                quality_issues.append(issue)

        return quality_issues

    def _generate_priorities(
        self,
        section: str,
        missing: List[Dict],
        quality: List[Dict]
    ) -> List[Dict]:
        """Generate priority fixes sorted by impact"""

        priorities = []

        for item in sorted(missing, key=lambda x: x.get("impact", 0), reverse=True)[:2]:
            priorities.append({
                "action": f"Add missing: {item.get('element', 'element')}",
                "why": item.get("why"),
                "estimated_gain": item.get("impact", 5),
                "effort": "medium"
            })

        for issue in quality[:2]:
            priorities.append({
                "action": f"Fix: {issue.get('issue')}",
                "current": issue.get("example"),
                "improved": issue.get("fix"),
                "estimated_gain": 5,
                "effort": "easy"
            })

        return priorities

    def _generate_quick_wins(
        self,
        section: str,
        quality: List[Dict],
        excessive: List[Dict]
    ) -> List[Dict]:
        """Generate easy, fast wins"""

        quick_wins = []

        for issue in quality[:3]:
            quick_wins.append({
                "action": issue.get("issue"),
                "how": f"Change from '{issue.get('example')}' to '{issue.get('fix')}'",
                "effort": "5 minutes",
                "estimated_gain": 3
            })

        for item in excessive[:2]:
            quick_wins.append({
                "action": f"Remove: {item.get('element')}",
                "why": item.get("why"),
                "effort": "2 minutes",
                "estimated_gain": 2
            })

        return quick_wins

    def _generate_suggestions(
        self,
        section: str,
        missing: List[Dict],
        quality: List[Dict],
        excessive: List[Dict]
    ) -> List[Dict]:
        """Generate full detailed suggestions"""

        suggestions = []

        for item in missing:
            suggestions.append({
                "type": "add",
                "element": item.get("element"),
                "reason": item.get("why"),
                "example": f"Include: {item.get('element')}",
                "impact": item.get("impact", 5)
            })

        for issue in quality:
            suggestions.append({
                "type": "improve",
                "issue": issue.get("issue"),
                "current_example": issue.get("example"),
                "improved_example": issue.get("fix"),
                "impact": 5
            })

        for item in excessive:
            suggestions.append({
                "type": "remove",
                "element": item.get("element"),
                "reason": item.get("why"),
                "action": item.get("action", "Remove"),
                "impact": 2
            })

        return suggestions

    def _get_examples(self, section: str) -> tuple:
        """Get before/after example text for each section"""

        examples = {
            "education": (
                "Computer Science\nMIT\n2020",
                "B.S. Computer Science | Massachusetts Institute of Technology | Graduated 2020\n"
                "Relevant Coursework: Algorithms, Machine Learning, Distributed Systems"
            ),
            "experience": (
                "Worked on backend systems and helped with optimization",
                "Engineered microservices architecture using Python and FastAPI, "
                "reducing deployment time by 80% and serving 2M+ daily requests"
            ),
            "skills": (
                "Programming, Database, Cloud",
                "Python • JavaScript • PostgreSQL • AWS EC2 • Docker • Kubernetes • React • Node.js"
            ),
            "summary": (
                "Software engineer with experience",
                "Results-driven Senior Software Engineer with 8+ years designing scalable Python systems. "
                "AWS & Docker expert. Shipped 50+ projects, 99.9% uptime."
            )
        }

        return examples.get(section, ("", ""))

    def _identify_section_strengths(self, section: str, resume: Dict) -> List[str]:
        """Identify what's already good about a section"""

        strengths = []
        section_data = resume.get(section)

        if not _is_section_present(section_data):
            return strengths

        data_str = str(section_data).lower()

        if section == "experience":
            if any(v in data_str for v in ["led", "managed", "spearheaded", "orchestrated"]):
                strengths.append("Strong leadership verbs")
            if any(char.isdigit() for char in str(section_data)):
                strengths.append("Includes quantifiable metrics")
            if len(str(section_data)) > 200:
                strengths.append("Detailed accomplishments")

        elif section == "skills":
            if any(tech in data_str for tech in ["python", "java", "react", "aws", "typescript", "node"]):
                strengths.append("Modern technical skills")
            if len(str(section_data).split()) > 10:
                strengths.append("Comprehensive skill set")

        elif section == "education":
            if "university" in data_str or "college" in data_str or "institute" in data_str:
                strengths.append("Clear institution information")
            if any(yr in data_str for yr in ["2020", "2021", "2022", "2023", "2024", "2025"]):
                strengths.append("Recent qualification")

        elif section == "summary":
            words = len(str(section_data).split())
            if words >= 25:
                strengths.append("Substantial summary")
            if any(char.isdigit() for char in str(section_data)):
                strengths.append("Includes specific achievements")

        return strengths

    # =========== OVERALL INSIGHTS ===========

    def _identify_top_priorities(self, sections: Dict) -> List[str]:
        """Top 3 priority improvements across all sections"""

        priorities = []

        for section_name, feedback in sorted(
            sections.items(),
            key=lambda x: x[1].impact_potential,
            reverse=True
        )[:3]:
            if feedback.top_priority_fixes:
                action = feedback.top_priority_fixes[0].get("action", "")
                gain = feedback.top_priority_fixes[0].get("estimated_gain", 0)
                priorities.append(f"[{section_name.title()}] {action} (+{gain} points)")

        return priorities

    def _identify_quick_wins(self, sections: Dict) -> List[str]:
        """Easy quick wins across sections"""

        quick_wins = []

        for section_name, feedback in sections.items():
            for win in feedback.quick_wins[:1]:
                quick_wins.append(
                    f"{win.get('action')} ({win.get('estimated_gain')} pts, {win.get('effort')})"
                )

        return quick_wins[:3]

    def _identify_strengths(self, sections: Dict) -> List[str]:
        """Overall strengths across all sections"""

        strengths = []
        for feedback in sections.values():
            strengths.extend(feedback.strengths)
        return strengths[:5]

    def _build_improvement_roadmap(self, sections: Dict) -> List[Dict]:
        """Step-by-step improvement plan sorted by impact"""

        roadmap = []
        step_num = 1

        for section_name, feedback in sorted(
            sections.items(),
            key=lambda x: x[1].impact_potential,
            reverse=True
        ):
            if feedback.top_priority_fixes:
                for fix in feedback.top_priority_fixes[:1]:
                    effort = fix.get("effort", "medium")
                    roadmap.append({
                        "step": step_num,
                        "section": section_name,
                        "action": fix.get("action"),
                        "effort": effort,
                        "estimated_gain": fix.get("estimated_gain", 5),
                        "time_estimate": "15 mins" if effort == "easy" else "30 mins"
                    })
                    step_num += 1

            if step_num > 6:
                break

        return roadmap


    def _get_status(self, score: int) -> str:
        """Map score to status label"""

        for status, (min_score, max_score) in self.SCORE_THRESHOLDS.items():
            if min_score <= score <= max_score:
                return status

        return "unknown"