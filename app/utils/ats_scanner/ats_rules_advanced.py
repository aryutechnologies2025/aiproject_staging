# /home/aryu_user/Arun/aiproject_staging/app/utils/ats_rules_advanced.py
"""
Production-grade ATS Rules Engine
Implements industry-standard resume validation and scoring
Author: Backend Architecture Team
"""

import re
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum

# =====================================================
# ENUMS & CONSTANTS
# =====================================================

class SeverityLevel(Enum):
    CRITICAL = "critical"      # Major ATS issues, blocks parsing
    HIGH = "high"              # Significant issues affecting scoring
    MEDIUM = "medium"          # Important but not blocking
    LOW = "low"                # Minor improvements

class IssueCategory(Enum):
    FORMAT = "format"
    STRUCTURE = "structure"
    CONTENT = "content"
    KEYWORDS = "keywords"
    QUALITY = "quality"
    COMPLIANCE = "compliance"


# ATS-Safe formatting standards
ATS_SAFE_FONTS = {
    "arial", "calibri", "times new roman", "helvetica", "georgia",
    "verdana", "courier", "courier new"
}

ATS_UNSAFE_FONTS = {
    "wingdings", "symbol", "dingbats", "impact", "comic sans"
}

# Action verbs for strong bullet points
STRONG_ACTION_VERBS = {
    # Leadership
    "led", "managed", "directed", "orchestrated", "spearheaded",
    "supervised", "coordinated", "oversaw",
    
    # Achievement
    "achieved", "delivered", "accomplished", "completed", "exceeded",
    "surpassed", "attained", "earned",
    
    # Creation
    "developed", "built", "created", "designed", "architected",
    "engineered", "constructed", "launched", "initiated",
    
    # Improvement
    "optimized", "improved", "enhanced", "streamlined", "refined",
    "accelerated", "elevated", "boosted", "increased", "reduced",
    "lowered", "minimized", "maximized",
    
    # Analysis
    "analyzed", "evaluated", "assessed", "examined", "identified",
    "discovered", "uncovered", "determined",
    
    # Implementation
    "implemented", "executed", "deployed", "installed", "integrated",
    "established", "instituted", "pioneered",
    
    # Transformation
    "transformed", "revolutionized", "modernized", "reformed",
    "repositioned", "restructured"
}

WEAK_ACTION_VERBS = {
    "responsible for", "involved in", "helped with", "assisted",
    "worked on", "was part of", "participated in", "contributed to"
}

# Metrics patterns
METRICS_PATTERNS = [
    r"\$[\d,]+\.?\d*[KMB]?",           # Currency: $50K, $1.2M
    r"\d+%",                            # Percentages: 25%
    r"\d+x",                            # Multiples: 3x
    r"\d+\+",                           # Plus numbers: 100+
    r"(increased|reduced|grew|grew|improved|decreased)\s+by\s+\d+%",
    r"\d+\s*(users|clients|customers|transactions|records|downloads)",
    r"(top\s+)?\d+\s*(in|of)",          # Rankings: top 5
    r"\d+\s*(hours?|days?|weeks?|months?|years?)\s+(saved|reduced)"
]

# Minimum requirements
MIN_BULLETS_PER_ROLE = 2
MAX_BULLETS_PER_ROLE = 8
MIN_BULLET_LENGTH = 12  # words
IDEAL_BULLET_LENGTH = 20  # words
MAX_BULLET_LENGTH = 120  # words

MIN_SKILLS = 5
IDEAL_SKILLS = 10
MAX_SKILLS = 50

MIN_RESUME_LENGTH = 200  # words
IDEAL_RESUME_LENGTH = 500
MAX_RESUME_LENGTH = 2000


# =====================================================
# DATA CLASSES
# =====================================================

@dataclass
class Issue:
    """Represents a single ATS issue"""
    severity: SeverityLevel
    category: IssueCategory
    section: str
    message: str
    suggestion: str
    impact_score: int  # 1-10, how much it affects ATS score


@dataclass
class SectionAnalysis:
    """Analysis result for a resume section"""
    name: str
    present: bool
    quality_score: int  # 0-100
    issues: List[Issue]
    strengths: List[str]
    metrics: Dict


@dataclass
class ORSScore:
    """Overall Resume Score with breakdown"""
    total_score: int
    format_score: int
    structure_score: int
    content_score: int
    keyword_score: int
    ats_compliance_score: int
    critical_issues_count: int
    all_issues: List[Issue]


# =====================================================
# MAIN ATS RULES ENGINE
# =====================================================

class ATSRulesEngine:
    """Production-grade ATS compliance validator"""
    
    def __init__(self):
        self.issues: List[Issue] = []
        self.strengths: List[str] = []
    
    def analyze(self, resume: Dict) -> ORSScore:
        """
        Comprehensive ATS analysis of resume
        """
        self.issues = []
        self.strengths = []
        
        # Individual scoring components
        format_score = self._analyze_format(resume)
        structure_score = self._analyze_structure(resume)
        content_score = self._analyze_content(resume)
        keyword_score = self._analyze_keywords(resume)
        compliance_score = self._analyze_ats_compliance(resume)
        
        # Calculate weighted total
        total_score = self._calculate_weighted_score(
            format_score=format_score,
            structure_score=structure_score,
            content_score=content_score,
            keyword_score=keyword_score,
            compliance_score=compliance_score
        )
        
        critical_count = sum(1 for i in self.issues if i.severity == SeverityLevel.CRITICAL)
        
        return ORSScore(
            total_score=max(total_score, 0),
            format_score=format_score,
            structure_score=structure_score,
            content_score=content_score,
            keyword_score=keyword_score,
            ats_compliance_score=compliance_score,
            critical_issues_count=critical_count,
            all_issues=self.issues
        )
    
    # =========== FORMAT VALIDATION ===========
    
    def _analyze_format(self, resume: Dict) -> int:
        """Check file format, fonts, spacing, structure"""
        score = 100
        
        # File type check
        file_type = resume.get("file_type", "").lower()
        if file_type and file_type not in {"pdf", "docx"}:
            self._add_issue(
                severity=SeverityLevel.CRITICAL,
                category=IssueCategory.FORMAT,
                section="summary",
                message=f"Unsupported file type: {file_type}",
                suggestion="Convert to PDF or DOCX format only",
                impact=15
            )
            score -= 15
        
        # Font check
        font = resume.get("font", "").lower()
        if font:
            if font in ATS_UNSAFE_FONTS:
                self._add_issue(
                    severity=SeverityLevel.CRITICAL,
                    category=IssueCategory.FORMAT,
                    section="summary",
                    message=f"ATS-unsafe font detected: {font}",
                    suggestion=f"Use standard fonts: {', '.join(list(ATS_SAFE_FONTS)[:3])}",
                    impact=15
                )
                score -= 15
            elif font not in ATS_SAFE_FONTS:
                self._add_issue(
                    severity=SeverityLevel.HIGH,
                    category=IssueCategory.FORMAT,
                    section="summary",
                    message=f"Non-standard font: {font}",
                    suggestion="Use Calibri, Arial, or Times New Roman",
                    impact=8
                )
                score -= 8
            else:
                self.strengths.append(f"ATS-safe font: {font}")
        
        # Layout issues
        if resume.get("uses_table"):
            self._add_issue(
                severity=SeverityLevel.HIGH,
                category=IssueCategory.FORMAT,
                section="experience",
                message="Tables detected in resume",
                suggestion="Remove tables; use bullet points instead",
                impact=12
            )
            score -= 12
        
        if resume.get("uses_columns"):
            self._add_issue(
                severity=SeverityLevel.HIGH,
                category=IssueCategory.FORMAT,
                section="summary",
                message="Multi-column layout detected",
                suggestion="Use single-column layout for ATS compatibility",
                impact=12
            )
            score -= 12
        
        if resume.get("uses_graphics"):
            self._add_issue(
                severity=SeverityLevel.MEDIUM,
                category=IssueCategory.FORMAT,
                section="summary",
                message="Graphics, images, or icons detected",
                suggestion="ATS systems can't parse visual elements; remove them",
                impact=5
            )
            score -= 5
        
        return max(score, 0)
    
    # =========== STRUCTURE VALIDATION ===========
    
    def _analyze_structure(self, resume: Dict) -> int:
        """Check section presence and organization"""
        score = 100
        
        required_sections = {
            "summary": (20, "Professional summary or objective"),
            "skills": (20, "Skills section"),
            "experience": (30, "Work experience"),
            "education": (20, "Education")
        }
        
        for section, (points, description) in required_sections.items():
            if section in resume and resume[section]:
                self.strengths.append(f"Strong {section} section")
            else:
                self._add_issue(
                    severity=SeverityLevel.HIGH,
                    category=IssueCategory.STRUCTURE,
                    section=section,
                    message=f"{section.title()} section missing or empty",
                    suggestion=f"Add {description}",
                    impact=points // 2
                )
                score -= points // 2
        
        # Check section ordering
        section_order = ["summary", "skills", "experience", "education"]
        provided_sections = [s for s in section_order if resume.get(s)]
        if provided_sections != sorted(provided_sections, key=lambda x: section_order.index(x)):
            self._add_issue(
                severity=SeverityLevel.LOW,
                category=IssueCategory.STRUCTURE,
                section="summary",
                message="Unconventional section ordering",
                suggestion="Use standard order: Summary → Skills → Experience → Education",
                impact=3
            )
            score -= 3
        
        return max(score, 0)
    
    # =========== CONTENT QUALITY ===========
    
    def _analyze_content(self, resume: Dict) -> int:
        """Check bullet quality, metrics, action verbs"""
        score = 100
        
        # Summary analysis
        summary = resume.get("summary", "")
        if summary:
            summary_words = len(summary.split())
            if summary_words < 20:
                self._add_issue(
                    severity=SeverityLevel.MEDIUM,
                    category=IssueCategory.CONTENT,
                    section="summary",
                    message="Summary too brief",
                    suggestion="Expand professional summary to 30-50 words",
                    impact=5
                )
                score -= 5
            elif summary_words > 150:
                self._add_issue(
                    severity=SeverityLevel.LOW,
                    category=IssueCategory.CONTENT,
                    section="summary",
                    message="Summary too long",
                    suggestion="Keep professional summary under 100 words",
                    impact=3
                )
                score -= 3
        
        # Experience bullet analysis
        experience = resume.get("experience", [])
        if experience:
            bullet_quality = self._analyze_bullets(experience)
            
            if bullet_quality["weak_percentage"] > 50:
                self._add_issue(
                    severity=SeverityLevel.HIGH,
                    category=IssueCategory.CONTENT,
                    section="experience",
                    message=f"{bullet_quality['weak_percentage']}% of bullets are weak",
                    suggestion="Expand bullets with action verbs, metrics, and impact",
                    impact=15
                )
                score -= 15
            elif bullet_quality["weak_percentage"] > 30:
                self._add_issue(
                    severity=SeverityLevel.MEDIUM,
                    category=IssueCategory.CONTENT,
                    section="experience",
                    message=f"{bullet_quality['weak_percentage']}% of bullets lack metrics",
                    suggestion="Add quantifiable results to more bullets",
                    impact=8
                )
                score -= 8
            
            if bullet_quality["strong_count"] > 0:
                self.strengths.append(f"Strong action verbs in {bullet_quality['strong_count']} bullets")
        
        return max(score, 0)
    
    def _analyze_bullets(self, experience: List[Dict]) -> Dict:
        """Analyze all experience bullets"""
        results = {
            "total": 0,
            "strong_count": 0,
            "weak_count": 0,
            "with_metrics": 0,
            "with_action_verb": 0,
            "average_length": 0,
            "weak_percentage": 0
        }
        
        all_bullets = []
        for exp in experience:
            bullets = exp.get("bullets", [])
            all_bullets.extend(bullets)
        
        if not all_bullets:
            return results
        
        results["total"] = len(all_bullets)
        
        for bullet in all_bullets:
            if not bullet:
                continue
            
            words = bullet.split()
            results["average_length"] += len(words)
            
            # Check for action verbs
            bullet_lower = bullet.lower()
            has_action_verb = any(verb in bullet_lower for verb in STRONG_ACTION_VERBS)
            has_weak_verb = any(verb in bullet_lower for verb in WEAK_ACTION_VERBS)
            
            if has_action_verb and not has_weak_verb:
                results["with_action_verb"] += 1
            
            # Check for metrics
            has_metrics = any(re.search(pattern, bullet) for pattern in METRICS_PATTERNS)
            if has_metrics:
                results["with_metrics"] += 1
            
            # Classify strength
            if len(words) >= MIN_BULLET_LENGTH and (has_action_verb or has_metrics):
                results["strong_count"] += 1
            else:
                results["weak_count"] += 1
        
        if results["total"] > 0:
            results["average_length"] = results["average_length"] // results["total"]
            results["weak_percentage"] = int((results["weak_count"] / results["total"]) * 100)
        
        return results
    
    # =========== KEYWORD ANALYSIS ===========
    
    def _analyze_keywords(self, resume: Dict) -> int:
        """Check for job-relevant keywords"""
        score = 100
        
        job_description = resume.get("job_description", "")
        if not job_description:
            return score  # No JD provided, assume okay
        
        skills = resume.get("skills", [])
        if not skills:
            self._add_issue(
                severity=SeverityLevel.HIGH,
                category=IssueCategory.KEYWORDS,
                section="skills",
                message="No skills listed in resume",
                suggestion="Add 8-10 relevant skills from job description",
                impact=20
            )
            score -= 20
        
        return max(score, 0)
    
    # =========== ATS COMPLIANCE ===========
    
    def _analyze_ats_compliance(self, resume: Dict) -> int:
        """Check ATS parsing safety"""
        score = 100
        
        # Special characters that might break ATS
        resume_text = " ".join([
            resume.get("summary", ""),
            " ".join(resume.get("skills", [])),
            " ".join([b for exp in resume.get("experience", []) 
                     for b in exp.get("bullets", [])])
        ])
        
        # Check for problematic characters
        problematic_chars = {
            "®": "Registered trademark symbol",
            "™": "Trademark symbol",
            "©": "Copyright symbol",
            "°": "Degree symbol"
        }
        
        for char, desc in problematic_chars.items():
            if char in resume_text:
                self._add_issue(
                    severity=SeverityLevel.LOW,
                    category=IssueCategory.COMPLIANCE,
                    section="summary",
                    message=f"Special character detected: {desc}",
                    suggestion="Use text equivalent or remove",
                    impact=2
                )
                score -= 2
        
        # Check for dates format consistency
        dates = re.findall(r"\d{4}", resume_text)
        if dates:
            self.strengths.append("Years/dates properly formatted")
        
        return max(score, 0)
    
    # =========== HELPER METHODS ===========
    
    def _add_issue(self, severity: SeverityLevel, category: IssueCategory,
                   section: str, message: str, suggestion: str, impact: int):
        """Add an issue to the list"""
        issue = Issue(
            severity=severity,
            category=category,
            section=section,
            message=message,
            suggestion=suggestion,
            impact_score=impact
        )
        self.issues.append(issue)
    
    def _calculate_weighted_score(self, format_score: int, structure_score: int,
                                 content_score: int, keyword_score: int,
                                 compliance_score: int) -> int:
        """Calculate weighted ATS score"""
        weights = {
            "format": 0.10,          # 10% - Format compliance
            "structure": 0.15,       # 15% - Section structure
            "content": 0.30,         # 30% - Content quality
            "keywords": 0.30,        # 30% - Keyword relevance
            "compliance": 0.15       # 15% - ATS compatibility
        }
        
        total = (
            format_score * weights["format"] +
            structure_score * weights["structure"] +
            content_score * weights["content"] +
            keyword_score * weights["keywords"] +
            compliance_score * weights["compliance"]
        )
        
        return int(total)