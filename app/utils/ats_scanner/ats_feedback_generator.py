# /home/aryu_user/Arun/aiproject_staging/app/utils/ats_feedback_generator.py
"""
ATS Feedback Generator - Creates detailed, actionable recommendations
Generates human-readable suggestions based on ATS analysis
"""

from typing import Dict, List
from dataclasses import dataclass, asdict
from enum import Enum
from app.utils.ats_scanner.ats_rules_advanced import ORSScore, Issue, SeverityLevel, IssueCategory
from app.utils.ats_scanner.ats_keyword_engine import KeywordAnalysis


# =====================================================
# DATA CLASSES
# =====================================================

@dataclass
class FeedbackCategory:
    """Structured feedback for a resume section"""
    section: str
    status: str  # "excellent", "good", "needs_improvement", "critical"
    score: int
    issues: List[Dict]
    suggestions: List[str]
    strengths: List[str]


@dataclass
class DetailedFeedback:
    """Complete feedback report"""
    overall_score: int
    overall_status: str
    critical_issues_count: int
    section_feedback: List[FeedbackCategory]
    top_priorities: List[str]
    quick_wins: List[str]
    detailed_analysis: Dict
    estimated_improvement_potential: Dict  # Potential score gains


# =====================================================
# FEEDBACK GENERATOR
# =====================================================

class FeedbackGenerator:
    """Generate actionable feedback from ATS analysis"""
    
    # Score range classification
    SCORE_RANGES = {
        (85, 100): ("Excellent", "excellent"),
        (70, 84): ("Good", "good"),
        (55, 69): ("Needs Improvement", "needs_improvement"),
        (0, 54): ("Critical Issues", "critical")
    }
    
    # Generic suggestions by category
    GENERIC_SUGGESTIONS = {
        IssueCategory.FORMAT: [
            "Use ATS-safe fonts: Calibri, Arial, or Times New Roman",
            "Ensure file is saved as PDF or DOCX",
            "Avoid special formatting like tables, columns, or graphics"
        ],
        IssueCategory.STRUCTURE: [
            "Follow standard order: Summary → Skills → Experience → Education",
            "Use clear section headers",
            "Maintain consistent spacing and margins"
        ],
        IssueCategory.CONTENT: [
            "Use strong action verbs at the start of bullet points",
            "Add quantifiable results (%, $, numbers)",
            "Expand weak bullets to at least 15 words",
            "Include specific achievements, not just responsibilities"
        ],
        IssueCategory.KEYWORDS: [
            "Match job description keywords naturally",
            "Include specific tools and technologies mentioned in job posting",
            "Use industry-standard terminology"
        ],
        IssueCategory.QUALITY: [
            "Focus on impact and results, not just tasks",
            "Show career progression and growth",
            "Highlight relevant skills for the target role"
        ],
        IssueCategory.COMPLIANCE: [
            "Avoid special characters (®, ™, ©)",
            "Use consistent date formats (MM/YYYY)",
            "Ensure proper text encoding"
        ]
    }
    
    def generate_feedback(self, ors_score: ORSScore,
                         keyword_analysis: KeywordAnalysis,
                         resume: Dict) -> DetailedFeedback:
        """
        Generate comprehensive feedback report
        """
        # Determine overall status
        overall_status = self._get_status(ors_score.total_score)
        
        # Generate section-specific feedback
        section_feedback = self._generate_section_feedback(
            ors_score, keyword_analysis, resume
        )
        
        # Identify top priorities
        top_priorities = self._identify_priorities(ors_score.all_issues)
        
        # Quick wins - easy improvements
        quick_wins = self._identify_quick_wins(ors_score.all_issues)
        
        # Improvement potential
        improvement_potential = self._calculate_improvement_potential(
            ors_score.all_issues
        )
        
        return DetailedFeedback(
            overall_score=ors_score.total_score,
            overall_status=overall_status,
            critical_issues_count=ors_score.critical_issues_count,
            section_feedback=section_feedback,
            top_priorities=top_priorities,
            quick_wins=quick_wins,
            detailed_analysis=self._create_detailed_analysis(
                ors_score, keyword_analysis
            ),
            estimated_improvement_potential=improvement_potential
        )
    
    # =========== SECTION FEEDBACK ===========
    
    def _generate_section_feedback(self, ors_score: ORSScore,
                                   keyword_analysis: KeywordAnalysis,
                                   resume: Dict) -> List[FeedbackCategory]:
        """Generate feedback for each resume section"""
        
        sections = {}
        
        # Experience section
        if "experience" in resume and resume["experience"]:
            sections["experience"] = self._analyze_section(
                section="experience",
                issues=ors_score.all_issues,
                strengths=ors_score.__dict__.get("strengths", []),
                content=resume.get("experience", [])
            )
        
        # Skills section
        if "skills" in resume and resume["skills"]:
            sections["skills"] = self._analyze_section(
                section="skills",
                issues=ors_score.all_issues,
                strengths=[],
                content=resume.get("skills", []),
                keyword_analysis=keyword_analysis
            )
        
        # Education section
        if "education" in resume and resume["education"]:
            sections["education"] = self._analyze_section(
                section="education",
                issues=ors_score.all_issues,
                strengths=[],
                content=resume.get("education", [])
            )
        
        # Summary section
        if "summary" in resume and resume["summary"]:
            sections["summary"] = self._analyze_section(
                section="summary",
                issues=ors_score.all_issues,
                strengths=[],
                content=[resume.get("summary", "")]
            )
        
        return list(sections.values())
    
    def _analyze_section(self, section: str, issues: List[Issue],
                        strengths: List[str], content: any,
                        keyword_analysis: KeywordAnalysis = None) -> FeedbackCategory:
        """Analyze single section"""
        
        # Filter issues for this section
        section_issues = [i for i in issues if i.section == section]
        
        # Calculate score
        score = 100
        for issue in section_issues:
            score -= issue.impact_score
        score = max(score, 0)
        
        # Determine status
        status = self._get_status(score)
        
        # Build suggestions
        suggestions = []
        for issue in section_issues:
            suggestions.append(issue.suggestion)
        
        # Add generic suggestions if needed
        if not suggestions and score < 80:
            for issue in section_issues:
                category_suggestions = self.GENERIC_SUGGESTIONS.get(
                    issue.category, []
                )
                suggestions.extend(category_suggestions[:2])
        
        # Build issue list for response
        issue_list = [
            {
                "severity": issue.severity.value,
                "message": issue.message,
                "suggestion": issue.suggestion
            }
            for issue in section_issues
        ]
        
        return FeedbackCategory(
            section=section,
            status=status,
            score=score,
            issues=issue_list,
            suggestions=list(set(suggestions)),  # Deduplicate
            strengths=strengths
        )
    
    # =========== PRIORITY IDENTIFICATION ===========
    
    def _identify_priorities(self, issues: List[Issue]) -> List[str]:
        """Identify top 3-5 priority improvements"""
        
        # Sort by severity and impact
        sorted_issues = sorted(
            issues,
            key=lambda x: (
                0 if x.severity == SeverityLevel.CRITICAL else 1,
                -x.impact_score
            )
        )
        
        priorities = []
        for issue in sorted_issues[:5]:
            if issue.severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH]:
                priorities.append(issue.suggestion)
        
        return priorities
    
    def _identify_quick_wins(self, issues: List[Issue]) -> List[str]:
        """Identify easy improvements that yield good results"""
        
        quick_win_issues = [
            i for i in issues
            if i.severity == SeverityLevel.LOW
            and i.impact_score <= 5
        ]
        
        return [i.suggestion for i in quick_win_issues[:3]]
    
    # =========== IMPROVEMENT POTENTIAL ===========
    
    def _calculate_improvement_potential(self, issues: List[Issue]) -> Dict:
        """Calculate potential score improvements by fixing issues"""
        
        improvements = {
            "critical_gains": 0,
            "high_gains": 0,
            "medium_gains": 0,
            "low_gains": 0,
            "total_potential": 0,
            "breakdown": {}
        }
        
        for issue in issues:
            if issue.severity == SeverityLevel.CRITICAL:
                improvements["critical_gains"] += issue.impact_score
            elif issue.severity == SeverityLevel.HIGH:
                improvements["high_gains"] += issue.impact_score
            elif issue.severity == SeverityLevel.MEDIUM:
                improvements["medium_gains"] += issue.impact_score
            else:
                improvements["low_gains"] += issue.impact_score
            
            # Track by category
            category = issue.category.value
            if category not in improvements["breakdown"]:
                improvements["breakdown"][category] = 0
            improvements["breakdown"][category] += issue.impact_score
        
        improvements["total_potential"] = sum([
            improvements["critical_gains"],
            improvements["high_gains"],
            improvements["medium_gains"],
            improvements["low_gains"]
        ])
        
        return improvements
    
    # =========== DETAILED ANALYSIS ===========
    
    def _create_detailed_analysis(self, ors_score: ORSScore,
                                 keyword_analysis: KeywordAnalysis) -> Dict:
        """Create detailed analysis breakdown"""
        
        return {
            "score_breakdown": {
                "format_score": ors_score.format_score,
                "structure_score": ors_score.structure_score,
                "content_score": ors_score.content_score,
                "keyword_score": ors_score.keyword_score,
                "ats_compliance_score": ors_score.ats_compliance_score
            },
            "keyword_analysis": {
                "total_jd_keywords": keyword_analysis.total_jd_keywords,
                "matched_keywords": keyword_analysis.matched_keywords,
                "match_percentage": keyword_analysis.match_percentage,
                "keyword_density": round(keyword_analysis.keyword_density, 3),
                "missing_critical": keyword_analysis.missing_critical_skills,
                "found_strengths": keyword_analysis.found_strengths
            },
            "issue_summary": {
                "critical_issues": sum(1 for i in ors_score.all_issues
                                       if i.severity == SeverityLevel.CRITICAL),
                "high_issues": sum(1 for i in ors_score.all_issues
                                   if i.severity == SeverityLevel.HIGH),
                "medium_issues": sum(1 for i in ors_score.all_issues
                                     if i.severity == SeverityLevel.MEDIUM),
                "low_issues": sum(1 for i in ors_score.all_issues
                                  if i.severity == SeverityLevel.LOW)
            }
        }
    
    # =========== HELPER METHODS ===========
    
    def _get_status(self, score: int) -> str:
        """Get status label for score"""
        for (low, high), (label, _) in self.SCORE_RANGES.items():
            if low <= score <= high:
                return label
        return "Unknown"
    
    # =========== SCORE CONVERSION ===========
    
    def to_dict(self, feedback: DetailedFeedback) -> Dict:
        """Convert feedback to dictionary for JSON serialization"""
        return {
            "overall_score": feedback.overall_score,
            "overall_status": feedback.overall_status,
            "critical_issues_count": feedback.critical_issues_count,
            "section_feedback": [asdict(sf) for sf in feedback.section_feedback],
            "top_priorities": feedback.top_priorities,
            "quick_wins": feedback.quick_wins,
            "detailed_analysis": feedback.detailed_analysis,
            "estimated_improvement_potential": feedback.estimated_improvement_potential
        }


# =====================================================
# SCORE INTERPRETATION
# =====================================================

class ScoreInterpreter:
    """Interpret ATS scores and provide guidance"""
    
    SCORE_INTERPRETATION = {
        (85, 100): {
            "message": "Excellent - Ready to Apply",
            "description": "Your resume is highly optimized for ATS systems",
            "action": "Ready to submit! Consider tailoring for specific roles"
        },
        (70, 84): {
            "message": "Good - Minor Improvements Needed",
            "description": "Your resume passes ATS but has improvement opportunities",
            "action": "Focus on quick wins identified above"
        },
        (55, 69): {
            "message": "Needs Improvement - Major Changes Required",
            "description": "Your resume may have ATS parsing issues",
            "action": "Address high-priority items before applying"
        },
        (0, 54): {
            "message": "Critical Issues - Significant Redesign Needed",
            "description": "Your resume likely has serious ATS compatibility problems",
            "action": "Complete overhaul recommended - focus on critical issues"
        }
    }
    
    @classmethod
    def interpret(cls, score: int) -> Dict:
        """Get interpretation for a score"""
        for (low, high), interpretation in cls.SCORE_INTERPRETATION.items():
            if low <= score <= high:
                return interpretation
        return {"message": "Unknown", "description": "", "action": ""}