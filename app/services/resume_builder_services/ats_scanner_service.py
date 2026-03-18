# /home/aryu_user/Arun/aiproject_staging/app/services/ats_scanner_service.py
"""
Production-Grade ATS Scanner Service v2
UPDATED: Works with enhanced v2 components
- Integrates DetailedFeedbackGenerator (v2)
- Uses enhanced ATSRulesEngine (v2 with fixed education + summary detection)
- Enhanced keyword matching
- Detailed section-by-section analysis
- None-safe throughout (FIX APPLIED)
"""

import logging
import json
from typing import Dict, Optional, Callable
from sqlalchemy.ext.asyncio import AsyncSession

from app.utils.ats_scanner.ats_rules_advanced import ATSRulesEngine
from app.utils.ats_scanner.ats_keyword_engine import KeywordEngine
from app.utils.ats_scanner.ats_feedback_generator import DetailedFeedbackGenerator

logger = logging.getLogger(__name__)


# =====================================================
# EXPERT PROMPTS
# =====================================================

EXPERT_ATS_ANALYSIS_PROMPT = """You are an expert ATS consultant with 20+ years of experience in recruitment and applicant tracking systems.

RESUME DATA:
Skills: {skills}
Experience: {experience_bullets}
Education: {education}
Summary: {summary}
Target Job Description: {job_description}

CRITICAL ANALYSIS REQUIRED:
1. ATS Compatibility: Will this resume parse correctly through ATS systems?
2. Content Quality: Are bullets strong, specific, and results-oriented?
3. Keyword Alignment: How well does it match the job requirements?
4. Missing Elements: What critical information is missing?
5. Strengths: What stands out positively?

PROVIDE ANALYSIS AS JSON ONLY:
{{
  "ats_compatibility_assessment": "Brief assessment of how well ATS will parse this resume",
  "content_strengths": ["Strength 1", "Strength 2", "Strength 3"],
  "critical_weaknesses": ["Weakness 1", "Weakness 2"],
  "missing_keywords": ["Critical keyword 1", "Critical keyword 2"],
  "missing_information": ["Missing element 1", "Missing element 2"],
  "specific_improvements": [
    {{"bullet": "Old bullet text", "improvement": "Better version with metrics"}},
    {{"section": "skills", "issue": "Problem", "fix": "Solution"}}
  ],
  "priority_fixes": [
    "Fix 1 (estimated +X points)",
    "Fix 2 (estimated +Y points)"
  ],
  "overall_assessment": "2-3 sentence assessment of resume readiness"
}}

OUTPUT ONLY VALID JSON, NO MARKDOWN, NO EXPLANATION."""


# =====================================================
# MAIN ATS SCANNER SERVICE v2
# =====================================================

class ATSScannerService:
    """
    Production-grade resume ATS scanner v2

    Combines:
    - Enhanced rule-based ATS compliance checking
    - Advanced keyword extraction and semantic matching
    - AI-powered content analysis (optional)
    - Detailed section-by-section feedback generation
    - Specific improvement suggestions
    - Improvement roadmap
    """

    def __init__(self):
        self.rules_engine = ATSRulesEngine()
        self.keyword_engine = KeywordEngine()
        self.feedback_generator = DetailedFeedbackGenerator()

    async def scan(
        self,
        resume: Dict,
        job_description: Optional[str] = None,
        llm_client: Optional[Callable] = None,
        db: Optional[AsyncSession] = None
    ) -> Dict:
        """
        Comprehensive ATS scan with rule-based and optional AI analysis

        Args:
            resume: Resume data dictionary
            job_description: Target job posting (optional)
            llm_client: LLM function for AI analysis (optional)
            db: Database session (optional)

        Returns:
            Complete ATS analysis with scores, feedback, and recommendations
        """

        try:
            logger.info("Starting comprehensive ATS scan v2")

            # ============================================================
            # STEP 1: RULE-BASED ANALYSIS
            # ============================================================
            logger.info("Step 1: Running rule-based ATS analysis")
            ors_score = self.rules_engine.analyze(resume)

            logger.info(f"  Format score: {ors_score.format_score}")
            logger.info(f"  Structure score: {ors_score.structure_score}")
            logger.info(f"  Content score: {ors_score.content_score}")
            logger.info(f"  Rule-based total: {ors_score.total_score}")
            logger.info(f"  Critical issues: {ors_score.critical_issues_count}")

            # ============================================================
            # STEP 2: KEYWORD ANALYSIS
            # ============================================================
            keyword_analysis = None
            keyword_score = 0

            if job_description:
                logger.info("Step 2: Running keyword analysis")
                keyword_analysis = self.keyword_engine.match_skills(resume, job_description)
                keyword_score = self.keyword_engine.calculate_keyword_score(keyword_analysis)
                logger.info(f"  Keyword match: {keyword_analysis.match_percentage}%")
                logger.info(f"  Keyword score: {keyword_score}")
            else:
                logger.info("Step 2: Skipping keyword analysis (no job description)")

            # ============================================================
            # STEP 3: AI ANALYSIS (Optional)
            # ============================================================
            ai_insights: Dict = {}

            if llm_client and db and job_description:
                logger.info("Step 3: Running AI-powered analysis")
                try:
                    ai_insights = await self._get_ai_insights(
                        resume, job_description, llm_client, db
                    )
                    if ai_insights.get("success"):
                        logger.info("  AI analysis completed successfully")
                    else:
                        logger.warning("  AI analysis failed, using rule-based only")
                except Exception as e:
                    logger.warning(f"  AI analysis error (continuing): {e}")
                    ai_insights = {"success": False}
            else:
                logger.info("Step 3: Skipping AI analysis")

            # ============================================================
            # STEP 4: GENERATE DETAILED FEEDBACK
            # ============================================================
            logger.info("Step 4: Generating detailed feedback")

            section_scores = {
                "education": ors_score.content_score,
                "experience": ors_score.content_score,
                "skills": ors_score.content_score,
                "summary": ors_score.content_score,
            }

            detailed_feedback = self.feedback_generator.generate_detailed_feedback(
                ats_score=ors_score.total_score,
                section_scores=section_scores,
                resume=resume,
                ats_issues=ors_score.all_issues
            )

            logger.info(f"  Overall status: {detailed_feedback.overall_status}")
            logger.info(f"  Ready to apply: {detailed_feedback.ready_to_apply}")
            logger.info(
                f"  Improvement potential: {detailed_feedback.estimated_improvement_potential} points"
            )

            # ============================================================
            # STEP 5: CALCULATE FINAL SCORE
            # ============================================================
            final_ats_score = self._calculate_final_score(
                ors_score.total_score,
                keyword_score
            )
            logger.info(f"Step 5: Final score: {final_ats_score}")

            # ============================================================
            # STEP 6: BUILD RESPONSE
            # ============================================================
            logger.info("Step 6: Building response")

            result = self._build_response(
                ors_score=ors_score,
                keyword_analysis=keyword_analysis,
                keyword_score=keyword_score,
                final_score=final_ats_score,
                detailed_feedback=detailed_feedback,
                ai_insights=ai_insights,
                resume=resume
            )

            logger.info(f"ATS scan completed. Final score: {final_ats_score}")
            return result

        except Exception as e:
            logger.error(f"ATS scan failed: {str(e)}", exc_info=True)
            raise

    # =========== AI ANALYSIS ===========

    async def _get_ai_insights(
        self,
        resume: Dict,
        job_description: str,
        llm_client: Callable,
        db: AsyncSession
    ) -> Dict:
        """Get AI-powered insights about resume"""

        try:
            skills_str = ", ".join((resume.get("skills") or [])[:15])

            experience_bullets = []
            for exp in (resume.get("experience") or [])[:3]:
                if isinstance(exp, dict):
                    experience_bullets.extend((exp.get("bullets") or [])[:2])
            experience_str = "\n".join(experience_bullets[:6])

            education_list = resume.get("education") or []
            if isinstance(education_list, list) and education_list:
                education_str = "; ".join([
                    f"{e.get('degree', '')} from {e.get('institution', '')}"
                    for e in education_list
                    if isinstance(e, dict) and (e.get("degree") or e.get("institution"))
                ])
            else:
                education_str = str(education_list)

            summary_str = (resume.get("summary") or "")[:200]

            prompt = EXPERT_ATS_ANALYSIS_PROMPT.format(
                skills=skills_str,
                experience_bullets=experience_str,
                education=education_str,
                summary=summary_str,
                job_description=job_description[:1000]
            )

            logger.debug("Calling LLM for AI analysis")
            response = await llm_client(prompt)

            # Strip markdown fences if present
            clean = response.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            clean = clean.strip()

            ai_result = json.loads(clean)

            return {
                "success": True,
                "ats_compatibility": ai_result.get("ats_compatibility_assessment"),
                "content_strengths": ai_result.get("content_strengths", []),
                "critical_weaknesses": ai_result.get("critical_weaknesses", []),
                "missing_keywords": ai_result.get("missing_keywords", []),
                "missing_information": ai_result.get("missing_information", []),
                "specific_improvements": ai_result.get("specific_improvements", []),
                "priority_fixes": ai_result.get("priority_fixes", []),
                "overall_assessment": ai_result.get("overall_assessment")
            }

        except json.JSONDecodeError as e:
            logger.warning(f"AI response parsing failed: {e}")
            return {"success": False, "error": "Invalid response format"}
        except Exception as e:
            logger.warning(f"AI analysis error: {e}")
            return {"success": False, "error": str(e)}

    # =========== RESPONSE BUILDING ===========

    def _build_response(
        self,
        ors_score,
        keyword_analysis,
        keyword_score: int,
        final_score: int,
        detailed_feedback,
        ai_insights: Dict,
        resume: Dict
    ) -> Dict:
        """Build comprehensive response"""

        section_analysis = {}
        if detailed_feedback and detailed_feedback.section_feedback:
            for section_name, feedback in detailed_feedback.section_feedback.items():
                section_analysis[section_name] = {
                    "score": feedback.current_score,
                    "target_score": feedback.target_score,
                    "status": feedback.status,
                    "is_present": feedback.is_present,
                    "is_complete": feedback.is_complete,
                    "quality_level": feedback.quality_level,
                    "impact_potential": feedback.impact_potential,
                    "missing_elements": feedback.missing_elements,
                    "incomplete_elements": feedback.incomplete_elements,
                    "quality_issues": feedback.quality_issues,
                    "excessive_elements": feedback.excessive_elements,
                    "top_priority_fixes": feedback.top_priority_fixes,
                    "quick_wins": feedback.quick_wins,
                    "detailed_suggestions": feedback.detailed_suggestions,
                    "example_current": feedback.example_current,
                    "example_improved": feedback.example_improved,
                    "strengths": feedback.strengths
                }

        return {
            "ats_score": final_score,
            "score_status": detailed_feedback.overall_status if detailed_feedback else "unknown",
            "ready_to_apply": final_score >= 75,

            "score_breakdown": {
                "format_compliance": ors_score.format_score,
                "structure_quality": ors_score.structure_score,
                "content_quality": ors_score.content_score,
                "ats_compatibility": ors_score.ats_compliance_score,
                "keyword_alignment": keyword_score if keyword_analysis else None,
            },

            "issues": self._format_issues(ors_score.all_issues),
            "critical_issues_count": ors_score.critical_issues_count,

            "section_analysis": section_analysis,

            "keyword_analysis": (
                {
                    "total_required_keywords": keyword_analysis.total_jd_keywords,
                    "matched_keywords": keyword_analysis.matched_keywords,
                    "match_percentage": keyword_analysis.match_percentage,
                    "critical_gaps": keyword_analysis.missing_critical_skills,
                    "strength_keywords": keyword_analysis.found_strengths
                }
                if keyword_analysis else None
            ),

            "recommendations": (
                {
                    "top_3_priorities": detailed_feedback.top_3_priorities,
                    "quick_wins": detailed_feedback.quick_wins_summary,
                    "improvement_roadmap": detailed_feedback.improvement_roadmap,
                    "estimated_improvement_potential": detailed_feedback.estimated_improvement_potential
                }
                if detailed_feedback else {}
            ),

            "ai_analysis": ai_insights if ai_insights.get("success") else None,

            "summary": {
                "ready_to_apply": final_score >= 75,
                "main_strengths": (
                    detailed_feedback.strengths_summary if detailed_feedback else []
                ),
                "main_weaknesses": (
                    detailed_feedback.top_3_priorities if detailed_feedback else []
                ),
                "key_findings": self._generate_key_findings(
                    final_score, ors_score.critical_issues_count
                ),
                "next_steps": self._generate_next_steps(
                    final_score, ors_score.critical_issues_count
                ),
                "estimated_ats_compatibility": (
                    "Excellent" if final_score >= 85 else
                    "Good" if final_score >= 70 else
                    "Moderate" if final_score >= 55 else
                    "Poor"
                )
            }
        }

    def _format_issues(self, issues) -> Dict:
        """Format issues by severity"""
        formatted = {
            "critical": [],
            "high": [],
            "medium": [],
            "low": []
        }

        for issue in issues:
            severity = (
                issue.severity.value
                if hasattr(issue.severity, "value")
                else str(issue.severity)
            )

            issue_dict = {
                "section": issue.section,
                "message": issue.message,
                "suggestion": issue.suggestion,
                "impact": issue.impact_score,
            }

            if hasattr(issue, "specific_example") and issue.specific_example:
                issue_dict["example"] = issue.specific_example

            if hasattr(issue, "improvement_example") and issue.improvement_example:
                issue_dict["improvement"] = issue.improvement_example

            if severity in formatted:
                formatted[severity].append(issue_dict)

        return formatted

    def _calculate_final_score(self, rule_score: int, keyword_score: int) -> int:
        """
        Weighted final score:
        - No JD: rule score is final
        - With JD: 40% rules + 60% keywords
        """
        if keyword_score == 0:
            return rule_score

        final = (rule_score * 0.40) + (keyword_score * 0.60)
        return min(int(final), 100)

    def _generate_key_findings(self, final_score: int, critical_count: int) -> list:
        findings = []

        if critical_count > 0:
            findings.append(f"⚠️ {critical_count} critical ATS issues detected")

        if final_score >= 85:
            findings.append("✅ Resume is well-optimized for ATS systems")
        elif final_score >= 70:
            findings.append("✅ Resume passes ATS with minor room for improvement")
        elif final_score >= 55:
            findings.append("⚠️ Resume has significant ATS issues to address")
        else:
            findings.append("❌ Resume needs major ATS improvements")

        return findings

    def _generate_next_steps(self, final_score: int, critical_count: int) -> list:
        steps = []

        if critical_count > 0:
            steps.append(f"1. Fix {critical_count} critical issue(s) (see Issues section)")

        if final_score < 75:
            steps.append("2. Review and implement high-priority recommendations")

        if final_score < 60:
            steps.append("3. Follow the improvement roadmap for structured fixes")

        if final_score >= 75:
            steps.append(
                "1. ✅ You're ready to apply! Consider implementing remaining suggestions "
                "for stronger positioning"
            )

        return steps

    def get_score_explanation(self, score: int) -> Dict:
        """Get explanation for a specific score"""
        thresholds = [
            (85, 100, "Excellent", "Your resume is highly optimized for ATS systems",
             "You're ready to submit! Consider tailoring for specific roles"),
            (70, 84, "Good", "Your resume passes ATS but has improvement opportunities",
             "Focus on quick wins for even better results"),
            (55, 69, "Needs Improvement", "Your resume has ATS issues to address",
             "Follow high-priority fixes before applying"),
            (0, 54, "Critical Issues", "Your resume needs significant improvements",
             "Complete the improvement roadmap"),
        ]

        for low, high, status, message, recommendation in thresholds:
            if low <= score <= high:
                return {
                    "status": status,
                    "message": message,
                    "recommendation": recommendation
                }

        return {"status": "Unknown", "message": "", "recommendation": ""}

    def estimate_improvement(self, resume: Dict) -> Dict:
        """Estimate potential score improvement by issue category"""

        ors_score = self.rules_engine.analyze(resume)
        improvements: Dict = {}

        for issue in ors_score.all_issues:
            category = (
                issue.category.value
                if hasattr(issue.category, "value")
                else str(issue.category)
            )

            if category not in improvements:
                improvements[category] = {
                    "current_issues": 0,
                    "potential_gain": 0
                }

            improvements[category]["current_issues"] += 1
            improvements[category]["potential_gain"] += issue.impact_score

        return improvements


# =====================================================
# CONVENIENCE FUNCTION
# =====================================================
async def create_ats_scan(
    resume: Dict,
    job_description: Optional[str] = None,
    llm_client: Optional[Callable] = None,
    db: Optional[AsyncSession] = None
) -> Dict:
    """
    Convenience function to run an ATS scan.

    Args:
        resume: Structured resume dictionary
        job_description: Optional job posting for keyword matching
        llm_client: Optional LLM client for AI analysis
        db: Optional database session

    Returns:
        Complete ATS scan results with scores and recommendations
    """
    scanner = ATSScannerService()
    return await scanner.scan(resume, job_description, llm_client, db)