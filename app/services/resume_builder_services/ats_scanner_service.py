# /home/aryu_user/Arun/aiproject_staging/app/services/ats_scanner_service.py
"""
Production-Grade ATS Scanner Service
Integrates rule-based analysis, keyword matching, AI evaluation, and feedback generation
Author: Backend Architecture Team
"""

import logging
import json
from typing import Dict, Optional, Callable
from sqlalchemy.ext.asyncio import AsyncSession

from app.utils.ats_scanner.ats_rules_advanced import ATSRulesEngine
from app.utils.ats_scanner.ats_keyword_engine import KeywordEngine
from app.utils.ats_scanner.ats_feedback_generator import FeedbackGenerator, ScoreInterpreter

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
1. **ATS Compatibility**: Will this resume parse correctly through ATS systems?
2. **Content Quality**: Are bullets strong, specific, and results-oriented?
3. **Keyword Alignment**: How well does it match the job requirements?
4. **Missing Elements**: What critical information is missing?
5. **Strengths**: What stands out positively?

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


QUICK_AI_EVALUATION_PROMPT = """Analyze this resume for ATS compliance and job fit. Return JSON only.

Resume summary: {summary}
Skills listed: {skills}
Experience highlights: {experience_summary}
Target role: {job_description}

Return ONLY this JSON (no markdown, no explanation):
{{
  "ats_safe": true/false,
  "critical_issues": ["Issue 1"],
  "keyword_gaps": ["Missing skill 1"],
  "top_suggestion": "Single most important improvement",
  "estimated_score_impact": "+X points by fixing main issue"
}}"""


# =====================================================
# MAIN ATS SCANNER SERVICE
# =====================================================

class ATSScannerService:
    """
    Production-grade resume ATS scanner
    
    Combines:
    - Rule-based ATS compliance checking
    - Keyword extraction and semantic matching
    - AI-powered content analysis
    - Detailed feedback generation
    """
    
    def __init__(self):
        self.rules_engine = ATSRulesEngine()
        self.keyword_engine = KeywordEngine()
        self.feedback_generator = FeedbackGenerator()
    
    async def scan(self, resume: Dict, job_description: Optional[str] = None,
                   llm_client: Optional[Callable] = None,
                   db: Optional[AsyncSession] = None) -> Dict:
        """
        Comprehensive ATS scan with rule-based and AI analysis
        
        Args:
            resume: Resume data dictionary
            job_description: Target job posting
            llm_client: LLM function for AI analysis
            db: Database session for LLM
        
        Returns:
            Complete ATS analysis with scores and feedback
        """
        
        try:
            logger.info("Starting comprehensive ATS scan")
            
            # Step 1: Rule-Based Analysis
            ors_score = self.rules_engine.analyze(resume)
            logger.info(f"Rule-based score: {ors_score.total_score}")
            
            # Step 2: Keyword Analysis
            keyword_analysis = self.keyword_engine.match_skills(
                resume, job_description or ""
            )
            logger.info(f"Keyword match: {keyword_analysis.match_percentage}%")
            
            # Step 3: AI Analysis (if available and has JD)
            ai_insights = {}
            if llm_client and db and job_description:
                try:
                    ai_insights = await self._get_ai_insights(
                        resume, job_description, llm_client, db
                    )
                    logger.info("AI insights generated successfully")
                except Exception as e:
                    logger.warning(f"AI analysis failed, continuing with rule-based: {e}")
            
            # Step 4: Generate Detailed Feedback
            feedback = self.feedback_generator.generate_feedback(
                ors_score, keyword_analysis, resume
            )
            
            # Step 5: Score Interpretation
            interpretation = ScoreInterpreter.interpret(ors_score.total_score)
            
            # Step 6: Combine Results
            result = self._combine_results(
                ors_score, keyword_analysis, feedback,
                ai_insights, interpretation, resume
            )
            
            logger.info(f"ATS scan complete. Final score: {result['ats_score']}")
            return result
            
        except Exception as e:
            logger.error(f"ATS scan failed: {str(e)}", exc_info=True)
            raise
    
    # =========== AI ANALYSIS ===========
    
    async def _get_ai_insights(self, resume: Dict, job_description: str,
                               llm_client: Callable, db: AsyncSession) -> Dict:
        """Get AI-powered insights about resume"""
        
        try:
            # Prepare data for prompt
            skills_str = ", ".join(resume.get("skills", [])[:15])
            
            experience_bullets = []
            for exp in resume.get("experience", [])[:3]:
                experience_bullets.extend(exp.get("bullets", [])[:2])
            experience_str = "\n".join(experience_bullets[:6])
            
            education_str = "; ".join([
                e.get("degree", "") for e in resume.get("education", [])
            ])
            
            summary_str = resume.get("summary", "")[:200]
            
            # Build prompt
            prompt = EXPERT_ATS_ANALYSIS_PROMPT.format(
                skills=skills_str,
                experience_bullets=experience_str,
                education=education_str,
                summary=summary_str,
                job_description=job_description[:1000]
            )
            
            # Call LLM
            response = await llm_client(prompt)
            
            # Parse response
            ai_result = json.loads(response)
            
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
    
    # =========== RESULT COMPILATION ===========
    
    def _combine_results(self, ors_score, keyword_analysis, feedback,
                        ai_insights, interpretation, resume) -> Dict:
        """Combine all analysis results into final output"""
        
        # Adjust scores based on keyword analysis
        final_ats_score = self._calculate_final_score(
            ors_score.total_score,
            keyword_analysis.match_percentage
        )
        
        return {
            # Main Score
            "ats_score": final_ats_score,
            "score_status": interpretation.get("message"),
            "score_interpretation": interpretation,
            
            # Score Breakdown
            "score_breakdown": {
                "format_compliance": ors_score.format_score,
                "structure_quality": ors_score.structure_score,
                "content_quality": ors_score.content_score,
                "keyword_alignment": keyword_analysis.match_percentage,
                "ats_compatibility": ors_score.ats_compliance_score
            },
            
            # Detailed Issues
            "issues": self._format_issues(ors_score.all_issues),
            "critical_issues_count": ors_score.critical_issues_count,
            
            # Section Analysis
            "section_analysis": [
                {
                    "section": sf.section,
                    "score": sf.score,
                    "status": sf.status,
                    "issues": sf.issues,
                    "suggestions": sf.suggestions
                }
                for sf in feedback.section_feedback
            ],
            
            # Keyword Analysis
            "keyword_analysis": {
                "total_required_keywords": keyword_analysis.total_jd_keywords,
                "matched_keywords": keyword_analysis.matched_keywords,
                "match_percentage": keyword_analysis.match_percentage,
                "critical_gaps": keyword_analysis.missing_critical_skills,
                "strength_keywords": keyword_analysis.found_strengths
            },
            
            # Recommendations
            "recommendations": {
                "top_priorities": feedback.top_priorities[:3],
                "quick_wins": feedback.quick_wins,
                "improvement_potential": {
                    "critical_gains": feedback.estimated_improvement_potential["critical_gains"],
                    "total_potential": feedback.estimated_improvement_potential["total_potential"]
                }
            },
            
            # AI Insights (if available)
            "ai_analysis": ai_insights if ai_insights.get("success") else None,
            
            # Summary
            "summary": {
                "ready_to_apply": final_ats_score >= 75,
                "main_strengths": feedback.section_feedback[0].strengths if feedback.section_feedback else [],
                "main_weaknesses": feedback.top_priorities,
                "estimated_ats_compatibility": "Good" if final_ats_score >= 75 else
                                              "Moderate" if final_ats_score >= 60 else
                                              "Poor"
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
            formatted[issue.severity.value].append({
                "section": issue.section,
                "message": issue.message,
                "suggestion": issue.suggestion,
                "impact": issue.impact_score
            })
        
        return formatted
    
    def _calculate_final_score(self, rule_score: int,
                              keyword_score: int) -> int:
        """
        Calculate final ATS score with weighting
        
        Weighted formula:
        - Rule compliance: 40%
        - Keyword matching: 60%
        """
        final = (rule_score * 0.40) + (keyword_score * 0.60)
        return min(int(final), 100)
    
    # =========== UTILITY METHODS ===========
    
    def get_score_explanation(self, score: int) -> Dict:
        """Get explanation for a specific score"""
        return ScoreInterpreter.interpret(score)
    
    def estimate_improvement(self, resume: Dict) -> Dict:
        """Estimate potential score improvement by category"""
        
        # Quick analysis
        ors_score = self.rules_engine.analyze(resume)
        improvements = {}
        
        for issue in ors_score.all_issues:
            category = issue.category.value
            if category not in improvements:
                improvements[category] = {
                    "current_issues": 0,
                    "potential_gain": 0
                }
            improvements[category]["current_issues"] += 1
            improvements[category]["potential_gain"] += issue.impact_score
        
        return improvements


# =====================================================
# HELPER FUNCTIONS
# =====================================================

async def create_ats_scan(resume: Dict, job_description: Optional[str] = None,
                         llm_client: Optional[Callable] = None,
                         db: Optional[AsyncSession] = None) -> Dict:
    """
    Convenience function to run ATS scan
    
    Args:
        resume: Resume data
        job_description: Optional job posting
        llm_client: Optional LLM for AI analysis
        db: Optional database session
    
    Returns:
        Complete ATS scan results
    """
    
    scanner = ATSScannerService()
    return await scanner.scan(resume, job_description, llm_client, db)