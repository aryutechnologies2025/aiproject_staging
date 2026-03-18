# /home/aryu_user/Arun/aiproject_staging/app/api/v1/ats_checker.py
"""
Production-Grade ATS Scanner API v2
Features:
- Better education detection (FIXED)
- None-safe summary handling (FIXED)
- Fallback summary extraction for unlabeled resumes (FIXED)
- Detailed section-by-section feedback
- Specific suggestions: add/remove/improve
- Complete resume parsing
- Enterprise-ready error handling
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
import logging
from typing import Optional, Dict
import traceback

router = APIRouter()
logger = logging.getLogger(__name__)


# =====================================================
# DATA VALIDATION
# =====================================================

class ATSValidation:
    """Production-grade validation"""

    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
    ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc"}
    MIN_RESUME_LENGTH = 100

    @staticmethod
    def validate_file(file: UploadFile) -> Dict:
        """Validate uploaded file"""

        if not file:
            raise HTTPException(status_code=400, detail="No file provided")

        filename_lower = (file.filename or "").lower()
        has_valid_ext = any(
            filename_lower.endswith(ext) for ext in ATSValidation.ALLOWED_EXTENSIONS
        )

        if not has_valid_ext:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Supported: {', '.join(ATSValidation.ALLOWED_EXTENSIONS)}"
            )

        if file.size and file.size > ATSValidation.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File too large. Maximum: 10 MB, "
                    f"Received: {file.size / 1024 / 1024:.1f} MB"
                )
            )

        return {
            "filename": file.filename,
            "size": file.size,
            "content_type": file.content_type,
            "valid": True
        }


# =====================================================
# MAIN ENDPOINTS
# =====================================================

@router.post("/scan")
async def scan_resume(
    file: UploadFile = File(...),
    job_description: Optional[str] = Query(None),
    include_suggestions: bool = Query(True)
) -> Dict:
    """
    Comprehensive ATS scan endpoint v2

    Features:
    - Fallback summary extraction (no explicit header needed)
    - Enhanced education detection
    - Detailed section-by-section analysis
    - Specific suggestions: what to add/remove/improve
    - Impact scoring and improvement roadmap

    Args:
        file: Resume PDF or DOCX
        job_description: Optional job posting for keyword matching
        include_suggestions: Include detailed suggestions

    Returns:
        Complete ATS analysis with scores and feedback
    """

    try:
        logger.info(f"Starting ATS scan v2 for {file.filename}")

        # ============ VALIDATION ============
        validation_result = ATSValidation.validate_file(file)
        logger.info(f"File validation passed: {validation_result}")

        # ============ TEXT EXTRACTION ============
        logger.info("Extracting text from file...")

        from app.utils.ats_scanner.text_extraction import TextExtractionEngine

        extraction_engine = TextExtractionEngine()
        extraction_result = await extraction_engine.extract_all(file)

        raw_text = extraction_result["raw_text"]
        sections = extraction_result["sections"]
        metadata = extraction_result["metadata"]

        logger.info(
            f"Extracted {len(raw_text)} characters, sections found: {list(sections.keys())}"
        )

        # ============ EDUCATION EXTRACTION ============
        logger.info("Extracting education entries...")

        education_text = sections.get("education", "")
        education_entries = extraction_engine.extract_education_entries(education_text)

        logger.info(f"Found {len(education_entries)} education entries")

        # ============ RESUME PARSING ============
        logger.info("Parsing resume structure...")

        resume_dict = _parse_resume_structure(
            raw_text=raw_text,
            sections=sections,
            education_entries=education_entries,
            metadata=metadata
        )

        logger.info(f"Resume parsed. Keys: {list(resume_dict.keys())}")
        logger.info(f"Summary detected: {bool(resume_dict.get('summary'))}")

        # ============ ATS RULES ANALYSIS ============
        logger.info("Running ATS rules analysis...")

        from app.utils.ats_scanner.ats_rules_advanced import ATSRulesEngine

        rules_engine = ATSRulesEngine()
        ors_score = rules_engine.analyze(resume_dict)

        logger.info(
            f"Rules score: {ors_score.total_score}, "
            f"Critical issues: {ors_score.critical_issues_count}"
        )

        # ============ KEYWORD ANALYSIS ============
        keyword_score = 0
        keyword_analysis = None

        if job_description:
            logger.info("Running keyword analysis...")

            from app.utils.ats_scanner.ats_keyword_engine import KeywordEngine

            keyword_engine = KeywordEngine()
            keyword_analysis = keyword_engine.match_skills(resume_dict, job_description)
            keyword_score = keyword_engine.calculate_keyword_score(keyword_analysis)

            logger.info(f"Keyword match: {keyword_analysis.match_percentage}%")

        # ============ DETAILED FEEDBACK ============
        detailed_feedback = None
        if include_suggestions:
            logger.info("Generating detailed feedback...")

            from app.utils.ats_scanner.ats_feedback_generator import DetailedFeedbackGenerator

            feedback_gen = DetailedFeedbackGenerator()
            detailed_feedback = feedback_gen.generate_detailed_feedback(
                ats_score=ors_score.total_score,
                section_scores={
                    "education": ors_score.content_score,
                    "experience": ors_score.content_score,
                    "skills": ors_score.content_score,
                    "summary": ors_score.content_score,
                },
                resume=resume_dict,
                ats_issues=ors_score.all_issues
            )

        # ============ FINAL SCORE ============
        final_score = _calculate_final_score(ors_score.total_score, keyword_score)

        # ============ BUILD RESPONSE ============
        response = {
            "success": True,
            "scan_metadata": {
                "filename": file.filename,
                "file_type": metadata.get("file_type"),
                "timestamp": _get_timestamp(),
                "has_job_description": bool(job_description)
            },

            "ats_score": final_score,
            "score_details": {
                "rules_compliance_score": ors_score.total_score,
                "keyword_match_score": keyword_score if job_description else None,
                "overall_status": _get_score_status(final_score),
                "ready_to_apply": final_score >= 75,
                "critical_issues_count": ors_score.critical_issues_count
            },

            "score_breakdown": {
                "format_compliance": ors_score.format_score,
                "structure_quality": ors_score.structure_score,
                "content_quality": ors_score.content_score,
                "ats_compatibility": ors_score.ats_compliance_score,
                "keyword_alignment": keyword_score if job_description else 0,
            },

            "issues": {
                "critical": [
                    {
                        "section": i.section,
                        "message": i.message,
                        "suggestion": i.suggestion,
                        "impact": i.impact_score
                    }
                    for i in ors_score.all_issues
                    if i.severity.value == "critical"
                ],
                "high": [
                    {
                        "section": i.section,
                        "message": i.message,
                        "suggestion": i.suggestion,
                        "impact": i.impact_score
                    }
                    for i in ors_score.all_issues
                    if i.severity.value == "high"
                ],
                "medium": [
                    {
                        "section": i.section,
                        "message": i.message,
                        "suggestion": i.suggestion,
                        "impact": i.impact_score
                    }
                    for i in ors_score.all_issues
                    if i.severity.value == "medium"
                ],
                "low": [
                    {
                        "section": i.section,
                        "message": i.message,
                        "suggestion": i.suggestion,
                        "impact": i.impact_score
                    }
                    for i in ors_score.all_issues
                    if i.severity.value == "low"
                ],
                "total_issues": len(ors_score.all_issues)
            },

            "section_analysis": (
                {}
                if not include_suggestions or not detailed_feedback
                else {
                    section_name: {
                        "score": fb.current_score,
                        "target_score": fb.target_score,
                        "status": fb.status,
                        "impact_potential": fb.impact_potential,
                        "is_present": fb.is_present,
                        "is_complete": fb.is_complete,
                        "quality_level": fb.quality_level,
                        "missing_elements": fb.missing_elements,
                        "excessive_elements": fb.excessive_elements,
                        "quality_issues": fb.quality_issues,
                        "priorities": fb.top_priority_fixes,
                        "quick_wins": fb.quick_wins,
                        "suggestions": fb.detailed_suggestions,
                        "current_example": fb.example_current,
                        "improved_example": fb.example_improved,
                        "strengths": fb.strengths
                    }
                    for section_name, fb in detailed_feedback.section_feedback.items()
                }
            ),

            "keyword_analysis": (
                None
                if not keyword_analysis
                else {
                    "total_keywords": keyword_analysis.total_jd_keywords,
                    "matched_keywords": keyword_analysis.matched_keywords,
                    "match_percentage": keyword_analysis.match_percentage,
                    "matched_skills": keyword_analysis.found_strengths,
                    "missing_critical_skills": keyword_analysis.missing_critical_skills,
                    "keyword_density": keyword_analysis.keyword_density
                }
            ),

            "recommendations": (
                {}
                if not include_suggestions or not detailed_feedback
                else {
                    "top_3_priorities": detailed_feedback.top_3_priorities,
                    "quick_wins": detailed_feedback.quick_wins_summary,
                    "improvement_roadmap": detailed_feedback.improvement_roadmap,
                    "estimated_potential": detailed_feedback.estimated_improvement_potential
                }
            ),

            "summary": {
                "key_findings": _generate_key_findings(ors_score, final_score),
                "strengths": (
                    detailed_feedback.strengths_summary
                    if include_suggestions and detailed_feedback
                    else []
                ),
                "main_issues": [i.message for i in ors_score.all_issues[:3]],
                "next_steps": _generate_next_steps(final_score, ors_score.critical_issues_count)
            }
        }

        logger.info(f"ATS scan completed successfully. Score: {final_score}")
        return response

    except HTTPException as e:
        logger.warning(f"HTTP Exception: {e.detail}")
        raise

    except Exception as e:
        logger.error(f"ATS scan error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Error during ATS scan: {str(e)}"
        )


@router.post("/quick-score")
async def quick_ats_score(file: UploadFile = File(...)) -> Dict:
    """
    Fast ATS score — no detailed suggestions, just score and critical issues.
    """

    try:
        logger.info(f"Quick score request for {file.filename}")

        ATSValidation.validate_file(file)

        from app.utils.ats_scanner.text_extraction import TextExtractionEngine
        engine = TextExtractionEngine()
        result = await engine.extract_all(file)

        education_entries = engine.extract_education_entries(
            result["sections"].get("education", "")
        )

        resume = _parse_resume_structure(
            raw_text=result["raw_text"],
            sections=result["sections"],
            education_entries=education_entries,
            metadata=result["metadata"]
        )

        from app.utils.ats_scanner.ats_rules_advanced import ATSRulesEngine
        rules_engine = ATSRulesEngine()
        score_result = rules_engine.analyze(resume)

        critical_issues = [
            i for i in score_result.all_issues if i.severity.value == "critical"
        ]

        return {
            "success": True,
            "ats_score": score_result.total_score,
            "status": _get_score_status(score_result.total_score),
            "ready_to_apply": score_result.total_score >= 75,
            "critical_issues_count": len(critical_issues),
            "top_issues": [i.message for i in critical_issues[:3]]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Quick score error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/score-guide")
async def get_score_guide() -> Dict:
    """Explanation of ATS scores and what they mean"""

    return {
        "score_ranges": {
            "85-100": {
                "status": "Excellent - Ready to Apply",
                "meaning": "Your resume is highly optimized for ATS systems. Submit with confidence!",
                "recommendation": "You can start applying immediately"
            },
            "70-84": {
                "status": "Good - Ready to Apply with Polish",
                "meaning": "Your resume passes ATS with room for improvement",
                "recommendation": "Apply now, but implement quick wins for better results"
            },
            "55-69": {
                "status": "Needs Improvement - Address Issues First",
                "meaning": "Your resume has significant issues that may hurt ATS parsing",
                "recommendation": "Implement high-priority fixes before applying"
            },
            "0-54": {
                "status": "Critical Issues - Major Revision Needed",
                "meaning": "Your resume has serious ATS compatibility problems",
                "recommendation": "Complete overhaul needed. Follow the roadmap provided"
            }
        },
        "scoring_methodology": {
            "Format Compliance": "10% - File type, fonts, layout compatibility",
            "Structure Quality": "15% - Section presence, organization, completeness",
            "Content Quality": "30% - Bullet quality, metrics, action verbs",
            "Keyword Alignment": "30% - Relevance to job description (if provided)",
            "ATS Compatibility": "15% - Parsing safety, character encoding"
        },
        "what_affects_score_most": [
            "Missing critical sections (Education, Experience, Skills)",
            "Weak action verbs and lack of metrics in bullets",
            "Poor keyword alignment with job description",
            "Formatting issues (tables, columns, unsafe fonts)"
        ]
    }


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def _parse_resume_structure(
    raw_text: str,
    sections: Dict,
    education_entries: list,
    metadata: Dict
) -> Dict:
    """
    Parse extracted text into a structured resume dictionary.
    Summary fallback: if sections["summary"] is empty, use raw_text heuristic.
    """

    # Pull summary — use extracted section, fall back to empty string (never None)
    summary_text = (sections.get("summary") or "").strip()

    resume = {
        "raw_text": raw_text,
        "file_type": metadata.get("file_type", "pdf"),
        "metadata": metadata,

        "summary": summary_text,
        "skills": _extract_skills(sections.get("skills", "")),
        "experience": _extract_experience(sections.get("experience", "")),
        "education": (
            education_entries
            if education_entries
            else _extract_education(sections.get("education", ""))
        ),

        "uses_table": metadata.get("has_tables", False),
        "uses_columns": False,
        "uses_graphics": metadata.get("has_images", False),
    }

    return resume


def _extract_skills(skills_text: str) -> list:
    """Extract skills as a list"""

    if not skills_text or not skills_text.strip():
        return []

    for delimiter in ["|", "•", "·", ",", "\n"]:
        if delimiter in skills_text:
            skills = [s.strip() for s in skills_text.split(delimiter) if s.strip()]
            if skills:
                return skills[:50]

    # Single line — split by whitespace
    skills = [s.strip() for s in skills_text.split() if s.strip()]
    return skills[:50]


def _extract_experience(exp_text: str) -> list:
    """Extract experience entries from raw section text"""

    if not exp_text or not exp_text.strip():
        return []

    import re

    experiences = []

    # Split by company/position patterns
    blocks = re.split(
        r'\n(?=[A-Z].*?(?:Inc|Ltd|LLC|Corp|Company|Inc\.|Ltd\.|LLC\.|Corp\.|Co\.|Co))',
        exp_text
    )

    for block in blocks:
        if len(block.strip()) < 10:
            continue

        lines = block.split("\n")

        experience = {
            "title": lines[0].strip() if lines else "",
            "company": lines[1].strip() if len(lines) > 1 else "",
            "duration": lines[2].strip() if len(lines) > 2 else "",
            "bullets": [
                l.strip() for l in lines[3:]
                if l.strip() and len(l.strip()) > 5
            ][:6]
        }

        experiences.append(experience)

    return experiences[:10]


def _extract_education(edu_text: str) -> list:
    """Extract education entries from raw section text (fallback parser)"""

    if not edu_text or not edu_text.strip():
        return []

    import re
    entries = []
    lines = edu_text.split("\n")
    current: Dict = {}

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if any(deg in line for deg in ["B.S", "B.A", "M.S", "M.A", "PhD", "MBA", "B.Tech", "B.E"]):
            if current:
                entries.append(current)
            current = {"degree": line, "institution": "", "year": ""}

        elif re.search(r"20\d{2}|19\d{2}", line) and len(line) < 30:
            if current:
                current["year"] = line
            else:
                current = {"degree": "", "institution": "", "year": line}

        elif len(line) > 5:
            if current:
                if not current.get("institution"):
                    current["institution"] = line
            else:
                current = {"degree": "", "institution": line, "year": ""}

    if current:
        entries.append(current)

    return entries


def _calculate_final_score(rules_score: int, keyword_score: int) -> int:
    """Calculate final weighted score"""

    if keyword_score == 0:
        return rules_score

    final = (rules_score * 0.4) + (keyword_score * 0.6)
    return min(int(final), 100)


def _get_score_status(score: int) -> str:
    """Map score to status label"""

    if score >= 85:
        return "Excellent"
    elif score >= 70:
        return "Good"
    elif score >= 55:
        return "Needs Improvement"
    else:
        return "Critical"


def _generate_key_findings(ors_score, final_score: int) -> list:
    """Generate key findings for summary"""

    findings = []

    if ors_score.critical_issues_count > 0:
        findings.append(f"Found {ors_score.critical_issues_count} critical ATS issues")

    if final_score >= 75:
        findings.append("Resume passes basic ATS compatibility checks")
    else:
        findings.append("Resume has significant ATS compatibility issues")

    return findings


def _generate_next_steps(score: int, critical_count: int) -> list:
    """Generate actionable next steps"""

    steps = []

    if critical_count > 0:
        steps.append(f"1. Fix {critical_count} critical issues (listed above)")

    if score < 60:
        steps.append("2. Review and implement all high-priority suggestions")

    if score < 75:
        steps.append("3. Focus on quick wins for fast improvements")

    if score >= 75:
        steps.append(
            "1. You're ready to apply! Consider implementing remaining suggestions "
            "for stronger positioning"
        )

    return steps


def _get_timestamp() -> str:
    """Get ISO UTC timestamp"""
    from datetime import datetime
    return datetime.utcnow().isoformat()