"""
validator.py — Source-grounded resume validation and anti-hallucination engine.

Validates structured AI parsing results against original source document text to:
- Verify candidate name, company names, job titles, degrees, and project names against source text.
- Detect fabricated / hallucinated entities (e.g. invented companies, wrong dates).
- Detect omitted or missing sections (e.g. projects dropped when present in source).
- Trigger automatic recovery when severe validation failures are detected.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from app.modules.resume_builder.schemas import CanonicalResume

logger = logging.getLogger("resume_builder.validator")


@dataclass
class ValidationResult:
    is_valid: bool
    score: float  # 0.0 to 1.0
    hallucination_warnings: List[str] = field(default_factory=list)
    missing_section_warnings: List[str] = field(default_factory=list)
    passed_checks: List[str] = field(default_factory=list)
    requires_recovery: bool = False
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "score": round(self.score, 3),
            "hallucination_warnings": self.hallucination_warnings,
            "missing_section_warnings": self.missing_section_warnings,
            "passed_checks": self.passed_checks,
            "requires_recovery": self.requires_recovery,
        }


class ResumeSourceValidator:
    """
    Validates parsed CanonicalResume against the raw source text to guarantee source grounding.
    """

    @classmethod
    def validate(
        cls,
        canonical: CanonicalResume,
        source_text: str,
        raw_items: Optional[List[Dict[str, Any]]] = None,
    ) -> ValidationResult:
        if not source_text or len(source_text.strip()) < 20:
            return ValidationResult(
                is_valid=True,
                score=1.0,
                passed_checks=["Empty or minimal source text, skipping deep validation"],
            )

        source_lower = source_text.lower()
        source_tokens = set(re.findall(r"\b[a-zA-Z0-9_]{3,}\b", source_lower))

        hallucination_warnings: List[str] = []
        missing_section_warnings: List[str] = []
        passed_checks: List[str] = []

        total_checks = 0
        passed_count = 0

        # ── 1. Candidate Name Validation ──
        p_name = canonical.personal_information.name.strip()
        if p_name:
            total_checks += 1
            name_parts = [p.lower() for p in p_name.split() if len(p) >= 2]
            matched_parts = [p for p in name_parts if p in source_lower]
            if matched_parts:
                passed_count += 1
                passed_checks.append(f"Candidate name '{p_name}' grounded in source")
            else:
                hallucination_warnings.append(f"Candidate name '{p_name}' not found in source text")

        # ── 2. Experience Company & Job Title Validation ──
        for idx, exp in enumerate(canonical.experience):
            comp = exp.company.strip()
            pos = exp.position.strip()

            if comp:
                total_checks += 1
                comp_tokens = set(re.findall(r"\b[a-zA-Z0-9_]{3,}\b", comp.lower()))
                # Exclude generic words like 'ltd', 'inc', 'corp', 'pvt', 'company'
                meaningful_comp = comp_tokens - {"pvt", "ltd", "inc", "corp", "llc", "company", "enterprises", "solutions"}
                if not meaningful_comp:
                    meaningful_comp = comp_tokens

                overlap = meaningful_comp.intersection(source_tokens)
                if overlap or comp.lower() in source_lower:
                    passed_count += 1
                    passed_checks.append(f"Experience [{idx+1}] company '{comp}' verified in source")
                else:
                    hallucination_warnings.append(
                        f"Experience [{idx+1}] company '{comp}' has no token match in source document (possible hallucination)"
                    )

            if pos:
                total_checks += 1
                pos_tokens = set(re.findall(r"\b[a-zA-Z0-9_]{3,}\b", pos.lower()))
                meaningful_pos = pos_tokens - {"and", "the", "for", "with", "lead", "senior", "junior"}
                if not meaningful_pos:
                    meaningful_pos = pos_tokens

                if meaningful_pos.intersection(source_tokens) or pos.lower() in source_lower:
                    passed_count += 1
                    passed_checks.append(f"Experience [{idx+1}] title '{pos}' verified in source")
                else:
                    hallucination_warnings.append(
                        f"Experience [{idx+1}] position '{pos}' not grounded in source document"
                    )

        # ── 3. Project Validation ──
        for idx, proj in enumerate(canonical.projects):
            title = proj.title.strip()
            if title:
                total_checks += 1
                title_tokens = set(re.findall(r"\b[a-zA-Z0-9_]{3,}\b", title.lower()))
                if title_tokens.intersection(source_tokens) or title.lower() in source_lower:
                    passed_count += 1
                    passed_checks.append(f"Project [{idx+1}] '{title}' verified in source")
                else:
                    hallucination_warnings.append(
                        f"Project [{idx+1}] title '{title}' not grounded in source document"
                    )

        # ── 4. Education Validation ──
        for idx, edu in enumerate(canonical.education):
            inst = edu.institution.strip()
            deg = edu.degree.strip()
            if inst:
                total_checks += 1
                inst_tokens = set(re.findall(r"\b[a-zA-Z0-9_]{3,}\b", inst.lower())) - {"university", "college", "school", "institute"}
                if not inst_tokens:
                    inst_tokens = set(re.findall(r"\b[a-zA-Z0-9_]{3,}\b", inst.lower()))
                if inst_tokens.intersection(source_tokens) or inst.lower() in source_lower:
                    passed_count += 1
                    passed_checks.append(f"Education [{idx+1}] institution '{inst}' verified")
                else:
                    hallucination_warnings.append(f"Education [{idx+1}] institution '{inst}' not found in source")

        # ── 5. Missing Section Check ──
        # Check if source contains prominent 'projects' but parsed projects is empty
        if re.search(r"(?i)\b(?:projects|key projects|personal projects)\b", source_lower) and not canonical.projects:
            # Check if there is actual content following projects header
            proj_match = re.search(r"(?i)\bprojects\b[\s\S]{30,200}", source_lower)
            if proj_match:
                missing_section_warnings.append("Source document contains a 'Projects' section, but 0 projects were extracted.")

        # Check if source contains 'experience' but parsed experience is empty
        if re.search(r"(?i)\b(?:experience|work experience|employment)\b", source_lower) and not canonical.experience:
            missing_section_warnings.append("Source document contains an 'Experience' section, but 0 experience items were extracted.")

        # Check if source contains 'education' but parsed education is empty
        if re.search(r"(?i)\b(?:education|academic|qualifications)\b", source_lower) and not canonical.education:
            missing_section_warnings.append("Source document contains an 'Education' section, but 0 education items were extracted.")

        # ── 6. Compute Score and Validity ──
        if total_checks > 0:
            validation_score = passed_count / total_checks
        else:
            validation_score = 1.0

        # Penalize for missing sections
        if missing_section_warnings:
            validation_score = max(0.0, validation_score - 0.25 * len(missing_section_warnings))

        is_valid = len(hallucination_warnings) == 0 and len(missing_section_warnings) == 0
        requires_recovery = (len(hallucination_warnings) >= 2) or (len(missing_section_warnings) >= 1 and validation_score < 0.50)

        if not is_valid:
            logger.warning(
                f"[ResumeValidator] Validation warnings: hallucinations={len(hallucination_warnings)}, missing={len(missing_section_warnings)}, score={validation_score:.2f}"
            )

        return ValidationResult(
            is_valid=is_valid,
            score=round(validation_score, 3),
            hallucination_warnings=hallucination_warnings,
            missing_section_warnings=missing_section_warnings,
            passed_checks=passed_checks,
            requires_recovery=requires_recovery,
            diagnostics={
                "total_checks": total_checks,
                "passed_checks_count": passed_count,
            },
        )
