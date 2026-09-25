"""
quality_evaluator.py — Extraction quality evaluation and anomaly detector for resumes.

Evaluates:
- Character and word density per page
- Printable character ratio and font corruption artifacts (e.g. `(cid:123)` or replacement chars `\ufffd`)
- Standard resume section coverage (Summary, Experience, Education, Skills, Projects, Contact)
- Bounding box overlap and text layer layering anomalies
- Reading-order coherence and column interleaving risks
- Determines if extraction is reliable or requires OCR / Multimodal Vision fallback.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from app.modules.resume_builder.pdf_inspector import PDFInspectionReport, DocumentType

logger = logging.getLogger("resume_builder.quality_evaluator")

# Standard resume section keywords
SECTION_KEYWORDS: Dict[str, List[str]] = {
    "contact": ["email", "phone", "linkedin", "github", "contact", "@", ".com"],
    "summary": ["summary", "profile", "objective", "about me", "overview", "highlights"],
    "experience": ["experience", "employment", "work history", "career", "internship", "positions held"],
    "education": ["education", "academic", "degree", "university", "college", "school", "b.sc", "b.tech", "b.s", "m.s", "m.tech", "bachelor", "master", "high school"],
    "skills": ["skills", "technologies", "tech stack", "tools", "competencies", "proficiencies", "programming"],
    "projects": ["projects", "personal projects", "key projects", "academic projects", "case studies"],
    "certifications": ["certifications", "certificates", "licenses", "courses"],
}


@dataclass
class ExtractionQualityReport:
    quality_score: float  # 0.0 to 1.0
    is_reliable: bool
    char_count_score: float
    printable_ratio: float
    suspicious_char_ratio: float
    section_coverage_score: float
    detected_sections: List[str]
    missing_sections: List[str]
    bbox_overlap_score: float
    reading_order_score: float
    flags: List[str] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "quality_score": round(self.quality_score, 3),
            "is_reliable": self.is_reliable,
            "char_count_score": round(self.char_count_score, 3),
            "printable_ratio": round(self.printable_ratio, 3),
            "suspicious_char_ratio": round(self.suspicious_char_ratio, 3),
            "section_coverage_score": round(self.section_coverage_score, 3),
            "detected_sections": self.detected_sections,
            "missing_sections": self.missing_sections,
            "bbox_overlap_score": round(self.bbox_overlap_score, 3),
            "reading_order_score": round(self.reading_order_score, 3),
            "flags": self.flags,
            "diagnostics": self.diagnostics,
        }


class ExtractionQualityEvaluator:
    """
    Evaluates extraction quality deterministically to trigger OCR or Vision fallbacks.
    """

    @classmethod
    def evaluate(
        cls,
        blocks: List[Dict[str, Any]],
        raw_text: str,
        inspection_report: Optional[PDFInspectionReport] = None,
    ) -> ExtractionQualityReport:
        flags: List[str] = []
        clean_text = raw_text.strip() if raw_text else ""
        total_chars = len(clean_text)
        page_count = inspection_report.page_count if inspection_report and inspection_report.page_count > 0 else 1

        # ── 1. Character Density Score ──
        # Normal 1-page resume has 800 - 3500 chars. Less than 200 chars/page is suspicious.
        avg_chars_per_page = total_chars / max(page_count, 1)
        if avg_chars_per_page >= 800:
            char_count_score = 1.0
        elif avg_chars_per_page >= 400:
            char_count_score = 0.85
        elif avg_chars_per_page >= 200:
            char_count_score = 0.60
            flags.append("LOW_CHAR_DENSITY")
        elif avg_chars_per_page >= 50:
            char_count_score = 0.30
            flags.append("VERY_LOW_CHAR_DENSITY")
        else:
            char_count_score = 0.05
            flags.append("EMPTY_OR_NEGLIGIBLE_TEXT")

        # ── 2. Printable & Clean Character Ratio ──
        if total_chars > 0:
            printable_count = sum(1 for c in clean_text if c.isprintable() or c in "\n\r\t")
            printable_ratio = printable_count / total_chars
        else:
            printable_ratio = 0.0

        if printable_ratio < 0.90:
            flags.append("NON_PRINTABLE_ARTIFACTS")

        # ── 3. Suspicious Glyph / Font Encoding Artifacts ──
        # Check for (cid:xxx), \ufffd (replacement char), or repeated unrecognized tokens
        cid_matches = len(re.findall(r"\(cid:\d+\)", clean_text))
        replacement_matches = clean_text.count("\ufffd")
        suspicious_count = cid_matches * 5 + replacement_matches

        if total_chars > 0:
            suspicious_char_ratio = min(suspicious_count / total_chars, 1.0)
        else:
            suspicious_char_ratio = 0.0

        if suspicious_count >= 5 or suspicious_char_ratio > 0.02:
            flags.append("CORRUPTED_FONT_ENCODING")

        # ── 4. Resume Section Coverage ──
        text_lower = clean_text.lower()
        detected_sections: List[str] = []
        missing_sections: List[str] = []

        for sec, keywords in SECTION_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                detected_sections.append(sec)
            else:
                missing_sections.append(sec)

        # Resumes should have at least 3-4 standard sections (e.g. experience, education, skills, contact)
        core_sections = {"experience", "education", "skills", "contact"}
        detected_core = set(detected_sections).intersection(core_sections)
        section_coverage_score = len(detected_core) / max(len(core_sections), 1)

        if len(detected_core) < 2 and total_chars > 100:
            flags.append("POOR_SECTION_STRUCTURE")

        # ── 5. Bounding Box Overlap Score ──
        # Checks if multiple blocks on same page overlap significantly (Canva/layered text error)
        bbox_overlap_score = cls._calculate_bbox_overlap(blocks)
        if bbox_overlap_score < 0.70:
            flags.append("HIGH_BBOX_OVERLAP")

        # ── 6. Reading Order Coherence ──
        reading_order_score = cls._calculate_reading_order_coherence(blocks)
        if reading_order_score < 0.65:
            flags.append("COLUMN_INTERLEAVING_RISK")

        # ── 7. Overall Quality Score Calculation ──
        quality_score = (
            0.30 * char_count_score
            + 0.20 * printable_ratio
            + 0.20 * (1.0 - suspicious_char_ratio)
            + 0.15 * section_coverage_score
            + 0.08 * bbox_overlap_score
            + 0.07 * reading_order_score
        )

        # Scanned override: if inspection marked document as SCANNED_PDF, quality is near 0
        if inspection_report and inspection_report.is_scanned:
            quality_score = min(quality_score, 0.10)
            flags.append("SCANNED_DOCUMENT")

        if inspection_report and inspection_report.is_image_heavy and total_chars < 300:
            quality_score = min(quality_score, 0.40)
            flags.append("IMAGE_HEAVY_LOW_TEXT")

        # Reliability threshold
        is_reliable = (
            quality_score >= 0.60
            and "EMPTY_OR_NEGLIGIBLE_TEXT" not in flags
            and "CORRUPTED_FONT_ENCODING" not in flags
            and "SCANNED_DOCUMENT" not in flags
        )

        return ExtractionQualityReport(
            quality_score=round(quality_score, 3),
            is_reliable=is_reliable,
            char_count_score=round(char_count_score, 3),
            printable_ratio=round(printable_ratio, 3),
            suspicious_char_ratio=round(suspicious_char_ratio, 3),
            section_coverage_score=round(section_coverage_score, 3),
            detected_sections=detected_sections,
            missing_sections=missing_sections,
            bbox_overlap_score=round(bbox_overlap_score, 3),
            reading_order_score=round(reading_order_score, 3),
            flags=flags,
            diagnostics={
                "total_chars": total_chars,
                "avg_chars_per_page": round(avg_chars_per_page, 1),
                "cid_count": cid_matches,
                "replacement_chars": replacement_matches,
                "block_count": len(blocks),
            },
        )

    @classmethod
    def _calculate_bbox_overlap(cls, blocks: List[Dict[str, Any]]) -> float:
        if not blocks or len(blocks) < 3:
            return 1.0

        # Group by page
        pages: Dict[int, List[Dict[str, Any]]] = {}
        for b in blocks:
            pages.setdefault(b.get("page", 1), []).append(b)

        overlapping_pairs = 0
        total_pairs = 0

        for p, p_blocks in pages.items():
            n = len(p_blocks)
            for i in range(n):
                b1 = p_blocks[i]
                x1_0, y1_0 = b1.get("x", 0.0), b1.get("y", 0.0)
                x1_1, y1_1 = x1_0 + b1.get("w", 0.0), y1_0 + b1.get("h", 0.0)
                a1 = max((x1_1 - x1_0) * (y1_1 - y1_0), 1.0)

                for j in range(i + 1, min(i + 15, n)):
                    b2 = p_blocks[j]
                    x2_0, y2_0 = b2.get("x", 0.0), b2.get("y", 0.0)
                    x2_1, y2_1 = x2_0 + b2.get("w", 0.0), y2_0 + b2.get("h", 0.0)
                    a2 = max((x2_1 - x2_0) * (y2_1 - y2_0), 1.0)

                    # Check intersection area
                    ix0 = max(x1_0, x2_0)
                    iy0 = max(y1_0, y2_0)
                    ix1 = min(x1_1, x2_1)
                    iy1 = min(y1_1, y2_1)

                    if ix1 > ix0 and iy1 > iy0:
                        intersection = (ix1 - ix0) * (iy1 - iy0)
                        min_area = min(a1, a2)
                        if intersection / min_area > 0.40:
                            overlapping_pairs += 1
                    total_pairs += 1

        if total_pairs == 0:
            return 1.0

        overlap_ratio = overlapping_pairs / total_pairs
        return max(0.0, 1.0 - (overlap_ratio * 4.0))

    @classmethod
    def _calculate_reading_order_coherence(cls, blocks: List[Dict[str, Any]]) -> float:
        if not blocks or len(blocks) < 4:
            return 1.0

        # Check if Y coordinates within the same column flow monotonically downwards
        violations = 0
        comparisons = 0

        for i in range(len(blocks) - 1):
            b1 = blocks[i]
            b2 = blocks[i + 1]

            # If same page and same column
            if b1.get("page", 1) == b2.get("page", 1) and b1.get("column", 0) == b2.get("column", 0):
                y1 = b1.get("y", 0.0)
                y2 = b2.get("y", 0.0)
                comparisons += 1
                if y2 < y1 - 30.0:  # Significant upward jump within same column
                    violations += 1

        if comparisons == 0:
            return 1.0

        coherence = 1.0 - (violations / comparisons)
        return max(0.0, coherence)
