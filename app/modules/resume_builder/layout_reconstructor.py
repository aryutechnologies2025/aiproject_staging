"""
layout_reconstructor.py — Advanced Layout-Aware Reconstruction and Entity Association.

Provides:
- Span-level line extraction from PyMuPDF dict blocks with font styles and bounding boxes.
- Robust multi-column and sidebar detection.
- Natural reading order preservation for single-column and multi-column documents.
- Elimination of horizontal column interleaving.
- Visual heading and section boundary detection.
- Structured Markdown generation preserving visual document hierarchy.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import fitz  # PyMuPDF

logger = logging.getLogger("resume_builder.layout_reconstructor")

# Known resume section heading keywords
SECTION_HEADING_KEYWORDS: Set[str] = {
    "summary", "professional summary", "career summary", "executive summary",
    "profile", "professional profile", "about me", "overview", "objective",
    "experience", "work experience", "professional experience", "employment",
    "employment history", "work history", "career history", "internships",
    "internship experience", "relevant experience", "positions held",
    "education", "academic background", "qualifications", "academic qualifications",
    "academic history", "degrees", "educational details",
    "skills", "technical skills", "tech stack", "technical stack",
    "core competencies", "competencies", "key skills", "technologies",
    "tools", "tools & technologies", "software skills", "proficiencies",
    "projects", "key projects", "personal projects", "academic projects",
    "project experience", "portfolio", "featured projects",
    "certifications", "certificates", "licenses", "courses",
    "languages", "language proficiency",
    "achievements", "awards", "honors", "publications", "references",
    "contact", "contact information", "contact details",
}

DATE_RANGE_PATTERN = re.compile(
    r"(?i)\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december)?\.?\s*(?:19\d\d|20\d\d)\s*(?:-|–|—|to)\s*(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december)?\.?\s*(?:19\d\d|20\d\d)|present|current|ongoing|now)\b"
)


class LayoutReconstructor:
    """
    Reconstructs document layout, extracts columns, isolates spanning headers,
    and produces correctly-ordered structured blocks and clean Markdown.
    """

    @classmethod
    def extract_structured_document(cls, doc: fitz.Document) -> List[Dict[str, Any]]:
        """
        Extract all blocks across all pages in the PDF, sorted by reading order with column preservation.
        """
        all_blocks: List[Dict[str, Any]] = []

        for page_idx, page in enumerate(doc):
            page_num = page_idx + 1
            page_lines = cls._extract_page_lines(page, page_num)
            if not page_lines:
                continue

            page_blocks = cls._reconstruct_page_layout(
                lines=page_lines,
                page_width=page.rect.width,
                page_height=page.rect.height,
                page_num=page_num,
            )
            all_blocks.extend(page_blocks)

        return all_blocks

    @classmethod
    def _extract_page_lines(cls, page: fitz.Page, page_num: int) -> List[Dict[str, Any]]:
        """
        Extract lines from PyMuPDF dict with exact bounding boxes, max font size, and font weight.
        """
        p_dict = page.get_text("dict") or {}
        lines: List[Dict[str, Any]] = []

        for b in p_dict.get("blocks", []):
            if "lines" not in b:
                continue
            for l in b["lines"]:
                bbox = l.get("bbox", (0, 0, 0, 0))
                x0, y0, x1, y1 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])

                spans_text: List[str] = []
                max_size = 0.0
                is_bold = False
                font_names: List[str] = []

                for s in l.get("spans", []):
                    stext = s.get("text", "")
                    if stext.strip():
                        spans_text.append(stext)
                        sz = float(s.get("size", 0.0))
                        max_size = max(max_size, sz)
                        fn = s.get("font", "").lower()
                        font_names.append(fn)
                        if (s.get("flags", 0) & 2) or ("bold" in fn) or ("black" in fn) or ("heavy" in fn):
                            is_bold = True

                line_text = "".join(spans_text).strip()
                if line_text:
                    lines.append({
                        "text": line_text,
                        "x0": x0,
                        "y0": y0,
                        "x1": x1,
                        "y1": y1,
                        "w": x1 - x0,
                        "h": y1 - y0,
                        "font_size": max_size or 10.0,
                        "is_bold": is_bold,
                        "font": font_names[0] if font_names else "",
                        "page": page_num,
                    })

        return lines

    @classmethod
    def _reconstruct_page_layout(
        cls,
        lines: List[Dict[str, Any]],
        page_width: float,
        page_height: float,
        page_num: int,
    ) -> List[Dict[str, Any]]:
        """
        Detects multi-column structure, assigns column indices, and sorts lines in true reading order.
        """
        if not lines:
            return []

        # Find page margin bounds
        min_x = min(l["x0"] for l in lines)
        max_x = max(l["x1"] for l in lines)
        content_width = max(max_x - min_x, 300.0)

        # Check if lines form multiple columns
        col_splits = cls._find_column_gutters(lines, min_x, max_x, content_width, page_width)

        if not col_splits:
            # Single-column page: Natural top-to-bottom reading order
            for l in lines:
                l["region"] = 0
                l["column"] = 0
            blocks = cls._group_lines_into_blocks(lines, page_num)
            return sorted(blocks, key=lambda b: (b.get("page", 1), b.get("y", 0), b.get("x", 0)))

        # Multi-column layout present
        # Identify top header cutoff and bottom footer cutoff
        header_y_max = 0.0
        footer_y_min = page_height

        for line in lines:
            lw = line["w"]
            lx0 = line["x0"]
            lx1 = line["x1"]
            ly0 = line["y0"]

            # Line spans across columns if width is wide or straddles the split
            is_spanning = (lw >= 0.55 * content_width) or any(
                (lx0 < s - 15 and lx1 > s + 15) for s in col_splits
            )

            if is_spanning and ly0 <= page_height * 0.30:
                line["region"] = 0  # Spanning top header
                line["column"] = 0
            elif is_spanning and ly0 >= page_height * 0.85:
                line["region"] = 99  # Spanning footer
                line["column"] = 0
            else:
                col_idx = 0
                for s_idx, split_x in enumerate(col_splits):
                    if lx0 >= split_x - 10:
                        col_idx = s_idx + 1
                line["region"] = 10 + col_idx
                line["column"] = col_idx

        blocks = cls._group_lines_into_blocks(lines, page_num)
        return sorted(blocks, key=lambda b: (b.get("page", 1), b.get("region", 0), b.get("y", 0), b.get("x", 0)))

    @classmethod
    def _find_column_gutters(
        cls,
        lines: List[Dict[str, Any]],
        min_x: float,
        max_x: float,
        content_width: float,
        page_width: float,
    ) -> List[float]:
        """
        Finds vertical gutter X coordinates dividing genuine multi-column layouts.
        """
        if len(lines) < 8:
            return []

        # Count how many lines span across the middle 50% of the page
        mid_x = min_x + 0.50 * content_width
        spanning_lines = [l for l in lines if l["x0"] < mid_x - 30 and l["x1"] > mid_x + 30]

        # If 30%+ of lines span across the center, this is a single column page with paragraph text
        if len(spanning_lines) / len(lines) >= 0.28:
            return []

        # Filter out wide lines
        narrow_lines = [l for l in lines if l["w"] <= 0.60 * content_width]
        if len(narrow_lines) < 6:
            return []

        # Test candidate split points from 22% to 78% of content width
        best_split: Optional[float] = None
        min_crossing_count = len(narrow_lines)

        step = 10.0
        start_x = min_x + 0.22 * content_width
        end_x = min_x + 0.78 * content_width

        curr_x = start_x
        while curr_x <= end_x:
            left_count = 0
            right_count = 0
            crossing_count = 0

            for l in narrow_lines:
                if l["x1"] <= curr_x + 5:
                    left_count += 1
                elif l["x0"] >= curr_x - 5:
                    right_count += 1
                else:
                    crossing_count += 1

            if left_count >= 3 and right_count >= 3:
                if crossing_count < min_crossing_count and crossing_count <= max(1, int(len(narrow_lines) * 0.12)):
                    min_crossing_count = crossing_count
                    best_split = curr_x

            curr_x += step

        if best_split is not None:
            return [best_split]

        return []

    @classmethod
    def _group_lines_into_blocks(
        cls,
        lines: List[Dict[str, Any]],
        page_num: int,
    ) -> List[Dict[str, Any]]:
        """
        Groups consecutive lines within the same column/region into coherent semantic blocks.
        """
        if not lines:
            return []

        # Sort lines first by region then y0 with small tolerance for same-line elements
        sorted_lines = sorted(lines, key=lambda l: (l.get("region", 0), round(l.get("y0", 0) / 4) * 4, l.get("x0", 0)))

        blocks: List[Dict[str, Any]] = []
        current_block: Optional[Dict[str, Any]] = None

        for line in sorted_lines:
            text = line["text"]
            is_bullet = text.startswith(("•", "-", "*", "–", "—", "·", "+", "►", "▶", "→", "✓")) or bool(re.match(r"^\d+\.\s+", text))
            is_heading = cls._is_section_heading(line)

            clean_text = text
            if is_bullet:
                clean_text = re.sub(r"^[\s•\-\*–—·+►▶→✓]+\s*", "", text).strip()

            should_break = False
            if current_block is None:
                should_break = True
            elif line.get("region") != current_block.get("region"):
                should_break = True
            elif is_heading:
                should_break = True
            elif current_block.get("type") == "heading":
                should_break = True
            elif is_bullet:
                should_break = True
            elif abs(line["y0"] - current_block["y1"]) > 16.0:
                should_break = True

            if should_break:
                if current_block:
                    blocks.append(cls._finalize_block(current_block))

                block_type = "heading" if is_heading else ("list" if is_bullet else "text")
                current_block = {
                    "text": clean_text if is_bullet else text,
                    "lines": [text],
                    "type": block_type,
                    "items": [clean_text] if is_bullet else [],
                    "x0": line["x0"],
                    "y0": line["y0"],
                    "x1": line["x1"],
                    "y1": line["y1"],
                    "font_size": line["font_size"],
                    "is_bold": line["is_bold"],
                    "page": page_num,
                    "column": line.get("column", 0),
                    "region": line.get("region", 0),
                }
            else:
                current_block["lines"].append(text)
                current_block["text"] = "\n".join(current_block["lines"])
                if is_bullet:
                    current_block["items"].append(clean_text)
                    current_block["type"] = "list"
                current_block["x0"] = min(current_block["x0"], line["x0"])
                current_block["y0"] = min(current_block["y0"], line["y0"])
                current_block["x1"] = max(current_block["x1"], line["x1"])
                current_block["y1"] = max(current_block["y1"], line["y1"])
                current_block["font_size"] = max(current_block["font_size"], line["font_size"])
                if line["is_bold"]:
                    current_block["is_bold"] = True

        if current_block:
            blocks.append(cls._finalize_block(current_block))

        return blocks

    @classmethod
    def _finalize_block(cls, block: Dict[str, Any]) -> Dict[str, Any]:
        """Convert bounding box into normalized block format."""
        return {
            "text": block["text"].strip(),
            "type": block["type"],
            "items": block["items"],
            "x": block["x0"],
            "y": block["y0"],
            "w": block["x1"] - block["x0"],
            "h": block["y1"] - block["y0"],
            "page": block["page"],
            "column": block["column"],
            "region": block["region"],
            "font_size": block["font_size"],
            "is_bold": block["is_bold"],
        }

    @classmethod
    def _is_section_heading(cls, line: Dict[str, Any]) -> bool:
        """
        Detects whether a line is a resume section heading using font size, weight, and keywords.
        """
        text = line["text"].strip()
        if not text or len(text) > 45:
            return False

        if text.startswith(("•", "-", "*", "–", "—", "·")) or "@" in text or "http" in text.lower() or DATE_RANGE_PATTERN.search(text):
            return False

        text_lower = text.lower().rstrip(":").strip()

        # 1. Exact keyword match
        if text_lower in SECTION_HEADING_KEYWORDS:
            return True

        # 2. All-caps short line with large font or bold
        if (text.isupper() or line.get("is_bold")) and len(text.split()) <= 4:
            if any(kw in text_lower for kw in ("summary", "experience", "education", "skills", "projects", "certifications", "contact")):
                return True
            if line.get("font_size", 10.0) >= 11.5:
                return True

        return False

    @classmethod
    def blocks_to_markdown(cls, blocks: List[Dict[str, Any]]) -> str:
        """
        Converts ordered layout blocks into rich, clean Markdown.
        """
        md_lines: List[str] = []
        prev_page: Optional[int] = None

        for b in blocks:
            page = b.get("page", 1)
            if prev_page is not None and page != prev_page:
                md_lines.append(f"\n--- [Page {page}] ---\n")
            prev_page = page

            b_type = b.get("type", "text")
            text = b.get("text", "").strip()
            if not text:
                continue

            if b_type == "heading":
                clean_h = text.rstrip(":-")
                md_lines.append(f"\n## {clean_h}\n")
            elif b_type == "list":
                items = b.get("items", [])
                if items:
                    for item in items:
                        md_lines.append(f"- {item}")
                else:
                    for l in text.split("\n"):
                        clean_l = re.sub(r"^[\s•\-\*–—·+►▶→✓]+\s*", "", l).strip()
                        if clean_l:
                            md_lines.append(f"- {clean_l}")
            else:
                md_lines.append(text)

        md = "\n".join(md_lines).strip()
        return re.sub(r"\n{3,}", "\n\n", md)
