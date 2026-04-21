"""
ATS Resume Extractor — LlamaParse items → Markdown pipeline.
Local fallback uses density-based column detection to handle
multi-column PDFs correctly (columns that overlap in x-coordinates).
"""

from __future__ import annotations

import os
import io
import re
import logging
from typing import List, Dict, Optional, Tuple

from fastapi import UploadFile
from llama_cloud import AsyncLlamaCloud

logger = logging.getLogger(__name__)

LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")


def _sanitise(text: str) -> str:
    if not text:
        return ""
    return text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

async def extract_resume_markdown(file: UploadFile) -> str:
    filename = file.filename or "resume.pdf"
    await file.seek(0)
    file_bytes = await file.read()
    await file.seek(0)

    content_type = file.content_type or (
        "application/pdf" if filename.lower().endswith(".pdf")
        else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )

    if LLAMA_CLOUD_API_KEY:
        try:
            md = await _llamaparse_to_markdown(file_bytes, filename, content_type)
            if md and len(md.strip()) > 100:
                logger.info(f"LlamaParse markdown: {len(md)} chars")
                return _sanitise(md)
            logger.warning("LlamaParse short/empty, using local fallback")
        except Exception as e:
            logger.warning(f"LlamaParse failed: {e}, using local fallback")

    md = _local_extract_markdown(file_bytes, filename)
    logger.info(f"Local fallback markdown: {len(md)} chars")
    return _sanitise(md)


# ─────────────────────────────────────────────────────────────────────────────
# LLAMAPARSE
# ─────────────────────────────────────────────────────────────────────────────

async def _llamaparse_to_markdown(file_bytes: bytes, filename: str, content_type: str) -> str:
    client   = AsyncLlamaCloud(api_key=LLAMA_CLOUD_API_KEY)
    uploaded = await client.files.create(
        file=(filename, file_bytes, content_type), purpose="parse"
    )
    result = await client.parsing.parse(
        file_id=uploaded.id, tier="agentic", version="latest", expand=["items"]
    )
    items = _flat_items(result)
    r= _items_to_markdown(items) if items else ""
    print(r)
    return r


def _flat_items(result) -> List:
    items_obj = getattr(result, "items", None)
    if not items_obj:
        return []
    pages = getattr(items_obj, "pages", None) or []
    flat  = []
    for page in pages:
        pnum = getattr(page, "page_number", 1)
        for item in (getattr(page, "items", None) or []):
            if not hasattr(item, "page_number"):
                setattr(item, "page_number", pnum)
            flat.append(item)
    return flat


def _items_to_markdown(items: List) -> str:
    lines: List[str] = []
    prev_page = None
    for item in items:
        pnum = getattr(item, "page_number", 1)
        if prev_page is not None and pnum != prev_page:
            lines.append("")
        prev_page = pnum

        text = (
            str(item.md).strip()    if getattr(item, "md", None)    else
            str(item.value).strip() if getattr(item, "value", None) else
            str(item.text).strip()  if getattr(item, "text", None)  else ""
        )
        if not text:
            for n in _nested_texts(item):
                lines.append(f"- {n}")
            continue

        btype = str(getattr(item, "type", "text") or "text").lower()

        if btype in ("heading", "h1", "h2", "h3", "h4", "title"):
            clean_text = re.sub(r'^#+\s*', '', text).strip()
            lines.append(f"\n## {clean_text}\n")

        elif btype in ("list", "bullet", "li"):
            clean_text = re.sub(r'^[-•*]\s*', '', text).strip()
            lines.append(f"- {clean_text}")
            for n in _nested_texts(item):
                lines.append(f"- {n}")
        elif btype == "table":
            lines.append(text)
        else:
            lines.append(text)

    return "\n".join(lines)


def _nested_texts(item) -> List[str]:
    out = []
    for sub in (getattr(item, "items", None) or []):
        v = (
            str(sub.value).strip() if getattr(sub, "value", None) else
            str(sub.md).strip()    if getattr(sub, "md",    None) else
            str(sub.text).strip()  if getattr(sub, "text",  None) else ""
        )
        if v:
            out.append(v)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# LOCAL FALLBACK
# ─────────────────────────────────────────────────────────────────────────────

def _local_extract_markdown(file_bytes: bytes, filename: str) -> str:
    fname = filename.lower()
    if fname.endswith(".pdf"):
        return _pdf_to_markdown(file_bytes)
    elif fname.endswith(".docx"):
        return _docx_to_markdown(file_bytes)
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# PDF EXTRACTION — density-based column detection
# ─────────────────────────────────────────────────────────────────────────────

def _pdf_to_markdown(file_bytes: bytes) -> str:
    try:
        import pdfplumber
    except ImportError:
        logger.error("pdfplumber not installed")
        return ""

    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            pages_md: List[str] = []
            for page in pdf.pages:
                pages_md.append(_page_to_markdown(page))
            return "\n\n".join(p for p in pages_md if p.strip())
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        return ""


def _page_to_markdown(page) -> str:
    words = page.extract_words(x_tolerance=3, y_tolerance=3,
                               keep_blank_chars=False, use_text_flow=False)
    if not words:
        return ""

    split = _density_column_split(words, page.width)

    if split is None:
        # Single column
        lines = _words_to_lines(words)
        return _lines_to_markdown(lines)
    else:
        # Two columns — left then right
        left_words  = [w for w in words if w["x0"] < split]
        right_words = [w for w in words if w["x0"] >= split]
        left_lines  = _words_to_lines(left_words)
        right_lines = _words_to_lines(right_words)
        left_md     = _lines_to_markdown(left_lines)
        right_md    = _lines_to_markdown(right_lines)
        return left_md + "\n\n" + right_md


def _density_column_split(words: List[Dict], page_width: float) -> Optional[float]:
    """
    Find the x-coordinate that splits two columns using a density approach.

    For each x position in the central 25-75% of the page width, count how
    many word x0 values fall in a 10pt window around it.  The position with
    the lowest density is the inter-column gap.

    Returns None for single-column pages (minimum density > threshold).
    """
    lo = page_width * 0.25
    hi = page_width * 0.75

    # Build an x0 histogram in 5pt buckets
    bucket: Dict[int, int] = {}
    for w in words:
        x0 = w["x0"]
        if lo <= x0 <= hi:
            b = int(x0 / 5) * 5
            bucket[b] = bucket.get(b, 0) + 1

    if not bucket:
        return None

    # Look for a run of low-density buckets (the gap zone)
    # The gap is where density drops to ≤1 word per bucket for several
    # consecutive buckets while having dense buckets on both sides.
    xs = sorted(bucket)

    # Find the emptiest contiguous run in the central zone
    best_gap_mid   = None
    best_gap_score = 9999   # lower = emptier = better gap

    for i, x in enumerate(xs):
        # Count words in a 20pt window centred on x
        window_count = sum(
            bucket.get(b, 0)
            for b in range(int(x/5)*5 - 2, int(x/5)*5 + 6)
        )
        if window_count < best_gap_score:
            best_gap_score = window_count
            best_gap_mid   = x

    # Also check: are there clearly dense regions on both sides?
    if best_gap_mid is None:
        return None

    left_dense  = sum(bucket.get(b, 0) for b in range(0, int(best_gap_mid/5)*5)   if lo/5 < b)
    right_dense = sum(bucket.get(b, 0) for b in range(int(best_gap_mid/5)*5, 9999) if b < hi/5*5)

    # Two columns: both sides must have significant word density
    # and the gap must be substantially emptier than the sides
    if left_dense < 5 or right_dense < 5:
        return None

    avg_side_density = (left_dense + right_dense) / (len(xs) or 1)
    if best_gap_score > avg_side_density * 0.3:
        return None   # Gap not empty enough → single column

    gap_split = float(best_gap_mid + 5)
    logger.info(f"Density column split: x={gap_split:.1f} "
                f"(gap_score={best_gap_score}, left={left_dense}, right={right_dense})")
    return gap_split


def _words_to_lines(words: List[Dict], ytol: float = 4) -> List[str]:
    """Group words into text lines by y-proximity, return list of strings."""
    if not words:
        return []

    sw = sorted(words, key=lambda w: (round(w["top"] / ytol) * ytol, w["x0"]))
    lines: List[str] = []
    cur: List[Dict]  = [sw[0]]
    cy               = sw[0]["top"]

    for w in sw[1:]:
        if abs(w["top"] - cy) <= ytol:
            cur.append(w)
        else:
            lines.append(" ".join(x["text"] for x in sorted(cur, key=lambda x: x["x0"])))
            cur, cy = [w], w["top"]

    if cur:
        lines.append(" ".join(x["text"] for x in sorted(cur, key=lambda x: x["x0"])))

    return lines


# ─────────────────────────────────────────────────────────────────────────────
# TEXT → MARKDOWN CONVERSION
# ─────────────────────────────────────────────────────────────────────────────

# Complete set of section heading keywords
_SECTION_HEADING_KEYWORDS = {
    # Experience — every possible variant
    "experience", "professional experience", "work experience",
    "employment", "employment history", "work history", "career history",
    "career", "internship", "internships", "industrial training",
    "training", "professional background", "positions held",
    "relevant experience", "prior experience",
    # Education
    "education", "educational background", "academic background",
    "qualifications", "academic qualifications", "academic",
    # Skills
    "skills", "technical skills", "tech stack", "technical stack",
    "core competencies", "competencies", "expertise", "technologies",
    "tools", "tools & technologies", "tools and technologies",
    "programming languages", "software skills",
    # Projects
    "projects", "key projects", "personal projects", "project experience",
    "portfolio",
    # Summary
    "summary", "professional summary", "objective", "career objective",
    "profile", "about me", "overview", "highlights",
    # Contact
    "contact", "contact information", "contact details",
    # Certifications
    "certifications", "certificates", "licenses", "credentials", "courses",
    # Languages
    "languages", "language proficiency",
    # Other
    "awards", "achievements", "honors", "honours",
    "volunteer", "volunteering", "publications", "research",
    "hobbies", "interests", "references",
}

# Regex: ALL-CAPS line, 3-55 chars, at least 3 consecutive capital letters
_ALLCAPS_RE   = re.compile(r"^[A-Z][A-Z\s&/\-]{2,53}[A-Z]$")
_BULLET_RE    = re.compile(r"^[\s]*[•\-\*\u2022\u2023►▶→✓✔]+\s*")


def _lines_to_markdown(lines: List[str]) -> str:
    """Convert a list of plain text lines into structured markdown."""
    md: List[str] = []

    for line in lines:
        s = line.strip()
        if not s:
            md.append("")
            continue

        # ── 1. Exact keyword match (case-insensitive) ─────────────────────
        s_lower = s.lower().rstrip(":").strip()
        if s_lower in _SECTION_HEADING_KEYWORDS:
            md.append(f"\n## {s.rstrip(':')}\n")
            continue

        # ── 2. ALL-CAPS heading (PROFESSIONAL EXPERIENCE, EDUCATION, etc.) ─
        if _ALLCAPS_RE.match(s) and len(s) <= 55:
            # Additional check: not a data line masquerading as caps
            # (e.g. acronyms in content like "CGPA 8.9" are caught by digit check)
            if not re.search(r"\d", s[:4]):
                md.append(f"\n## {s.title()}\n")
                continue

        # ── 3. Contains a heading keyword as a significant portion ─────────
        # Handles "PROFESSIONAL EXPERIENCE" when above ALLCAPS didn't fire
        if len(s.split()) <= 6:
            for kw in _SECTION_HEADING_KEYWORDS:
                if kw in s_lower and len(kw) >= 6:
                    md.append(f"\n## {s.rstrip(':')}\n")
                    break
            else:
                # Not a heading — check bullet
                if _BULLET_RE.match(s):
                    clean = _BULLET_RE.sub("", s).strip()
                    md.append(f"- {clean}")
                else:
                    md.append(s)
            continue

        # ── 4. Bullet line ─────────────────────────────────────────────────
        if _BULLET_RE.match(s):
            clean = _BULLET_RE.sub("", s).strip()
            md.append(f"- {clean}")
            continue

        # ── 5. Regular content ─────────────────────────────────────────────
        md.append(s)

    return "\n".join(md)


# ─────────────────────────────────────────────────────────────────────────────
# DOCX EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def _docx_to_markdown(file_bytes: bytes) -> str:
    try:
        import docx as python_docx
        doc   = python_docx.Document(io.BytesIO(file_bytes))
        lines: List[str] = []
        for para in doc.paragraphs:
            if para.text.strip():
                lines.append(para.text.strip())
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(c.text.strip() for c in row.cells if c.text.strip())
                if row_text:
                    lines.append(row_text)
        return _lines_to_markdown(lines)
    except Exception as e:
        logger.error(f"DOCX extraction failed: {e}")
        return ""