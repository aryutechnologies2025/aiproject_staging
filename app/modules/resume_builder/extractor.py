import io
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import pdfplumber
from docx import Document

LINE_TOLERANCE = 4   # px: vertical tolerance for grouping words into lines
COL_GAP_MIN_PCT = 0.04  # minimum gap (as fraction of page width) to split cols

def extract_text(file_bytes: bytes, filename: str) -> Tuple[str, Dict[str, Any]]:
    """Extract clean text and metadata from PDF/DOCX."""
    suffix = Path(filename).suffix.lower()
    meta = {"filename": filename}

    if suffix == ".pdf":
        return _extract_pdf(file_bytes, meta)
    elif suffix in (".docx", ".doc"):
        return _extract_docx(file_bytes, meta)
    else:
        raise ValueError("Unsupported file type. Only .pdf and .docx allowed.")


def _extract_pdf(file_bytes: bytes, meta: dict) -> Tuple[str, dict]:
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        meta["pages"] = len(pdf.pages)
        meta["extractor"] = "pdfplumber"
        pages_out = []
        col_counts = []
        for page in pdf.pages:
            # Try to get words with attributes; fallback to extract_words
            try:
                # Newer pdfplumber (>=0.10.0)
                words = page.words(extra_attrs=["fontname", "size"])
                # Normalize keys (pdfplumber returns 'x0', 'top', 'x1', 'bottom')
            except AttributeError:
                # Older pdfplumber (<=0.9.0) uses extract_words
                words = page.extract_words(
                    keep_blank_chars=False,
                    extra_attrs=["fontname", "size"]
                )
                # In older versions, keys are 'x0', 'y0', 'x1', 'y1', etc.
                # We'll map them to consistent names
                for w in words:
                    # Rename 'y0' to 'top', 'y1' to 'bottom' if needed
                    if 'y0' in w and 'top' not in w:
                        w['top'] = w['y0']
                        w['bottom'] = w['y1']
                    if 'x0' not in w:
                        w['x0'] = w['x0']  # already there
                    if 'x1' not in w:
                        w['x1'] = w['x1']
            if not words:
                # Fallback: extract raw text
                pages_out.append(page.extract_text() or "")
                col_counts.append(1)
                continue

            # Determine column layout
            page_width = float(page.width)
            col_text, n_cols = _column_aware_text(words, page_width)
            pages_out.append(col_text)
            col_counts.append(n_cols)

        meta["columns_detected"] = max(col_counts) if col_counts else 1
        return "\n\n".join(pages_out), meta


def _column_aware_text(words: List[dict], page_width: float) -> Tuple[str, int]:
    """Detect columns and reassemble text in reading order."""
    mid = page_width / 2
    gap_x = _find_column_gap(words, page_width, mid)

    if gap_x is not None:
        left = [w for w in words if w["x1"] <= gap_x + 2]
        right = [w for w in words if w["x0"] >= gap_x - 2]
        left_text = _words_to_text(left)
        right_text = _words_to_text(right)
        # Emit left column first, then right
        combined = left_text + "\n\n" + right_text
        return combined, 2

    return _words_to_text(words), 1


def _find_column_gap(words: List[dict], page_width: float, mid: float) -> Optional[float]:
    """Find a vertical gap near the page midpoint (multi‑column)."""
    pw = page_width
    search_lo = mid - 0.15 * pw
    search_hi = mid + 0.15 * pw
    min_gap = COL_GAP_MIN_PCT * pw

    spans = []
    for w in words:
        if w["x0"] < search_hi and w["x1"] > search_lo:
            spans.append((w["x0"], w["x1"]))
    if not spans:
        return None

    spans.sort()
    max_right = spans[0][1]
    for x0, x1 in spans[1:]:
        gap = x0 - max_right
        if gap >= min_gap:
            return max_right + gap / 2
        max_right = max(max_right, x1)
    return None


def _words_to_text(words: List[dict]) -> str:
    """Group words into lines (by Y coordinate) and sort within line."""
    if not words:
        return ""
    # Group by rounded Y (top coordinate)
    lines_y = {}
    for w in words:
        y_center = (w["top"] + w["bottom"]) / 2
        key = round(y_center / LINE_TOLERANCE)
        lines_y.setdefault(key, []).append(w)

    lines = []
    for y_group in sorted(lines_y.keys()):
        group = lines_y[y_group]
        group.sort(key=lambda w: w["x0"])
        line_text = " ".join(w["text"] for w in group).strip()
        if line_text:
            lines.append(line_text)
    return "\n".join(lines)


def _extract_docx(file_bytes: bytes, meta: dict) -> Tuple[str, dict]:
    doc = Document(io.BytesIO(file_bytes))
    meta["extractor"] = "python-docx"
    meta["pages"] = None
    meta["columns_detected"] = 1

    lines = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        # Detect bold paragraphs as possible headers (add ** marker)
        all_bold = bool(para.runs) and all(run.bold for run in para.runs if run.text.strip())
        if all_bold:
            lines.append(f"**{text}**")
        else:
            lines.append(text)

    # Also extract tables (skills grids)
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                lines.append(" | ".join(cells))

    return "\n".join(lines), meta