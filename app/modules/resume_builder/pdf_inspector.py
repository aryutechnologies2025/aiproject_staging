"""
pdf_inspector.py — Document inspection and layout classification for resume documents.

Inspects PDF and document bytes before extraction to determine:
- Page count and dimensions
- Text layer presence, character count, and word count
- Image count, bounding boxes, and image area coverage ratio
- Vector drawings and separator lines
- Font sizes and typography distribution
- Multi-column and sidebar layout indicators
- Document classification (TEXT_PDF, SCANNED_PDF, IMAGE_HEAVY_PDF, COMPLEX_LAYOUT_PDF, MIXED_PDF, DOCX, TXT)
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF

logger = logging.getLogger("resume_builder.pdf_inspector")


class DocumentType(str, Enum):
    TEXT_PDF = "text_pdf"
    SCANNED_PDF = "scanned_pdf"
    IMAGE_HEAVY_PDF = "image_heavy_pdf"
    COMPLEX_LAYOUT_PDF = "complex_layout_pdf"
    MIXED_PDF = "mixed_pdf"
    DOCX = "docx"
    TXT = "txt"
    UNKNOWN = "unknown"


@dataclass
class PageInspection:
    page_number: int
    width: float
    height: float
    char_count: int
    word_count: int
    block_count: int
    image_count: int
    image_area_ratio: float
    drawing_count: int
    is_scanned: bool
    is_image_heavy: bool
    has_multiple_columns: bool
    column_count: int
    font_sizes: List[float] = field(default_factory=list)


@dataclass
class PDFInspectionReport:
    document_type: DocumentType
    page_count: int
    total_chars: int
    total_words: int
    total_blocks: int
    total_images: int
    total_drawings: int
    avg_image_area_ratio: float
    avg_chars_per_page: float
    has_text_layer: bool
    is_scanned: bool
    is_image_heavy: bool
    has_multiple_columns: bool
    detected_columns_count: int
    has_positioned_boxes: bool
    page_inspections: List[PageInspection] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_type": self.document_type.value,
            "page_count": self.page_count,
            "total_chars": self.total_chars,
            "total_words": self.total_words,
            "total_blocks": self.total_blocks,
            "total_images": self.total_images,
            "total_drawings": self.total_drawings,
            "avg_image_area_ratio": round(self.avg_image_area_ratio, 3),
            "avg_chars_per_page": round(self.avg_chars_per_page, 1),
            "has_text_layer": self.has_text_layer,
            "is_scanned": self.is_scanned,
            "is_image_heavy": self.is_image_heavy,
            "has_multiple_columns": self.has_multiple_columns,
            "detected_columns_count": self.detected_columns_count,
            "has_positioned_boxes": self.has_positioned_boxes,
            "diagnostics": self.diagnostics,
        }


class PDFInspector:
    """
    Fast pre-extraction document inspector and classifier.
    """

    @classmethod
    def inspect(
        cls,
        file_bytes: bytes,
        filename: str = "document.pdf",
        content_type: str = "application/pdf",
    ) -> PDFInspectionReport:
        fname_lower = filename.lower()
        if fname_lower.endswith(".docx") or "wordprocessingml" in content_type.lower():
            return cls._inspect_docx(file_bytes)
        elif fname_lower.endswith(".txt") or "text/plain" in content_type.lower():
            return cls._inspect_txt(file_bytes)
        elif fname_lower.endswith(".pdf") or "pdf" in content_type.lower():
            return cls._inspect_pdf(file_bytes)
        else:
            return cls._inspect_generic(file_bytes)

    @classmethod
    def _inspect_pdf(cls, file_bytes: bytes) -> PDFInspectionReport:
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
        except Exception as e:
            logger.warning(f"Failed to open PDF stream for inspection: {e}")
            return PDFInspectionReport(
                document_type=DocumentType.UNKNOWN,
                page_count=0,
                total_chars=0,
                total_words=0,
                total_blocks=0,
                total_images=0,
                total_drawings=0,
                avg_image_area_ratio=0.0,
                avg_chars_per_page=0.0,
                has_text_layer=False,
                is_scanned=False,
                is_image_heavy=False,
                has_multiple_columns=False,
                detected_columns_count=1,
                has_positioned_boxes=False,
                diagnostics={"error": str(e)},
            )

        page_inspections: List[PageInspection] = []
        total_chars = 0
        total_words = 0
        total_blocks = 0
        total_images = 0
        total_drawings = 0
        all_font_sizes: List[float] = []
        scanned_pages_count = 0
        image_heavy_pages_count = 0
        multi_col_pages_count = 0
        max_cols = 1
        total_image_area_ratio = 0.0
        small_box_count = 0

        for page_idx, page in enumerate(doc):
            p_width = page.rect.width
            p_height = page.rect.height
            p_area = max(p_width * p_height, 1.0)

            # Text extraction
            page_text = page.get_text() or ""
            p_chars = len(page_text.strip())
            words = page.get_text("words") or []
            p_words = len(words)
            blocks = page.get_text("blocks") or []
            p_blocks = len([b for b in blocks if b[4].strip()])

            # Images
            images = page.get_images(full=True) or []
            p_images = len(images)
            total_images += p_images

            # Image area coverage
            img_area = 0.0
            for img_info in images:
                xref = img_info[0]
                rects = page.get_image_rects(xref)
                for r in rects:
                    img_area += r.width * r.height
            p_img_ratio = min(img_area / p_area, 1.0)
            total_image_area_ratio += p_img_ratio

            # Drawings / vectors
            drawings = page.get_drawings() or []
            p_drawings = len(drawings)
            total_drawings += p_drawings

            # Dict extraction for fonts and line coordinates
            p_dict = page.get_text("dict") or {}
            page_font_sizes = []
            page_lines_bboxes: List[Tuple[float, float, float, float]] = []

            for b in p_dict.get("blocks", []):
                if "lines" in b:
                    for l in b["lines"]:
                        l_bbox = l.get("bbox", (0, 0, 0, 0))
                        page_lines_bboxes.append((float(l_bbox[0]), float(l_bbox[1]), float(l_bbox[2]), float(l_bbox[3])))
                        if (l_bbox[2] - l_bbox[0]) < 35 and (l_bbox[3] - l_bbox[1]) < 18:
                            small_box_count += 1
                        for s in l.get("spans", []):
                            sz = s.get("size", 0)
                            if sz > 0:
                                page_font_sizes.append(round(sz, 1))

            all_font_sizes.extend(page_font_sizes)

            # Scanned detection
            p_is_scanned = bool((p_chars < 50) and (p_images > 0 or p_img_ratio > 0.40))
            if p_is_scanned:
                scanned_pages_count += 1

            p_is_image_heavy = bool(p_img_ratio > 0.35 and not p_is_scanned)
            if p_is_image_heavy:
                image_heavy_pages_count += 1

            # Multi-column detection per page using line bounding boxes
            p_has_multi_col, p_col_count = cls._detect_columns_from_lines(page_lines_bboxes, p_width)
            if p_has_multi_col:
                multi_col_pages_count += 1
                max_cols = max(max_cols, p_col_count)

            total_chars += p_chars
            total_words += p_words
            total_blocks += p_blocks

            page_inspections.append(
                PageInspection(
                    page_number=page_idx + 1,
                    width=p_width,
                    height=p_height,
                    char_count=p_chars,
                    word_count=p_words,
                    block_count=p_blocks,
                    image_count=p_images,
                    image_area_ratio=p_img_ratio,
                    drawing_count=p_drawings,
                    is_scanned=p_is_scanned,
                    is_image_heavy=p_is_image_heavy,
                    has_multiple_columns=p_has_multi_col,
                    column_count=p_col_count,
                    font_sizes=list(set(page_font_sizes)),
                )
            )

        page_count = len(doc)
        doc.close()

        if page_count == 0:
            return PDFInspectionReport(
                document_type=DocumentType.UNKNOWN,
                page_count=0,
                total_chars=0,
                total_words=0,
                total_blocks=0,
                total_images=0,
                total_drawings=0,
                avg_image_area_ratio=0.0,
                avg_chars_per_page=0.0,
                has_text_layer=False,
                is_scanned=False,
                is_image_heavy=False,
                has_multiple_columns=False,
                detected_columns_count=1,
                has_positioned_boxes=False,
            )

        avg_chars_per_page = total_chars / page_count
        avg_img_ratio = total_image_area_ratio / page_count
        has_text = total_chars >= 50
        is_all_scanned = scanned_pages_count == page_count or (total_chars < 50 and total_images >= page_count)
        is_mixed = scanned_pages_count > 0 and scanned_pages_count < page_count
        is_img_heavy = (avg_img_ratio > 0.40 or image_heavy_pages_count > 0) and not is_all_scanned
        has_multi_cols = multi_col_pages_count > 0
        has_pos_boxes = small_box_count >= max(page_count * 25, 30)

        # Classification decision tree
        if is_all_scanned or not has_text:
            doc_type = DocumentType.SCANNED_PDF
        elif is_mixed:
            doc_type = DocumentType.MIXED_PDF
        elif is_img_heavy:
            doc_type = DocumentType.IMAGE_HEAVY_PDF
        elif has_multi_cols or has_pos_boxes:
            doc_type = DocumentType.COMPLEX_LAYOUT_PDF
        else:
            doc_type = DocumentType.TEXT_PDF

        return PDFInspectionReport(
            document_type=doc_type,
            page_count=page_count,
            total_chars=total_chars,
            total_words=total_words,
            total_blocks=total_blocks,
            total_images=total_images,
            total_drawings=total_drawings,
            avg_image_area_ratio=avg_img_ratio,
            avg_chars_per_page=avg_chars_per_page,
            has_text_layer=has_text,
            is_scanned=is_all_scanned,
            is_image_heavy=is_img_heavy,
            has_multiple_columns=has_multi_cols,
            detected_columns_count=max_cols,
            has_positioned_boxes=has_pos_boxes,
            page_inspections=page_inspections,
            diagnostics={
                "scanned_pages": scanned_pages_count,
                "multi_col_pages": multi_col_pages_count,
                "image_heavy_pages": image_heavy_pages_count,
                "unique_font_sizes": sorted(list(set(all_font_sizes))),
            },
        )

    @classmethod
    def _detect_columns_from_lines(
        cls,
        lines: List[Tuple[float, float, float, float]],
        page_width: float,
    ) -> Tuple[bool, int]:
        """
        Analyze line bounding boxes (x0, y0, x1, y1) to detect true multi-column layouts.
        """
        if len(lines) < 6:
            return False, 1

        min_x = min(l[0] for l in lines)
        max_x = max(l[2] for l in lines)
        content_width = max(max_x - min_x, 300.0)

        # Check if 30%+ of lines span across the middle (single-column indicator)
        mid_x = min_x + 0.50 * content_width
        spanning_lines = [l for l in lines if l[0] < mid_x - 30 and l[2] > mid_x + 30]
        if len(spanning_lines) / len(lines) >= 0.28:
            return False, 1

        # Check for candidate vertical gutters between 20% and 80% of width
        step = 10.0
        curr_x = min_x + 0.20 * content_width
        end_x = min_x + 0.80 * content_width

        while curr_x <= end_x:
            left_lines = [l for l in lines if l[2] <= curr_x + 8]
            right_lines = [l for l in lines if l[0] >= curr_x - 8]
            crossing_lines = [l for l in lines if l[0] < curr_x - 8 and l[2] > curr_x + 8]

            if len(left_lines) >= 4 and len(right_lines) >= 4:
                if len(crossing_lines) <= max(1, int(len(lines) * 0.08)):
                    return True, 2

            curr_x += step

        # Check 3 columns
        s1 = min_x + 0.33 * content_width
        s2 = min_x + 0.66 * content_width
        c1 = [l for l in lines if l[2] <= s1 + 10]
        c2 = [l for l in lines if l[0] >= s1 - 10 and l[2] <= s2 + 10]
        c3 = [l for l in lines if l[0] >= s2 - 10]
        if len(c1) >= 3 and len(c2) >= 3 and len(c3) >= 3:
            return True, 3

        return False, 1

    @classmethod
    def _inspect_docx(cls, file_bytes: bytes) -> PDFInspectionReport:
        try:
            import docx
            doc = docx.Document(io.BytesIO(file_bytes))
            p_count = len(doc.paragraphs)
            t_count = len(doc.tables)
            total_chars = sum(len(p.text) for p in doc.paragraphs)
            for t in doc.tables:
                for row in t.rows:
                    for cell in row.cells:
                        total_chars += len(cell.text)
            return PDFInspectionReport(
                document_type=DocumentType.DOCX,
                page_count=max(1, p_count // 30),
                total_chars=total_chars,
                total_words=total_chars // 5,
                total_blocks=p_count + t_count,
                total_images=0,
                total_drawings=0,
                avg_image_area_ratio=0.0,
                avg_chars_per_page=float(total_chars),
                has_text_layer=True,
                is_scanned=False,
                is_image_heavy=False,
                has_multiple_columns=t_count > 0,
                detected_columns_count=2 if t_count > 0 else 1,
                has_positioned_boxes=False,
                diagnostics={"paragraphs": p_count, "tables": t_count},
            )
        except Exception as e:
            return PDFInspectionReport(
                document_type=DocumentType.DOCX,
                page_count=1,
                total_chars=len(file_bytes),
                total_words=len(file_bytes) // 5,
                total_blocks=1,
                total_images=0,
                total_drawings=0,
                avg_image_area_ratio=0.0,
                avg_chars_per_page=float(len(file_bytes)),
                has_text_layer=True,
                is_scanned=False,
                is_image_heavy=False,
                has_multiple_columns=False,
                detected_columns_count=1,
                has_positioned_boxes=False,
                diagnostics={"error": str(e)},
            )

    @classmethod
    def _inspect_txt(cls, file_bytes: bytes) -> PDFInspectionReport:
        text = file_bytes.decode("utf-8", errors="ignore")
        chars = len(text)
        words = len(text.split())
        lines = [l for l in text.split("\n") if l.strip()]
        return PDFInspectionReport(
            document_type=DocumentType.TXT,
            page_count=max(1, len(lines) // 40),
            total_chars=chars,
            total_words=words,
            total_blocks=len(lines),
            total_images=0,
            total_drawings=0,
            avg_image_area_ratio=0.0,
            avg_chars_per_page=float(chars),
            has_text_layer=True,
            is_scanned=False,
            is_image_heavy=False,
            has_multiple_columns=False,
            detected_columns_count=1,
            has_positioned_boxes=False,
        )

    @classmethod
    def _inspect_generic(cls, file_bytes: bytes) -> PDFInspectionReport:
        chars = len(file_bytes)
        return PDFInspectionReport(
            document_type=DocumentType.UNKNOWN,
            page_count=1,
            total_chars=chars,
            total_words=chars // 5,
            total_blocks=1,
            total_images=0,
            total_drawings=0,
            avg_image_area_ratio=0.0,
            avg_chars_per_page=float(chars),
            has_text_layer=chars > 20,
            is_scanned=False,
            is_image_heavy=False,
            has_multiple_columns=False,
            detected_columns_count=1,
            has_positioned_boxes=False,
        )
