"""
universal_extractor.py — Layout-aware multi-stage document extractor and normalizer.

Provides:
- Unified extraction engine orchestrating PDF inspection, local PyMuPDF extraction,
  quality scoring, OCR fallback, and page rendering.
- Layout-aware block flattening with semantic tags ([HEADING], [LIST], [TEXT], [LINK]).
- Aggregated content extraction and robust contact details extraction.
- Normalized document representation (raw_items, clean raw_text, structured Markdown).
"""

from __future__ import annotations

import io
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF
import docx

from app.modules.resume_builder.pdf_inspector import (
    PDFInspector,
    PDFInspectionReport,
    DocumentType,
)
from app.modules.resume_builder.quality_evaluator import (
    ExtractionQualityEvaluator,
    ExtractionQualityReport,
)
from app.modules.resume_builder.page_renderer import PageRenderer
from app.modules.resume_builder.ocr_engine import OCREngine, OCRExtractionResult
from app.modules.resume_builder.layout_reconstructor import LayoutReconstructor

logger = logging.getLogger("resume_builder.universal_extractor")


@dataclass
class ExtractedDocument:
    """
    Standardized, layout-aware document extraction result.
    """
    raw_items: List[Dict[str, Any]]
    raw_text: str
    markdown: str
    document_type: str
    quality_score: float
    ocr_used: bool
    vision_used: bool
    detected_columns: int
    page_count: int
    page_images: List[bytes] = field(default_factory=list)
    inspection_report: Optional[PDFInspectionReport] = None
    quality_report: Optional[ExtractionQualityReport] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_items_count": len(self.raw_items),
            "raw_text_length": len(self.raw_text),
            "markdown_length": len(self.markdown),
            "document_type": self.document_type,
            "quality_score": round(self.quality_score, 3),
            "ocr_used": self.ocr_used,
            "vision_used": self.vision_used,
            "detected_columns": self.detected_columns,
            "page_count": self.page_count,
            "has_page_images": len(self.page_images) > 0,
            "metadata": self.metadata,
        }


class UniversalDocumentExtractor:
    """
    Unified multi-stage document extraction engine.
    """

    @classmethod
    def extract_document(
        cls,
        file_bytes: bytes,
        filename: str = "resume.pdf",
        content_type: str = "application/pdf",
    ) -> ExtractedDocument:
        """
        Main extraction entry point.
        Executes:
        1. Fast PDF inspection & classification
        2. Fast local layout extraction
        3. Extraction quality evaluation
        4. Automatic OCR fallback if scanned or low quality
        5. Reading order & multi-column reconstruction
        6. Clean Markdown formatting
        """
        fname_lower = filename.lower()
        if fname_lower.endswith(".docx") or "wordprocessingml" in content_type.lower():
            return cls._extract_docx(file_bytes, filename)
        elif fname_lower.endswith(".txt") or "text/plain" in content_type.lower():
            return cls._extract_txt(file_bytes, filename)
        elif fname_lower.endswith(".pdf") or "pdf" in content_type.lower():
            return cls._extract_pdf(file_bytes, filename)
        else:
            return cls._extract_generic(file_bytes, filename)

    @classmethod
    def _extract_pdf(cls, file_bytes: bytes, filename: str) -> ExtractedDocument:
        # Stage 1: Fast Inspection
        inspection = PDFInspector.inspect(file_bytes, filename, "application/pdf")
        doc_type = inspection.document_type.value
        page_count = inspection.page_count or 1
        detected_cols = inspection.detected_columns_count

        ocr_used = False
        vision_used = False
        page_images: List[bytes] = []
        raw_items: List[Dict[str, Any]] = []
        raw_text = ""
        markdown = ""
        quality_rep: Optional[ExtractionQualityReport] = None

        # Case A: Scanned / Image-Only PDF (Zero or negligible text layer)
        if inspection.is_scanned or inspection.total_chars < 50:
            logger.info(
                f"[UniversalExtractor] '{filename}' detected as SCANNED_PDF ({inspection.total_chars} chars, {inspection.total_images} images). Triggering OCR engine..."
            )
            ocr_res = OCREngine.extract_from_pdf(file_bytes)
            ocr_used = True
            raw_items = ocr_res.blocks
            raw_text = ocr_res.raw_text
            page_images = PageRenderer.render_pdf_to_images(file_bytes, dpi=150)
            markdown = LayoutReconstructor.blocks_to_markdown(raw_items)
            quality_rep = ExtractionQualityEvaluator.evaluate(raw_items, raw_text, inspection)
            quality_score = max(quality_rep.quality_score, 0.85 if raw_text else 0.10)

        # Case B: PDF with Text Layer
        else:
            try:
                doc = fitz.open(stream=file_bytes, filetype="pdf")
                raw_items = LayoutReconstructor.extract_structured_document(doc)
                doc.close()
                raw_text = "\n\n".join(b["text"] for b in raw_items if b.get("text"))
                markdown = LayoutReconstructor.blocks_to_markdown(raw_items)
            except Exception as e:
                logger.warning(f"[UniversalExtractor] Error during layout extraction for '{filename}': {e}")
                raw_items = []
                raw_text = ""
                markdown = ""

            # Stage 2: Extraction Quality Evaluation
            quality_rep = ExtractionQualityEvaluator.evaluate(raw_items, raw_text, inspection)
            quality_score = quality_rep.quality_score

            # If text layer is poor, corrupted, or missing sections -> trigger OCR fallback
            if not quality_rep.is_reliable or quality_score < 0.55:
                logger.info(
                    f"[UniversalExtractor] Quality check failed (score={quality_score:.2f}, flags={quality_rep.flags}). Activating OCR fallback..."
                )
                ocr_res = OCREngine.extract_from_pdf(file_bytes)
                ocr_used = True
                page_images = PageRenderer.render_pdf_to_images(file_bytes, dpi=150)

                # Adopt OCR text if it extracted more content
                if len(ocr_res.raw_text) > len(raw_text) * 0.8:
                    raw_items = ocr_res.blocks
                    raw_text = ocr_res.raw_text
                    markdown = LayoutReconstructor.blocks_to_markdown(raw_items)
                    quality_score = max(quality_score, 0.80)

            # For image-heavy or complex design resumes, pre-render page images in case multimodal vision is needed
            if inspection.is_image_heavy or inspection.document_type == DocumentType.COMPLEX_LAYOUT_PDF:
                if not page_images:
                    page_images = PageRenderer.render_pdf_to_images(file_bytes, dpi=150)

        return ExtractedDocument(
            raw_items=raw_items,
            raw_text=raw_text,
            markdown=markdown,
            document_type=doc_type,
            quality_score=quality_score,
            ocr_used=ocr_used,
            vision_used=vision_used,
            detected_columns=detected_cols,
            page_count=page_count,
            page_images=page_images,
            inspection_report=inspection,
            quality_report=quality_rep,
            metadata={
                "total_chars": len(raw_text),
                "total_blocks": len(raw_items),
                "image_count": inspection.total_images,
            },
        )

    @classmethod
    def _extract_docx(cls, file_bytes: bytes, filename: str) -> ExtractedDocument:
        inspection = PDFInspector.inspect(file_bytes, filename, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        raw_items: List[Dict[str, Any]] = []
        md_lines: List[str] = []

        try:
            doc = docx.Document(io.BytesIO(file_bytes))
            for p_idx, p in enumerate(doc.paragraphs):
                t = p.text.strip()
                if not t:
                    continue
                is_bullet = t.startswith(("•", "-", "*", "–", "—", "·")) or p.style.name.startswith("List")
                clean_t = re.sub(r"^[\s•\-\*–—·]+\s*", "", t) if is_bullet else t

                raw_items.append({
                    "text": clean_t if is_bullet else t,
                    "type": "list" if is_bullet else "text",
                    "items": [clean_t] if is_bullet else [],
                    "x": 0.0,
                    "y": float(p_idx * 20),
                    "w": 600.0,
                    "h": 20.0,
                    "page": 1,
                    "column": 0,
                    "font_size": 11.0,
                    "is_bold": any(r.bold for r in p.runs),
                })
                if is_bullet:
                    md_lines.append(f"- {clean_t}")
                else:
                    md_lines.append(t)

            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                    if row_text:
                        raw_items.append({
                            "text": row_text,
                            "type": "table_row",
                            "items": [],
                            "x": 0.0,
                            "y": 0.0,
                            "w": 600.0,
                            "h": 20.0,
                            "page": 1,
                            "column": 0,
                            "font_size": 10.0,
                            "is_bold": False,
                        })
                        md_lines.append(f"| {row_text} |")

        except Exception as e:
            logger.warning(f"Error reading docx '{filename}': {e}")
            raw_text = file_bytes.decode("utf-8", errors="ignore")
            for l in raw_text.split("\n"):
                if l.strip():
                    raw_items.append({
                        "text": l.strip(),
                        "type": "text",
                        "items": [],
                        "x": 0.0,
                        "y": 0.0,
                        "w": 600.0,
                        "h": 20.0,
                        "page": 1,
                        "column": 0,
                        "font_size": 10.0,
                        "is_bold": False,
                    })
                    md_lines.append(l.strip())

        raw_text = "\n".join(b["text"] for b in raw_items)
        markdown = "\n".join(md_lines)
        quality_rep = ExtractionQualityEvaluator.evaluate(raw_items, raw_text, inspection)

        return ExtractedDocument(
            raw_items=raw_items,
            raw_text=raw_text,
            markdown=markdown,
            document_type="docx",
            quality_score=quality_rep.quality_score,
            ocr_used=False,
            vision_used=False,
            detected_columns=1,
            page_count=inspection.page_count or 1,
            inspection_report=inspection,
            quality_report=quality_rep,
        )

    @classmethod
    def _extract_txt(cls, file_bytes: bytes, filename: str) -> ExtractedDocument:
        text = file_bytes.decode("utf-8", errors="ignore")
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        raw_items = [
            {
                "text": line,
                "type": "list" if line.startswith(("-", "*", "•")) else "text",
                "items": [line.lstrip("-*• ")] if line.startswith(("-", "*", "•")) else [],
                "x": 0.0,
                "y": float(idx * 16),
                "w": 500.0,
                "h": 16.0,
                "page": 1,
                "column": 0,
                "font_size": 10.0,
                "is_bold": False,
            }
            for idx, line in enumerate(lines)
        ]
        return ExtractedDocument(
            raw_items=raw_items,
            raw_text=text,
            markdown=text,
            document_type="txt",
            quality_score=0.95 if text else 0.0,
            ocr_used=False,
            vision_used=False,
            detected_columns=1,
            page_count=max(1, len(lines) // 40),
        )

    @classmethod
    def _extract_generic(cls, file_bytes: bytes, filename: str) -> ExtractedDocument:
        text = file_bytes.decode("utf-8", errors="ignore")
        return cls._extract_txt(file_bytes, filename)


class UniversalExtractor:
    """
    Universal content extractor preserving block layout hints and contact details.
    (Maintained for 100% backward compatibility with all existing imports).
    """

    @staticmethod
    def get_all_items_flat(raw_items: List[Dict[str, Any]]) -> List[str]:
        """
        Return text lines annotated with block types for layout-aware parsing without duplication.
        """
        items: List[str] = []

        for item in raw_items:
            text = item.get("text", "").strip()
            block_type = item.get("type", "text").lower()

            if not text and not item.get("items"):
                continue

            if block_type in ("heading", "h1", "h2", "h3", "title"):
                tag = "[HEADING]"
            elif block_type in ("list", "bullet", "li"):
                tag = "[LIST]"
            elif block_type in ("link", "url"):
                tag = "[LINK]"
            else:
                tag = "[TEXT]"

            if text:
                items.append(f"{tag} {text}")

            nested = item.get("items", [])
            text_lines = [l.strip() for l in text.split("\n") if l.strip()]
            for sub in nested:
                val = sub.strip() if isinstance(sub, str) else ""
                if val and val not in text_lines and val not in text:
                    items.append(f"[LIST] {val}")

        return items

    @staticmethod
    def extract_all_content(raw_items: List[Dict[str, Any]]) -> str:
        """
        Aggregates all raw layout items into a clean multiline document string without duplication.
        """
        lines = []
        for item in raw_items:
            text = item.get("text", "").strip()
            nested_items = item.get("items", [])
            clean_nested = [
                n.strip() for n in nested_items
                if isinstance(n, str) and n.strip()
            ]

            if not text:
                for val in clean_nested:
                    lines.append(f"• {val}")
            elif not clean_nested:
                lines.append(text)
            else:
                text_lines = [l.strip() for l in text.split("\n") if l.strip()]
                if clean_nested == text_lines:
                    lines.append(text)
                else:
                    lines.append(text)
                    for val in clean_nested:
                        if val not in text and f"• {val}" not in text:
                            lines.append(f"• {val}")
        return "\n".join(lines).strip()

    @staticmethod
    def extract_contact_info_raw(raw_items: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Extract contact details directly from raw layout items.
        """
        contact = {"name": "", "email": "", "phone": "", "location": "", "link": ""}

        full_text = UniversalExtractor.extract_all_content(raw_items[:10])

        # Email
        email_match = re.search(r'\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b', full_text)
        if email_match:
            contact["email"] = email_match.group(0)

        # Phone
        phone_matches = re.finditer(r"(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,5}\)?[\s.-]?)?\d{3,5}[\s.-]?\d{3,5}[\s.-]?\d{0,5}", full_text)
        candidates = []
        for m in phone_matches:
            cand = m.group(0).strip()
            digits = re.findall(r"\d", cand)
            if len(digits) >= 10 or (len(digits) >= 7 and cand.startswith("+")):
                candidates.append(cand)
        if candidates:
            candidates.sort(key=lambda c: (c.startswith("+"), len(re.findall(r"\d", c))), reverse=True)
            contact["phone"] = candidates[0]

        # Name heuristic
        for item in raw_items[:5]:
            text = item.get("text", "").strip()
            if text and "@" not in text and not any(c.isdigit() for c in text) and len(text.split()) <= 4:
                contact["name"] = text
                break

        return contact
