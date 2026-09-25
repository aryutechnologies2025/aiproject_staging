"""
ocr_engine.py — Structured OCR extraction preserving layout and bounding boxes.

Provides:
- PyMuPDF native Tesseract OCR integration (`get_textpage_ocr`)
- Fallback page-level OCR on rendered pixmaps
- Line and block bounding box preservation (x, y, w, h, page)
- Structural heading and bullet list detection from OCR text lines
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF

logger = logging.getLogger("resume_builder.ocr_engine")

DEFAULT_OCR_DPI = 150


@dataclass
class OCRExtractionResult:
    blocks: List[Dict[str, Any]]
    raw_text: str
    page_count: int
    confidence: float
    ocr_engine: str = "pymupdf_tesseract"
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "blocks_count": len(self.blocks),
            "raw_text_length": len(self.raw_text),
            "page_count": self.page_count,
            "confidence": round(self.confidence, 3),
            "ocr_engine": self.ocr_engine,
        }


class OCREngine:
    """
    Layout-aware OCR engine that converts image-based PDFs into structured layout blocks.
    """

    @classmethod
    def extract_from_pdf(
        cls,
        file_bytes: bytes,
        dpi: int = DEFAULT_OCR_DPI,
        language: str = "eng",
    ) -> OCRExtractionResult:
        """
        Extract structured layout blocks and text from PDF using PyMuPDF OCR.
        """
        blocks: List[Dict[str, Any]] = []
        raw_text_parts: List[str] = []
        total_pages = 0

        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            total_pages = len(doc)

            for page_idx, page in enumerate(doc):
                page_num = page_idx + 1
                page_blocks = cls._extract_page_ocr(page, page_num, dpi, language)
                blocks.extend(page_blocks)
                page_text = "\n".join(b["text"] for b in page_blocks if b.get("text"))
                if page_text:
                    raw_text_parts.append(page_text)

            doc.close()
            full_raw_text = "\n\n".join(raw_text_parts).strip()

            logger.info(
                f"✓ OCR extraction completed: {len(blocks)} blocks from {total_pages} pages ({len(full_raw_text)} chars)"
            )

            return OCRExtractionResult(
                blocks=blocks,
                raw_text=full_raw_text,
                page_count=total_pages,
                confidence=0.90 if full_raw_text else 0.0,
                ocr_engine="pymupdf_tesseract",
            )

        except Exception as e:
            logger.warning(f"PyMuPDF OCR failed: {e}. Attempting CLI tesseract fallback...")
            return cls._fallback_cli_ocr(file_bytes, total_pages)

    @classmethod
    def _extract_page_ocr(
        cls,
        page: fitz.Page,
        page_num: int,
        dpi: int,
        language: str,
    ) -> List[Dict[str, Any]]:
        page_blocks: List[Dict[str, Any]] = []
        try:
            # Native PyMuPDF Tesseract OCR textpage
            tp = page.get_textpage_ocr(language=language, dpi=dpi, full=True)
            raw_blocks = page.get_text("blocks", textpage=tp) or []

            for b in raw_blocks:
                text = b[4].strip()
                if not text:
                    continue
                x0, y0, x1, y1 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
                lines = [l.strip() for l in text.split("\n") if l.strip()]

                # Check if this block is a bullet list
                bullet_lines = [
                    l for l in lines
                    if l.startswith(("•", "-", "*", "–", "—", "·", "+")) or re.match(r"^\d+\.\s+", l)
                ]
                is_list = bool(bullet_lines and len(bullet_lines) >= len(lines) * 0.6)

                # Heading detection heuristic based on line length, uppercase, or short header keywords
                is_heading = False
                if len(lines) == 1 and len(text) < 45:
                    if text.isupper() or text.endswith(":"):
                        is_heading = True

                block_type = "heading" if is_heading else ("list" if is_list else "text")

                page_blocks.append({
                    "text": text,
                    "type": block_type,
                    "items": bullet_lines if is_list else [],
                    "x": x0,
                    "y": y0,
                    "w": x1 - x0,
                    "h": y1 - y0,
                    "page": page_num,
                    "column": 0,
                    "font_size": max(10.0, (y1 - y0) / max(len(lines), 1)),
                    "is_ocr": True,
                })

        except Exception as ocr_err:
            logger.warning(f"Error during page {page_num} get_textpage_ocr: {ocr_err}")

        return page_blocks

    @classmethod
    def _fallback_cli_ocr(cls, file_bytes: bytes, page_count: int) -> OCRExtractionResult:
        """Fallback when native PyMuPDF OCR fails: renders images and calls tesseract CLI."""
        blocks: List[Dict[str, Any]] = []
        raw_text_parts: List[str] = []
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for page_idx, page in enumerate(doc):
                pix = page.get_pixmap(dpi=150)
                img_path = f"/tmp/ocr_page_{page_idx}.png"
                pix.save(img_path)
                try:
                    res = subprocess.run(
                        ["tesseract", img_path, "stdout", "--oem", "1", "-l", "eng"],
                        capture_output=True,
                        text=True,
                        timeout=15,
                    )
                    out_text = res.stdout.strip()
                    if out_text:
                        raw_text_parts.append(out_text)
                        for line_idx, line in enumerate(out_text.split("\n")):
                            if line.strip():
                                blocks.append({
                                    "text": line.strip(),
                                    "type": "text",
                                    "items": [],
                                    "x": 40.0,
                                    "y": float(line_idx * 16),
                                    "w": 500.0,
                                    "h": 16.0,
                                    "page": page_idx + 1,
                                    "column": 0,
                                    "font_size": 10.0,
                                    "is_ocr": True,
                                })
                finally:
                    if os.path.exists(img_path):
                        os.remove(img_path)
            doc.close()
        except Exception as e:
            logger.error(f"Fallback CLI tesseract failed: {e}")

        full_text = "\n\n".join(raw_text_parts)
        return OCRExtractionResult(
            blocks=blocks,
            raw_text=full_text,
            page_count=max(page_count, 1),
            confidence=0.80 if full_text else 0.0,
            ocr_engine="cli_tesseract",
        )
