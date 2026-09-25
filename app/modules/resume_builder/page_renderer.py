"""
page_renderer.py — High-performance PDF page rendering and image optimization.

Renders PDF pages to high-clarity PNG/JPEG images at controlled DPI (150-200 DPI)
for OCR engines and Gemini Multimodal Vision extraction.
"""

from __future__ import annotations

import io
import logging
from typing import List, Optional, Tuple
import fitz  # PyMuPDF
from PIL import Image

logger = logging.getLogger("resume_builder.page_renderer")

DEFAULT_DPI = 150
MAX_IMAGE_DIMENSION = 1800
JPEG_QUALITY = 90


class PageRenderer:
    """
    Renders PDF pages into optimized raster images for OCR and Multimodal AI.
    """

    @classmethod
    def render_pdf_to_images(
        cls,
        file_bytes: bytes,
        dpi: int = DEFAULT_DPI,
        max_pages: int = 10,
        max_dimension: int = MAX_IMAGE_DIMENSION,
        format: str = "PNG",
    ) -> List[bytes]:
        """
        Render each page of a PDF into image bytes.
        """
        images: List[bytes] = []
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for page_idx in range(min(len(doc), max_pages)):
                page = doc[page_idx]
                img_bytes = cls.render_page_to_bytes(
                    page=page,
                    dpi=dpi,
                    max_dimension=max_dimension,
                    format=format,
                )
                if img_bytes:
                    images.append(img_bytes)
            doc.close()
            logger.info(f"Rendered {len(images)} pages from PDF (dpi={dpi})")
        except Exception as e:
            logger.warning(f"Failed to render PDF pages: {e}")
        return images

    @classmethod
    def render_page_to_bytes(
        cls,
        page: fitz.Page,
        dpi: int = DEFAULT_DPI,
        max_dimension: int = MAX_IMAGE_DIMENSION,
        format: str = "PNG",
    ) -> Optional[bytes]:
        """
        Render a single fitz.Page to optimized image bytes.
        """
        try:
            pix = page.get_pixmap(dpi=dpi)
            # Check dimensions and scale down if exceeding max_dimension
            if pix.width > max_dimension or pix.height > max_dimension:
                pil_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                pil_img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
                buf = io.BytesIO()
                if format.upper() in ("JPEG", "JPG"):
                    pil_img.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
                else:
                    pil_img.save(buf, format="PNG", optimize=True)
                return buf.getvalue()
            else:
                if format.upper() in ("JPEG", "JPG"):
                    return pix.tobytes("jpeg")
                return pix.tobytes("png")
        except Exception as e:
            logger.error(f"Failed to render single page: {e}")
            return None
