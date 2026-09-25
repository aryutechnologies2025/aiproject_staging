"""
extractor.py — Layout-aware document extractor for resume_builder.

Supports:
- Multi-stage layout-aware PDF extraction (zero network latency, 0 tokens, <50ms).
- Automatic OCR fallback for scanned and low-quality documents.
- Multi-column and sidebar reconstruction preserving reading order.
- Python-docx extraction for DOCX resumes.
- Optional fallback to LlamaCloud if local extraction is incomplete and LlamaCloud is configured.
"""

from __future__ import annotations

import io
import logging
import os
from typing import Any, Dict, List, Optional
import fitz  # PyMuPDF
import docx

from app.modules.resume_builder.universal_extractor import UniversalDocumentExtractor, ExtractedDocument
from app.modules.resume_builder.layout_reconstructor import LayoutReconstructor

try:
    from llama_cloud import AsyncLlamaCloud
    LLAMA_CLOUD_AVAILABLE = True
except ImportError:
    LLAMA_CLOUD_AVAILABLE = False

logger = logging.getLogger("resume_builder.extractor")
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")


def extract_local_pdf(file_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Extract structured text blocks and layout metadata from PDF bytes using PyMuPDF.
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    blocks = LayoutReconstructor.extract_structured_document(doc)
    doc.close()
    return blocks


def extract_local_docx(file_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Extract structured paragraphs and tables from DOCX bytes using python-docx.
    """
    file_stream = io.BytesIO(file_bytes)
    doc = docx.Document(file_stream)
    blocks: List[Dict[str, Any]] = []

    for p_idx, p in enumerate(doc.paragraphs):
        text = p.text.strip()
        if not text:
            continue
        blocks.append({
            "text": text,
            "type": "text",
            "items": [],
            "x": 0.0,
            "y": float(p_idx * 20),
            "w": 600.0,
            "h": 20.0,
            "page": 1,
            "column": 0,
        })

    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
            if row_text:
                blocks.append({
                    "text": row_text,
                    "type": "table_row",
                    "items": [],
                    "x": 0.0,
                    "y": 0.0,
                    "w": 600.0,
                    "h": 20.0,
                    "page": 1,
                    "column": 0,
                })

    return blocks


def normalize_items(items: List) -> List[Dict[str, Any]]:
    """Convert LlamaParse items to normalized dict format"""
    normalized = []
    for item in items:
        text = ""
        if hasattr(item, "value") and item.value:
            text = item.value
        elif hasattr(item, "md") and item.md:
            text = item.md
        elif hasattr(item, "text") and item.text:
            text = item.text

        if not text:
            continue

        text = str(text).strip()
        if not text:
            continue

        bbox = None
        if hasattr(item, "bbox") and item.bbox:
            bbox = item.bbox[0]

        item_type = getattr(item, "type", "text")
        block_type = str(item_type).lower()

        nested_items = []
        if hasattr(item, "items") and item.items:
            for sub in item.items:
                val = getattr(sub, "value", None) or getattr(sub, "md", None) or getattr(sub, "text", None)
                if val:
                    nested_items.append(str(val).strip())

        normalized.append({
            "text": text,
            "type": block_type,
            "items": nested_items,
            "x": float(getattr(bbox, "x", 0)) if bbox else 0.0,
            "y": float(getattr(bbox, "y", 0)) if bbox else 0.0,
            "w": float(getattr(bbox, "w", 0)) if bbox else 0.0,
            "h": float(getattr(bbox, "h", 0)) if bbox else 0.0,
            "page": int(getattr(item, "page_number", 1)),
            "column": 0,
        })
    return normalized


def detect_columns(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Detect columns per page using gutter analysis.
    """
    if not blocks:
        return blocks

    pages: Dict[int, List[Dict[str, Any]]] = {}
    for b in blocks:
        p = b.get("page", 1)
        pages.setdefault(p, []).append(b)

    for p, page_blocks in pages.items():
        if len(page_blocks) < 4:
            for b in page_blocks:
                b["column"] = 0
            continue

        min_x = min(b["x"] for b in page_blocks)
        max_x = max(b["x"] + b.get("w", 0) for b in page_blocks)
        page_width = max(max_x - min_x, 400.0)

        total_text_len = sum(len(b.get("text", "")) for b in page_blocks)
        if total_text_len == 0:
            for b in page_blocks:
                b["column"] = 0
            continue

        best_gutter = None
        min_crossing_blocks = len(page_blocks)

        step = 15.0
        start_x = min_x + 0.20 * page_width
        end_x = min_x + 0.75 * page_width

        curr_split = start_x
        while curr_split <= end_x:
            left_blocks = []
            right_blocks = []
            crossing_blocks = []

            for b in page_blocks:
                bx = b["x"]
                bw = b.get("w", 0)
                br = bx + bw

                if br <= curr_split + 5:
                    left_blocks.append(b)
                elif bx >= curr_split - 5:
                    right_blocks.append(b)
                else:
                    crossing_blocks.append(b)

            left_text_len = sum(len(b.get("text", "")) for b in left_blocks)
            right_text_len = sum(len(b.get("text", "")) for b in right_blocks)

            if (
                left_text_len >= 0.20 * total_text_len
                and right_text_len >= 0.20 * total_text_len
                and len(left_blocks) >= 3
                and len(right_blocks) >= 3
            ):
                if len(crossing_blocks) < min_crossing_blocks and len(crossing_blocks) <= max(2, int(len(page_blocks) * 0.25)):
                    min_crossing_blocks = len(crossing_blocks)
                    best_gutter = curr_split

            curr_split += step

        if best_gutter is not None:
            for b in page_blocks:
                bx = b["x"]
                bw = b.get("w", 0)
                br = bx + bw
                if bx < best_gutter and br > best_gutter + 30:
                    b["column"] = 0
                elif bx >= best_gutter - 10:
                    b["column"] = 1
                else:
                    b["column"] = 0
        else:
            for b in page_blocks:
                b["column"] = 0

    return blocks


def sort_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(blocks, key=lambda b: (b.get("page", 1), b.get("column", 0), b.get("y", 0), b.get("x", 0)))


def reconstruct_layout(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    blocks_with_cols = detect_columns(blocks)
    return sort_blocks(blocks_with_cols)


def expand_list_items(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    expanded = []
    for b in blocks:
        items = b.get("items", [])
        if b.get("type") == "list" and items:
            for item in items:
                expanded.append({
                    "text": item,
                    "type": "list_item",
                    "items": [],
                    "x": b["x"],
                    "y": b["y"],
                    "w": b["w"],
                    "h": b["h"],
                    "page": b["page"],
                    "column": b["column"],
                })
        else:
            expanded.append(b)
    return expanded


def merge_links(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged = []
    i = 0
    while i < len(blocks):
        current = blocks[i]
        if current.get("type") == "link" and (i + 1) < len(blocks):
            next_block = blocks[i + 1]
            if (next_block.get("page") == current.get("page")
                    and next_block.get("column") == current.get("column")
                    and abs(next_block.get("y", 0) - current.get("y", 0)) < 15):
                merged.append({
                    "text": f"{current['text']} ({next_block['text']})",
                    "type": "text",
                    "items": [],
                    "x": current["x"],
                    "y": current["y"],
                    "w": current["w"],
                    "h": current["h"],
                    "page": current["page"],
                    "column": current["column"],
                })
                i += 2
                continue
        merged.append(current)
        i += 1
    return merged


async def extract_with_llamaparse(file_bytes: bytes, filename: str, content_type: str) -> Dict[str, Any]:
    """
    Unified extraction entry point.
    Executes local multi-stage extraction (inspection -> layout reconstruction -> quality check -> OCR fallback).
    Falls back to LlamaCloud only if local extraction returns empty content and LlamaCloud is configured.
    """
    try:
        extracted = UniversalDocumentExtractor.extract_document(
            file_bytes=file_bytes,
            filename=filename,
            content_type=content_type,
        )
        if extracted and extracted.raw_items:
            logger.info(
                f"✓ UniversalExtractor extracted {len(extracted.raw_items)} blocks from '{filename}' "
                f"(type={extracted.document_type}, quality={extracted.quality_score:.2f}, ocr={extracted.ocr_used})"
            )
            return {
                "raw_items": extracted.raw_items,
                "raw_text": extracted.raw_text,
                "markdown": extracted.markdown,
                "document_type": extracted.document_type,
                "quality_score": extracted.quality_score,
                "ocr_used": extracted.ocr_used,
                "page_count": extracted.page_count,
                "page_images": extracted.page_images,
                "success": True,
            }
    except Exception as e:
        logger.warning(f"Universal extraction error for '{filename}': {e}. Trying fallback...")

    # ── LlamaCloud Fallback (If configured and local extraction returned empty) ──
    if LLAMA_CLOUD_AVAILABLE and LLAMA_CLOUD_API_KEY:
        try:
            logger.info(f"Invoking LlamaCloud fallback for '{filename}'...")
            client = AsyncLlamaCloud(api_key=LLAMA_CLOUD_API_KEY)
            file = await client.files.create(
                file=(filename, file_bytes, content_type),
                purpose="parse",
            )
            result = await client.parsing.parse(
                file_id=file.id,
                tier="agentic",
                version="latest",
                expand=["items"],
            )
            items = []
            if hasattr(result, "items") and result.items:
                items = result.items
            elif isinstance(result, dict) and "items" in result:
                items = result["items"]

            normalized = normalize_items(items)
            if normalized:
                ordered = reconstruct_layout(normalized)
                ordered = expand_list_items(ordered)
                ordered = merge_links(ordered)
                logger.info(f"✓ LlamaCloud extracted {len(ordered)} blocks from '{filename}'")
                return {"raw_items": ordered, "success": True}
        except Exception as e:
            logger.error(f"LlamaCloud fallback failed for '{filename}': {e}")

    # ── Fallback Raw String Extraction ──
    try:
        raw_text = file_bytes.decode("utf-8", errors="ignore").strip()
        if raw_text:
            lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
            blocks = [{
                "text": line,
                "type": "text",
                "items": [line],
                "x": 0.0,
                "y": float(idx * 15),
                "w": 500.0,
                "h": 15.0,
                "page": 1,
                "column": 0,
            } for idx, line in enumerate(lines)]
            return {"raw_items": blocks, "raw_text": raw_text, "markdown": raw_text, "success": True}
    except Exception:
        pass

    logger.error(f"Failed to extract content from '{filename}'")
    return {"raw_items": [], "raw_text": "", "markdown": "", "success": False}