# /home/aryu_user/Arun/aiproject_staging/app/modules/resume_builder/extractor.py
import os
import logging
from typing import Dict, Any, List
from llama_cloud import AsyncLlamaCloud

logger = logging.getLogger(__name__)

LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")


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
        block_type = item_type if isinstance(item_type, str) else str(item_type)
        block_type = block_type.lower()

        nested_items = []
        if hasattr(item, "items") and item.items:
            for sub in item.items:
                val = ""

                if hasattr(sub, "value") and sub.value:
                    val = sub.value
                elif hasattr(sub, "md") and sub.md:
                    val = sub.md
                elif hasattr(sub, "text") and sub.text:
                    val = sub.text

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
            "column": 0
        })

    return normalized


def detect_columns(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Detect number of columns using x-coordinate clustering"""
    if not blocks:
        return blocks

    x_positions = sorted(set(b["x"] for b in blocks if b["x"] > 0))

    if len(x_positions) <= 1:
        for b in blocks:
            b["column"] = 0
        return blocks

    gaps = [x_positions[i+1] - x_positions[i] for i in range(len(x_positions)-1)]

    if not gaps:
        for b in blocks:
            b["column"] = 0
        return blocks

    threshold = max(gaps) * 0.5

    current_col = 0
    col_map = {}

    for i, x in enumerate(x_positions):
        if i > 0 and (x - x_positions[i-1]) > threshold:
            current_col += 1
        col_map[x] = current_col

    for b in blocks:
        if b["x"] in col_map:
            b["column"] = col_map[b["x"]]
        else:
            closest_x = min(col_map.keys(), key=lambda k: abs(k - b["x"])) if col_map else 0
            b["column"] = col_map.get(closest_x, 0)

    return blocks


def merge_fragments(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge blocks on same line (same y, same column, same page)"""
    if not blocks:
        return blocks

    merged = []
    current = blocks[0]

    for b in blocks[1:]:
        if (
            abs(b["y"] - current["y"]) < 10 and
            b["column"] == current["column"] and
            b["page"] == current["page"]
            and current["type"] != "heading"
            and b["type"] != "heading"
        ):
            current["text"] += " " + b["text"]
        else:
            merged.append(current)
            current = b

    merged.append(current)
    return merged


def expand_list_items(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Expand list items into individual blocks while preserving order"""
    expanded = []

    for b in blocks:
        expanded.append(b)

        if b.get("type") == "list":
            items = b.get("items", [])
            
            for idx, item in enumerate(items):
                if isinstance(item, dict):
                    val = item.get("value") or item.get("md") or item.get("text") or ""
                else:
                    val = getattr(item, "value", None) or getattr(item, "md", None) or ""

                if val and isinstance(val, str):
                    val = val.strip()
                    if val:
                        expanded.append({
                            "text": val,
                            "type": "text",
                            "x": b.get("x", 0),
                            "y": b.get("y", 0) + (idx * 0.01),
                            "w": b.get("w", 0),
                            "h": b.get("h", 0),
                            "column": b.get("column", 0),
                            "page": b.get("page", 1)
                        })

    return expanded


def merge_links(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge email/link text fragments that are split across blocks"""
    if not blocks:
        return blocks

    merged = []
    i = 0

    while i < len(blocks):
        b = blocks[i]

        if i + 1 < len(blocks):
            next_b = blocks[i + 1]

            if (
                b["type"] == "text"
                and next_b["type"] == "text"
                and abs(b["y"] - next_b["y"]) < 5
                and b["column"] == next_b["column"]
                and b["page"] == next_b["page"]
            ):
                if "@" in next_b["text"] or "linkedin" in next_b["text"].lower() or "github" in next_b["text"].lower():
                    b["text"] = b["text"] + " " + next_b["text"]
                    i += 1

        merged.append(b)
        i += 1

    return merged


def reconstruct_layout(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Reconstruct proper reading order: page → column → y position"""
    blocks = detect_columns(blocks)
    blocks.sort(key=lambda b: (b["page"], b["column"], b["y"]))
    return merge_fragments(blocks)


def extract_items_from_result(result) -> List:
    """Extract items from LlamaParse result"""
    if not result:
        return []

    items_obj = getattr(result, "items", None)
    if not items_obj:
        return []

    pages = getattr(items_obj, "pages", None)
    if not pages:
        return []

    flat_items = []

    for page in pages:
        page_items = getattr(page, "items", [])
        page_num = getattr(page, "page_number", 1)
        
        for item in page_items:
            if not hasattr(item, "page_number"):
                setattr(item, "page_number", page_num)
            flat_items.append(item)

    return flat_items


async def extract_with_llamaparse(file_bytes: bytes, filename: str, content_type: str) -> Dict[str, Any]:
    """
    Extract resume using LlamaParse
    
    Returns:
        {
            "raw_items": [normalized blocks],
            "success": bool
        }
    """
    try:
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

        items = extract_items_from_result(result)
        
        if not items:
            logger.error("No items extracted from LlamaParse")
            return {"raw_items": [], "success": False}

        normalized = normalize_items(items)

        if not normalized:
            logger.error("No items after normalization")
            return {"raw_items": [], "success": False}

        ordered_blocks = reconstruct_layout(normalized)
        ordered_blocks = expand_list_items(ordered_blocks)
        ordered_blocks = merge_links(ordered_blocks)

        logger.info(f"Extracted {len(ordered_blocks)} blocks from {filename}")

        return {
            "raw_items": ordered_blocks,
            "success": True
        }

    except Exception as e:
        logger.error(f"LlamaParse extraction failed: {repr(e)}")
        return {"raw_items": [], "success": False}
    
    