# /app/modules/resume_builder/extractor.py

import os
import logging
from typing import Dict, Any
from llama_cloud import AsyncLlamaCloud

logger = logging.getLogger(__name__)

LLAMA_API_KEY = os.getenv("LLAMAPARSE_API_KEY")


async def extract_with_llamaparse(file_bytes: bytes, filename: str, content_type: str) -> Dict[str, Any]:
    """
    Upload → parse → return MARKDOWN output
    """

    try:
        client = AsyncLlamaCloud(api_key=LLAMA_API_KEY)

        file = await client.files.create(
            file=(filename, file_bytes, content_type),
            purpose="parse",
        )

        result = await client.parsing.parse(
            file_id=file.id,
            tier="agentic",
            version="latest",
            expand=["markdown"],
        )

        logger.info(f"LlamaParse success for {filename}")

        # FIXED ACCESS
        markdown_obj = getattr(result, "markdown", None)

        if not markdown_obj or not hasattr(markdown_obj, "pages"):
            raise Exception("Markdown output not found")

        full_markdown = "\n\n".join(
            page.markdown for page in markdown_obj.pages if getattr(page, "markdown", None)
        )

        return {
            "markdown": full_markdown
        }

    except Exception as e:
        logger.error(f"LlamaParse extraction failed: {repr(e)}")
        raise