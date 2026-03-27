# /app/modules/resume_builder/parser.py

from typing import Dict, Any


def parse_llama_to_resume_json(llama_response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return markdown response
    """

    return {
        "markdown": llama_response.get("markdown", ""),
        "metadata": {
            "source": "llamaparse"
        }
    }