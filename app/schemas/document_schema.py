"""
Document schema definitions.
"""
from pydantic import BaseModel
from typing import Optional


class DocumentBase(BaseModel):
    filename: str
    content_type: str
    size_bytes: int


__all__ = ["DocumentBase"]
