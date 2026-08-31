"""
Message schema definitions.
"""
from pydantic import BaseModel
from typing import Optional


class MessageBase(BaseModel):
    role: str
    content: str


__all__ = ["MessageBase"]
