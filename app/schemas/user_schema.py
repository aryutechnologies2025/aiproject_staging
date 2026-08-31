"""
User schema definitions.
"""
from pydantic import BaseModel, EmailStr
from typing import Optional


class UserBase(BaseModel):
    id: Optional[str] = None
    email: Optional[str] = None
    name: Optional[str] = None


__all__ = ["UserBase"]
