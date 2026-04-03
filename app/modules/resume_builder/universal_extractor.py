import json
import logging
import re
from typing import Dict, List, Any, Tuple

logger = logging.getLogger(__name__)


class UniversalExtractor:
    """
    Universal content extractor that works without assumptions about resume structure.
    Extracts raw content blocks and identifies section boundaries through intelligent analysis.
    """
    
    @staticmethod
    def extract_all_content(raw_items: List[Dict[str, Any]]) -> str:
        """
        Extract all content preserving structure but without format assumptions.
        Handles: markdown, plain text, multi-column, any layout, any font
        """
        if not raw_items:
            return ""
        
        # Preserve raw text exactly as provided
        content_blocks = []
        
        for item in raw_items:
            text = item.get("text", "").strip()
            block_type = item.get("type", "text")
            
            if not text:
                continue
            
            # Preserve all text with type markers for later analysis
            content_blocks.append({
                "text": text,
                "type": block_type,
                "items": item.get("items", [])
            })
        
        # Build raw content without any filtering
        raw_content = "\n".join([b["text"] for b in content_blocks])
        
        return raw_content
    
    @staticmethod
    def get_all_items_flat(raw_items: List[Dict[str, Any]]) -> List[str]:
        """Get all text items as flat list for LLM analysis"""
        items = []
        
        for item in raw_items:
            text = item.get("text", "").strip()
            if text:
                items.append(text)
            
            # Also include nested items from lists
            nested = item.get("items", [])
            if nested:
                for nested_item in nested:
                    if isinstance(nested_item, str) and nested_item.strip():
                        items.append(nested_item.strip())
        
        return items
    
    @staticmethod
    def extract_contact_info_raw(raw_items: List[Dict[str, Any]]) -> Dict[str, str]:
        """Extract contact info without LLM - regex based fallback"""
        all_text = "\n".join([item.get("text", "") for item in raw_items])
        
        email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        phone_pattern = r"(?:\+?(?:91|1|44|86|81|33|39|34|49|31|46|45|47|358|41|43|43))?[\s.-]?(?:\(?\d{1,5}\)?[\s.-]?\d{1,5}[\s.-]?\d{1,5})"
        linkedin_pattern = r"linkedin\.com/in/[\w-]+"
        github_pattern = r"github\.com/[\w-]+"
        portfolio_pattern = r"[\w-]+\.(?:com|net|io|app|netlify|vercel|github\.io)"
        
        email_match = re.search(email_pattern, all_text)
        phone_matches = re.findall(phone_pattern, all_text)
        linkedin_match = re.search(linkedin_pattern, all_text, re.IGNORECASE)
        github_match = re.search(github_pattern, all_text, re.IGNORECASE)
        portfolio_match = re.search(portfolio_pattern, all_text)
        
        return {
            "email": email_match.group(0) if email_match else "",
            "phone": phone_matches[0] if phone_matches else "",
            "linkedin": linkedin_match.group(0) if linkedin_match else "",
            "github": github_match.group(0) if github_match else "",
            "portfolio": portfolio_match.group(0) if portfolio_match else "",
        }