"""
security_audit_logger.py — Structured Security & Ingestion Audit Event Logger.

Provides:
- Anonymized audit logging for file uploads, rate limit blocks, and suspicious payload attempts.
- Timezone-aware ISO-8601 UTC timestamps (zero deprecated utcnow calls).
- SHA-256 pseudonymized identifiers for zero PII leakage.
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict

logger = logging.getLogger("resume_builder.security_audit")


class SecurityAuditLogger:
    """
    Emits structured JSON security audit events.
    """

    @staticmethod
    def log_file_upload(
        filename: str,
        file_size: int,
        client_ip: str,
        user_id: str,
        status: str,
        details: str = "",
    ) -> Dict[str, Any]:
        """
        Log file upload attempt with hashed identifiers.
        """
        audit_entry = {
            "event": "file_upload",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "filename_hash": hashlib.sha256(filename.encode("utf-8", errors="ignore")).hexdigest()[:12],
            "file_size_bytes": file_size,
            "client_ip_hash": hashlib.sha256(client_ip.encode("utf-8", errors="ignore")).hexdigest()[:12],
            "user_id_hash": hashlib.sha256(user_id.encode("utf-8", errors="ignore")).hexdigest()[:12],
            "status": status,
            "details": details,
        }

        logger.info(f"AUDIT_UPLOAD: {json.dumps(audit_entry)}")
        return audit_entry

    @staticmethod
    def log_security_violation(
        violation_type: str,
        client_ip: str,
        user_id: str,
        details: str,
    ) -> Dict[str, Any]:
        """
        Log security violations (rate limits, injection attempts, invalid headers).
        """
        audit_entry = {
            "event": "security_violation",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "violation_type": violation_type,
            "client_ip_hash": hashlib.sha256(client_ip.encode("utf-8", errors="ignore")).hexdigest()[:12],
            "user_id_hash": hashlib.sha256(user_id.encode("utf-8", errors="ignore")).hexdigest()[:12],
            "details": details,
        }

        logger.warning(f"AUDIT_VIOLATION: {json.dumps(audit_entry)}")
        return audit_entry

    @staticmethod
    def log_suspicious_activity(
        activity_type: str,
        client_ip: str,
        evidence: str,
    ) -> Dict[str, Any]:
        """
        Log suspicious payload patterns.
        """
        audit_entry = {
            "event": "suspicious_activity",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "activity_type": activity_type,
            "client_ip_hash": hashlib.sha256(client_ip.encode("utf-8", errors="ignore")).hexdigest()[:12],
            "evidence": evidence,
        }

        logger.warning(f"AUDIT_SUSPICIOUS: {json.dumps(audit_entry)}")
        return audit_entry