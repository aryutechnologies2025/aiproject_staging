import logging
import json
from datetime import datetime
from typing import Dict, Any
import hashlib

logger = logging.getLogger(__name__)


class SecurityAuditLogger:
    """
    Log security events for monitoring and auditing
    """
    
    @staticmethod
    def log_file_upload(
        filename: str,
        file_size: int,
        client_ip: str,
        user_id: str,
        status: str,
        details: str = ""
    ):
        """
        Log file upload attempt
        """
        
        audit_entry = {
            "event": "file_upload",
            "timestamp": datetime.utcnow().isoformat(),
            "filename_hash": hashlib.sha256(filename.encode()).hexdigest()[:8],
            "file_size": file_size,
            "client_ip": client_ip,
            "user_id_hash": hashlib.sha256(user_id.encode()).hexdigest()[:8],
            "status": status,  # success, failure, blocked
            "details": details
        }
        
        logger.info(f"AUDIT: {json.dumps(audit_entry)}")
    
    @staticmethod
    def log_security_violation(
        violation_type: str,
        client_ip: str,
        user_id: str,
        details: str
    ):
        """
        Log security violations
        """
        
        audit_entry = {
            "event": "security_violation",
            "timestamp": datetime.utcnow().isoformat(),
            "violation_type": violation_type,  # malware, injection, rate_limit, etc
            "client_ip": client_ip,
            "user_id_hash": hashlib.sha256(user_id.encode()).hexdigest()[:8],
            "details": details
        }
        
        logger.warning(f"SECURITY: {json.dumps(audit_entry)}")
    
    @staticmethod
    def log_suspicious_activity(
        activity_type: str,
        client_ip: str,
        evidence: str
    ):
        """
        Log suspicious activity
        """
        
        audit_entry = {
            "event": "suspicious_activity",
            "timestamp": datetime.utcnow().isoformat(),
            "activity_type": activity_type,
            "client_ip": client_ip,
            "evidence": evidence
        }
        
        logger.warning(f"SUSPICIOUS: {json.dumps(audit_entry)}")
        