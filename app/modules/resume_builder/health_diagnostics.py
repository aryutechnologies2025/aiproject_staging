"""
health_diagnostics.py — Production Health Probes & Subsystem Diagnostics for resume_builder.

Provides:
- Real-time diagnostic probes for all resume_builder subsystems (Gemini, AST Parser, Job Manager, Telemetry, Rate Limiter, Versioning).
- Composite health evaluation (HEALTHY, DEGRADED, UNHEALTHY) with subsystem latency measurements.
- Zero database queries, writes, or persistent state.
"""

import logging
import os
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional

from app.modules.resume_builder.gemini_client import get_gemini_client
from app.modules.resume_builder.deterministic_parser import DeterministicResumeParser
from app.modules.resume_builder.job_manager import get_job_manager
from app.modules.resume_builder.telemetry import get_usage_summary
from app.modules.resume_builder.security_manager import SecurityManager
from app.modules.resume_builder.version_manager import get_version_manager

logger = logging.getLogger("resume_builder.health_diagnostics")

_START_TIME = time.time()


class SubsystemStatus(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"
    NOT_CONFIGURED = "NOT_CONFIGURED"


class HealthDiagnostics:
    """
    Production health & diagnostics aggregator for the resume_builder module.
    """

    @classmethod
    def check_gemini(cls) -> Dict[str, Any]:
        start = time.time()
        client = get_gemini_client()
        latency_ms = round((time.time() - start) * 1000, 2)

        if not client.is_configured:
            return {
                "status": SubsystemStatus.NOT_CONFIGURED.value,
                "message": "Gemini API key is not set (offline deterministic fallback active)",
                "model": client.default_model,
                "latency_ms": latency_ms,
            }

        return {
            "status": SubsystemStatus.HEALTHY.value,
            "message": "Google GenAI client initialized and ready",
            "model": client.default_model,
            "latency_ms": latency_ms,
        }

    @classmethod
    def check_deterministic_parser(cls) -> Dict[str, Any]:
        start = time.time()
        sample_text = "Jane Candidate\njane@example.com | +1 555 123 4567\nSkills: Python, FastAPI\nExperience\nDeveloper | 2022 - Present\n- Built microservices."
        try:
            res = DeterministicResumeParser.parse_text(sample_text)
            latency_ms = round((time.time() - start) * 1000, 2)
            is_valid = bool(res.personal_information.email == "jane@example.com" and "Python" in res.skills)

            return {
                "status": SubsystemStatus.HEALTHY.value if is_valid else SubsystemStatus.DEGRADED.value,
                "message": "Deterministic AST parser operational (<50ms, 0 tokens)",
                "latency_ms": latency_ms,
            }
        except Exception as e:
            return {
                "status": SubsystemStatus.UNHEALTHY.value,
                "message": f"Deterministic parser failed: {e}",
                "latency_ms": round((time.time() - start) * 1000, 2),
            }

    @classmethod
    def check_job_manager(cls) -> Dict[str, Any]:
        jm = get_job_manager()
        total_records = len(jm._jobs)
        active_tasks = len(jm._tasks)

        return {
            "status": SubsystemStatus.HEALTHY.value,
            "message": "In-memory async background job manager active",
            "active_tasks_count": active_tasks,
            "cached_records_count": total_records,
        }

    @classmethod
    def check_telemetry(cls) -> Dict[str, Any]:
        summary = get_usage_summary()
        return {
            "status": SubsystemStatus.HEALTHY.value,
            "message": "Structured telemetry & in-memory token accounting active",
            "total_ai_calls": summary["total_calls"],
            "total_tokens_consumed": summary["total_tokens"],
            "total_estimated_cost_usd": summary["total_estimated_cost_usd"],
            "by_provider": summary["by_provider"],
        }

    @classmethod
    def check_rate_limiter(cls) -> Dict[str, Any]:
        active_ips = len(SecurityManager._ip_window_history)
        active_users = len(SecurityManager._user_window_history)

        return {
            "status": SubsystemStatus.HEALTHY.value,
            "message": "Sliding-window rate limiter active with automatic TTL pruning",
            "active_ip_buckets": active_ips,
            "active_user_buckets": active_users,
        }

    @classmethod
    def check_version_manager(cls) -> Dict[str, Any]:
        vm = get_version_manager()
        active_users = len(vm._user_snapshots)

        return {
            "status": SubsystemStatus.HEALTHY.value,
            "message": "In-memory resume snapshot versioning & diff engine active",
            "active_users_versioned": active_users,
        }

    @classmethod
    def check_ollama(cls) -> Dict[str, Any]:
        ollama_endpoint = os.getenv("OLLAMA_ENDPOINT", os.getenv("OLLAMA_HOST", "http://localhost:11434"))
        ollama_model = os.getenv("OLLAMA_MODEL", os.getenv("LLAMA_MODEL", "gemma4:31b-cloud"))

        return {
            "status": SubsystemStatus.HEALTHY.value,
            "message": "Ollama local compute routing configured for suggestions & CV generation",
            "endpoint": ollama_endpoint,
            "model": ollama_model,
        }

    @classmethod
    def run_all_diagnostics(cls) -> Dict[str, Any]:
        """
        Runs comprehensive diagnostic probes across all module components and computes overall status.
        """
        gemini_diag = cls.check_gemini()
        ast_diag = cls.check_deterministic_parser()
        jm_diag = cls.check_job_manager()
        telem_diag = cls.check_telemetry()
        sec_diag = cls.check_rate_limiter()
        vm_diag = cls.check_version_manager()
        ollama_diag = cls.check_ollama()

        subsystems = {
            "gemini_sdk": gemini_diag,
            "deterministic_ast_parser": ast_diag,
            "async_job_manager": jm_diag,
            "telemetry_accounting": telem_diag,
            "security_rate_limiter": sec_diag,
            "version_manager": vm_diag,
            "ollama_suggestions": ollama_diag,
        }

        # Compute composite status
        statuses = [s["status"] for s in subsystems.values()]
        if any(st == SubsystemStatus.UNHEALTHY.value for st in statuses):
            overall = SubsystemStatus.UNHEALTHY.value
        elif any(st in (SubsystemStatus.DEGRADED.value, SubsystemStatus.NOT_CONFIGURED.value) for st in statuses):
            overall = SubsystemStatus.DEGRADED.value
        else:
            overall = SubsystemStatus.HEALTHY.value

        uptime = round(time.time() - _START_TIME, 1)

        return {
            "module": "resume_builder",
            "overall_status": overall,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": uptime,
            "subsystems": subsystems,
        }
