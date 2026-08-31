"""
telemetry.py — Structured AI telemetry, token accounting, and cost observability for resume_builder.

Features:
- Emits structured JSON log records for every AI call (Gemini and Ollama) with estimated USD costs.
- In-memory aggregation for user-level, provider-level, and operation-level accounting (NO database writes).
- Strict PII filtering: Never logs raw resume text, names, emails, phones, or API keys.
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import uuid

from app.modules.resume_builder.model_router import ModelRouter

logger = logging.getLogger("resume_builder.telemetry")

# In-memory token metrics store (Process lifetime, zero database dependency)
_USAGE_RECORDS: List[Dict[str, Any]] = []
_MAX_IN_MEMORY_RECORDS = 5000


def sanitize_value(val: Any) -> Any:
    """Ensure no raw PII or nested sensitive structures leak into telemetry logs."""
    if isinstance(val, (int, float, bool)) or val is None:
        return val
    s = str(val)
    # Mask emails
    if "@" in s and "." in s:
        return "[MASKED_EMAIL]"
    # Mask long text payloads
    if len(s) > 120:
        return s[:40] + "...[TRUNCATED]"
    return s


def log_ai_usage(
    *,
    provider: str,
    model: str,
    operation: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    total_tokens: int = 0,
    cached_tokens: int = 0,
    thinking_tokens: int = 0,
    latency_ms: float = 0.0,
    status: str = "success",
    error_type: Optional[str] = None,
    retry_count: int = 0,
    user_id: Optional[str] = None,
    request_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
    endpoint: Optional[str] = None,
    workflow: Optional[str] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Log an AI usage event to structured JSON logger and in-memory aggregator with dynamic cost calculation.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    req_id = request_id or str(uuid.uuid4())
    uid = user_id or "anonymous"

    calculated_total = total_tokens if total_tokens > 0 else (input_tokens + output_tokens)

    # Compute estimated monetary cost
    estimated_cost_usd = ModelRouter.calculate_cost(
        provider=provider,
        model=model,
        input_tokens=int(input_tokens),
        output_tokens=int(output_tokens),
        cached_tokens=int(cached_tokens),
    )

    record = {
        "event": "ai_usage",
        "timestamp": now_iso,
        "request_id": req_id,
        "correlation_id": correlation_id or req_id,
        "user_id": uid,
        "provider": provider.lower(),
        "model": model,
        "operation": operation,
        "endpoint": endpoint or "unknown",
        "workflow": workflow or "resume_builder",
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "total_tokens": int(calculated_total),
        "cached_tokens": int(cached_tokens),
        "thinking_tokens": int(thinking_tokens),
        "estimated_cost_usd": estimated_cost_usd,
        "latency_ms": round(float(latency_ms), 2),
        "status": status,
        "error_type": error_type,
        "retry_count": int(retry_count),
    }

    if extra_meta:
        record["meta"] = {k: sanitize_value(v) for k, v in extra_meta.items()}

    # Structured JSON log output
    logger.info(json.dumps(record))

    # Append to in-memory store with bounded size
    global _USAGE_RECORDS
    if len(_USAGE_RECORDS) >= _MAX_IN_MEMORY_RECORDS:
        _USAGE_RECORDS = _USAGE_RECORDS[-int(_MAX_IN_MEMORY_RECORDS * 0.8):]
    _USAGE_RECORDS.append(record)

    return record


def get_usage_summary(user_id: Optional[str] = None, operation: Optional[str] = None) -> Dict[str, Any]:
    """
    Aggregates in-memory usage and cost metrics per user, operation, or overall.
    Zero database queries or table persistence required.
    """
    records = _USAGE_RECORDS
    if user_id:
        records = [r for r in records if r.get("user_id") == user_id]
    if operation:
        records = [r for r in records if r.get("operation") == operation]

    total_calls = len(records)
    successful_calls = sum(1 for r in records if r.get("status") == "success")
    failed_calls = sum(1 for r in records if r.get("status") != "success")

    total_input = sum(r.get("input_tokens", 0) for r in records)
    total_output = sum(r.get("output_tokens", 0) for r in records)
    total_tokens = sum(r.get("total_tokens", 0) for r in records)
    total_cached = sum(r.get("cached_tokens", 0) for r in records)
    total_cost_usd = round(sum(r.get("estimated_cost_usd", 0.0) for r in records), 6)

    latencies = [r.get("latency_ms", 0.0) for r in records if r.get("latency_ms", 0.0) > 0]
    avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0.0

    # Group by provider
    by_provider: Dict[str, Dict[str, Any]] = {}
    for r in records:
        prov = r.get("provider", "unknown")
        if prov not in by_provider:
            by_provider[prov] = {"calls": 0, "tokens": 0, "cost_usd": 0.0, "errors": 0}
        by_provider[prov]["calls"] += 1
        by_provider[prov]["tokens"] += r.get("total_tokens", 0)
        by_provider[prov]["cost_usd"] = round(by_provider[prov]["cost_usd"] + r.get("estimated_cost_usd", 0.0), 6)
        if r.get("status") != "success":
            by_provider[prov]["errors"] += 1

    return {
        "filter": {"user_id": user_id, "operation": operation},
        "total_calls": total_calls,
        "successful_calls": successful_calls,
        "failed_calls": failed_calls,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_tokens": total_tokens,
        "total_cached_tokens": total_cached,
        "total_estimated_cost_usd": total_cost_usd,
        "average_latency_ms": avg_latency,
        "by_provider": by_provider,
    }


def clear_telemetry_cache() -> int:
    """Clear in-memory telemetry records (for testing)."""
    global _USAGE_RECORDS
    count = len(_USAGE_RECORDS)
    _USAGE_RECORDS = []
    return count
