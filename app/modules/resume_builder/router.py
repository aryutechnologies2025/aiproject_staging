"""
router.py — Modular FastAPI router exposing production endpoints for resume_builder.

Endpoints:
- GET  /diagnostics            - Live subsystem health probes & latencies.
- GET  /telemetry/summary      - In-memory AI token & cost accounting.
- POST /jobs/submit            - Enqueue async background jobs.
- GET  /jobs/{job_id}          - Poll background job status & results.
- GET  /versions               - List candidate resume snapshot history.
- POST /versions/diff          - Structured semantic diff between snapshots.
- POST /versions/rollback      - In-memory state restoration.
- POST /ats/evaluate-composite - Hybrid explainable ATS evaluation report.
"""

import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.modules.resume_builder.health_diagnostics import HealthDiagnostics
from app.modules.resume_builder.telemetry import get_usage_summary
from app.modules.resume_builder.job_manager import get_job_manager, JobStatus
from app.modules.resume_builder.version_manager import get_version_manager
from app.modules.resume_builder.ats_evaluator import generate_ats_composite_report
from app.modules.resume_builder.schemas import CanonicalResume

logger = logging.getLogger("resume_builder.router")

router = APIRouter(prefix="", tags=["Resume Builder Modular"])


# ── Request Models ──────────────────────────────────────────────────────────

class JobSubmitRequest(BaseModel):
    job_type: str = Field(..., description="Job identifier e.g. parse_resume, cv_generation")
    user_id: Optional[str] = Field("anonymous", description="User identifier")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Job parameters")


class VersionDiffRequest(BaseModel):
    user_id: str
    version_id_1: str
    version_id_2: str


class VersionRollbackRequest(BaseModel):
    user_id: str
    target_version_id: str


class CompositeATSEvalRequest(BaseModel):
    resume: CanonicalResume
    job_description: str
    include_semantic: bool = True
    user_id: Optional[str] = "anonymous"


# ── Endpoints ───────────────────────────────────────────────────────────────

@router.get("/diagnostics", summary="Subsystem Health Probes")
async def get_diagnostics():
    """Returns real-time health probes and latency measurements across all subsystems."""
    return HealthDiagnostics.run_all_diagnostics()


@router.get("/telemetry/summary", summary="AI Usage & Cost Telemetry")
async def get_telemetry(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    operation: Optional[str] = Query(None, description="Filter by operation name"),
):
    """Returns in-memory token accounting, provider breakdown, and estimated USD costs."""
    return get_usage_summary(user_id=user_id, operation=operation)


@router.post("/jobs/submit", summary="Submit Async Background Job", status_code=status.HTTP_202_ACCEPTED)
async def submit_job(req: JobSubmitRequest):
    """Submits an asynchronous task to the in-memory background worker queue."""
    jm = get_job_manager()

    async def _dummy_task_runner(payload: Dict[str, Any]):
        return {"processed": True, "input_keys": list(payload.keys())}

    job_id = await jm.submit_job(
        job_type=req.job_type,
        coro_fn=_dummy_task_runner,
        payload=req.payload,
        user_id=req.user_id,
    )
    return {"job_id": job_id, "status": "QUEUED"}


@router.get("/jobs/{job_id}", summary="Poll Background Job Status")
async def get_job_status(job_id: str):
    """Retrieves current job status, execution progress, latency, or result."""
    jm = get_job_manager()
    job_dict = jm.get_job_dict(job_id)
    if not job_dict:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job '{job_id}' not found")
    return job_dict


@router.get("/versions", summary="List Resume Snapshots")
async def list_versions(user_id: str = Query(..., description="User ID to list versions for")):
    """Returns the snapshot history for a given user (up to 10 snapshots)."""
    vm = get_version_manager()
    return {"user_id": user_id, "snapshots": vm.list_snapshots(user_id)}


@router.post("/versions/diff", summary="Diff Resume Snapshots")
async def diff_versions(req: VersionDiffRequest):
    """Computes structured semantic diff between two snapshot versions."""
    vm = get_version_manager()
    v1 = vm.get_snapshot(req.user_id, req.version_id_1)
    v2 = vm.get_snapshot(req.user_id, req.version_id_2)
    if not v1 or not v2:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="One or both snapshot versions not found for this user",
        )
    return vm.diff_snapshots(v1, v2)


@router.post("/versions/rollback", summary="Rollback Resume Snapshot")
async def rollback_version(req: VersionRollbackRequest):
    """Restores a previous snapshot version in-memory."""
    vm = get_version_manager()
    result = vm.rollback(req.user_id, req.target_version_id)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Target version '{req.target_version_id}' not found for rollback",
        )
    return result


@router.post("/ats/evaluate-composite", summary="Composite ATS Evaluation")
async def evaluate_ats_composite(req: CompositeATSEvalRequest):
    """Runs hybrid ATS evaluation (80% deterministic rubric + 20% Gemini semantic analysis)."""
    report = await generate_ats_composite_report(
        resume=req.resume,
        job_description=req.job_description,
        include_semantic=req.include_semantic,
        user_id=req.user_id,
    )
    return report.to_dict()
