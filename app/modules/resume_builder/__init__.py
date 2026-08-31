"""
resume_builder — Production-Grade AI Resume Platform Module.

Architectural Highlights:
- Gemini GenAI SDK v1.69+ (`from google import genai`) for high-precision document extraction and ATS analysis.
- Single-Call Structured Resume Parsing replacing 11-step legacy parsing pipelines (90%+ token reduction).
- Offline Deterministic AST Parser (<50ms, 0 tokens) with transparent automatic fallback.
- Explainable ATS Evaluation: 80% deterministic rubric + 20% Gemini semantic fit.
- Ollama local compute integration for suggestions, rewriting, and CV generation with zero Groq dependencies.
- In-Memory Asynchronous Background Job Manager with state tracking and TTL pruning.
- Structured AI Telemetry emitting real-time JSON usage events and cost tracking (NO database writes).
- Sliding-Window Rate Limiting and Adversarial Prompt Injection defenses.
- In-Memory Resume Snapshot Versioning and Semantic Diff Engine.
- Production Health Probes & Subsystem Diagnostics.
"""

from app.modules.resume_builder.gemini_client import (
    GeminiClient,
    get_gemini_client,
    GeminiServiceError,
    GeminiRateLimitError,
)
from app.modules.resume_builder.telemetry import (
    log_ai_usage,
    get_usage_summary,
    clear_telemetry_cache,
    sanitize_value,
)
from app.modules.resume_builder.schemas import (
    CanonicalResume,
    PersonalInformation,
    ExperienceItem,
    EducationItem,
    ProjectItem,
    CertificationItem,
    map_to_legacy_parse_dict,
    map_to_legacy_flat_dict,
)
from app.modules.resume_builder.ai_parser import ImprovedUniversalResumeParser
from app.modules.resume_builder.deterministic_parser import DeterministicResumeParser
from app.modules.resume_builder.ats_evaluator import (
    evaluate_ats_deterministically,
    evaluate_ats_semantically,
    generate_ats_composite_report,
    DeterministicATSBreakdown,
    SemanticATSBreakdown,
    ATSCompositeReport,
)
from app.modules.resume_builder.job_manager import (
    AsyncJobManager,
    JobStatus,
    JobRecord,
    get_job_manager,
)
from app.modules.resume_builder.batch_processor import BatchProcessor
from app.modules.resume_builder.model_router import (
    ModelRouter,
    ProviderType,
    TaskCategory,
    ModelPricing,
)
from app.modules.resume_builder.security_manager import SecurityManager
from app.modules.resume_builder.input_sanitizer import InputSanitizer
from app.modules.resume_builder.version_manager import (
    ResumeVersionManager,
    ResumeSnapshot,
    get_version_manager,
)
from app.modules.resume_builder.health_diagnostics import (
    HealthDiagnostics,
    SubsystemStatus,
)
from app.modules.resume_builder.router import router

__all__ = [
    # Gemini Client
    "GeminiClient",
    "get_gemini_client",
    "GeminiServiceError",
    "GeminiRateLimitError",
    # Telemetry
    "log_ai_usage",
    "get_usage_summary",
    "clear_telemetry_cache",
    "sanitize_value",
    # Schemas & Adapters
    "CanonicalResume",
    "PersonalInformation",
    "ExperienceItem",
    "EducationItem",
    "ProjectItem",
    "CertificationItem",
    "map_to_legacy_parse_dict",
    "map_to_legacy_flat_dict",
    # Parsers
    "ImprovedUniversalResumeParser",
    "DeterministicResumeParser",
    # ATS Engine
    "evaluate_ats_deterministically",
    "evaluate_ats_semantically",
    "generate_ats_composite_report",
    "DeterministicATSBreakdown",
    "SemanticATSBreakdown",
    "ATSCompositeReport",
    # Background Jobs
    "AsyncJobManager",
    "JobStatus",
    "JobRecord",
    "get_job_manager",
    "BatchProcessor",
    # Model Router
    "ModelRouter",
    "ProviderType",
    "TaskCategory",
    "ModelPricing",
    # Security & Sanitization
    "SecurityManager",
    "InputSanitizer",
    # Versioning
    "ResumeVersionManager",
    "ResumeSnapshot",
    "get_version_manager",
    # Health Diagnostics
    "HealthDiagnostics",
    "SubsystemStatus",
    # Modular Router
    "router",
]
