"""
model_router.py — Intelligent Multi-Tier Model Router & Configurable Cost Calculator.

Responsibilities:
- Routes tasks strictly to their designated AI provider tier:
  - Gemini: Resume parsing, document understanding, and ATS semantic analysis.
  - Ollama: Experience, summary, project, and education suggestions, plus CV generation.
  - Deterministic: Zero-token AST parsing and rule evaluations.
- Dynamically computes estimated monetary costs based on configurable pricing tables without database dependencies.
"""

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

logger = logging.getLogger("resume_builder.model_router")


class ProviderType(str, Enum):
    GEMINI = "gemini"
    OLLAMA = "ollama"
    DETERMINISTIC = "deterministic"


class TaskCategory(str, Enum):
    PARSING = "resume_parsing"
    ATS_SEMANTIC = "ats_semantic_analysis"
    EXPERIENCE_SUGGESTION = "suggest_experience"
    SUMMARY_SUGGESTION = "suggest_summary"
    PROJECT_SUGGESTION = "suggest_project"
    EDUCATION_SUGGESTION = "suggest_education"
    SKILLS_SUGGESTION = "generate_skills"
    CV_GENERATION = "generate_professional_cv"
    TARGETED_CV = "generate_targeted_cv"
    COVER_LETTER = "generate_cover_letter"
    REFINEMENT = "refine_resume_section"


@dataclass
class ModelPricing:
    input_cost_per_m: float
    output_cost_per_m: float
    cached_cost_per_m: float = 0.0


# Default pricing per 1 Million tokens (USD)
DEFAULT_PRICING_TABLE: Dict[str, ModelPricing] = {
    "gemini-3.1-flash-lite": ModelPricing(
        input_cost_per_m=float(os.getenv("GEMINI_INPUT_COST_PER_M", "0.10")),
        output_cost_per_m=float(os.getenv("GEMINI_OUTPUT_COST_PER_M", "0.40")),
        cached_cost_per_m=0.025,
    ),
    "gemini-2.0-flash": ModelPricing(
        input_cost_per_m=0.10,
        output_cost_per_m=0.40,
        cached_cost_per_m=0.025,
    ),
    "gemini-1.5-flash": ModelPricing(
        input_cost_per_m=0.075,
        output_cost_per_m=0.30,
        cached_cost_per_m=0.01875,
    ),
    "gemini-1.5-pro": ModelPricing(
        input_cost_per_m=1.25,
        output_cost_per_m=5.00,
        cached_cost_per_m=0.3125,
    ),
    # Ollama is locally hosted / self-hosted compute
    "ollama": ModelPricing(input_cost_per_m=0.0, output_cost_per_m=0.0),
    "ollama-default": ModelPricing(input_cost_per_m=0.0, output_cost_per_m=0.0),
    "gemma4:31b-cloud": ModelPricing(input_cost_per_m=0.0, output_cost_per_m=0.0),
    "llama3.1:8b": ModelPricing(input_cost_per_m=0.0, output_cost_per_m=0.0),
}


class ModelRouter:
    """
    Centralized router resolving provider tier, target model, and cost calculation.
    """

    @classmethod
    def resolve_provider(cls, task: TaskCategory) -> Tuple[ProviderType, str]:
        """
        Determines the designated provider and target model for a given task.
        """
        if task in (TaskCategory.PARSING, TaskCategory.ATS_SEMANTIC):
            model = os.getenv("GEMINI_RESUME_MODEL", "gemini-3.1-flash-lite")
            return (ProviderType.GEMINI, model)

        # All suggestion and CV workflows route to Ollama
        ollama_model = os.getenv("OLLAMA_MODEL", os.getenv("LLAMA_MODEL", "ollama-default"))
        return (ProviderType.OLLAMA, ollama_model)

    @classmethod
    def calculate_cost(
        cls,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
    ) -> float:
        """
        Computes the estimated cost in USD for a model invocation.
        Returns 0.0 for Ollama or unpriced models.
        """
        if provider.lower() in ("ollama", "deterministic"):
            return 0.0

        model_key = model.lower().strip()
        pricing = DEFAULT_PRICING_TABLE.get(model_key)
        if not pricing:
            # Fallback to default flash pricing for any generic gemini model
            if "gemini" in model_key:
                pricing = DEFAULT_PRICING_TABLE.get("gemini-2.0-flash")
            else:
                return 0.0

        if not pricing:
            return 0.0

        uncached_input = max(0, input_tokens - cached_tokens)
        input_cost = (uncached_input / 1_000_000.0) * pricing.input_cost_per_m
        cached_cost = (cached_tokens / 1_000_000.0) * pricing.cached_cost_per_m
        output_cost = (output_tokens / 1_000_000.0) * pricing.output_cost_per_m

        total_cost = round(input_cost + cached_cost + output_cost, 6)
        return total_cost
