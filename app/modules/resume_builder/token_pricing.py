"""
token_pricing.py — Precise Financial Token Cost Calculator for AI Generations.

Provides:
- Decimal-precision cost calculations (no floating point rounding drift).
- Official Google Gemini model pricing in USD per 1M tokens with configurable overrides.
- Configurable USD-to-INR currency conversion via GEMINI_USD_TO_INR.
- Cached token cost calculation matching Google Gemini pricing specifications.
- Safe handling of unknown or unconfigured models (no invented prices).
- Clean dictionary serialization for API responses and telemetry logging.
"""

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger("resume_builder.token_pricing")

# Default USD to INR exchange rate (Configurable via environment)
DEFAULT_USD_TO_INR = Decimal("86.5")

# 1 Million tokens constant
ONE_MILLION = Decimal("1000000")


@dataclass
class GeminiModelPriceConfig:
    """Official Gemini pricing rates in USD per 1 Million tokens."""
    input_usd_per_m: Decimal
    output_usd_per_m: Decimal
    cached_usd_per_m: Decimal = Decimal("0.0")


def get_usd_to_inr_rate() -> Decimal:
    """Returns the configured USD-to-INR conversion rate as Decimal."""
    rate_str = os.getenv("GEMINI_USD_TO_INR", "").strip()
    if rate_str:
        try:
            return Decimal(rate_str)
        except Exception as e:
            logger.warning(f"Invalid GEMINI_USD_TO_INR '{rate_str}': {e}. Using default {DEFAULT_USD_TO_INR}")
    return DEFAULT_USD_TO_INR


def get_official_gemini_pricing_table() -> Dict[str, GeminiModelPriceConfig]:
    """
    Returns official Gemini pricing table in USD per 1M tokens.
    Supports environment variable overrides for custom tiers or dynamic rate updates.
    """
    custom_input = os.getenv("GEMINI_INPUT_COST_PER_M")
    custom_output = os.getenv("GEMINI_OUTPUT_COST_PER_M")
    custom_cached = os.getenv("GEMINI_CACHED_COST_PER_M")

    flash_lite_input = Decimal(custom_input) if custom_input else Decimal("0.075")
    flash_lite_output = Decimal(custom_output) if custom_output else Decimal("0.30")
    flash_lite_cached = Decimal(custom_cached) if custom_cached else Decimal("0.01875")

    return {
        # Configured staging / production alias
        "gemini-3.1-flash-lite": GeminiModelPriceConfig(
            input_usd_per_m=flash_lite_input,
            output_usd_per_m=flash_lite_output,
            cached_usd_per_m=flash_lite_cached,
        ),
        # Gemini 2.0 Flash (Official GA pricing: $0.10 input, $0.40 output, $0.025 cached)
        "gemini-2.0-flash": GeminiModelPriceConfig(
            input_usd_per_m=Decimal("0.10"),
            output_usd_per_m=Decimal("0.40"),
            cached_usd_per_m=Decimal("0.025"),
        ),
        "gemini-2.0-flash-exp": GeminiModelPriceConfig(
            input_usd_per_m=Decimal("0.10"),
            output_usd_per_m=Decimal("0.40"),
            cached_usd_per_m=Decimal("0.025"),
        ),
        # Gemini 2.0 Flash Lite ($0.075 input, $0.30 output, $0.01875 cached)
        "gemini-2.0-flash-lite": GeminiModelPriceConfig(
            input_usd_per_m=Decimal("0.075"),
            output_usd_per_m=Decimal("0.30"),
            cached_usd_per_m=Decimal("0.01875"),
        ),
        "gemini-2.0-flash-lite-preview-02-05": GeminiModelPriceConfig(
            input_usd_per_m=Decimal("0.075"),
            output_usd_per_m=Decimal("0.30"),
            cached_usd_per_m=Decimal("0.01875"),
        ),
        # Gemini 2.5 Flash / Pro
        "gemini-2.5-flash": GeminiModelPriceConfig(
            input_usd_per_m=Decimal("0.10"),
            output_usd_per_m=Decimal("0.40"),
            cached_usd_per_m=Decimal("0.025"),
        ),
        "gemini-2.5-pro": GeminiModelPriceConfig(
            input_usd_per_m=Decimal("1.25"),
            output_usd_per_m=Decimal("5.00"),
            cached_usd_per_m=Decimal("0.3125"),
        ),
        # Gemini 1.5 Flash (<= 128k prompt tokens)
        "gemini-1.5-flash": GeminiModelPriceConfig(
            input_usd_per_m=Decimal("0.075"),
            output_usd_per_m=Decimal("0.30"),
            cached_usd_per_m=Decimal("0.01875"),
        ),
        # Gemini 1.5 Pro (<= 128k prompt tokens)
        "gemini-1.5-pro": GeminiModelPriceConfig(
            input_usd_per_m=Decimal("1.25"),
            output_usd_per_m=Decimal("5.00"),
            cached_usd_per_m=Decimal("0.3125"),
        ),
    }


# Global pricing table reference
OFFICIAL_GEMINI_PRICING_USD_PER_1M = get_official_gemini_pricing_table()


@dataclass
class TokenCostResult:
    """Structured calculation result for generation token pricing in INR."""
    input_cost_inr: Optional[Decimal]
    output_cost_inr: Optional[Decimal]
    cached_cost_inr: Optional[Decimal]
    total_cost_inr: Optional[Decimal]
    currency: str = "INR"
    cost_status: str = "success"
    usd_to_inr_rate: Optional[Decimal] = None
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cached_tokens": self.cached_tokens,
            "input_cost_inr": self.input_cost_inr,
            "output_cost_inr": self.output_cost_inr,
            "cached_cost_inr": self.cached_cost_inr,
            "total_cost_inr": self.total_cost_inr,
            "currency": self.currency,
            "cost_status": self.cost_status,
        }


def get_model_pricing_usd(model: str) -> Optional[GeminiModelPriceConfig]:
    """Look up USD pricing rates per 1M tokens for a given model."""
    clean_model = (model or "").lower().strip()
    table = get_official_gemini_pricing_table()
    if clean_model in table:
        return table[clean_model]
    for key in table:
        if clean_model.startswith(key):
            return table[key]
    return None


def calculate_gemini_cost(
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
    provider: str = "gemini",
    usd_to_inr_rate: Optional[Decimal] = None,
) -> TokenCostResult:
    """
    Calculates exact INR cost for a Gemini generation using Python Decimal.
    
    If the model is not configured, returns cost_status="pricing_not_configured"
    and total_cost_inr=None without inventing prices.
    """
    calc_total_tokens = int(input_tokens) + int(output_tokens)

    if provider.lower() not in ("gemini", "google"):
        return TokenCostResult(
            input_cost_inr=Decimal("0.00"),
            output_cost_inr=Decimal("0.00"),
            cached_cost_inr=Decimal("0.00"),
            total_cost_inr=Decimal("0.00"),
            currency="INR",
            cost_status="free_or_self_hosted",
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            cached_tokens=int(cached_tokens),
            total_tokens=calc_total_tokens,
        )

    clean_model = (model or "").lower().strip()
    table = get_official_gemini_pricing_table()

    pricing = table.get(clean_model)
    if not pricing:
        # Check normalized prefix match e.g. "gemini-2.0-flash-001" -> "gemini-2.0-flash"
        for key in table:
            if clean_model.startswith(key):
                pricing = table[key]
                break

    if not pricing:
        logger.warning(f"[TokenPricing] Pricing not configured for model '{model}'. Returning null cost.")
        return TokenCostResult(
            input_cost_inr=None,
            output_cost_inr=None,
            cached_cost_inr=None,
            total_cost_inr=None,
            currency="INR",
            cost_status="pricing_not_configured",
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            cached_tokens=int(cached_tokens),
            total_tokens=calc_total_tokens,
        )

    usd_to_inr = usd_to_inr_rate if usd_to_inr_rate is not None else get_usd_to_inr_rate()

    # Calculate via Decimal
    dec_input_tokens = Decimal(str(max(0, input_tokens)))
    dec_output_tokens = Decimal(str(max(0, output_tokens)))
    dec_cached_tokens = Decimal(str(max(0, cached_tokens)))

    # Compute USD
    input_cost_usd = (dec_input_tokens / ONE_MILLION) * pricing.input_usd_per_m
    output_cost_usd = (dec_output_tokens / ONE_MILLION) * pricing.output_usd_per_m
    cached_cost_usd = (dec_cached_tokens / ONE_MILLION) * pricing.cached_usd_per_m

    # Convert to INR using Decimal arithmetic
    input_cost_inr = (input_cost_usd * usd_to_inr).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
    output_cost_inr = (output_cost_usd * usd_to_inr).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
    cached_cost_inr = (cached_cost_usd * usd_to_inr).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
    total_cost_inr = input_cost_inr + output_cost_inr + cached_cost_inr

    return TokenCostResult(
        input_cost_inr=input_cost_inr,
        output_cost_inr=output_cost_inr,
        cached_cost_inr=cached_cost_inr,
        total_cost_inr=total_cost_inr,
        currency="INR",
        cost_status="success",
        usd_to_inr_rate=usd_to_inr,
        input_tokens=int(input_tokens),
        output_tokens=int(output_tokens),
        cached_tokens=int(cached_tokens),
        total_tokens=calc_total_tokens,
    )


def calculate_llm_cost(
    *,
    provider: str = "gemini",
    model: str = "gemini-3.1-flash-lite",
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_tokens: int = 0,
    usd_to_inr_rate: Optional[Decimal] = None,
) -> TokenCostResult:
    """
    Universal reusable pricing utility function matching Section 17 specifications.
    """
    return calculate_gemini_cost(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
        provider=provider,
        usd_to_inr_rate=usd_to_inr_rate,
    )
