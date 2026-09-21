"""
test_production_resume_parser_fix.py — Comprehensive test suite covering all 25 production requirements:
1. Normal resume parse
2. Long resume parse
3. Large output JSON
4. MAX_TOKENS finish reason
5. Gemini successful response
6. Gemini validation failure
7. Gemini 429 retry
8. deterministic fallback
9. input token tracking
10. output token tracking
11. total token tracking
12. cached token tracking
13. input cost calculation
14. output cost calculation
15. cached cost calculation
16. total INR cost calculation
17. Decimal precision
18. model-specific pricing
19. configurable USD-to-INR rate
20. unknown model pricing
21. missing pricing configuration
22. missing usage metadata
23. multiple generations
24. aggregate usage
25. existing API response compatibility
"""

import asyncio
import json
import os
import unittest
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from app.modules.resume_builder.token_pricing import (
    calculate_gemini_cost,
    get_model_pricing_usd,
    DEFAULT_USD_TO_INR,
    OFFICIAL_GEMINI_PRICING_USD_PER_1M,
)
from app.modules.resume_builder.gemini_client import (
    GeminiClient,
    UsageMetadata,
    GenerationResult,
)
from app.modules.resume_builder.ai_parser import (
    ImprovedUniversalResumeParser,
    DEFAULT_MAX_OUTPUT_TOKENS,
)
from app.modules.resume_builder.deterministic_parser import DeterministicResumeParser
from app.modules.resume_builder.universal_extractor import UniversalExtractor
from app.modules.resume_builder.extractor import detect_columns, sort_blocks
from app.modules.resume_builder.input_sanitizer import InputSanitizer
from app.modules.resume_builder.schemas import (
    CanonicalResume,
    PersonalInformation,
    ExperienceItem,
    EducationItem,
    ProjectItem,
    CertificationItem,
    map_to_legacy_parse_dict,
)
from app.modules.resume_builder.telemetry import (
    log_ai_usage,
    get_usage_summary,
    clear_telemetry_cache,
)


class TestProductionResumeParserFix(unittest.TestCase):
    def setUp(self):
        clear_telemetry_cache()

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Normal resume parse
    # ─────────────────────────────────────────────────────────────────────────
    def test_01_normal_resume_parse(self):
        sample_resume = (
            "Alex Johnson\n"
            "alex@example.com | +1 555 123 4567 | San Francisco, CA\n"
            "Summary: Senior backend engineer with 8 years of experience building distributed systems.\n"
            "Experience\n"
            "Senior Backend Engineer | Stripe | 2021 - Present\n"
            "• Architected distributed payment processing pipeline handling 50k RPS.\n"
            "• Reduced database p99 query latency by 42% through query optimization.\n"
            "Education\n"
            "B.S. in Computer Science | UC Berkeley | 2017\n"
            "Skills\n"
            "Python, Go, PostgreSQL, Redis, Kubernetes, Docker\n"
        )
        canonical = DeterministicResumeParser.parse_text(sample_resume)
        legacy = map_to_legacy_parse_dict(canonical)

        self.assertEqual(canonical.personal_information.name, "Alex Johnson")
        self.assertEqual(canonical.personal_information.email, "alex@example.com")
        self.assertIn("Backend Engineer", canonical.experience[0].position)
        self.assertIn("Computer Science", canonical.education[0].degree)
        self.assertTrue(len(canonical.skills) >= 4)
        self.assertIn("header", legacy)
        self.assertIn("experience", legacy)

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Long resume parse (preserves full descriptions, responsibilities, bullets)
    # ─────────────────────────────────────────────────────────────────────────
    def test_02_long_resume_parse(self):
        long_description = (
            "Led the core platform infrastructure modernization initiative across 12 distributed engineering teams. "
            "Responsible for architectural direction, service reliability, Kubernetes orchestration, multi-region failover, "
            "and cross-cloud migration strategies."
        )
        exp = ExperienceItem(
            position="Principal Infrastructure Architect",
            company="Enterprise Cloud Corp",
            location="Remote",
            fromYear="2020",
            toYear="Present",
            isOngoing=True,
            description=long_description,
            responsibilities=[
                "Direct platform infrastructure roadmap",
                "Ensure 99.99% uptime SLA across 4 cloud regions",
            ],
            bullets=[
                "Migrated 140 microservices from on-premise VMs to AWS EKS with zero customer downtime.",
                "Designed zero-trust service mesh using Istio reducing inter-service security vulnerabilities by 100%.",
                "Cut cloud expenditure by $1.8M annually through autoscaling and spot instance policies.",
            ],
            achievements=["Awarded Technology Innovator of the Year 2023"],
        )

        canonical = CanonicalResume(
            personal_information=PersonalInformation(name="Dr. Elena Vance"),
            experience=[exp],
        )
        legacy = map_to_legacy_parse_dict(canonical)

        self.assertEqual(legacy["experience"][0]["description"], long_description)
        self.assertEqual(len(legacy["experience"][0]["bullets"]), 3)
        self.assertEqual(len(legacy["experience"][0]["responsibilities"]), 2)
        self.assertEqual(len(legacy["experience"][0]["achievements"]), 1)

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Large output JSON
    # ─────────────────────────────────────────────────────────────────────────
    def test_03_large_output_json(self):
        items = []
        for i in range(15):
            items.append(
                ProjectItem(
                    title=f"Microservice System {i}",
                    description=f"Extensive enterprise system {i} handling massive high-throughput workloads with full fault-tolerance and automated reconciliation pipelines.",
                    technologies=["Python", "FastAPI", "PostgreSQL", "Kafka", "Redis"],
                    bullets=[
                        f"Bullet 1 for service {i} showing detailed metric improvement of {i*10}%.",
                        f"Bullet 2 for service {i} describing fault-tolerance recovery under partition.",
                    ],
                )
            )
        canonical = CanonicalResume(projects=items)
        json_str = canonical.model_dump_json()
        self.assertTrue(len(json_str) > 3000)
        reconstructed = CanonicalResume.model_validate_json(json_str)
        self.assertEqual(len(reconstructed.projects), 15)

    # ─────────────────────────────────────────────────────────────────────────
    # 4. MAX_TOKENS finish reason
    # ─────────────────────────────────────────────────────────────────────────
    def test_04_max_tokens_finish_reason(self):
        mock_client = MagicMock()
        mock_client.is_configured = True

        mock_usage = UsageMetadata(
            provider="gemini",
            model="gemini-3.1-flash-lite",
            input_tokens=1500,
            output_tokens=8192,
            total_tokens=9692,
            finish_reason="MAX_TOKENS",
            input_cost_inr=Decimal("0.01"),
            output_cost_inr=Decimal("0.10"),
            cached_cost_inr=Decimal("0.00"),
            total_cost_inr=Decimal("0.11"),
            currency="INR",
            parser_source="gemini",
        )

        dummy_valid_json = CanonicalResume(
            personal_information=PersonalInformation(name="Truncated User")
        ).model_dump_json()

        mock_gen_result = GenerationResult(text=dummy_valid_json, usage=mock_usage)
        mock_client.generate = AsyncMock(return_value=mock_gen_result)

        with patch("app.modules.resume_builder.ai_parser.get_gemini_client", return_value=mock_client):
            res = asyncio.run(
                ImprovedUniversalResumeParser.parse_text(
                    "Dummy Resume Text That Truncates Over Maximum Output Tokens",
                    operation="resume_parse",
                )
            )

        self.assertEqual(res["finish_reason"], "MAX_TOKENS")
        self.assertEqual(res["parse_status"], "truncated")
        self.assertEqual(res["usage"][0]["finish_reason"], "MAX_TOKENS")

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Gemini successful response
    # ─────────────────────────────────────────────────────────────────────────
    def test_05_gemini_successful_response(self):
        mock_client = MagicMock()
        mock_client.is_configured = True

        mock_usage = UsageMetadata(
            provider="gemini",
            model="gemini-3.1-flash-lite",
            input_tokens=1000,
            output_tokens=500,
            total_tokens=1500,
            cached_tokens=0,
            finish_reason="STOP",
            input_cost_inr=Decimal("0.006488"),
            output_cost_inr=Decimal("0.012975"),
            cached_cost_inr=Decimal("0.000000"),
            total_cost_inr=Decimal("0.019463"),
            currency="INR",
            parser_source="gemini",
        )

        canonical = CanonicalResume(
            personal_information=PersonalInformation(name="Jane Doe", email="jane@example.com"),
            summary="Staff Engineer with extensive experience.",
        )
        mock_gen = GenerationResult(text=canonical.model_dump_json(), usage=mock_usage)
        mock_client.generate = AsyncMock(return_value=mock_gen)

        with patch("app.modules.resume_builder.ai_parser.get_gemini_client", return_value=mock_client):
            res = asyncio.run(
                ImprovedUniversalResumeParser.parse_text("Jane Doe\nStaff Engineer", operation="resume_parse")
            )

        self.assertTrue(res["success"])
        self.assertEqual(res["source"], "gemini")
        self.assertEqual(res["parser_source"], "gemini")
        self.assertEqual(res["finish_reason"], "STOP")
        self.assertEqual(res["parse_status"], "complete")
        self.assertEqual(res["parsed"]["header"]["name"], "Jane Doe")
        self.assertEqual(res["usage"][0]["currency"], "INR")
        self.assertEqual(res["usage"][0]["total_cost_inr"], Decimal("0.019463"))

    # ─────────────────────────────────────────────────────────────────────────
    # 6. Gemini validation failure
    # ─────────────────────────────────────────────────────────────────────────
    def test_06_gemini_validation_failure(self):
        mock_client = MagicMock()
        mock_client.is_configured = True

        mock_usage = UsageMetadata(
            provider="gemini",
            model="gemini-3.1-flash-lite",
            input_tokens=1200,
            output_tokens=300,
            total_tokens=1500,
            finish_reason="STOP",
            total_cost_inr=Decimal("0.02"),
            currency="INR",
        )
        # Returns invalid malformed JSON
        mock_gen = GenerationResult(text="INVALID_JSON_NON_PARSEABLE", usage=mock_usage)
        mock_client.generate = AsyncMock(return_value=mock_gen)

        with patch("app.modules.resume_builder.ai_parser.get_gemini_client", return_value=mock_client):
            res = asyncio.run(
                ImprovedUniversalResumeParser.parse_text(
                    "Sarah Connor\nsarah@sky.net\nExperience\nResistance Fighter | 2029",
                    operation="resume_parse",
                )
            )

        self.assertTrue(res["success"])
        self.assertEqual(res["source"], "deterministic_fallback")
        self.assertEqual(res["parser_source"], "deterministic_fallback")
        # Ensure Gemini's usage is preserved rather than dropped
        self.assertIsNotNone(res["usage"])
        self.assertEqual(res["usage"][0]["total_tokens"], 1500)

    # ─────────────────────────────────────────────────────────────────────────
    # 7. Gemini 429 retry
    # ─────────────────────────────────────────────────────────────────────────
    def test_07_gemini_429_retry(self):
        client = GeminiClient(api_key="test_key")

        mock_response = MagicMock()
        mock_response.text = '{"personal_information": {"name": "Retry Success"}}'
        mock_candidate = MagicMock()
        mock_candidate.finish_reason = "STOP"
        mock_response.candidates = [mock_candidate]
        mock_meta = MagicMock()
        mock_meta.prompt_token_count = 500
        mock_meta.candidates_token_count = 100
        mock_meta.total_token_count = 600
        mock_meta.cached_content_token_count = 0
        mock_response.usage_metadata = mock_meta

        mock_internal_client = MagicMock()
        mock_internal_client.aio.models.generate_content = AsyncMock(
            side_effect=[
                Exception("429 Resource Exhausted: Quota exceeded"),
                mock_response,
            ]
        )
        client._client = mock_internal_client

        with patch("asyncio.sleep", new_callable=AsyncMock):
            res = asyncio.run(
                client.generate(prompt="Test Prompt", operation="test_retry")
            )

        self.assertIn("Retry Success", res.text)
        self.assertEqual(mock_internal_client.aio.models.generate_content.call_count, 2)

    # ─────────────────────────────────────────────────────────────────────────
    # 8. Deterministic fallback
    # ─────────────────────────────────────────────────────────────────────────
    def test_08_deterministic_fallback(self):
        mock_client = MagicMock()
        mock_client.is_configured = False  # Offline mode

        with patch("app.modules.resume_builder.ai_parser.get_gemini_client", return_value=mock_client):
            res = asyncio.run(
                ImprovedUniversalResumeParser.parse_text(
                    "David Miller\ndavid@example.com\nSoftware Developer",
                    operation="resume_parse",
                )
            )

        self.assertTrue(res["success"])
        self.assertEqual(res["source"], "deterministic_fallback")
        self.assertEqual(res["parser_source"], "deterministic_fallback")
        self.assertIsNone(res["usage"])

    # ─────────────────────────────────────────────────────────────────────────
    # 9. Input token tracking
    # ─────────────────────────────────────────────────────────────────────────
    def test_09_input_token_tracking(self):
        cost_res = calculate_gemini_cost(
            model="gemini-3.1-flash-lite",
            input_tokens=2500,
            output_tokens=0,
            cached_tokens=0,
        )
        self.assertEqual(cost_res.input_tokens, 2500)

    # ─────────────────────────────────────────────────────────────────────────
    # 10. Output token tracking
    # ─────────────────────────────────────────────────────────────────────────
    def test_10_output_token_tracking(self):
        cost_res = calculate_gemini_cost(
            model="gemini-3.1-flash-lite",
            input_tokens=0,
            output_tokens=1850,
            cached_tokens=0,
        )
        self.assertEqual(cost_res.output_tokens, 1850)

    # ─────────────────────────────────────────────────────────────────────────
    # 11. Total token tracking
    # ─────────────────────────────────────────────────────────────────────────
    def test_11_total_token_tracking(self):
        cost_res = calculate_gemini_cost(
            model="gemini-3.1-flash-lite",
            input_tokens=2500,
            output_tokens=1200,
            cached_tokens=0,
        )
        self.assertEqual(cost_res.total_tokens, 3700)

    # ─────────────────────────────────────────────────────────────────────────
    # 12. Cached token tracking
    # ─────────────────────────────────────────────────────────────────────────
    def test_12_cached_token_tracking(self):
        cost_res = calculate_gemini_cost(
            model="gemini-3.1-flash-lite",
            input_tokens=3000,
            output_tokens=1000,
            cached_tokens=2000,
        )
        self.assertEqual(cost_res.cached_tokens, 2000)
        self.assertTrue(cost_res.cached_cost_inr > Decimal("0.0"))

    # ─────────────────────────────────────────────────────────────────────────
    # 13. Input cost calculation
    # ─────────────────────────────────────────────────────────────────────────
    def test_13_input_cost_calculation(self):
        # 1M input at $0.075/1M and 86.5 INR/USD = 6.487500 INR
        cost_res = calculate_gemini_cost(
            model="gemini-3.1-flash-lite",
            input_tokens=1_000_000,
            output_tokens=0,
            usd_to_inr_rate=Decimal("86.5"),
        )
        self.assertEqual(cost_res.input_cost_inr, Decimal("6.487500"))

    # ─────────────────────────────────────────────────────────────────────────
    # 14. Output cost calculation
    # ─────────────────────────────────────────────────────────────────────────
    def test_14_output_cost_calculation(self):
        # 1M output at $0.30/1M and 86.5 INR/USD = 25.950000 INR
        cost_res = calculate_gemini_cost(
            model="gemini-3.1-flash-lite",
            input_tokens=0,
            output_tokens=1_000_000,
            usd_to_inr_rate=Decimal("86.5"),
        )
        self.assertEqual(cost_res.output_cost_inr, Decimal("25.950000"))

    # ─────────────────────────────────────────────────────────────────────────
    # 15. Cached cost calculation
    # ─────────────────────────────────────────────────────────────────────────
    def test_15_cached_cost_calculation(self):
        # 1M cached at $0.01875/1M and 86.5 INR/USD = 1.621875 INR
        cost_res = calculate_gemini_cost(
            model="gemini-3.1-flash-lite",
            input_tokens=0,
            output_tokens=0,
            cached_tokens=1_000_000,
            usd_to_inr_rate=Decimal("86.5"),
        )
        self.assertEqual(cost_res.cached_cost_inr, Decimal("1.621875"))

    # ─────────────────────────────────────────────────────────────────────────
    # 16. Total INR cost calculation
    # ─────────────────────────────────────────────────────────────────────────
    def test_16_total_inr_cost_calculation(self):
        cost_res = calculate_gemini_cost(
            model="gemini-3.1-flash-lite",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            cached_tokens=1_000_000,
            usd_to_inr_rate=Decimal("86.5"),
        )
        self.assertEqual(cost_res.currency, "INR")
        expected_total = Decimal("6.487500") + Decimal("25.950000") + Decimal("1.621875")
        self.assertEqual(cost_res.total_cost_inr, expected_total)

    # ─────────────────────────────────────────────────────────────────────────
    # 17. Decimal precision (no float precision loss)
    # ─────────────────────────────────────────────────────────────────────────
    def test_17_decimal_precision(self):
        cost_res = calculate_gemini_cost(
            model="gemini-3.1-flash-lite",
            input_tokens=12,
            output_tokens=5,
            cached_tokens=0,
        )
        self.assertIsInstance(cost_res.input_cost_inr, Decimal)
        self.assertIsInstance(cost_res.output_cost_inr, Decimal)
        self.assertIsInstance(cost_res.total_cost_inr, Decimal)
        # Verify tiny costs are not rounded to 0.0 internally
        self.assertTrue(cost_res.total_cost_inr > Decimal("0.0"))

    # ─────────────────────────────────────────────────────────────────────────
    # 18. Model-specific pricing
    # ─────────────────────────────────────────────────────────────────────────
    def test_18_model_specific_pricing(self):
        cost_flash_lite = calculate_gemini_cost(
            model="gemini-3.1-flash-lite", input_tokens=10000, output_tokens=5000
        )
        cost_pro = calculate_gemini_cost(
            model="gemini-2.5-pro", input_tokens=10000, output_tokens=5000
        )
        # Pro model is priced significantly higher than Flash-Lite
        self.assertTrue(cost_pro.total_cost_inr > cost_flash_lite.total_cost_inr)

    # ─────────────────────────────────────────────────────────────────────────
    # 19. Configurable USD-to-INR rate
    # ─────────────────────────────────────────────────────────────────────────
    def test_19_configurable_usd_to_inr_rate(self):
        with patch.dict(os.environ, {"GEMINI_USD_TO_INR": "90.0"}):
            res_90 = calculate_gemini_cost(
                model="gemini-3.1-flash-lite", input_tokens=100000, output_tokens=50000
            )
        with patch.dict(os.environ, {"GEMINI_USD_TO_INR": "80.0"}):
            res_80 = calculate_gemini_cost(
                model="gemini-3.1-flash-lite", input_tokens=100000, output_tokens=50000
            )
        self.assertTrue(res_90.total_cost_inr > res_80.total_cost_inr)

    # ─────────────────────────────────────────────────────────────────────────
    # 20. Unknown model pricing
    # ─────────────────────────────────────────────────────────────────────────
    def test_20_unknown_model_pricing(self):
        cost_res = calculate_gemini_cost(
            model="gemini-unknown-future-model-v99",
            input_tokens=1000,
            output_tokens=500,
        )
        self.assertEqual(cost_res.cost_status, "pricing_not_configured")
        self.assertIsNone(cost_res.total_cost_inr)
        self.assertIsNone(cost_res.input_cost_inr)

    # ─────────────────────────────────────────────────────────────────────────
    # 21. Missing pricing configuration
    # ─────────────────────────────────────────────────────────────────────────
    def test_21_missing_pricing_configuration(self):
        pricing = get_model_pricing_usd("non_existent_model_123")
        self.assertIsNone(pricing)

    # ─────────────────────────────────────────────────────────────────────────
    # 22. Missing usage metadata
    # ─────────────────────────────────────────────────────────────────────────
    def test_22_missing_usage_metadata(self):
        client = GeminiClient(api_key="test_key")

        mock_response = MagicMock()
        mock_response.text = '{"personal_information": {"name": "No Usage User"}}'
        mock_response.candidates = []
        mock_response.usage_metadata = None  # Missing usage metadata

        mock_internal_client = MagicMock()
        mock_internal_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        client._client = mock_internal_client

        res = asyncio.run(client.generate(prompt="Test", operation="missing_meta"))
        self.assertIsNotNone(res.usage)
        self.assertEqual(res.usage.total_tokens, 0)
        self.assertIsNone(res.usage.total_cost_inr)

    # ─────────────────────────────────────────────────────────────────────────
    # 23. Multiple generations (each produces distinct usage record)
    # ─────────────────────────────────────────────────────────────────────────
    def test_23_multiple_generations(self):
        log_ai_usage(
            provider="gemini",
            model="gemini-3.1-flash-lite",
            operation="gen_1",
            input_tokens=1000,
            output_tokens=400,
            total_cost_inr=Decimal("0.015"),
            request_id="req-1",
        )
        log_ai_usage(
            provider="gemini",
            model="gemini-3.1-flash-lite",
            operation="gen_2",
            input_tokens=2000,
            output_tokens=800,
            total_cost_inr=Decimal("0.030"),
            request_id="req-2",
        )
        summary = get_usage_summary()
        self.assertEqual(summary["total_calls"], 2)
        self.assertEqual(summary["total_input_tokens"], 3000)
        self.assertEqual(summary["total_output_tokens"], 1200)
        self.assertEqual(summary["total_cost_inr"], Decimal("0.045"))

    # ─────────────────────────────────────────────────────────────────────────
    # 24. Aggregate usage summary
    # ─────────────────────────────────────────────────────────────────────────
    def test_24_aggregate_usage(self):
        clear_telemetry_cache()
        for i in range(5):
            log_ai_usage(
                provider="gemini",
                model="gemini-3.1-flash-lite",
                operation="resume_parse",
                input_tokens=1000,
                output_tokens=500,
                total_cost_inr=Decimal("0.02"),
                request_id=f"req-{i}",
            )
        summary = get_usage_summary(operation="resume_parse")
        self.assertEqual(summary["total_calls"], 5)
        self.assertEqual(summary["total_tokens"], 7500)
        self.assertEqual(summary["total_cost_inr"], Decimal("0.10"))
        self.assertEqual(summary["currency"], "INR")

    # ─────────────────────────────────────────────────────────────────────────
    # 25. Existing API response compatibility
    # ─────────────────────────────────────────────────────────────────────────
    def test_25_existing_api_response_compatibility(self):
        canonical = CanonicalResume(
            personal_information=PersonalInformation(
                name="Alice Wonder",
                title="Lead Architect",
                email="alice@wonder.org",
                phone="+1-555-0199",
                location="Seattle, WA",
                link="https://linkedin.com/in/alicew",
            ),
            summary="Accomplished enterprise software architect.",
            skills=["Python", "Cloud"],
            experience=[
                ExperienceItem(
                    position="Principal Engineer",
                    company="Acme Corp",
                    fromYear="2019",
                    toYear="Present",
                    bullets=["Led cloud transformation."],
                )
            ],
            education=[
                EducationItem(
                    degree="B.S. Software Engineering",
                    institution="UW",
                    toYear="2015",
                )
            ],
            projects=[
                ProjectItem(
                    title="Platform X",
                    description="Comprehensive multi-tenant platform.",
                    bullets=["Achieved 99.999% availability."],
                )
            ],
            certifications=[
                CertificationItem(title="AWS Solutions Architect", issuer="Amazon", year="2022")
            ],
            languages=["English"],
            other=["Speaker at PyCon 2023"],
        )

        legacy = map_to_legacy_parse_dict(canonical)

        # All existing legacy keys must exist without omission
        for key in ["header", "summary", "experience", "education", "skills", "projects", "certifications", "languages", "other"]:
            self.assertIn(key, legacy)

        self.assertIn("name", legacy["header"])
        self.assertIn("email", legacy["header"])
        self.assertIn("position", legacy["experience"][0])
        self.assertIn("bullets", legacy["experience"][0])
        self.assertIn("degree", legacy["education"][0])
        self.assertIn("institution", legacy["education"][0])

    # ─────────────────────────────────────────────────────────────────────────
    # Additional checks: Layout column detection & Input deduplication
    # ─────────────────────────────────────────────────────────────────────────
    def test_input_deduplication(self):
        raw_items = [
            {
                "text": "Full Block Text\nLine 1\nLine 2",
                "type": "text",
                "items": ["Line 1", "Line 2"],  # exact duplicate of lines in text
            }
        ]
        result = UniversalExtractor.extract_all_content(raw_items)
        # Content should appear exactly once, not twice
        self.assertEqual(result.count("Line 1"), 1)
        self.assertEqual(result.count("Line 2"), 1)

    def test_column_detection_single_column_preservation(self):
        # Single column resume with right-aligned dates and indented bullets
        blocks = [
            {"text": "John Doe", "x": 54.0, "w": 200.0, "y": 50.0, "page": 1},
            {"text": "Experience", "x": 54.0, "w": 100.0, "y": 80.0, "page": 1},
            {"text": "Senior Engineer", "x": 54.0, "w": 150.0, "y": 100.0, "page": 1},
            {"text": "Jan 2021 - Present", "x": 480.0, "w": 80.0, "y": 100.0, "page": 1},  # right-aligned date
            {"text": "• Bullet point 1 with indent", "x": 72.0, "w": 450.0, "y": 120.0, "page": 1},  # indented bullet
            {"text": "• Bullet point 2 with indent", "x": 72.0, "w": 450.0, "y": 140.0, "page": 1},
        ]
        detected = detect_columns(blocks)
        # All blocks should remain in column 0
        for b in detected:
            self.assertEqual(b.get("column", 0), 0)

        sorted_b = sort_blocks(detected)
        # Right-aligned date should remain next to Senior Engineer, NOT at the bottom of the page!
        texts = [b["text"] for b in sorted_b]
        date_idx = texts.index("Jan 2021 - Present")
        bullet_idx = texts.index("• Bullet point 1 with indent")
        self.assertTrue(date_idx < bullet_idx)

    def test_sanitizer_preserves_newlines_and_large_descriptions(self):
        multiline_text = "Paragraph 1 describing project scope.\n\nParagraph 2 describing achievements with 99.9% uptime."
        sanitized = InputSanitizer._sanitize_string(multiline_text)
        self.assertIn("\n\n", sanitized)
        self.assertIn("Paragraph 1", sanitized)
        self.assertIn("Paragraph 2", sanitized)

        # Ensure long text over 5000 characters is NOT truncated
        long_text = "A" * 12000
        sanitized_long = InputSanitizer._sanitize_string(long_text)
        self.assertEqual(len(sanitized_long), 12000)

    def test_column_detection_two_column_preservation(self):
        # Genuine 2-column layout: Left column (skills, education), Right column (experience, projects)
        blocks = [
            # Header spanning top
            {"text": "Jane Doe Header", "x": 50.0, "w": 500.0, "y": 30.0, "page": 1},
            # Left column: x=50, w=180
            {"text": "Skills Section Heading", "x": 50.0, "w": 180.0, "y": 80.0, "page": 1},
            {"text": "Python, TypeScript, AWS", "x": 50.0, "w": 180.0, "y": 100.0, "page": 1},
            {"text": "Education Section Heading", "x": 50.0, "w": 180.0, "y": 150.0, "page": 1},
            {"text": "B.S. in Computer Science", "x": 50.0, "w": 180.0, "y": 170.0, "page": 1},
            # Right column: x=280, w=280
            {"text": "Experience Section Heading", "x": 280.0, "w": 280.0, "y": 80.0, "page": 1},
            {"text": "Senior Software Architect at BigCo", "x": 280.0, "w": 280.0, "y": 100.0, "page": 1},
            {"text": "• Built high-scale cloud platforms", "x": 280.0, "w": 280.0, "y": 120.0, "page": 1},
            {"text": "Projects Section Heading", "x": 280.0, "w": 280.0, "y": 180.0, "page": 1},
            {"text": "Open Source Distributed Engine", "x": 280.0, "w": 280.0, "y": 200.0, "page": 1},
        ]
        detected = detect_columns(blocks)
        # Verify right column blocks were assigned column 1
        right_blocks = [b for b in detected if b["x"] >= 280]
        for b in right_blocks:
            self.assertEqual(b.get("column"), 1)

    def test_deterministic_parser_captures_descriptions_and_institutions(self):
        text = (
            "Alex Smith\nalex@example.com\n"
            "Education\n"
            "B.S. Computer Science | Stanford University 2020\n"
            "Projects\n"
            "Distributed Event Broker\n"
            "Engineered a fault-tolerant log replicating 100k events/sec.\n"
            "• Implemented Raft consensus protocol from scratch.\n"
        )
        canonical = DeterministicResumeParser.parse_text(text)
        self.assertEqual(len(canonical.education), 1)
        self.assertIn("Stanford University", canonical.education[0].institution)
        self.assertEqual(len(canonical.projects), 1)
        self.assertIn("fault-tolerant log", canonical.projects[0].description)
        self.assertEqual(len(canonical.projects[0].bullets), 1)


if __name__ == "__main__":
    unittest.main()
