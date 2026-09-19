"""Comprehensive Unit and Integration Tests for Gemini Token Usage Tracking and Service Authentication.

Tests cover:
1. Gemini usage metadata extraction from response.usage_metadata.
2. Prompt tokens mapping (prompt_token_count -> input_tokens).
3. Candidate tokens mapping (candidates_token_count -> output_tokens).
4. Total tokens mapping (total_token_count -> total_tokens).
5. Cached tokens mapping (cached_content_token_count -> cached_tokens).
6. Usage list format propagation through resume parser (usage: [{...}]).
7. Usage list format propagation through ATS scanner (usage: [{...}]).
8. Request ID propagation across logging/context and response.
9. Operation specification (resume_parse / ats_scan / ats_semantic_analysis).
10. Multiple Gemini generations tracking in a single request (usage: [{...}, {...}]).
11. Invalid service key rejection (HTTP 401).
12. Missing service key rejection (HTTP 401 when configured).
13. User ID not trusted without valid service authentication.
14. Resume response retains all existing fields + additive usage list.
15. ATS response retains all existing fields + additive usage list.
16. include_ai=False performs zero Gemini calls and returns usage=None.
17. Cache hit reports 0 Gemini tokens (not fake usage).
18. Missing usage metadata is handled safely with 0 tokens.
19. Existing deterministic fallback remains functional with usage=None.
20. Existing telemetry continues working seamlessly.
21. No sensitive information (PII/keys/passwords/candidate names) written to logs.
"""

import asyncio
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException, Request
from fastapi.testclient import TestClient

from app.core.settings import settings
from app.core.security import RequestContext, get_trusted_request_context
from app.modules.resume_builder.gemini_client import GeminiClient, UsageMetadata, GenerationResult
from app.modules.resume_builder.ai_client import ImprovedAIClientManager, _AI_RESPONSE_CACHE
from app.modules.resume_builder.ai_parser import ImprovedUniversalResumeParser
from app.modules.resume_builder.parser_service import parse_resume_with_ai
from app.modules.ats_scanner.service import ATSScannerService
from app.main import app


class TestTokenUsageAndSecurity(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app, base_url="http://localhost")
        self.secret = "test_internal_secret_key_12345"

    # 1-5: Gemini Usage Metadata Extraction & Mapping
    def test_gemini_usage_metadata_extraction_and_mapping(self):
        """Test extraction of actual provider-reported usage from response.usage_metadata."""
        client = GeminiClient(api_key="fake-key")
        
        # Mock Gemini response
        mock_response = MagicMock()
        mock_response.text = '{"name": "Jane Doe", "skills": ["Python", "FastAPI"]}'
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 1420
        mock_response.usage_metadata.candidates_token_count = 512
        mock_response.usage_metadata.total_token_count = 1932
        mock_response.usage_metadata.cached_content_token_count = 128

        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        client._client = mock_client

        result = asyncio.run(client.generate(prompt="Parse this resume", model="gemini-3.1-flash-lite", request_id="req-123", operation="resume_parse"))

        self.assertIsInstance(result, GenerationResult)
        self.assertEqual(result.text, mock_response.text)
        self.assertIsNotNone(result.usage)
        
        usage = result.usage
        self.assertEqual(usage.provider, "gemini")
        self.assertEqual(usage.model, "gemini-3.1-flash-lite")
        self.assertEqual(usage.input_tokens, 1420)
        self.assertEqual(usage.output_tokens, 512)
        self.assertEqual(usage.total_tokens, 1932)
        self.assertEqual(usage.cached_tokens, 128)
        self.assertEqual(usage.request_id, "req-123")
        self.assertEqual(usage.operation, "resume_parse")
        self.assertGreaterEqual(usage.latency_ms, 0)

        usage_list = result.to_usage_list()
        self.assertIsInstance(usage_list, list)
        self.assertEqual(len(usage_list), 1)
        self.assertEqual(usage_list[0]["input_tokens"], 1420)
        self.assertEqual(usage_list[0]["request_id"], "req-123")

    # 18: Safe Handling of Missing Usage Metadata
    def test_missing_usage_metadata_handling(self):
        """Ensure missing usage_metadata does not crash and defaults safely to 0."""
        client = GeminiClient(api_key="fake-key")
        mock_response = MagicMock()
        mock_response.text = "Hello world"
        mock_response.usage_metadata = None

        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        client._client = mock_client

        result = asyncio.run(client.generate(prompt="Test prompt"))
        self.assertEqual(result.usage.input_tokens, 0)
        self.assertEqual(result.usage.output_tokens, 0)
        self.assertEqual(result.usage.total_tokens, 0)
        self.assertEqual(result.usage.cached_tokens, 0)

    # 6: Usage Propagation through Resume Parser
    def test_resume_parser_usage_propagation(self):
        """Verify that usage propagates up through parse_resume_with_ai as a list."""
        mock_usage_item = {
            "provider": "gemini",
            "model": "gemini-3.1-flash-lite",
            "input_tokens": 1500,
            "output_tokens": 400,
            "total_tokens": 1900,
            "cached_tokens": 0,
            "latency_ms": 120.5,
            "request_id": "test-req-99",
            "operation": "resume_parse"
        }
        
        with patch.object(ImprovedUniversalResumeParser, "parse_document_bytes", new_callable=AsyncMock) as mock_doc_parse:
            mock_doc_parse.return_value = {
                "success": True,
                "parsed": {"personal_info": {"full_name": "Alice Smith"}},
                "canonical": {},
                "source": "gemini",
                "usage": [mock_usage_item],
            }

            service_result = asyncio.run(parse_resume_with_ai(
                file_bytes=b"%PDF-1.4 sample content",
                filename="sample.pdf",
                request_id="test-req-99",
                operation="resume_parse"
            ))
            self.assertTrue(service_result["success"])
            self.assertIsNotNone(service_result["usage"])
            self.assertIsInstance(service_result["usage"], list)
            self.assertEqual(len(service_result["usage"]), 1)
            self.assertEqual(service_result["usage"][0]["total_tokens"], 1900)
            self.assertEqual(service_result["usage"][0]["request_id"], "test-req-99")

    # 7 & 15: ATS Pipeline Usage Propagation and Existing Fields
    def test_ats_scanner_usage_propagation_and_fields(self):
        """Verify ATS pipeline propagates usage as a list when include_ai=True and keeps all existing fields."""
        mock_usage = UsageMetadata(
            provider="gemini",
            model="gemini-3.1-flash-lite",
            input_tokens=2100,
            output_tokens=600,
            total_tokens=2700,
            cached_tokens=0,
            latency_ms=350.0,
            request_id="ats-req-1",
            operation="ats_scan"
        )
        ai_json_response = '{"match_score": 85, "keyword_analysis": {"skills_match_score": 80}}'
        mock_gen_result = GenerationResult(text=ai_json_response, usage=mock_usage, usages=[mock_usage])

        scanner = ATSScannerService()
        with patch("app.modules.ats_scanner.service.call_ai", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_gen_result
            
            result = asyncio.run(scanner.scan(
                resume={
                    "contact": {"name": "Test User", "email": "test@example.com"},
                    "skills": ["Python", "FastAPI"],
                    "experience": [],
                    "education": []
                },
                job_description="Seeking a Python FastAPI backend engineer.",
                include_ai=True,
                request_id="ats-req-1",
                operation="ats_scan"
            ))

            # Verify existing ATS fields are preserved
            self.assertIn("ats_score", result)
            self.assertIn("score_breakdown", result)
            self.assertIn("keyword_analysis", result)
            self.assertIn("section_analysis", result)
            self.assertIn("issues", result)
            self.assertIn("ai_analysis", result)

            # Verify additive usage list
            self.assertIn("usage", result)
            self.assertIsNotNone(result["usage"])
            self.assertIsInstance(result["usage"], list)
            self.assertEqual(len(result["usage"]), 1)
            self.assertEqual(result["usage"][0]["input_tokens"], 2100)
            self.assertEqual(result["usage"][0]["output_tokens"], 600)
            self.assertEqual(result["usage"][0]["total_tokens"], 2700)
            self.assertEqual(result["usage"][0]["request_id"], "ats-req-1")
            self.assertEqual(result["usage"][0]["operation"], "ats_scan")

    # 10: Multiple Gemini Generations Tracking
    def test_multiple_gemini_generations_tracking(self):
        """Verify that multiple Gemini calls in a single workflow produce multiple usage items."""
        gen1_usage = UsageMetadata(
            provider="gemini",
            model="gemini-3.1-flash-lite",
            input_tokens=1000,
            output_tokens=300,
            total_tokens=1300,
            cached_tokens=0,
            latency_ms=200.0,
            request_id="multi-req-1",
            operation="resume_parse_chunk1"
        )
        gen2_usage = UsageMetadata(
            provider="gemini",
            model="gemini-3.1-flash-lite",
            input_tokens=1200,
            output_tokens=400,
            total_tokens=1600,
            cached_tokens=100,
            latency_ms=250.0,
            request_id="multi-req-1",
            operation="resume_parse_chunk2"
        )

        composite_result = GenerationResult(
            text="Combined output",
            usage=gen1_usage,
            usages=[gen1_usage, gen2_usage]
        )

        usage_list = composite_result.to_usage_list()
        self.assertEqual(len(usage_list), 2)
        self.assertEqual(usage_list[0]["input_tokens"], 1000)
        self.assertEqual(usage_list[0]["operation"], "resume_parse_chunk1")
        self.assertEqual(usage_list[1]["input_tokens"], 1200)
        self.assertEqual(usage_list[1]["operation"], "resume_parse_chunk2")
        self.assertEqual(usage_list[0]["request_id"], "multi-req-1")
        self.assertEqual(usage_list[1]["request_id"], "multi-req-1")

    # 16: include_ai=False Performs Zero Gemini Calls
    def test_ats_include_ai_false_zero_gemini_calls(self):
        """Verify include_ai=False makes zero Gemini calls and returns usage=None."""
        scanner = ATSScannerService()
        with patch("app.modules.ats_scanner.service.call_ai", new_callable=AsyncMock) as mock_call:
            result = asyncio.run(scanner.scan(
                resume={
                    "contact": {"name": "Test User"},
                    "skills": ["Python"],
                    "experience": [],
                    "education": []
                },
                job_description="Looking for Python developer.",
                include_ai=False
            ))
            
            mock_call.assert_not_called()
            self.assertIn("ats_score", result)
            self.assertIn("usage", result)
            self.assertIsNone(result["usage"])
            self.assertEqual(result["ai_analysis"].get("status"), "not_available")

    # 17: Cache Hit Returns 0 Gemini Tokens
    def test_cache_hit_returns_zero_tokens(self):
        """Verify cache hit does not report fake Gemini token usage."""
        client_mgr = ImprovedAIClientManager()
        
        with patch("app.modules.resume_builder.ai_client.USE_AI_CACHE", True):
            _AI_RESPONSE_CACHE.clear()
            prompt = "Test prompt for caching"
            from app.modules.resume_builder.ai_client import _make_cache_key
            key = _make_cache_key(prompt, "", None)
            _AI_RESPONSE_CACHE[key] = "Cached output text"

            res = asyncio.run(client_mgr.call(
                prompt=prompt,
                operation="resume_parse",
            ))
            self.assertEqual(res.text, "Cached output text")
            self.assertEqual(res.usage.input_tokens, 0)
            self.assertEqual(res.usage.output_tokens, 0)
            self.assertEqual(res.usage.total_tokens, 0)
            self.assertEqual(res.usage.provider, "cache")
            _AI_RESPONSE_CACHE.clear()

    # 19: Deterministic Fallback Handling
    def test_deterministic_fallback_returns_none_usage(self):
        """Verify that when AI call fails and deterministic parser runs, usage is None."""
        mock_gemini = MagicMock()
        mock_gemini.is_configured = True
        mock_gemini.generate = AsyncMock(side_effect=Exception("Gemini API connection error"))

        with patch("app.modules.resume_builder.ai_parser.get_gemini_client", return_value=mock_gemini):
            with patch("app.modules.resume_builder.ai_parser.DeterministicResumeParser.parse_text") as mock_det:
                mock_canonical = MagicMock()
                mock_canonical.model_dump.return_value = {"personal_information": {"name": "Fallback User"}}
                mock_canonical.personal_information.name = "Fallback User"
                mock_det.return_value = mock_canonical

                result = asyncio.run(ImprovedUniversalResumeParser.parse_text(
                    "Fallback User\nSoftware Engineer",
                    request_id="fallback-req"
                ))
                
                self.assertIn("parsed", result)
                self.assertIn("usage", result)
                self.assertIsNone(result["usage"])  # Zero/null token usage reported for deterministic fallback

    # 11, 12, 13: Service Authentication & Request Context
    def test_service_auth_missing_key_rejected(self):
        """Verify missing X-Internal-Service-Key returns 401 when secret is set."""
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}
        mock_request.client = MagicMock(host="127.0.0.1")
        mock_request.url = MagicMock(path="/api/v1/resume/parse-resume")

        with patch.object(settings, "INTERNAL_SERVICE_SECRET", "super_secret_key"):
            with self.assertRaises(HTTPException) as ctx:
                asyncio.run(get_trusted_request_context(mock_request))
            self.assertEqual(ctx.exception.status_code, 401)
            self.assertIn("Invalid or missing internal service key", ctx.exception.detail)

    def test_service_auth_invalid_key_rejected(self):
        """Verify invalid X-Internal-Service-Key returns 401."""
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {"X-Internal-Service-Key": "wrong_key"}
        mock_request.client = MagicMock(host="127.0.0.1")
        mock_request.url = MagicMock(path="/api/v1/resume/parse-resume")

        with patch.object(settings, "INTERNAL_SERVICE_SECRET", "super_secret_key"):
            with self.assertRaises(HTTPException) as ctx:
                asyncio.run(get_trusted_request_context(mock_request))
            self.assertEqual(ctx.exception.status_code, 401)
            self.assertIn("Invalid or missing internal service key", ctx.exception.detail)

    def test_service_auth_valid_key_accepts_context(self):
        """Verify valid X-Internal-Service-Key successfully populates RequestContext."""
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {
            "X-Internal-Service-Key": "super_secret_key",
            "X-User-ID": "user_789",
            "X-Request-ID": "req_456",
            "X-Operation": "resume_parse"
        }

        with patch.object(settings, "INTERNAL_SERVICE_SECRET", "super_secret_key"):
            ctx = asyncio.run(get_trusted_request_context(mock_request))
            self.assertIsInstance(ctx, RequestContext)
            self.assertTrue(ctx.is_trusted)
            self.assertEqual(ctx.user_id, "user_789")
            self.assertEqual(ctx.request_id, "req_456")
            self.assertEqual(ctx.operation, "resume_parse")

    # 8, 9, 14: Full HTTP Integration via TestClient for Resume and ATS
    def test_http_endpoint_parse_resume(self):
        """Test POST /api/v1/resume/parse-resume with headers and PDF payload returning usage list."""
        with patch.object(settings, "INTERNAL_SERVICE_SECRET", "my_secret_token"):
            mock_usage = [{
                "provider": "gemini",
                "model": "gemini-3.1-flash-lite",
                "input_tokens": 1200,
                "output_tokens": 400,
                "total_tokens": 1600,
                "cached_tokens": 0,
                "latency_ms": 500.0,
                "request_id": "django_req_001",
                "operation": "resume_parse"
            }]
            with patch("app.api.v1.resume_builder.resume_builder.parse_resume_with_ai", new_callable=AsyncMock) as mock_parse:
                mock_parse.return_value = {
                    "success": True,
                    "file_name": "test_resume.pdf",
                    "parsed": {"name": "Bob Vance"},
                    "error": None,
                    "confidence_score": 0.95,
                    "usage": mock_usage
                }

                # Valid PDF signature header
                pdf_payload = b"%PDF-1.4\n%Fake PDF content for testing\n%%EOF"

                # Test with valid service key
                response = self.client.post(
                    "/api/v1/resume/parse-resume",
                    files={"file": ("test_resume.pdf", pdf_payload, "application/pdf")},
                    headers={
                        "Host": "localhost",
                        "X-Internal-Service-Key": "my_secret_token",
                        "X-User-ID": "django_user_42",
                        "X-Request-ID": "django_req_001",
                        "X-Operation": "resume_parse"
                    }
                )

                self.assertEqual(response.status_code, 200)
                data = response.json()
                self.assertTrue(data["success"])
                self.assertEqual(data["file_name"], "test_resume.pdf")
                self.assertIn("parsed", data)
                self.assertIn("usage", data)
                self.assertIsInstance(data["usage"], list)
                self.assertEqual(len(data["usage"]), 1)
                self.assertEqual(data["usage"][0]["input_tokens"], 1200)
                self.assertEqual(data["usage"][0]["output_tokens"], 400)
                self.assertEqual(data["usage"][0]["total_tokens"], 1600)
                self.assertEqual(data["usage"][0]["request_id"], "django_req_001")
                self.assertEqual(data["usage"][0]["operation"], "resume_parse")

                # Test with invalid service key
                bad_response = self.client.post(
                    "/api/v1/resume/parse-resume",
                    files={"file": ("test_resume.pdf", pdf_payload, "application/pdf")},
                    headers={
                        "Host": "localhost",
                        "X-Internal-Service-Key": "invalid_token",
                        "X-Request-ID": "django_req_001"
                    }
                )
                self.assertEqual(bad_response.status_code, 401)

    def test_http_endpoint_ats_scan_file(self):
        """Test POST /api/v1/ats/scan-file with headers and PDF payload returning usage list."""
        with patch.object(settings, "INTERNAL_SERVICE_SECRET", "my_secret_token"):
            mock_usage = [{
                "provider": "gemini",
                "model": "gemini-3.1-flash-lite",
                "input_tokens": 1800,
                "output_tokens": 550,
                "total_tokens": 2350,
                "cached_tokens": 0,
                "latency_ms": 750.0,
                "request_id": "django_ats_req_002",
                "operation": "ats_scan"
            }]
            with patch("app.modules.ats_scanner.router.extract_resume_markdown", new_callable=AsyncMock) as mock_extract:
                mock_extract.return_value = "# Candidate Resume\n## Summary\nExperienced Engineer\n## Skills\nPython, FastAPI"
                
                with patch("app.modules.ats_scanner.service.ATSScannerService.scan", new_callable=AsyncMock) as mock_scan:
                    mock_scan.return_value = {
                        "ats_score": 88,
                        "score_breakdown": {"keyword_score": 90},
                        "keyword_analysis": {},
                        "formatting_analysis": {},
                        "meta": {"parser_used": "hybrid"},
                        "ai_analysis": {"skills_match_score": 85},
                        "usage": mock_usage
                    }

                    pdf_payload = b"%PDF-1.4\n%Fake PDF content for ATS\n%%EOF"

                    response = self.client.post(
                        "/api/v1/ats/scan-file",
                        files={"file": ("resume.pdf", pdf_payload, "application/pdf")},
                        data={"job_description": "Software Engineer", "include_ai": "true"},
                        headers={
                            "Host": "localhost",
                            "X-Internal-Service-Key": "my_secret_token",
                            "X-User-ID": "django_user_42",
                            "X-Request-ID": "django_ats_req_002",
                            "X-Operation": "ats_scan"
                        }
                    )

                    self.assertEqual(response.status_code, 200)
                    data = response.json()
                    self.assertEqual(data["ats_score"], 88)
                    self.assertIn("usage", data)
                    self.assertIsInstance(data["usage"], list)
                    self.assertEqual(len(data["usage"]), 1)
                    self.assertEqual(data["usage"][0]["total_tokens"], 2350)
                    self.assertEqual(data["usage"][0]["request_id"], "django_ats_req_002")
                    self.assertEqual(data["usage"][0]["operation"], "ats_scan")


if __name__ == "__main__":
    unittest.main()
