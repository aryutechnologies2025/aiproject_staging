"""
test_ai_suggestions.py — Test Suite for Grounded, ATS-Friendly AI Resume Suggestions

Implements:
TEST 1 — MBA Education (Grounded, concise, zero hallucinated subjects/coursework)
TEST 2 — Software Tester with QC title (Software QA domain detection, no manufacturing content)
TEST 3 — Manufacturing QC (Manufacturing QC context, no software testing assumptions)
TEST 4 — Ambiguous QC (Neutral / ambiguous context, no invented domain)
TEST 5 — User prompt (Style/tone guidance respected without altering factual truth)
TEST 6 — ATS keyword preservation (Technical keywords strictly retained without degradation)
Plus backward compatibility verification of API response schemas.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from app.modules.resume_builder.context_analyzer import (
    DomainType,
    ResumeContext,
    analyze_resume_context,
    build_grounded_education_prompt,
    build_grounded_experience_prompt,
    build_grounded_skills_prompt,
    build_grounded_summary_prompt,
    sanitize_education_output,
    sanitize_experience_output,
    sanitize_skills_output,
    sanitize_summary_output,
)
from app.modules.resume_builder.service import (
    build_skills_prompt,
    suggest_education,
    suggest_experience,
    suggest_skills,
    suggest_summary,
)


class TestAISuggestions(unittest.TestCase):
    """Unit and Integration Tests for AI Resume Suggestions."""

    # ─────────────────────────────────────────────────────────────────────────
    # TEST 1: MBA Education
    # ─────────────────────────────────────────────────────────────────────────
    def test_1_mba_education_factual_grounding(self):
        """
        Input: MBA from Rathinam Institute of Management, Coimbatore, 2023.
        Expected: Does not invent subjects or achievements, uses actual degree/college/year, concise, ATS-friendly.
        """
        payload = {
            "degree": "MBA",
            "college": "Rathinam Institute of Management",
            "location": "Coimbatore",
            "year": "2023",
            "prompt": "make it professional and human written style only i want"
        }

        ctx = analyze_resume_context(payload)
        prompt = build_grounded_education_prompt(payload, ctx)

        # Verify prompt rules prohibit hallucinated subjects
        self.assertIn("MBA", prompt)
        self.assertIn("Rathinam Institute of Management", prompt)
        self.assertIn("Coimbatore", prompt)
        self.assertIn("2023", prompt)
        self.assertIn("DO NOT invent subjects", prompt)
        self.assertIn("DO NOT invent fake GPA", prompt)

        # Verify style instruction parsed
        self.assertIn("human-written tone", ctx.style_guidelines)
        self.assertIn("professional tone", ctx.style_guidelines.lower())

        # Test bad response interception
        bad_response = """Studied advanced organizational behavior and strategic management principles to optimize business operations
Analyzed financial accounting and corporate finance frameworks to support data-driven decision making
Explored marketing strategies and consumer behavior patterns to enhance brand positioning and market reach
Examined human resource management practices and organizational development theories for workforce optimization
Applied operational research and supply chain management concepts to improve process efficiency and logistics"""

        cleaned = sanitize_education_output(bad_response, payload, ctx)

        # Must NOT contain hallucinated subjects
        for bad_subj in [
            "organizational behavior", "financial accounting", "corporate finance",
            "marketing strategies", "human resource management", "supply chain"
        ]:
            for line in cleaned:
                self.assertNotIn(bad_subj, line.lower())

        # Must produce concise, factual statement containing candidate credentials
        self.assertTrue(len(cleaned) >= 1)
        first_line = cleaned[0]
        self.assertIn("MBA", first_line)
        self.assertIn("Rathinam Institute of Management", first_line)
        self.assertIn("2023", first_line)

    # ─────────────────────────────────────────────────────────────────────────
    # TEST 2: Software Tester with QC Title
    # ─────────────────────────────────────────────────────────────────────────
    def test_2_software_tester_with_qc_title(self):
        """
        Job title: QC
        Experience: Performed manual testing of web applications, created test cases, executed regression testing and reported bugs in Jira.
        Skills: Selenium, Postman, Jira, SQL
        Expected: Detects software testing domain, treats QC as Software QA, does NOT generate manufacturing content, uses software testing keywords.
        """
        payload = {
            "job_title": "QC",
            "experience": "Performed manual testing of web applications, created test cases, executed regression testing and reported bugs in Jira.",
            "skills": ["Selenium", "Postman", "Jira", "SQL"]
        }

        ctx = analyze_resume_context(payload)

        # Domain inference
        self.assertEqual(ctx.inferred_domain, DomainType.SOFTWARE_TESTING)
        self.assertIn("Software", ctx.domain_label)
        self.assertIn("Software QA", ctx.resolved_role)

        # Prohibits manufacturing
        for bad_domain in ("manufacturing", "factory inspection", "production quality", "raw materials"):
            self.assertIn(bad_domain, ctx.prohibited_domains)

        # Grounded Experience prompt
        exp_prompt = build_grounded_experience_prompt(payload, ctx)
        self.assertIn("Software Quality Assurance & Testing", exp_prompt)
        self.assertIn("Selenium", exp_prompt)
        self.assertIn("Postman", exp_prompt)
        self.assertIn("Jira", exp_prompt)
        self.assertIn("SQL", exp_prompt)
        self.assertIn("STRICTLY PROHIBITED TOPICS", exp_prompt)

        # Verify sanitizer rejects manufacturing contamination
        contaminated_output = """Executed regression testing for web applications and logged defects in Jira.
Inspected raw materials and finished goods on the production floor.
Automated API tests using Postman and validated response payloads.
Conducted shop floor quality inspections according to ISO manufacturing standards."""

        sanitized_bullets = sanitize_experience_output(contaminated_output, payload, ctx)
        for bullet in sanitized_bullets:
            self.assertNotIn("raw materials", bullet.lower())
            self.assertNotIn("production floor", bullet.lower())
            self.assertNotIn("shop floor", bullet.lower())
            self.assertNotIn("iso manufacturing", bullet.lower())

        self.assertEqual(len(sanitized_bullets), 2)
        self.assertTrue(any("jira" in b.lower() for b in sanitized_bullets))
        self.assertTrue(any("postman" in b.lower() for b in sanitized_bullets))

    # ─────────────────────────────────────────────────────────────────────────
    # TEST 3: Manufacturing QC
    # ─────────────────────────────────────────────────────────────────────────
    def test_3_manufacturing_qc(self):
        """
        Job title: QC Inspector
        Experience: Inspected raw materials and finished products on the production floor.
        Expected: Manufacturing/production QC context, no Selenium/API/software-testing assumptions.
        """
        payload = {
            "job_title": "QC Inspector",
            "experience": "Inspected raw materials and finished products on the production floor."
        }

        ctx = analyze_resume_context(payload)

        # Domain inference
        self.assertEqual(ctx.inferred_domain, DomainType.MANUFACTURING_QC)
        self.assertIn("Manufacturing", ctx.domain_label)
        self.assertIn("QC Inspector", ctx.resolved_role)

        # Prohibits software testing
        for sw_term in ("selenium", "postman", "api testing", "software testing"):
            self.assertIn(sw_term, ctx.prohibited_domains)

        exp_prompt = build_grounded_experience_prompt(payload, ctx)
        self.assertIn("Manufacturing Quality Control", exp_prompt)
        self.assertIn("STRICTLY PROHIBITED TOPICS: Do NOT mention or invent: selenium, postman", exp_prompt)

        # Verify sanitizer rejects software testing contamination
        contaminated_output = """Inspected raw materials and verified vendor certifications upon receipt.
Performed automated regression testing using Selenium WebDriver.
Monitored finished goods on the production floor to guarantee adherence to quality standards.
Created test cases in Jira for REST API endpoints."""

        sanitized_bullets = sanitize_experience_output(contaminated_output, payload, ctx)
        for bullet in sanitized_bullets:
            self.assertNotIn("selenium", bullet.lower())
            self.assertNotIn("jira", bullet.lower())
            self.assertNotIn("api", bullet.lower())

        self.assertEqual(len(sanitized_bullets), 2)
        self.assertTrue(any("raw materials" in b.lower() for b in sanitized_bullets))
        self.assertTrue(any("finished goods" in b.lower() for b in sanitized_bullets))

    # ─────────────────────────────────────────────────────────────────────────
    # TEST 4: Ambiguous QC
    # ─────────────────────────────────────────────────────────────────────────
    def test_4_ambiguous_qc(self):
        """
        Job title: QC (Very limited information).
        Expected: Does not invent a domain, does not assume manufacturing, does not assume software testing, uses only available factual info.
        """
        payload = {
            "job_title": "QC"
        }

        ctx = analyze_resume_context(payload)

        self.assertEqual(ctx.inferred_domain, DomainType.AMBIGUOUS_NEUTRAL)
        self.assertTrue(ctx.is_ambiguous)
        self.assertEqual(ctx.resolved_role, "QC")

        # Prohibits assuming either software or manufacturing
        self.assertTrue(any("software testing" in p for p in ctx.prohibited_domains))
        self.assertTrue(any("raw materials" in p for p in ctx.prohibited_domains))

        exp_prompt = build_grounded_experience_prompt(payload, ctx)
        self.assertIn("Ambiguous / Neutral QC", exp_prompt)
        self.assertIn("use neutral quality control terminology", exp_prompt)

    # ─────────────────────────────────────────────────────────────────────────
    # TEST 5: User Prompt Handling
    # ─────────────────────────────────────────────────────────────────────────
    def test_5_user_prompt_handling(self):
        """
        Input contains: prompt = "Make it professional and human written style"
        Expected: Changes writing style, does not change factual content, does not introduce unsupported information.
        """
        payload = {
            "degree": "B.Tech",
            "college": "Anna University",
            "year": "2022",
            "prompt": "Make it professional and human written style"
        }

        ctx = analyze_resume_context(payload)

        # Style guidelines captured
        self.assertIn("human-written tone", ctx.style_guidelines)
        self.assertIn("professional tone", ctx.style_guidelines.lower())

        # Grounding remains strictly verified
        self.assertEqual(ctx.verified_education["degree"], "B.Tech")
        self.assertEqual(ctx.verified_education["college"], "Anna University")
        self.assertEqual(ctx.verified_education["year"], "2022")
        self.assertEqual(ctx.verified_education["coursework"], [])

        prompt = build_grounded_education_prompt(payload, ctx)
        self.assertIn("B.Tech", prompt)
        self.assertIn("Anna University", prompt)
        self.assertIn("DO NOT invent subjects", prompt)

    # ─────────────────────────────────────────────────────────────────────────
    # TEST 6: ATS Keyword Preservation
    # ─────────────────────────────────────────────────────────────────────────
    def test_6_ats_keyword_preservation(self):
        """
        Input: Selenium, Postman, Jira, API Testing, Regression Testing
        Expected: Important keywords remain present in generated Skills/Summary/Experience where relevant,
                  no keyword stuffing, no replacement with vague generic terminology.
        """
        input_keywords = ["Selenium", "Postman", "Jira", "API Testing", "Regression Testing"]
        payload = {
            "job_titles": ["QC"],
            "skills": input_keywords,
            "experience": "Executed manual and API testing using Postman; tracked bugs in Jira."
        }

        ctx = analyze_resume_context(payload)
        skills_prompt = build_grounded_skills_prompt(payload, ctx)

        # Prompt explicitly commands preserving input keywords
        for kw in input_keywords:
            self.assertIn(kw, skills_prompt)

        raw_skills_response = """Selenium
Postman
Jira
API Testing
Regression Testing
Test Case Design
Leadership
Teamwork"""

        sanitized_skills = sanitize_skills_output(raw_skills_response, payload, ctx)

        # Verify all input keywords preserved
        for kw in input_keywords:
            self.assertIn(kw, sanitized_skills)

        # Verify generic soft skills filtered out
        self.assertNotIn("Leadership", sanitized_skills)
        self.assertNotIn("Teamwork", sanitized_skills)

    # ─────────────────────────────────────────────────────────────────────────
    # INTEGRATION & ASYNC SERVICE TESTS (MOCKED LLM)
    # ─────────────────────────────────────────────────────────────────────────
    def test_service_suggest_education(self):
        """Verify suggest_education end-to-end with mocked LLM."""
        payload = {
            "degree": "MBA",
            "college": "Rathinam Institute of Management",
            "location": "Coimbatore",
            "year": "2023",
            "prompt": "make it professional and human written style only i want"
        }

        async def _test():
            mock_db = AsyncMock()
            with patch("app.modules.resume_builder.service._call_llm_with_telemetry", new_callable=AsyncMock) as mock_llm:
                mock_llm.return_value = "MBA, Rathinam Institute of Management, Coimbatore | 2023"
                result = await suggest_education(payload, mock_db)

                self.assertIn("education_bullets", result)
                self.assertIn("count", result)
                self.assertIn("quality_notes", result)
                self.assertIn("MBA", result["education_bullets"])
                self.assertIn("Rathinam Institute of Management", result["education_bullets"])
                self.assertEqual(result["count"], 1)

        asyncio.run(_test())

    def test_service_suggest_experience(self):
        """Verify suggest_experience end-to-end with mocked LLM and domain grounding."""
        payload = {
            "job_title": "QC",
            "company": "Tech Solutions",
            "experience": "Performed manual testing of web applications, created test cases, executed regression testing and reported bugs in Jira.",
            "skills": ["Selenium", "Postman", "Jira", "SQL"]
        }

        async def _test():
            mock_db = AsyncMock()
            with patch("app.modules.resume_builder.service._call_llm_with_telemetry", new_callable=AsyncMock) as mock_llm:
                mock_llm.return_value = """Executed regression testing for web applications and reported defects using Jira.
Validated REST API endpoints using Postman to ensure reliable service integration.
Created comprehensive test cases and executed functional testing across sprint cycles."""

                result = await suggest_experience(payload, mock_db)

                self.assertIn("experience_bullets", result)
                self.assertIn("count", result)
                self.assertIn("quality_notes", result)
                self.assertEqual(result["count"], 3)
                self.assertIn("Software Quality Assurance & Testing", result["quality_notes"])
                self.assertIn("Jira", result["experience_bullets"])
                self.assertIn("Postman", result["experience_bullets"])

        asyncio.run(_test())

    def test_service_suggest_skills(self):
        """Verify suggest_skills end-to-end with keyword preservation and soft-skill stripping."""
        payload = {
            "job_titles": ["QC"],
            "skills": ["Selenium", "Postman", "Jira", "SQL", "API Testing"],
            "experience": "Executed manual and automated testing for web applications."
        }

        async def _test():
            mock_db = AsyncMock()
            with patch("app.modules.resume_builder.service._call_llm_with_telemetry", new_callable=AsyncMock) as mock_llm:
                mock_llm.return_value = """Selenium
Postman
Jira
SQL
API Testing
Regression Testing
Functional Testing
Communication"""

                result = await suggest_skills(payload, mock_db)

                self.assertIn("skills", result)
                self.assertIn("count", result)
                self.assertIn("quality_notes", result)

                skills = result["skills"]
                self.assertIn("Selenium", skills)
                self.assertIn("Postman", skills)
                self.assertIn("Jira", skills)
                self.assertIn("SQL", skills)
                self.assertIn("API Testing", skills)
                self.assertNotIn("Communication", skills)

        asyncio.run(_test())

    def test_service_suggest_summary(self):
        """Verify suggest_summary end-to-end produces concise 2-4 sentences without generic fluff."""
        payload = {
            "job_title": "QC",
            "skills": ["Selenium", "Postman", "Jira", "SQL"],
            "experiences": [
                {
                    "job_title": "QC",
                    "company": "TechCorp",
                    "start_date": "2021-01-01",
                    "end_date": "2024-01-01",
                    "description": "Executed regression testing for web applications and reported defects in Jira."
                }
            ],
            "prompt": "make it professional and human written style only i want"
        }

        async def _test():
            mock_db = AsyncMock()
            with patch("app.modules.resume_builder.service._call_llm_with_telemetry", new_callable=AsyncMock) as mock_llm:
                mock_llm.return_value = """Results-driven professional Software Tester with 3+ years of experience in functional, regression, and API testing of web applications. Skilled in Selenium, Postman, Jira, and SQL with a proven track record in defect identification and test execution."""

                result = await suggest_summary(payload, mock_db)

                self.assertIn("summary", result)
                self.assertIn("line_count", result)
                self.assertIn("quality_notes", result)

                summary = result["summary"]
                self.assertNotIn("Results-driven professional", summary)
                self.assertIn("Software Tester", summary)
                self.assertIn("Selenium", summary)
                self.assertIn("Postman", summary)
                self.assertIn("Jira", summary)
                self.assertIn("SQL", summary)

        asyncio.run(_test())

    def test_build_skills_prompt_backward_compatibility(self):
        """Verify build_skills_prompt signature remains 100% backward compatible."""
        # Old caller syntax with only positional args
        prompt = build_skills_prompt(["Software Engineer"], "senior")
        self.assertIsInstance(prompt, str)
        self.assertIn("Software Engineer", prompt)
        self.assertIn("senior", prompt)

    def test_api_endpoints_via_testclient(self):
        """Test API endpoints via TestClient to verify backward compatibility of HTTP contract."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from app.api.v1.resume_builder.resume_builder import router as resume_router
        from app.core.database import get_db

        app = FastAPI()
        mock_db = AsyncMock()

        async def override_get_db():
            yield mock_db

        app.dependency_overrides[get_db] = override_get_db
        app.include_router(resume_router, prefix="/api/v1/resume")
        client = TestClient(app)

        with patch("app.modules.resume_builder.service._call_llm_with_telemetry", new_callable=AsyncMock) as mock_llm:
            # 1. Education endpoint
            mock_llm.return_value = "MBA, Rathinam Institute of Management, Coimbatore | 2023"
            res = client.post("/api/v1/resume/education", json={
                "degree": "MBA",
                "college": "Rathinam Institute of Management",
                "location": "Coimbatore",
                "year": "2023",
                "prompt": "make it professional and human written style only i want"
            })
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertIn("education_bullets", data)
            self.assertIn("count", data)
            self.assertIn("quality_notes", data)
            self.assertIn("MBA", data["education_bullets"])

            # 2. Experience endpoint
            mock_llm.return_value = "Executed regression testing for web applications and reported bugs in Jira.\nAutomated API tests using Postman."
            res = client.post("/api/v1/resume/experience", json={
                "job_title": "QC",
                "company": "TechCorp",
                "experience": "Performed manual testing of web applications, created test cases, executed regression testing and reported bugs in Jira.",
                "skills": ["Selenium", "Postman", "Jira", "SQL"]
            })
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertIn("experience_bullets", data)
            self.assertIn("count", data)
            self.assertIn("quality_notes", data)

            # 3. Skills endpoint
            mock_llm.return_value = "Selenium\nPostman\nJira\nSQL\nAPI Testing"
            res = client.post("/api/v1/resume/skills", json={
                "job_titles": ["QC"],
                "skills": ["Selenium", "Postman", "Jira", "SQL", "API Testing"]
            })
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertIn("skills", data)
            self.assertIn("count", data)
            self.assertIn("quality_notes", data)
            self.assertIn("Selenium", data["skills"])

            # 4. Summary endpoint
            mock_llm.return_value = "Software Tester with experience in functional, regression, and API testing of web applications. Skilled in Selenium, Postman, Jira, and SQL."
            res = client.post("/api/v1/resume/summary", json={
                "job_title": "QC",
                "skills": ["Selenium", "Postman", "Jira", "SQL"],
                "experiences": [{"job_title": "QC", "company": "TechCorp", "description": "Executed regression testing"}]
            })
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertIn("summary", data)
            self.assertIn("line_count", data)
            self.assertIn("quality_notes", data)
            self.assertIn("Software Tester", data["summary"])


if __name__ == "__main__":
    unittest.main()

