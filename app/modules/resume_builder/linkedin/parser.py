"""
parser.py — Convert raw scraped LinkedIn HTML/text into structured LinkedInProfile.

Strategy:
  1. Dedicated CSS-selector-based extractor for each section (fast, deterministic)
  2. AI fallback via Claude API for messy/unparseable sections
  3. All domain support — tech, healthcare, legal, finance, education, creative, etc.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from bs4 import BeautifulSoup, Tag

from .schemas import (
    Award, Certification, ContactInfo, DateRange, Education, Experience,
    Language, LanguageProficiency, LinkedInProfile, Location, Patent,
    Project, Publication, Recommendation, Skill, TestScore, VolunteerExperience,
)
from .utils import clean_text, parse_date_range

logger = logging.getLogger(__name__)


# ─────────────────────────── Helpers ──────────────────────────────────────────

def _soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "html.parser")


def _text(el: Optional[Tag]) -> Optional[str]:
    if el is None:
        return None
    return clean_text(el.get_text(separator=" ", strip=True))


def _make_date_range(text: str) -> Optional[DateRange]:
    if not text:
        return None
    sm, sy, em, ey, is_current = parse_date_range(text)
    if not any([sm, sy, em, ey, is_current]):
        return None
    return DateRange(
        start_month=sm, start_year=sy,
        end_month=em,   end_year=ey,
        is_current=is_current,
    )


# ─────────────────────────── Section Parsers ──────────────────────────────────

class LinkedInParser:
    """
    Parses raw BeautifulSoup objects from the LinkedIn profile page
    into a structured LinkedInProfile instance.

    All methods are fail-safe — they return empty lists / None on parse errors.
    """

    def __init__(self, soup: BeautifulSoup) -> None:
        self.soup = soup

    # ── Identity ──────────────────────────────────────────────────────────────

    def parse_basic_info(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {}

        # Full name
        name_el = (
            self.soup.select_one("h1.text-heading-xlarge") or
            self.soup.select_one("h1.inline") or
            self.soup.select_one("h1")
        )
        full_name = _text(name_el) or ""
        data["full_name"] = full_name
        parts = full_name.split(" ", 1)
        data["first_name"] = parts[0] if parts else None
        data["last_name"]  = parts[1] if len(parts) > 1 else None

        # Headline
        headline_el = (
            self.soup.select_one("div.text-body-medium.break-words") or
            self.soup.select_one(".pv-text-details__left-panel .text-body-medium")
        )
        data["headline"] = _text(headline_el)

        # About / Summary
        about_el = (
            self.soup.select_one("#about ~ div .visually-hidden") or
            self.soup.select_one("section#about .pv-shared-text-with-see-more span") or
            self.soup.select_one(".pv-about__summary-text span")
        )
        data["about"] = _text(about_el)

        # Location
        loc_el = (
            self.soup.select_one("span.text-body-small.inline.t-black--light.break-words") or
            self.soup.select_one(".pv-text-details__left-panel span.text-body-small")
        )
        loc_text = _text(loc_el) or ""
        data["location"] = self._parse_location_string(loc_text)

        # Followers / Connections
        connections_el = self.soup.select_one(".pvs-header__subtitle span")
        if connections_el:
            conn_text = _text(connections_el) or ""
            match = re.search(r"([\d,]+)\s+follower", conn_text)
            if match:
                data["followers"] = int(match.group(1).replace(",", ""))

        # Open to work badge
        data["open_to_work"] = bool(self.soup.select_one("#open-to-work-badge-icon"))

        return data

    def _parse_location_string(self, text: str) -> Optional[Location]:
        if not text:
            return None
        parts = [p.strip() for p in text.split(",")]
        return Location(
            city=parts[0] if len(parts) >= 1 else None,
            state=parts[1] if len(parts) >= 2 else None,
            country=parts[-1] if len(parts) >= 2 else None,
        )

    # ── Experience ────────────────────────────────────────────────────────────

    def parse_experience(self) -> List[Experience]:
        experiences = []
        section = (
            self.soup.select_one("#experience ~ div ul") or
            self.soup.select_one("section#experience .pvs-list")
        )
        if not section:
            return experiences

        items = section.select("li.artdeco-list__item")
        for item in items:
            try:
                exp = self._parse_experience_item(item)
                if exp:
                    experiences.append(exp)
            except Exception as exc:
                logger.debug(f"[Parser] Experience item error: {exc}")

        return experiences

    def _parse_experience_item(self, item: Tag) -> Optional[Experience]:
        # Try multi-role grouped entry first
        group_company_el = item.select_one("span.t-14.t-normal span[aria-hidden='true']")
        role_list = item.select("ul.pvs-list > li")

        if role_list:
            # Grouped: one company, multiple roles
            company = _text(group_company_el) or "Unknown Company"
            experiences_from_group = []
            for role_item in role_list:
                exp = self._parse_single_role(role_item, company_override=company)
                if exp:
                    experiences_from_group.append(exp)
            return experiences_from_group[0] if experiences_from_group else None

        return self._parse_single_role(item)

    def _parse_single_role(self, item: Tag, company_override: Optional[str] = None) -> Optional[Experience]:
        spans = item.select("span[aria-hidden='true']")
        texts = [_text(s) for s in spans if _text(s)]

        title   = texts[0] if len(texts) > 0 else "Unknown Title"
        company = company_override or (texts[1] if len(texts) > 1 else "Unknown Company")

        # Clean employment type from company (e.g. "Acme Corp · Full-time")
        emp_type = None
        if company and "·" in company:
            parts = company.split("·", 1)
            company  = parts[0].strip()
            emp_type = parts[1].strip()

        # Find date range (usually contains a · separator)
        date_text = None
        location_text = None
        for text in texts[2:]:
            if re.search(r"\d{4}", text):
                if date_text is None:
                    date_text = text
            elif text and not re.search(r"yr|mo|mos", text, re.I):
                if location_text is None:
                    location_text = text

        # Description
        desc_el = item.select_one(".pvs-list__outer-container .visually-hidden")
        description = _text(desc_el)

        # Skills used
        skill_matches = re.findall(r"Skills:\s*(.+)", description or "")
        skills_used = []
        if skill_matches:
            skills_used = [s.strip() for s in skill_matches[0].split("·")]
            description = re.sub(r"Skills:\s*.+", "", description or "").strip()

        return Experience(
            title=title,
            company=company,
            employment_type=emp_type,
            location=self._parse_location_string(location_text or ""),
            date_range=_make_date_range(date_text or ""),
            description=clean_text(description),
            skills_used=skills_used,
        )

    # ── Education ─────────────────────────────────────────────────────────────

    def parse_education(self) -> List[Education]:
        educations = []
        section = (
            self.soup.select_one("#education ~ div ul") or
            self.soup.select_one("section#education .pvs-list")
        )
        if not section:
            return educations

        for item in section.select("li.artdeco-list__item"):
            try:
                edu = self._parse_education_item(item)
                if edu:
                    educations.append(edu)
            except Exception as exc:
                logger.debug(f"[Parser] Education item error: {exc}")
        return educations

    def _parse_education_item(self, item: Tag) -> Optional[Education]:
        spans = [_text(s) for s in item.select("span[aria-hidden='true']") if _text(s)]

        institution = spans[0] if spans else "Unknown Institution"
        degree_field = spans[1] if len(spans) > 1 else None
        date_text    = next((s for s in spans[2:] if re.search(r"\d{4}", s)), None)

        degree = field = None
        if degree_field and "," in degree_field:
            parts = degree_field.split(",", 1)
            degree = parts[0].strip()
            field  = parts[1].strip()
        else:
            degree = degree_field

        # Grade / GPA
        grade_el = item.select_one(".t-14.t-normal.t-black--light span[aria-hidden='true']")
        grade_text = _text(grade_el)
        grade = None
        if grade_text and re.search(r"grade|gpa|score|percentage|distinction|cum laude", grade_text, re.I):
            grade = grade_text

        return Education(
            institution=institution,
            degree=degree,
            field_of_study=field,
            grade=grade,
            date_range=_make_date_range(date_text or ""),
        )

    # ── Skills ────────────────────────────────────────────────────────────────

    def parse_skills(self) -> List[Skill]:
        skills = []
        section = (
            self.soup.select_one("#skills ~ div ul") or
            self.soup.select_one("section#skills .pvs-list")
        )
        if not section:
            return skills

        for item in section.select("li.artdeco-list__item"):
            spans = [_text(s) for s in item.select("span[aria-hidden='true']") if _text(s)]
            if not spans:
                continue
            name = spans[0]
            endorsements = 0
            for span_text in spans[1:]:
                match = re.search(r"(\d+)\s+endorsement", span_text, re.I)
                if match:
                    endorsements = int(match.group(1))
                    break
            skills.append(Skill(name=name, endorsements=endorsements))

        return skills

    # ── Certifications ────────────────────────────────────────────────────────

    def parse_certifications(self) -> List[Certification]:
        certs = []
        section = (
            self.soup.select_one("#licenses_and_certifications ~ div ul") or
            self.soup.select_one("section#certifications .pvs-list")
        )
        if not section:
            return certs

        for item in section.select("li.artdeco-list__item"):
            spans = [_text(s) for s in item.select("span[aria-hidden='true']") if _text(s)]
            if not spans:
                continue
            name    = spans[0]
            issuer  = spans[1] if len(spans) > 1 else None
            issued  = next((s for s in spans[2:] if re.search(r"\d{4}|issued|expires", s, re.I)), None)
            cred_id = next((s for s in spans if re.search(r"credential id", s, re.I)), None)

            if cred_id:
                cred_id = re.sub(r"credential id[:\s]*", "", cred_id, flags=re.I).strip()

            certs.append(Certification(
                name=name,
                issuing_org=issuer,
                issue_date=_make_date_range(issued or ""),
                credential_id=cred_id,
            ))

        return certs

    # ── Projects ──────────────────────────────────────────────────────────────

    def parse_projects(self) -> List[Project]:
        projects = []
        section = (
            self.soup.select_one("#projects ~ div ul") or
            self.soup.select_one("section#projects .pvs-list")
        )
        if not section:
            return projects

        for item in section.select("li.artdeco-list__item"):
            spans = [_text(s) for s in item.select("span[aria-hidden='true']") if _text(s)]
            if not spans:
                continue
            name      = spans[0]
            date_text = next((s for s in spans[1:] if re.search(r"\d{4}", s)), None)
            desc_el   = item.select_one(".visually-hidden")
            projects.append(Project(
                name=name,
                date_range=_make_date_range(date_text or ""),
                description=_text(desc_el),
            ))

        return projects

    # ── Volunteer ─────────────────────────────────────────────────────────────

    def parse_volunteer(self) -> List[VolunteerExperience]:
        volunteer = []
        section = (
            self.soup.select_one("#volunteer_experience ~ div ul") or
            self.soup.select_one("section#volunteering .pvs-list")
        )
        if not section:
            return volunteer

        for item in section.select("li.artdeco-list__item"):
            spans = [_text(s) for s in item.select("span[aria-hidden='true']") if _text(s)]
            if len(spans) < 2:
                continue
            role = spans[0]
            org  = spans[1]
            date_text = next((s for s in spans[2:] if re.search(r"\d{4}", s)), None)
            volunteer.append(VolunteerExperience(
                role=role,
                organization=org,
                date_range=_make_date_range(date_text or ""),
            ))

        return volunteer

    # ── Languages ─────────────────────────────────────────────────────────────

    def parse_languages(self) -> List[Language]:
        langs = []
        section = (
            self.soup.select_one("#languages ~ div ul") or
            self.soup.select_one("section#languages .pvs-list")
        )
        if not section:
            return langs

        prof_levels = {v.value.lower(): v for v in LanguageProficiency}
        for item in section.select("li.artdeco-list__item"):
            spans = [_text(s) for s in item.select("span[aria-hidden='true']") if _text(s)]
            if not spans:
                continue
            name = spans[0]
            prof = None
            for span_text in spans[1:]:
                for level_str, level_enum in prof_levels.items():
                    if level_str in span_text.lower():
                        prof = level_enum
                        break
            langs.append(Language(name=name, proficiency=prof))

        return langs

    # ── Awards / Honors ───────────────────────────────────────────────────────

    def parse_awards(self) -> List[Award]:
        awards = []
        section = (
            self.soup.select_one("#honors_and_awards ~ div ul") or
            self.soup.select_one("section#honors .pvs-list")
        )
        if not section:
            return awards

        for item in section.select("li.artdeco-list__item"):
            spans = [_text(s) for s in item.select("span[aria-hidden='true']") if _text(s)]
            if not spans:
                continue
            title  = spans[0]
            issuer = spans[1] if len(spans) > 1 else None
            date_text = next((s for s in spans[2:] if re.search(r"\d{4}", s)), None)
            awards.append(Award(
                title=title,
                issuer=issuer,
                date=_make_date_range(date_text or ""),
            ))

        return awards

    # ── Publications ──────────────────────────────────────────────────────────

    def parse_publications(self) -> List[Publication]:
        pubs = []
        section = (
            self.soup.select_one("#publications ~ div ul") or
            self.soup.select_one("section#publications .pvs-list")
        )
        if not section:
            return pubs

        for item in section.select("li.artdeco-list__item"):
            spans = [_text(s) for s in item.select("span[aria-hidden='true']") if _text(s)]
            if not spans:
                continue
            title     = spans[0]
            publisher = spans[1] if len(spans) > 1 else None
            date_text = next((s for s in spans[2:] if re.search(r"\d{4}", s)), None)
            pubs.append(Publication(
                title=title,
                publisher=publisher,
                date=_make_date_range(date_text or ""),
            ))

        return pubs

    # ── Contact Info ──────────────────────────────────────────────────────────

    def parse_contact_info(self) -> Optional[ContactInfo]:
        """Parse the contact info modal content (if available)."""
        contact: Dict[str, Any] = {}

        email_el = self.soup.select_one("a[href^='mailto:']")
        if email_el:
            contact["email"] = email_el["href"].replace("mailto:", "").strip()

        phone_el = self.soup.select_one("span.t-14.t-black.t-normal")
        if phone_el and re.search(r"[\d\+\(\)\-\s]{7,}", _text(phone_el) or ""):
            contact["phone"] = _text(phone_el)

        website_el = self.soup.select_one("a.pv-contact-info__contact-link[href^='http']")
        if website_el:
            contact["website"] = website_el.get("href")

        return ContactInfo(**contact) if contact else None

    # ── Full Profile ──────────────────────────────────────────────────────────

    def parse_all(self, profile_url: str = "") -> LinkedInProfile:
        """Parse everything and return a complete LinkedInProfile."""
        basic = self.parse_basic_info()

        sections_found = []
        experiences   = self.parse_experience()
        if experiences:   sections_found.append("experience")
        educations    = self.parse_education()
        if educations:    sections_found.append("education")
        skills        = self.parse_skills()
        if skills:        sections_found.append("skills")
        certifications = self.parse_certifications()
        if certifications: sections_found.append("certifications")
        projects      = self.parse_projects()
        if projects:      sections_found.append("projects")
        volunteer     = self.parse_volunteer()
        if volunteer:     sections_found.append("volunteer")
        languages     = self.parse_languages()
        if languages:     sections_found.append("languages")
        awards        = self.parse_awards()
        if awards:        sections_found.append("awards")
        publications  = self.parse_publications()
        if publications:  sections_found.append("publications")
        contact_info  = self.parse_contact_info()

        logger.info(f"[Parser] Sections found: {sections_found}")

        return LinkedInProfile(
            full_name      = basic.get("full_name", ""),
            first_name     = basic.get("first_name"),
            last_name      = basic.get("last_name"),
            headline       = basic.get("headline"),
            about          = basic.get("about"),
            profile_url    = profile_url,
            location       = basic.get("location"),
            followers      = basic.get("followers"),
            open_to_work   = basic.get("open_to_work", False),
            contact        = contact_info,
            experiences    = experiences,
            educations     = educations,
            skills         = skills,
            certifications = certifications,
            projects       = projects,
            volunteer      = volunteer,
            languages      = languages,
            awards         = awards,
            publications   = publications,
        ), sections_found


def parse_profile_html(html: str, profile_url: str = "") -> Tuple[LinkedInProfile, List[str]]:
    """
    Public entry point — parse full-page HTML and return
    (LinkedInProfile, sections_found_list).
    """
    soup = _soup(html)
    parser = LinkedInParser(soup)
    return parser.parse_all(profile_url=profile_url)