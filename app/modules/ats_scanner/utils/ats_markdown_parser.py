from __future__ import annotations

import re
import logging
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

_CANONICAL: Dict[str, str] = {}

_RAW_ALIASES: Dict[str, List[str]] = {
    "contact": [
        "contact", "contact information", "contact info", "contact details",
        "personal details", "personal information", "personal info",
        "personal data", "personal profile", "contact me",
    ],
    "summary": [
        "summary", "professional summary", "career summary", "executive summary",
        "objective", "career objective", "professional objective",
        "profile", "professional profile", "about", "about me",
        "overview", "introduction", "personal statement", "highlights",
        "career highlights", "bio", "snapshot", "professional overview",
        "professional profile", "professional background summary",
        "career profile", "personal overview", "who i am",
    ],
    "experience": [
        "experience", "experiences",
        "work experience", "work experiences",
        "professional experience", "professional experiences",
        "employment", "employment history", "employment record",
        "work history", "career history", "career", "career experience",
        "professional background", "professional history",
        "industry experience", "relevant experience", "related experience",
        "practical experience", "hands-on experience",
        "job experience", "job history", "positions held", "positions",
        "roles", "roles held",
        "internship", "internships", "internship experience",
        "intern experience", "industrial training", "industry training",
        "training", "training experience",
        "clinical experience", "teaching experience", "research experience",
        "consulting experience", "freelance experience", "project experience",
        "field experience", "work & experience", "career details",
        "professional work experience", "prior experience", "past experience",
        "work background", "career background", "occupation", "occupations",
        "job profile", "employment details", "work details",
        "professional details", "corporate experience",
        "apprenticeship", "apprenticeships", "placement", "placements",
        "work placements", "on the job experience", "organizational experience",
        "operational experience", "industry background",
    ],
    "education": [
        "education", "educational background", "educational history",
        "educational qualifications", "educational details",
        "academic background", "academic history", "academic qualifications",
        "academic details", "qualifications", "qualification", "academic",
        "degrees", "degree", "schooling", "university", "college", "studies",
        "formal education", "academic training",
        "scholastic background", "scholastic details", "scholastic achievements",
        "scholastic qualifications", "academic achievements",
        "educational achievements", "academic profile",
        "educational profile", "learning background",
        "academic credentials", "educational credentials",
        "degree details", "academic record", "educational record",
        "school background", "university background", "college background",
        "technical education", "professional education",
    ],
    "skills": [
        "skills", "skill", "skill set", "skillset",
        "technical skills", "tech skills", "technical competencies",
        "technical expertise", "core competencies", "core skills", "key skills",
        "competencies", "expertise", "areas of expertise",
        "technologies", "technology", "tech stack", "technical stack",
        "tools", "tools & technologies", "tools and technologies",
        "tools & frameworks", "frameworks", "programming languages",
        "coding skills", "software skills", "software proficiency",
        "platforms", "knowledge", "technical knowledge", "proficiencies",
        "abilities", "strengths", "it skills", "computer skills",
        "digital skills", "hard skills", "key competencies",
        "professional skills", "functional skills", "domain skills",
        "domain expertise", "technical proficiency", "technical abilities",
        "core technical skills", "software expertise",
        "systems", "systems knowledge", "tools used",
        "languages and tools", "languages & tools",
        "programming skills", "development skills",
    ],
    "projects": [
        "projects", "project", "key projects", "major projects",
        "personal projects", "academic projects", "side projects",
        "project experience", "project work", "project portfolio",
        "portfolio", "notable projects", "selected projects",
        "capstone", "open source", "open-source contributions", "case studies",
        "works", "project highlights", "featured projects",
        "independent projects", "individual projects",
    ],
    "certifications": [
        "certifications", "certification", "certifications & licenses",
        "certificates", "certificate", "licenses", "license",
        "licences", "licence", "credentials", "professional certifications",
        "courses", "course", "training & certifications",
        "professional development", "continuing education", "accreditations",
        "online courses", "badges", "qualifications & certifications",
        "professional licenses", "professional credentials",
        "technical certifications", "industry certifications",
        "completed courses", "course completions",
    ],
    "languages": [
        "languages", "language", "language proficiency", "spoken languages",
        "human languages", "foreign languages", "linguistic skills",
        "language skills", "languages known", "languages spoken",
        "communication languages", "verbal communication",
    ],
    "awards": [
        "awards", "award", "awards & honors", "awards and honors",
        "achievements", "achievement", "honors", "honours",
        "recognitions", "accomplishments", "accolades", "scholarships",
        "fellowships", "prizes", "distinctions", "recognition",
        "awards and recognition", "honors and awards",
        "academic awards", "professional awards",
        "achievements and awards", "notable achievements",
    ],
    "volunteer": [
        "volunteer", "volunteering", "volunteer experience", "volunteer work",
        "voluntary work", "community service", "community involvement",
        "extracurricular", "extra-curricular", "activities", "civic involvement",
        "social work", "ngo work", "non-profit", "nonprofit",
        "community activities", "outreach",
    ],
    "publications": [
        "publications", "publication", "research", "research work", "papers",
        "articles", "journal articles", "patents", "presentations",
        "published work", "research papers", "research publications",
        "conference papers", "conference presentations",
    ],
    "hobbies": [
        "hobbies", "hobby", "hobbies & interests", "interests", "interest",
        "personal interests", "leisure", "recreational activities", "passions",
        "outside interests", "other interests",
    ],
    "references": [
        "references", "reference", "referees", "referee",
        "professional references", "character references",
    ],
}

for _sec, _aliases in _RAW_ALIASES.items():
    for _a in _aliases:
        _CANONICAL[_a.lower().strip()] = _sec

_SORTED_ALIASES = sorted(_CANONICAL.keys(), key=len, reverse=True)

EMAIL_RE      = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", re.I)
PHONE_RE      = re.compile(r"(\+\d{1,3}[\s\-]?)?(\(?\d{2,5}\)?[\s\-]?)?\d{3,5}[\s\-]?\d{3,5}[\s\-]?\d{0,5}")
LINKEDIN_RE   = re.compile(r"(?:https?://)?(?:www\.)?linkedin\.com/in/[\w\-]+", re.I)
GITHUB_RE     = re.compile(r"(?:https?://)?(?:www\.)?github\.com/[\w\-]+", re.I)
URL_RE        = re.compile(r"https?://[^\s]+", re.I)
YEAR_RE       = re.compile(r"\b(19|20)\d{2}\b")
DATE_RANGE_RE = re.compile(
    r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s,\.]*\d{4}|\d{4})"
    r"\s*[-–—to]+\s*"
    r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s,\.]*\d{4}|"
    r"\d{4}|[Pp]resent|[Cc]urrent|[Nn]ow|[Tt]ill\s*[Dd]ate)", re.I,
)
DEGREE_RE = re.compile(
    r"(Bachelor[^,\n|]{0,80}|B\.?\s?[ESTech]{1,5}\.?[^,\n|]{0,50}|"
    r"B\.?Sc\.?[^,\n|]{0,50}|B\.?A\.?[^,\n|]{0,40}|B\.?Tech\.?[^,\n|]{0,50}|"
    r"B\.?E\.?[^,\n|]{0,40}|B\.?Com\.?[^,\n|]{0,40}|B\.?C\.?A\.?[^,\n|]{0,40}|"
    r"Master[^,\n|]{0,80}|M\.?Sc\.?[^,\n|]{0,50}|M\.?Tech\.?[^,\n|]{0,50}|"
    r"M\.?B\.?A\.?[^,\n|]{0,40}|MBA[^,\n|]{0,30}|M\.?A\.?[^,\n|]{0,30}|"
    r"M\.?C\.?A\.?[^,\n|]{0,40}|Ph\.?D\.?[^,\n|]{0,60}|Doctor[^,\n|]{0,60}|"
    r"Associate[^,\n|]{0,50}|Diploma[^,\n|]{0,50}|Certificate[^,\n|]{0,50}|"
    r"High School[^,\n|]{0,40}|10th|12th|S\.?S\.?C\.?|H\.?S\.?C\.?)", re.I,
)
INSTITUTION_RE = re.compile(
    r"([\w\s&'\-\.]+(?:University|College|Institute|School|Academy|"
    r"Engineering College|Polytechnic|IIT|NIT|BITS|SASTRA|VIT|SRM|"
    r"Anna University|Madras University)[\w\s&'\-\.]{0,60})", re.I,
)
GPA_RE     = re.compile(r"(?:CGPA|GPA|Grade|Percentage)[:\s]*([0-9]+\.?[0-9]*)", re.I)
MD_H_RE    = re.compile(r"^(#{1,4})\s+(.+)$")
BULLET_RE  = re.compile(r"^[\s]*[-*•▶►→✓✔\u2022\u2023]+\s+(.+)$")
SEP_RE     = re.compile(r"^[-*_=]{3,}$")
BOLD_H_RE  = re.compile(r"^\*{1,2}([^*]{2,60})\*{1,2}\s*:?\s*$")
ALLCAPS_RE = re.compile(r"^[A-Z][A-Z\s&/\-]{2,53}[A-Z]$")

# ── Role-keyword detector used to identify which header line is the job
#    title regardless of line order (Role/Company/Dates can appear in any
#    sequence, on any number of lines, or pipe-delimited on one line). ─────
_ROLE_KEYWORDS_RE = re.compile(
    r"\b(engineer|manager|developer|designer|analyst|specialist|executive|"
    r"consultant|director|officer|coordinator|administrator|associate|"
    r"intern|internship|trainee|apprentice|lead|head|architect|scientist|"
    r"representative|technician|nurse|teacher|professor|recruiter|"
    r"accountant|auditor|strategist|marketer|writer|editor|researcher|"
    r"supervisor|assistant|founder|owner|freelancer|freelance|"
    r"self[\s\-]?employed|graduate engineer trainee|ceo|cto|cfo|coo|"
    r"president|vp\b|svp|evp|chief)\b", re.I,
)


def _resolve(raw: str) -> Optional[str]:
    key = raw.lower().strip()
    key = re.sub(r"^#+\s*", "", key)
    key = re.sub(r"\*{1,2}", "", key)
    key = re.sub(r"[:\-–—_]+$", "", key)
    key = re.sub(r"\s{2,}", " ", key).strip()

    if not key or len(key) < 2:
        return None

    if key in _CANONICAL:
        return _CANONICAL[key]

    for alias in _SORTED_ALIASES:
        if alias in key:
            return _CANONICAL[alias]

    _ANCHORS = [
        ("experience",          "experience"),
        ("employment",          "experience"),
        ("work history",        "experience"),
        ("career history",      "experience"),
        ("career background",   "experience"),
        ("work background",     "experience"),
        ("internship",          "experience"),
        ("industrial training", "experience"),
        ("apprenticeship",      "experience"),
        ("placement",           "experience"),
        ("training",            "experience"),
        ("scholastic",          "education"),
        ("education",           "education"),
        ("academic",            "education"),
        ("qualification",       "education"),
        ("schooling",           "education"),
        ("degree",              "education"),
        ("skills",              "skills"),
        ("competenc",           "skills"),
        ("expertise",           "skills"),
        ("technolog",           "skills"),
        ("stack",               "skills"),
        ("proficienc",          "skills"),
        ("project",             "projects"),
        ("portfolio",           "projects"),
        ("certification",       "certifications"),
        ("certificate",         "certifications"),
        ("licence",             "certifications"),
        ("license",             "certifications"),
        ("course",              "certifications"),
        ("accreditation",       "certifications"),
        ("summary",             "summary"),
        ("objective",           "summary"),
        ("profile",             "summary"),
        ("overview",            "summary"),
        ("about",               "summary"),
        ("highlight",           "summary"),
        ("contact",             "contact"),
        ("publication",         "publications"),
        ("research",            "publications"),
        ("award",               "awards"),
        ("achievement",         "awards"),
        ("honor",               "awards"),
        ("honour",              "awards"),
        ("recognition",         "awards"),
        ("accomplishment",      "awards"),
        ("volunteer",           "volunteer"),
        ("community",           "volunteer"),
        ("language",            "languages"),
        ("hobby",               "hobbies"),
        ("interest",            "hobbies"),
        ("reference",           "references"),
    ]
    for anchor, canonical in _ANCHORS:
        if anchor in key:
            return canonical

    return None


class ATSMarkdownParser:

    def parse(self, markdown: str) -> Dict[str, Any]:
        if not markdown or not markdown.strip():
            return self._empty()

        sections = self._split_sections(markdown)
        logger.info(f"Sections detected: {list(sections.keys())}")

        contact = self._contact(markdown, sections.get("contact", ""))

        result = {
            "name":     contact["name"],
            "email":    contact["email"],
            "phone":    contact["phone"],
            "location": contact["location"],
            "linkedin": contact["linkedin"],
            "github":   contact["github"],

            "summary":        self._summary(sections.get("summary", ""), markdown),
            "experience":     self._experience(sections.get("experience", "")),
            "education":      self._education(sections.get("education", "")),
            "skills":         self._skills(sections.get("skills", "")),
            "projects":       self._projects(sections.get("projects", "")),
            "certifications": self._certifications(sections.get("certifications", "")),
            "languages":      self._languages(sections.get("languages", "")),
            "awards":         self._list_section(sections.get("awards", "")),
            "volunteer":      self._list_section(sections.get("volunteer", "")),
            "publications":   self._list_section(sections.get("publications", "")),
            "hobbies":        self._list_section(sections.get("hobbies", "")),
            "raw_text":       markdown,
        }

        logger.info(
            f"name='{result['name']}' exp={len(result['experience'])} "
            f"edu={len(result['education'])} skills={len(result['skills'])} "
            f"proj={len(result['projects'])}"
        )
        return result

    def _split_sections(self, markdown: str) -> Dict[str, str]:
        lines    = markdown.split("\n")
        sections: Dict[str, str] = {}
        cur_sec: Optional[str]   = None
        cur_lines: List[str]     = []

        def flush():
            nonlocal cur_lines
            if cur_sec:
                content = "\n".join(cur_lines).strip()
                if content:
                    sections[cur_sec] = (
                        sections[cur_sec] + "\n" + content
                        if cur_sec in sections else content
                    )
            cur_lines.clear()

        for line in lines:
            stripped = line.strip()
            canonical = self._detect_heading(line, stripped)

            if canonical is not None:
                flush()
                cur_sec   = canonical
                cur_lines = []
                continue

            if cur_sec is not None:
                cur_lines.append(line)
            else:
                sections.setdefault("_header", "")
                sections["_header"] += "\n" + line

        flush()

        if "_header" in sections:
            hc = sections.pop("_header").strip()
            if hc:
                sections["contact"] = (
                    (sections.get("contact", "") + "\n" + hc).strip()
                )

        return sections

    def _detect_heading(self, line: str, stripped: str) -> Optional[str]:
        md_m = MD_H_RE.match(line)
        if md_m:
            resolved = _resolve(md_m.group(2).strip())
            return resolved or md_m.group(2).strip().lower()[:40]

        if ALLCAPS_RE.match(stripped):
            result = _resolve(stripped)
            if result:
                return result
            if 3 <= len(stripped) <= 55 and not re.search(r"\d", stripped[:3]):
                return stripped.lower().replace(" ", "_")[:30]

        bm = BOLD_H_RE.match(stripped)
        if bm:
            result = _resolve(bm.group(1))
            if result:
                return result

        if (
            not stripped.startswith(("-", "•", "*", "+"))
            and 3 <= len(stripped) <= 60
            and not stripped[0].isdigit()
        ):
            result = _resolve(stripped)
            if result:
                words     = stripped.split()
                has_at    = "@" in stripped
                has_pipe  = "|" in stripped
                has_digit = bool(re.search(r"\d", stripped))
                if not has_at and not has_pipe and len(words) <= 6:
                    return result

        return None

    def _contact(self, full_md: str, section: str) -> Dict[str, str]:
        src   = section if section.strip() else full_md[:2000]
        email = EMAIL_RE.search(src)
        li    = LINKEDIN_RE.search(src)
        gh    = GITHUB_RE.search(src)

        phone = ""
        for m in PHONE_RE.finditer(src):
            d = re.sub(r"\D", "", m.group(0))
            if 7 <= len(d) <= 15:
                phone = m.group(0).strip()
                break

        return {
            "name":     self._name(full_md),
            "email":    email.group(0).strip() if email else "",
            "phone":    phone,
            "linkedin": li.group(0).strip() if li else "",
            "github":   gh.group(0).strip() if gh else "",
            "location": self._location(src),
        }

    def _name(self, md: str) -> str:
        skip = re.compile(
            r"@|http|www\.|linkedin|github|\.com|\.io|\+\d|\d{5,}|"
            r"resume|curriculum|vitae|objective|summary|profile|"
            r"experience|education|skill|project|certification", re.I,
        )
        for line in md.split("\n")[:25]:
            c = re.sub(r"^#+\s*", "", line).strip()
            c = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", c)
            if not c or skip.search(c):
                continue
            if ALLCAPS_RE.match(c) and len(c.split()) > 4:
                continue
            w = c.split()
            if 1 <= len(w) <= 6 and re.match(r"^[A-Za-z][A-Za-z\s\-'.]{1,50}$", c):
                return c
        return ""

    def _location(self, text: str) -> str:
        for pat in [
            r"\b([A-Z][a-zA-Z\s\-]+),\s*([A-Z][a-zA-Z\s]{2,}(?:\s+\d{5,6})?)\b",
            r"\b([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*),?\s*"
            r"(India|Tamil Nadu|Maharashtra|Karnataka|Telangana|Gujarat|"
            r"Delhi|Kerala|Punjab|USA|UK|Canada|Australia|Germany|Singapore|UAE)\b",
        ]:
            m = re.search(pat, text)
            if m:
                return m.group(0).strip()
        return ""

    def _summary(self, sec: str, full_md: str) -> str:
        if sec.strip():
            return " ".join(
                re.sub(r"[*_`#]+", "", l).strip()
                for l in sec.split("\n")
                if l.strip() and not SEP_RE.match(l.strip())
            )
        skip = re.compile(r"@|linkedin|github|http|www\.|\.com|\+\d|\d{5,}|^#{1,4}\s", re.I)
        found = []
        for line in full_md.split("\n")[:50]:
            s = line.strip()
            if not s or skip.search(s) or (ALLCAPS_RE.match(s) and len(s) < 50):
                continue
            if len(s.split()) >= 6:
                found.append(re.sub(r"[*_`#]+", "", s).strip())
            if len(found) >= 3:
                break
        return " ".join(found)

    # ── EXPERIENCE PARSING ───────────────────────────────────────────────
    #
    # v2: block-based parsing. Previously, this method only recognised a
    # role when (a) the date range appeared on the SAME line as the title
    # AND (b) the date regex matched bare years / Present-Current-Now only.
    # Real resumes routinely have:
    #   - Title, Company, Dates on three SEPARATE lines (any order)
    #   - "Mon YYYY – Mon YYYY" dates (month on both ends)
    #   - "Role | Company | Dates" pipe-delimited single lines
    #   - A blank line between the date line and the bullet list
    # All four caused the previous implementation to silently drop the
    # entire entry (and its bullets), producing exp_count == 0.
    #
    # Fix: split the section into per-role BLOCKS first (header lines +
    # bullets), using bullet-transitions / "header already has a date" as
    # block boundaries — blank lines are ignored for boundary purposes
    # since they're just visual spacing, not entry separators. Then parse
    # each block's header lines using DATE_RANGE_RE (which supports
    # month-year on both ends) and a role-keyword heuristic that finds the
    # title regardless of which line it's on.
    # ─────────────────────────────────────────────────────────────────────

    def _experience(self, text: str) -> List[Dict]:
        if not text.strip():
            return []
        blocks = self._split_experience_blocks(text.split("\n"))
        entries: List[Dict] = []
        for header_lines, bullets in blocks:
            entry = self._parse_experience_block(header_lines, bullets)
            if entry:
                entries.append(entry)
        return entries

    def _split_experience_blocks(
        self, raw_lines: List[str]
    ) -> List[Tuple[List[str], List[str]]]:
        blocks: List[Tuple[List[str], List[str]]] = []
        header: List[str] = []
        bullets: List[str] = []
        header_has_date = False

        def flush():
            nonlocal header, bullets, header_has_date
            if header or bullets:
                blocks.append((header, bullets))
            header, bullets, header_has_date = [], [], False

        for raw_line in raw_lines:
            line = raw_line.strip()
            if not line:
                continue  # blank lines are spacing, not boundaries

            bm = BULLET_RE.match(line)
            if bm:
                bullets.append(bm.group(1).strip())
                continue

            # Non-bullet (header-type) line
            if bullets:
                # Bullets already collected for the current role —
                # a new header line means a new role is starting.
                flush()
                header.append(line)
            elif header_has_date and header:
                # Current role's header is already "complete" (has a
                # date) and no bullets were seen yet — back-to-back
                # roles listed without bullets.
                flush()
                header.append(line)
            else:
                header.append(line)

            if DATE_RANGE_RE.search(line):
                header_has_date = True

        flush()
        return blocks

    def _parse_experience_block(
        self, header_lines: List[str], bullets: List[str]
    ) -> Optional[Dict]:
        if not header_lines and not bullets:
            return None

        # Expand pipe-delimited single-line headers: "Role | Company | Dates"
        expanded: List[str] = []
        for line in header_lines:
            if "|" in line:
                parts = [p.strip() for p in line.split("|") if p.strip()]
                expanded.extend(parts if parts else [line])
            else:
                expanded.append(line)

        start_date = end_date = ""
        remaining: List[str] = []
        for line in expanded:
            if not start_date:
                dm = DATE_RANGE_RE.search(line)
                if dm:
                    start_date = dm.group(1).strip()
                    end_date   = dm.group(2).strip()
                    prefix = line[:dm.start()].strip(" |,-")
                    suffix = line[dm.end():].strip(" |,-")
                    if prefix:
                        remaining.append(prefix)
                    if suffix:
                        remaining.append(suffix)
                    continue
            remaining.append(line)

        # Identify title via role-keyword match (order-independent), then
        # assign the next leftover lines to company / location.
        title = company = location = ""
        for line in remaining[:5]:
            if not title and _ROLE_KEYWORDS_RE.search(line):
                title = line
            elif not company:
                company = line
            elif not location:
                location = line

        # No line matched a role keyword — fall back to positional guess.
        if not title and remaining:
            title = remaining[0]
            if not company:
                for line in remaining[1:]:
                    company = line
                    break
        elif title and not company:
            for line in remaining:
                if line != title:
                    company = line
                    break

        if not (title or company or bullets):
            return None

        return {
            "title":      title,
            "company":    company,
            "location":   location,
            "start_date": start_date,
            "end_date":   end_date,
            "bullets":    bullets,
        }

    def _education(self, sec: str) -> List[Dict[str, Any]]:
        if not sec.strip():
            return []
        entries = self._split_edu(sec)
        return [p for e in entries if (p := self._parse_edu(e)) and (p.get("degree") or p.get("institution"))]

    def _split_edu(self, text: str) -> List[str]:
        lines   = text.split("\n")
        entries: List[List[str]] = []
        cur:     List[str]       = []
        for line in lines:
            s     = line.strip()
            clean = re.sub(r"[*_`#|]+", "", s).strip()
            is_new = (
                MD_H_RE.match(line)
                or BOLD_H_RE.match(s)
                or bool(DEGREE_RE.match(clean))
            )
            if is_new and cur and any(l.strip() for l in cur):
                entries.append("\n".join(cur))
                cur = []
            cur.append(line)
        if cur and any(l.strip() for l in cur):
            entries.append("\n".join(cur))
        return entries or [text]

    def _parse_edu(self, text: str) -> Dict[str, Any]:
        c  = re.sub(r"[*_`#]+", "", text).strip()
        dm = DEGREE_RE.search(c)
        im = INSTITUTION_RE.search(c)
        ym = YEAR_RE.search(c)
        gm = GPA_RE.search(c)
        degree      = dm.group(0).strip().rstrip(",;|") if dm else ""
        institution = im.group(1).strip().rstrip(",;|") if im else ""
        if not degree and not institution:
            parts       = [p.strip() for p in re.split(r"[|\n,]", c) if p.strip()]
            degree      = parts[0][:100] if parts else ""
            institution = parts[1][:100] if len(parts) > 1 else ""
        return {
            "degree":      degree,
            "institution": institution,
            "college":     institution,
            "year":        ym.group(0) if ym else "",
            "gpa":         gm.group(1) if gm else "",
            "raw_text":    c[:300],
        }

    def _skills(self, sec: str) -> List[str]:
        if not sec.strip():
            return []
        clean  = re.sub(r"[*_`#]+", "", sec)
        skills: List[str] = []
        for line in clean.split("\n"):
            line = line.strip().lstrip("-•*▶►→✓✔ ").strip()
            if not line:
                continue
            cm  = re.match(r"^[A-Za-z\s&/\-]{2,35}:\s*(.+)$", line)
            src = cm.group(1) if cm else line
            for part in re.split(r"[,|/•·\t]+", src):
                part = part.strip().strip("•·*-–—()[]{}").strip()
                if part and 1 < len(part) <= 60 and not part.isdigit():
                    skills.append(part)
        seen: set = set()
        return [s for s in skills if not (s.lower() in seen or seen.add(s.lower()))]  # type: ignore

    def _projects(self, sec: str) -> List[Dict[str, Any]]:
        if not sec.strip():
            return []
        entries = self._split_by_bold(sec)
        results = []
        for e in entries:
            clean = re.sub(r"[*_`]+", "", e).strip()
            if not clean:
                continue
            lines   = [l.strip() for l in clean.split("\n") if l.strip()]
            title   = lines[0][:120] if lines else ""
            desc    = ""
            techs:   List[str] = []
            bullets: List[str] = []
            for line in lines[1:]:
                lc = line.lower()
                if re.match(r"^(tech|stack|built with|tools|technologies)[:\s]", lc):
                    after = re.sub(r"^[^:]+:\s*", "", line)
                    techs = [t.strip() for t in re.split(r"[,|/]", after) if t.strip()]
                elif BULLET_RE.match(line) or line.startswith(("-", "•")):
                    bullets.append(line.lstrip("-•*▶►→ ").strip())
                elif not desc:
                    desc = line
            um = URL_RE.search(e)
            results.append({
                "title":        title,
                "description":  desc,
                "technologies": techs,
                "bullets":      bullets,
                "url":          um.group(0) if um else "",
            })
        return results

    def _certifications(self, sec: str) -> List[Dict[str, Any]]:
        if not sec.strip():
            return []
        clean   = re.sub(r"[*_`#]+", "", sec)
        results = []
        for line in clean.split("\n"):
            line = line.strip().lstrip("-•*▶►→ ").strip()
            if not line or len(line) < 3:
                continue
            ym   = YEAR_RE.search(line)
            year = ym.group(0) if ym else ""
            base = line[:ym.start()].strip().rstrip("(,|") if ym else line
            ps   = re.split(r"\||-|,|\bby\b", base, maxsplit=1, flags=re.I)
            t    = ps[0].strip()
            iss  = ps[1].strip() if len(ps) > 1 else ""
            if t:
                results.append({"title": t, "name": t, "issuer": iss, "year": year})
        return results

    def _languages(self, sec: str) -> List[str]:
        if not sec.strip():
            return []
        clean = re.sub(r"[*_`#]+", "", sec)
        langs = []
        for line in clean.split("\n"):
            line = line.strip().lstrip("-•*▶►→ ").strip()
            if not line:
                continue
            line = re.sub(r"\([^)]+\)", "", line).strip()
            for p in re.split(r"[,|/]", line):
                p = p.strip()
                if p and 2 < len(p) < 30:
                    langs.append(p)
        return langs

    def _list_section(self, sec: str) -> List[str]:
        if not sec.strip():
            return []
        clean = re.sub(r"[*_`#]+", "", sec)
        return [
            l.strip().lstrip("-•*▶►→ ").strip()
            for l in clean.split("\n")
            if l.strip().lstrip("-•*▶►→ ").strip() and len(l.strip()) > 3
        ]

    def _split_by_bold(self, text: str) -> List[str]:
        lines   = text.split("\n")
        entries: List[List[str]] = []
        cur:     List[str]       = []
        for line in lines:
            s = line.strip()
            if MD_H_RE.match(line) or (BOLD_H_RE.match(s) and 4 < len(s) < 100):
                if cur and any(l.strip() for l in cur):
                    entries.append("\n".join(cur))
                cur = []
            cur.append(line)
        if cur and any(l.strip() for l in cur):
            entries.append("\n".join(cur))
        return entries or [text]

    def _empty(self) -> Dict[str, Any]:
        return {
            "name": "", "email": "", "phone": "", "location": "",
            "linkedin": "", "github": "",
            "summary": "", "experience": [], "education": [],
            "skills": [], "projects": [], "certifications": [],
            "languages": [], "awards": [], "volunteer": [],
            "publications": [], "hobbies": [], "raw_text": "",
        }


_parser = ATSMarkdownParser()


def parse_resume_markdown(markdown: str) -> Dict[str, Any]:
    return _parser.parse(markdown)