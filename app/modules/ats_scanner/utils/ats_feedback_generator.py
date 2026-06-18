
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SectionFeedback:
    section_name:      str
    current_score:     int
    target_score:      int
    status:            str
    impact_potential:  int
    is_present:        bool
    is_complete:       bool
    quality_level:     str             # "high" / "medium" / "low"

    missing_elements:    List[Dict]    = field(default_factory=list)
    elements_to_remove:  List[Dict]    = field(default_factory=list)
    quality_issues:      List[Dict]    = field(default_factory=list)
    top_priority_fixes:  List[Dict]    = field(default_factory=list)
    quick_wins:          List[Dict]    = field(default_factory=list)
    detailed_suggestions: List[Dict]   = field(default_factory=list)
    ats_passing_tips:    List[str]     = field(default_factory=list)
    rewrite_examples:    List[Dict]    = field(default_factory=list)
    strengths:           List[str]     = field(default_factory=list)


@dataclass
class ComprehensiveFeedback:
    overall_score:                 int
    overall_status:                str
    ready_to_apply:                bool
    estimated_improvement_potential: int
    grade:                         str
    percentile_estimate:           str

    section_feedback:   Dict[str, SectionFeedback] = field(default_factory=dict)

    top_3_priorities:    List[str]  = field(default_factory=list)
    quick_wins_summary:  List[str]  = field(default_factory=list)
    strengths_summary:   List[str]  = field(default_factory=list)
    ats_passing_tactics: List[str]  = field(default_factory=list)

    improvement_roadmap: List[Dict] = field(default_factory=list)
    recruiter_tips:      List[str]  = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION TEMPLATES
# ─────────────────────────────────────────────────────────────────────────────

SECTION_TEMPLATES: Dict[str, Dict] = {

    "contact": {
        "add_items": [
            {"element": "Professional email address",
             "why":     "ATS stores email as primary contact; unprofessional emails reduce recruiter response",
             "impact":  8,
             "example": "john.doe@gmail.com (not: j_dogg99@hotmail.com)"},
            {"element": "Phone number with country code",
             "why":     "Required for interview scheduling",
             "impact":  8,
             "example": "+91 98765 43210"},
            {"element": "City and State/Country",
             "why":     "Many ATS filter by location; missing location = likely rejection",
             "impact":  7,
             "example": "Chennai, Tamil Nadu, India"},
            {"element": "LinkedIn profile URL",
             "why":     "90% of recruiters check LinkedIn — link makes it frictionless",
             "impact":  6,
             "example": "linkedin.com/in/your-name"},
            {"element": "Portfolio / GitHub / Professional website",
             "why":     "Evidence of work — highly valued in creative, tech, and research roles",
             "impact":  5,
             "example": "github.com/yourname | yourportfolio.com"},
        ],
        "remove_items": [
            {"element": "Date of birth",
             "why":     "Illegal to request in most regions; ATS may flag it",
             "action":  "Delete entirely"},
            {"element": "Nationality / Religion / Marital status",
             "why":     "Protected class information — creates legal risk and bias",
             "action":  "Delete entirely"},
            {"element": "Full home address",
             "why":     "Privacy risk; City + State is sufficient",
             "action":  "Replace with City, State only"},
        ],
        "quality_items": [
            {"issue":   "Unprofessional email domain",
             "example": "coolkid1995@yahoo.com",
             "fix":     "Create firstname.lastname@gmail.com",
             "impact":  8},
        ],
    },

    "summary": {
        "add_items": [
            {"element": "Professional title / Role level",
             "why":     "ATS uses job title matching; leading with title improves parsing",
             "impact":  10,
             "example": "Senior Financial Analyst | Registered Nurse | Marketing Manager"},
            {"element": "Years of experience",
             "why":     "Signals seniority; ATS often filters by experience range",
             "impact":  8,
             "example": "8+ years in clinical nursing"},
            {"element": "2-3 core skills or specialisations",
             "why":     "Keyword-rich summary boosts ATS match score significantly",
             "impact":  12,
             "example": "Specialising in Python, AWS, and distributed systems"},
            {"element": "One quantified achievement",
             "why":     "Numbers in summary catch recruiters' eyes within 6 seconds",
             "impact":  10,
             "example": "Delivered $4M in annual savings | Maintained 99.2% patient satisfaction"},
        ],
        "remove_items": [
            {"element": "First-person pronouns (I, me, my)",
             "why":     "Non-standard in resumes; wastes characters",
             "action":  "Delete 'I am' → start directly with your title or achievement"},
            {"element": "Generic clichés",
             "why":     "Every resume uses them; they add zero value",
             "action":  "Replace 'hard-working team player' with specific evidence"},
            {"element": "Summaries over 150 words",
             "why":     "Recruiters spend <10s on summary; length kills impact",
             "action":  "Trim to 50-80 words maximum"},
        ],
        "quality_items": [
            {"issue": "No numbers or metrics",
             "example": "Experienced professional with strong communication skills",
             "fix":     "Results-driven professional with X years in [field]. Expert in [skill 1], [skill 2]. Delivered [metric achievement].",
             "impact":  12},
            {"issue": "Too long (over 150 words)",
             "example": "(paragraph exceeding 150 words)",
             "fix":     "Condense to 3-4 impactful sentences. Use bullets if listing multiple achievements.",
             "impact":  5},
        ],
    },

    "experience": {
        "add_items": [
            {"element": "Strong action verb at start of every bullet",
             "why":     "ATS parses verbs to classify achievement type; weak openers hurt score",
             "impact":  12,
             "example": "Led / Managed / Designed / Reduced / Achieved / Delivered / Built"},
            {"element": "Quantified results in every bullet",
             "why":     "Numbers are the #1 signal ATS uses to assess impact",
             "impact":  15,
             "example": "Reduced patient wait time by 35% | Generated $1.2M in new revenue"},
            {"element": "Employment dates (Month Year – Month Year)",
             "why":     "ATS calculates total experience duration from dates; missing = parse error",
             "impact":  10,
             "example": "Oct 2024 – Jan 2025"},
            {"element": "Job title, Company name, Location",
             "why":     "Three required ATS fields — missing any one causes parse failure",
             "impact":  12,
             "example": "Digital Marketing Executive | Wedzat | Chennai"},
            {"element": "Industry keywords in bullet context",
             "why":     "Embedding JD keywords in bullets doubles keyword hit rate",
             "impact":  10,
             "example": "Executed SEO activities that improved organic reach by 40%"},
        ],
        "remove_items": [
            {"element": "Passive / weak openers",
             "why":     "Drops ATS content quality score",
             "action":  "Replace 'Responsible for' → 'Managed'; 'Helped with' → 'Collaborated'"},
            {"element": "More than 8 bullets per role",
             "why":     "ATS truncates; recruiter attention drops after bullet 6",
             "action":  "Keep 4-6 strongest, most relevant bullets per role"},
            {"element": "Duties-focused language",
             "why":     "Describes what the role required, not what YOU delivered",
             "action":  "Flip from 'duties included X' to 'achieved/delivered X'"},
        ],
        "quality_items": [
            {"issue": "No measurable results",
             "example": "Managed customer relationships and handled complaints",
             "fix":     "Managed 85-account portfolio worth $3.2M, achieving 94% retention rate",
             "impact":  15},
            {"issue": "Weak action verb",
             "example": "Helped with the marketing campaign",
             "fix":     "Co-led 6-channel digital campaign that generated 4,200 qualified leads in Q3",
             "impact":  10},
            {"issue": "Missing dates",
             "example": "Marketing Coordinator | ABC Corp",
             "fix":     "Marketing Coordinator | ABC Corp | Jun 2021 – Dec 2023",
             "impact":  10},
        ],
    },

    # v4.1 — fresher-specific template used in place of "experience" when
    # candidate_type == "fresher" AND no experience entries exist.
    "experience_fresher": {
        "add_items": [
            {"element": "Any internship, training, or apprenticeship",
             "why":     "Internships are treated as valid experience and signal job-readiness",
             "impact":  12,
             "example": "Marketing Intern | StartupCo | Jun 2025 - Aug 2025"},
            {"element": "Freelance or self-employed work",
             "why":     "Freelance/contract work counts as real experience for ATS and recruiters",
             "impact":  10,
             "example": "Freelance Graphic Designer | Self-employed | 2024 - Present"},
            {"element": "Academic, personal, or capstone projects with outcomes",
             "why":     "Projects substitute for work history when paired with measurable results",
             "impact":  12,
             "example": "Built a task-tracker app (React, Firebase) used by 50+ classmates"},
            {"element": "Relevant certifications",
             "why":     "Certifications reinforce skill claims when work history is thin",
             "impact":  8,
             "example": "Google Data Analytics Certificate | Coursera | 2025"},
            {"element": "Volunteer or hackathon participation",
             "why":     "Demonstrates initiative and teamwork without requiring paid work history",
             "impact":  6,
             "example": "Participated in [University] Hackathon 2025 — built a campus-navigation app"},
        ],
        "remove_items": [],
        "quality_items": [
            {"issue": "Project with no outcome",
             "example": "Built a web app for task management",
             "fix":     "Built task management app (React, Firebase) adopted by 3 teams; reduced missed deadlines by 45%",
             "impact":  10},
        ],
    },

    "education": {
        "add_items": [
            {"element": "Full degree name (not abbreviated)",
             "why":     "ATS matches 'Bachelor of Science in Nursing' — abbreviations may miss",
             "impact":  10,
             "example": "Bachelor of Science in Computer Science, not just 'B.Sc CS'"},
            {"element": "Full institution name",
             "why":     "ATS may filter by institution; abbreviations cause mismatches",
             "impact":  8,
             "example": "Anna University, Chennai — not 'AU'"},
            {"element": "Graduation year (or expected graduation)",
             "why":     "ATS calculates experience timeline from education dates",
             "impact":  8,
             "example": "May 2023 | Expected Dec 2024"},
        ],
        "remove_items": [
            {"element": "GPA below 3.5 / CGPA below 8.0",
             "why":     "Below-average GPA hurts more than it helps",
             "action":  "Remove unless the job posting specifically requests GPA"},
            {"element": "High school diploma (if you have a degree)",
             "why":     "Redundant once higher education is listed; wastes space",
             "action":  "Delete high school entry entirely"},
        ],
        "quality_items": [
            {"issue": "Abbreviated degree name",
             "example": "B.E, VIT, 2020",
             "fix":     "Bachelor of Engineering (Electronics) | VIT University | May 2020",
             "impact":  8},
            {"issue": "No graduation year",
             "example": "B.A English | State University",
             "fix":     "B.A. English | State University | May 2019",
             "impact":  8},
        ],
    },

    "skills": {
        "add_items": [
            {"element": "Exact keywords from the job description",
             "why":     "ATS performs exact-match keyword scoring on skills section",
             "impact":  18,
             "example": "If JD says 'Tableau' — list 'Tableau', not 'data visualisation'"},
            {"element": "Both abbreviation and full form",
             "why":     "Some ATS match on one form only; listing both doubles hit chance",
             "impact":  8,
             "example": "SEO (Search Engine Optimization) | CRM (Customer Relationship Management)"},
            {"element": "Tools and software",
             "why":     "Tool names are among the highest-frequency JD keywords",
             "impact":  10,
             "example": "Jira | Slack | HubSpot | Zoom | Google Analytics | Canva"},
        ],
        "remove_items": [
            {"element": "Soft skills without evidence",
             "why":     "Listed alone in skills, soft skills score near zero in ATS",
             "action":  "Move to summary or embed in experience bullets with examples"},
            {"element": "Outdated technologies (10+ years old)",
             "why":     "May signal outdated knowledge; dilutes keyword relevance",
             "action":  "Remove unless the target role explicitly lists them"},
            {"element": "Skills list exceeding 40 items",
             "why":     "Keyword dilution — ATS drops confidence score on over-stuffed lists",
             "action":  "Curate to 12-25 most relevant, role-specific skills"},
        ],
        "quality_items": [
            {"issue": "Vague skills",
             "example": "Good with computers, databases, cloud",
             "fix":     "Python • PostgreSQL • AWS EC2 • Docker • Kubernetes",
             "impact":  12},
            {"issue": "Inconsistent naming",
             "example": "NodeJS, node.js, Node, node",
             "fix":     "Standardise: 'Node.js' — use official product casing",
             "impact":  5},
        ],
    },

    "projects": {
        "add_items": [
            {"element": "Project name and brief description",
             "why":     "ATS needs a title to categorise; vague entries score low",
             "impact":  7,
             "example": "SEO Optimization Project | Social Media Campaign Dashboard"},
            {"element": "Technologies / tools used",
             "why":     "These are keyword-rich and directly match JD skill requirements",
             "impact":  10,
             "example": "Built using React, Node.js, PostgreSQL, and AWS Lambda"},
            {"element": "Quantified outcome",
             "why":     "Numbers differentiate a project from a class assignment",
             "impact":  10,
             "example": "Reduced scheduling conflicts by 70% | Served 10,000 monthly users"},
        ],
        "remove_items": [
            {"element": "Projects older than 5 years (unless landmark)",
             "why":     "Old tech stack signals outdated skills",
             "action":  "Remove or consolidate under 'Earlier Projects'"},
        ],
        "quality_items": [
            {"issue": "Project with no outcome",
             "example": "Built a web app for task management",
             "fix":     "Built task management app (React, Firebase) adopted by 3 teams; reduced missed deadlines by 45%",
             "impact":  10},
        ],
    },

    "certifications": {
        "add_items": [
            {"element": "Exact certification name (full, official)",
             "why":     "ATS matches certification names precisely; short forms may miss",
             "impact":  10,
             "example": "AWS Certified Solutions Architect – Associate (not just 'AWS cert')"},
            {"element": "Issuing body",
             "why":     "Validates legitimacy; ATS may filter by issuing organisation",
             "impact":  8,
             "example": "Issued by: Google | HubSpot | Coursera | PMI"},
            {"element": "Date obtained",
             "why":     "ATS and recruiters check recency; expired certs raise red flags",
             "impact":  8,
             "example": "Obtained: Mar 2023"},
        ],
        "remove_items": [
            {"element": "Expired certifications (without noting renewal plan)",
             "why":     "Raises questions about currency of knowledge",
             "action":  "Remove or note: '(Renewal in progress, expected Jan 2025)'"},
        ],
        "quality_items": [
            {"issue": "Missing issue date",
             "example": "Digital Marketing Certification",
             "fix":     "Digital Marketing Certification | Google | Obtained Jan 2023",
             "impact":  8},
        ],
    },

    "languages": {
        "add_items": [
            {"element": "Proficiency level for every language",
             "why":     "ATS and recruiters need to know if you can actually do the job in that language",
             "impact":  8,
             "example": "English (Professional Proficiency) | Tamil (Native)"},
        ],
        "remove_items": [
            {"element": "Languages you cannot use professionally",
             "why":     "Listing a language at the wrong level misleads",
             "action":  "Only list if you can hold professional conversations"},
        ],
        "quality_items": [
            {"issue": "Language without proficiency level",
             "example": "English, Tamil",
             "fix":     "English (Professional) | Tamil (Native)",
             "impact":  8},
        ],
    },

    "volunteer":     {"add_items": [], "remove_items": [], "quality_items": []},
    "awards":        {"add_items": [], "remove_items": [], "quality_items": []},
    "publications":  {"add_items": [], "remove_items": [], "quality_items": []},
    "hobbies":       {"add_items": [], "remove_items": [], "quality_items": []},
    "references":    {"add_items": [], "remove_items": [], "quality_items": []},
}


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL TACTICS
# ─────────────────────────────────────────────────────────────────────────────

GLOBAL_ATS_TACTICS = [
    "Submit as PDF (unless the ATS explicitly asks for DOCX) — PDF preserves formatting.",
    "Use standard section headings: 'Work Experience' not 'My Journey' — ATS searches for exact headings.",
    "Mirror keywords verbatim from the job description — ATS does exact-string matching.",
    "Place the most important keywords in the top third of the first page (highest ATS weight).",
    "Use both the spelled-out form AND abbreviation for every key term: SEO (Search Engine Optimization).",
    "Avoid graphics, charts, text boxes, and images — ATS cannot parse them.",
    "Use a single-column layout — multi-column resumes parse out of order in ATS.",
    "Save the file as 'FirstName-LastName-Resume.pdf' — readable filename helps HR filing.",
    "Avoid headers and footers for critical info (name, contact) — some ATS skip header/footer regions.",
    "Apply fresh to each role — tailor your resume keywords to every specific job posting.",
    "Keep ATS version clean (no colour, fancy fonts, tables) — maintain a separate visual version for humans.",
    "Use standard bullet characters (•) not custom symbols — non-standard bullets render as garbage in ATS.",
]

RECRUITER_TIPS = [
    "Recruiters spend an average of 7.4 seconds on initial resume scan — your name, title, and top bullet must stand out.",
    "The 'F-pattern' reading means recruiters see: top-left first, then skim down the left edge — front-load your value.",
    "Metrics > adjectives, always. '40% faster delivery' beats 'excellent delivery skills' every time.",
    "LinkedIn profile views spike 40% when your resume and LinkedIn are consistent — keep them aligned.",
    "Tailoring takes 15 minutes per application and can triple your callback rate.",
    "ATS passes an average of 25% of applicants to human review — your goal is to be in that 25%.",
]


# ─────────────────────────────────────────────────────────────────────────────
# FEEDBACK GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

class DetailedFeedbackGenerator:
    """
    Generates comprehensive, actionable ATS-passing feedback.

    Core invariant (v4.0):
      EVERY field in SectionFeedback is derived from the SAME
      section_score and is_present values — no contradictions.
    """

    SCORE_THRESHOLDS = {
        "excellent":         (85, 100),
        "good":              (70, 84),
        "needs_improvement": (50, 69),
        "critical":          (20, 49),
        "missing":           (0,  19),
    }

    ALL_SECTIONS = [
        "contact", "summary", "experience", "education", "skills",
        "projects", "certifications", "languages", "volunteer",
        "publications", "awards", "hobbies", "references",
    ]

    def generate_detailed_feedback(
        self,
        ats_score:        int,
        section_scores:   Dict[str, int],
        resume:           Dict,
        ats_issues:       List,
        section_analyses: Dict = None,
        candidate_type:    str = "experienced",
    ) -> ComprehensiveFeedback:
        logger.info(f"Generating detailed feedback — overall score: {ats_score}, candidate_type={candidate_type}")

        section_feedback: Dict[str, SectionFeedback] = {}

        for section in self.ALL_SECTIONS:
            score         = section_scores.get(section, 0)
            section_data  = resume.get(section) or resume.get(self._alt_key(section))
            issues_for_sec = [i for i in ats_issues if getattr(i, "section", "") == section]
            analysis      = (section_analyses or {}).get(section)

            feedback = self._build_section_feedback(
                section, score, section_data, issues_for_sec, analysis, resume, candidate_type
            )
            section_feedback[section] = feedback

        top_3      = self._top_priorities(section_feedback)
        quick_wins = self._quick_wins(section_feedback)
        strengths  = self._strengths(section_feedback)
        roadmap    = self._build_roadmap(section_feedback)
        total_pot  = sum(sf.impact_potential for sf in section_feedback.values())

        grade         = self._grade(ats_score)
        percentile    = self._percentile(ats_score)
        overall_status = self._status(ats_score)

        return ComprehensiveFeedback(
            overall_score=ats_score,
            overall_status=overall_status,
            ready_to_apply=ats_score >= 72,
            estimated_improvement_potential=min(total_pot, 100 - ats_score),
            grade=grade,
            percentile_estimate=percentile,
            section_feedback=section_feedback,
            top_3_priorities=top_3,
            quick_wins_summary=quick_wins,
            strengths_summary=strengths,
            ats_passing_tactics=GLOBAL_ATS_TACTICS[:6],
            improvement_roadmap=roadmap,
            recruiter_tips=RECRUITER_TIPS[:4],
        )

    # ── Per-section builder ───────────────────────────────────────────────────

    def _build_section_feedback(
        self,
        section:      str,
        score:        int,
        data:         Any,
        issues:       List,
        analysis=None,
        resume:       Dict = None,
        candidate_type: str = "experienced",
    ) -> SectionFeedback:
        """
        Build SectionFeedback with guaranteed internal consistency:
          • is_present is derived from score AND data presence
          • status is always derived from the SAME score
          • missing_elements only lists fields that are genuinely absent

        BUG 4: contact feedback reads actual contact fields from resume.
        BUG 5: is_present / status / score are mutually consistent.
        BUG 7: projects — is_present=True when score > 0.
        BUG 8: languages — status derived from score, never "missing" when present.

        v4.1: for section == "experience" with candidate_type == "fresher"
        and no data present, swap in the fresher-appropriate template and
        cap status at "needs_improvement" instead of "critical"/"missing".
        """
        is_fresher_no_exp = (
            section == "experience"
            and candidate_type == "fresher"
            and not self._is_present(data)
        )

        template_key = "experience_fresher" if is_fresher_no_exp else section
        template     = SECTION_TEMPLATES.get(template_key, SECTION_TEMPLATES.get(section, {}))
        add_items    = template.get("add_items", [])
        remove_items = template.get("remove_items", [])
        quality_items = template.get("quality_items", [])

        # ── Pull overrides from rules analysis ───────────────────────────────
        rule_missing:    List[str] = []
        rule_quality:    List[str] = []
        rule_strengths:  List[str] = []
        rule_tips:       List[str] = []
        rule_rewrites:   List[Dict] = []

        if analysis:
            rule_missing   = list(getattr(analysis, "missing_fields", []) or [])
            rule_quality   = list(getattr(analysis, "quality_issues",   []) or [])
            rule_strengths = list(getattr(analysis, "strengths",         []) or [])
            rule_tips      = list(getattr(analysis, "ats_tips",          []) or [])
            rule_rewrites  = list(getattr(analysis, "rewrite_examples",  []) or [])
            # Use rules score only if it's credible (> 0)
            rs = getattr(analysis, "current_score", 0)
            if rs and rs > 0:
                score = rs

        # ── Derive is_present from score + data + analysis ───────────────────
        # BUG 5: is_present is True whenever the score is positive OR data exists
        data_present = self._is_present(data)
        is_present   = score > 0 or data_present

        # If analysis explicitly says the section is present, trust it
        if analysis and getattr(analysis, "is_present", None) is not None:
            is_present = getattr(analysis, "is_present")

        # ── Contact: build missing list from ACTUAL resume fields ─────────────
        # BUG 4: never recommend adding a field that already exists
        if section == "contact" and resume:
            missing_contact = self._contact_missing_fields(resume, rule_missing, analysis)
            rule_missing = missing_contact

        # ── Missing elements: only show if section is absent ─────────────────
        if not is_present:
            # Section is genuinely absent — show what to add
            missing_elements = [
                {
                    "element": item.get("element", ""),
                    "why":     item.get("why", "Required for ATS compliance"),
                    "impact":  item.get("impact", 5),
                    "type":    "add",
                }
                for item in add_items
                if item.get("impact", 0) >= 8      # only high-impact items
            ][:4]
        else:
            # Section is present — only show fields genuinely missing (from rules)
            missing_elements = [
                {"element": m, "impact": 5, "type": "add"}
                for m in rule_missing[:4]
            ]

        # ── Quality issues from template (only when section is present) ───────
        data_str = self._to_text(data).lower()
        quality_issues_found: List[Dict] = []
        if is_present:
            for qi in quality_items:
                ex = qi.get("example", "").lower()
                if ex and len(ex) > 3 and ex[:20] in data_str:
                    quality_issues_found.append({**qi, "type": "quality"})
            # Add rule engine quality issues
            for rq in rule_quality[:3]:
                if not any(rq.lower() in str(q).lower() for q in quality_issues_found):
                    quality_issues_found.append({"issue": rq, "impact": 4, "type": "quality"})

        # ── is_complete ───────────────────────────────────────────────────────
        is_complete = is_present and not bool(missing_elements)
        if analysis and getattr(analysis, "complete", None) is not None:
            is_complete = getattr(analysis, "complete")

        # ── ATS tips ──────────────────────────────────────────────────────────
        ats_tips = rule_tips[:5] if rule_tips else []

        # ── Priorities ────────────────────────────────────────────────────────
        priorities = self._build_priorities(
            section, missing_elements, quality_issues_found
        )

        # ── Quick wins ────────────────────────────────────────────────────────
        quick_wins = self._build_quick_wins(quality_items, remove_items)

        # ── Detailed suggestions ──────────────────────────────────────────────
        detailed = self._build_detailed(
            add_items if not is_present else [],
            quality_issues_found,
            remove_items if is_present else [],
        )

        # ── Rewrite examples ──────────────────────────────────────────────────
        rewrites = rule_rewrites or []
        if not rewrites:
            _template_rewrites = {
                "summary": {
                    "before": "Experienced professional with strong communication skills",
                    "after":  "[Title] with [X] years in [field]. Expert in [skill 1], [skill 2]. Delivered [metric achievement].",
                },
                "experience": {
                    "before": "Responsible for managing team and handling projects",
                    "after":  "Led cross-functional team of [N], delivering [outcome] [X]% [better/faster/cheaper]",
                },
                "skills": {
                    "before": "Good communication, teamwork, computers",
                    "after":  "[Specific Tool] • [Specific Tool] • [Industry Keyword] • [Certification]",
                },
                "education": {
                    "before": "B.E CS, VIT, 2020",
                    "after":  "Bachelor of Engineering (Computer Science) | VIT University | May 2020",
                },
            }
            if is_fresher_no_exp:
                rewrites = [{
                    "before": "No work experience listed",
                    "after":  "Marketing Intern | StartupCo | Jun 2025 - Aug 2025\n• Assisted in running social media campaigns reaching 10K+ users",
                }]
            elif section in _template_rewrites:
                rewrites = [_template_rewrites[section]]

        # ── Impact potential ──────────────────────────────────────────────────
        missing_impact  = sum(i.get("impact", 5) for i in missing_elements[:5])
        quality_impact  = sum(i.get("impact", 4) for i in quality_issues_found[:5])
        impact_potential = min(missing_impact + quality_impact, 30)

        # ── BUG 5: derive target_score and status from the SAME score ─────────
        target_score  = min(score + impact_potential, 95)
        # is_present=False → status="missing"; otherwise → threshold-based
        # v4.1: fresher-with-no-experience is never "missing"/"critical" —
        # it's a normal, expected state, capped at "needs_improvement".
        if is_fresher_no_exp:
            status = "needs_improvement"
        else:
            status = "missing" if not is_present else (self._status(score) if score > 0 else "missing")
        quality_level = "high" if score >= 80 else ("medium" if score >= 55 else "low")

        # ── Strengths ─────────────────────────────────────────────────────────
        if rule_strengths:
            strengths = rule_strengths
        elif is_present:
            label = section.replace("_", " ").title()
            strengths = [f"{label} section present"]
        else:
            strengths = []

        return SectionFeedback(
            section_name=section,
            current_score=score,
            target_score=target_score,
            status=status,
            impact_potential=impact_potential,
            is_present=is_present,
            is_complete=is_complete,
            quality_level=quality_level,
            missing_elements=missing_elements,
            elements_to_remove=[
                {
                    "element": r.get("element", ""),
                    "why":     r.get("why", ""),
                    "action":  r.get("action", "Remove"),
                }
                for r in remove_items[:4]
            ] if is_present else [],
            quality_issues=[
                {
                    "issue":   q.get("issue", q) if isinstance(q, dict) else q,
                    "example": q.get("example", "") if isinstance(q, dict) else "",
                    "fix":     q.get("fix", "") if isinstance(q, dict) else "",
                    "impact":  q.get("impact", 4) if isinstance(q, dict) else 4,
                }
                for q in quality_issues_found
            ][:6],
            top_priority_fixes=priorities,
            quick_wins=quick_wins,
            detailed_suggestions=detailed,
            ats_passing_tips=ats_tips,
            rewrite_examples=rewrites[:3],
            strengths=strengths,
        )

    # ── Contact field checker (BUG 4) ─────────────────────────────────────────

    def _contact_missing_fields(
        self,
        resume: Dict,
        rule_missing: List[str],
        analysis,
    ) -> List[str]:
        """
        Returns ONLY fields that are genuinely absent from the contact section.
        Never recommends adding a field that already has a value.
        """
        # Prefer the analysis-provided missing list from the rules engine
        # (which was built from NormalizedContact by _build_contact_section_proxy)
        if analysis:
            fields = getattr(analysis, "missing_fields", None)
            if fields is not None:
                return list(fields)

        # Fallback: inspect the resume dict directly
        missing: List[str] = []
        if not (resume.get("name") or "").strip():
            missing.append("Full name")
        if not (resume.get("email") or "").strip():
            missing.append("Email address")
        if not (resume.get("phone") or "").strip():
            missing.append("Phone number")
        if not (resume.get("location") or "").strip():
            missing.append("Location (City, Country)")
        if not (resume.get("linkedin") or "").strip():
            missing.append("LinkedIn profile URL")
        return missing

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _alt_key(self, section: str) -> str:
        """Alternate resume dict keys for the same section."""
        return {
            "contact": "header",
        }.get(section, section)

    def _build_priorities(
        self, section: str, missing: List[Dict], quality: List[Dict],
    ) -> List[Dict]:
        prios: List[Dict] = []
        for item in sorted(missing, key=lambda x: x.get("impact", 0), reverse=True)[:2]:
            prios.append({
                "action":         f"Add: {item.get('element', 'missing element')}",
                "why":            item.get("why", "Required for ATS compliance"),
                "estimated_gain": item.get("impact", 5),
                "effort":         "easy",
                "time":           "5-10 minutes",
            })
        for item in quality[:2]:
            prios.append({
                "action":         f"Fix: {item.get('issue', 'quality issue')}",
                "current":        item.get("example", ""),
                "improved":       item.get("fix", ""),
                "estimated_gain": item.get("impact", 4),
                "effort":         "easy",
                "time":           "10-15 minutes",
            })
        return prios[:4]

    def _build_quick_wins(self, quality_items: List, remove_items: List) -> List[Dict]:
        wins: List[Dict] = []
        for qi in quality_items[:3]:
            wins.append({
                "action":         qi.get("issue", "Fix quality issue"),
                "how":            f"Change: '{qi.get('example', '')}' → '{qi.get('fix', '')}'",
                "estimated_gain": qi.get("impact", 3),
                "effort":         "5 minutes",
            })
        for ri in remove_items[:2]:
            wins.append({
                "action":         f"Remove: {ri.get('element', '')}",
                "why":            ri.get("why", ""),
                "estimated_gain": 2,
                "effort":         "2 minutes",
            })
        return wins[:4]

    def _build_detailed(
        self, add_items: List, quality_items: List, remove_items: List,
    ) -> List[Dict]:
        suggestions: List[Dict] = []
        for item in add_items:
            suggestions.append({
                "type":    "add",
                "element": item.get("element"),
                "reason":  item.get("why"),
                "example": item.get("example"),
                "impact":  item.get("impact", 5),
            })
        for qi in quality_items:
            suggestions.append({
                "type":             "improve",
                "issue":            qi.get("issue"),
                "current_example":  qi.get("example"),
                "improved_example": qi.get("fix"),
                "impact":           qi.get("impact", 4),
            })
        for ri in remove_items:
            suggestions.append({
                "type":    "remove",
                "element": ri.get("element"),
                "reason":  ri.get("why"),
                "action":  ri.get("action"),
                "impact":  2,
            })
        return suggestions

    # ── Overall insight helpers ───────────────────────────────────────────────

    def _top_priorities(self, sections: Dict[str, SectionFeedback]) -> List[str]:
        prios = []
        for name, fb in sorted(sections.items(), key=lambda x: x[1].impact_potential, reverse=True)[:3]:
            if fb.top_priority_fixes:
                action = fb.top_priority_fixes[0].get("action", "")
                gain   = fb.top_priority_fixes[0].get("estimated_gain", 0)
                prios.append(f"[{name.title()}] {action} (estimated +{gain} pts)")
        return prios

    def _quick_wins(self, sections: Dict[str, SectionFeedback]) -> List[str]:
        wins = []
        for name, fb in sections.items():
            for win in fb.quick_wins[:1]:
                wins.append(
                    f"[{name.title()}] {win.get('action', '')} "
                    f"(~{win.get('estimated_gain', 2)} pts, {win.get('effort', '5 min')})"
                )
        return wins[:5]

    def _strengths(self, sections: Dict[str, SectionFeedback]) -> List[str]:
        all_strengths = []
        for fb in sections.values():
            all_strengths.extend(fb.strengths)
        return all_strengths[:6]

    def _build_roadmap(self, sections: Dict[str, SectionFeedback]) -> List[Dict]:
        roadmap = []
        step = 1
        for name, fb in sorted(sections.items(), key=lambda x: x[1].impact_potential, reverse=True):
            if fb.top_priority_fixes and fb.impact_potential > 0:
                fix = fb.top_priority_fixes[0]
                roadmap.append({
                    "step":           step,
                    "section":        name,
                    "action":         fix.get("action", ""),
                    "effort":         fix.get("effort", "medium"),
                    "estimated_gain": fix.get("estimated_gain", 5),
                    "time_estimate":  fix.get("time", "15 minutes"),
                    "why_now":        f"Fixing {name} has the highest remaining impact potential ({fb.impact_potential} pts)",
                })
                step += 1
            if step > 8:
                break
        return roadmap

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _is_present(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return len(value.strip()) > 0
        if isinstance(value, (list, dict)):
            return len(value) > 0
        return bool(value)

    @staticmethod
    def _to_text(value: Any) -> str:
        if not value:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts = []
            for item in value:
                if isinstance(item, dict):
                    parts.append(" ".join(str(v) for v in item.values() if v))
                else:
                    parts.append(str(item))
            return " ".join(parts)
        if isinstance(value, dict):
            return " ".join(str(v) for v in value.values() if v)
        return str(value)

    def _status(self, score: int) -> str:
        """Derive status label from score.  Always consistent with score."""
        for status, (lo, hi) in self.SCORE_THRESHOLDS.items():
            if lo <= score <= hi:
                return status
        return "missing"

    @staticmethod
    def _grade(score: int) -> str:
        if score >= 90: return "A+"
        if score >= 85: return "A"
        if score >= 80: return "A-"
        if score >= 75: return "B+"
        if score >= 70: return "B"
        if score >= 65: return "B-"
        if score >= 60: return "C+"
        if score >= 55: return "C"
        if score >= 50: return "C-"
        if score >= 40: return "D"
        return "F"

    @staticmethod
    def _percentile(score: int) -> str:
        if score >= 90: return "Top 5% of applicants"
        if score >= 80: return "Top 15% of applicants"
        if score >= 70: return "Top 30% of applicants"
        if score >= 60: return "Top 50% of applicants"
        if score >= 50: return "Bottom 50% — significant improvements needed"
        return "High risk of ATS rejection — immediate action required"