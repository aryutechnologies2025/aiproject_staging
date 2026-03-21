"""
ATS Feedback Generator v3
─────────────────────────────────────────────────────────────────────────────
Generates ATS-passing suggestions designed to get the resume through
automated screening AND impress the human recruiter after.

For every section it produces:
  • What to add (with examples)
  • What to remove (padding/ATS-unfriendly elements)
  • How to rewrite (before → after)
  • Prioritised quick wins
  • An improvement roadmap sorted by impact/effort ratio

All output is industry-agnostic and works for:
  Healthcare, Finance, Legal, Marketing, Education, Design,
  Engineering, Sales, HR, Science, Government, and more.
─────────────────────────────────────────────────────────────────────────────
"""

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
    ats_passing_tips:    List[str]     = field(default_factory=list)  # concrete ATS tactics
    rewrite_examples:    List[Dict]    = field(default_factory=list)  # before/after
    strengths:           List[str]     = field(default_factory=list)


@dataclass
class ComprehensiveFeedback:
    overall_score:                 int
    overall_status:                str
    ready_to_apply:                bool
    estimated_improvement_potential: int
    grade:                         str     # A / B / C / D / F
    percentile_estimate:           str     # "Top 20% of applicants"

    section_feedback:   Dict[str, SectionFeedback] = field(default_factory=dict)

    # Actionable summaries
    top_3_priorities:   List[str]  = field(default_factory=list)
    quick_wins_summary: List[str]  = field(default_factory=list)
    strengths_summary:  List[str]  = field(default_factory=list)
    ats_passing_tactics: List[str] = field(default_factory=list)  # top global tactics

    # Step-by-step roadmap
    improvement_roadmap: List[Dict] = field(default_factory=list)

    # Recruiter intelligence
    recruiter_tips:      List[str]  = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION TEMPLATES  (add / remove / quality / rewrite examples)
# ─────────────────────────────────────────────────────────────────────────────

# add_items:    [{"element", "why", "impact", "example"}]
# remove_items: [{"element", "why", "action"}]
# quality_items:[{"issue", "example", "fix", "impact"}]

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
             "example": "+1 (555) 234-5678"},
            {"element": "City and State/Country",
             "why":     "Many ATS filter by location; missing location = likely rejection",
             "impact":  7,
             "example": "Austin, TX, USA"},
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
            {"element": "Multiple phone numbers",
             "why":     "Confusing; ATS may parse both incorrectly",
             "action":  "Keep one primary number only"},
        ],
        "quality_items": [
            {"issue": "Unprofessional email domain",
             "example": "coolkid1995@yahoo.com",
             "fix":     "Create firstname.lastname@gmail.com",
             "impact":  8},
            {"issue": "No LinkedIn link",
             "example": "(no link listed)",
             "fix":     "Add linkedin.com/in/yourname next to your email",
             "impact":  6},
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
             "example": "8+ years in clinical nursing" or "5 years managing B2B accounts"},
            {"element": "2-3 core skills or specialisations",
             "why":     "Keyword-rich summary boosts ATS match score significantly",
             "impact":  12,
             "example": "Specialising in Python, AWS, and distributed systems"},
            {"element": "One quantified achievement",
             "why":     "Numbers in summary catch recruiters' eyes within 6 seconds",
             "impact":  10,
             "example": "Delivered $4M in annual savings | Maintained 99.2% patient satisfaction"},
            {"element": "Target role / Career direction",
             "why":     "Aligns resume to job posting; ATS semantic matching benefits",
             "impact":  7,
             "example": "Seeking senior-level roles in healthcare administration"},
        ],
        "remove_items": [
            {"element": "First-person pronouns (I, me, my)",
             "why":     "Non-standard in resumes; wastes characters",
             "action":  "Delete 'I am' → start directly with your title or achievement"},
            {"element": "Generic clichés",
             "why":     "Every resume uses them; they add zero value",
             "action":  "Replace 'hard-working team player' with specific evidence"},
            {"element": "Objective statement (old format)",
             "why":     "Outdated; focuses on what YOU want, not what you offer",
             "action":  "Replace with results-focused professional summary"},
            {"element": "Summaries over 150 words",
             "why":     "Recruiters spend <10s on summary; length kills impact",
             "action":  "Trim to 50-80 words maximum"},
        ],
        "quality_items": [
            {"issue": "No numbers or metrics",
             "example": "Experienced professional with strong communication skills",
             "fix":     "Results-driven RN with 6 years in ICU care. Maintained 98% medication accuracy across 200+ patients monthly. BLS/ACLS certified.",
             "impact":  12},
            {"issue": "Generic / vague",
             "example": "Dynamic self-starter with a passion for excellence",
             "fix":     "Certified CPA with 7 years in Big 4 audit. Led $15M SOX compliance programme, reducing audit deficiencies by 40%.",
             "impact":  10},
            {"issue": "Too long",
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
             "example": "Jan 2020 – Mar 2023"},
            {"element": "Job title, Company name, Location",
             "why":     "Three required ATS fields — missing any one causes parse failure",
             "impact":  12,
             "example": "Senior Nurse Practitioner | Mayo Clinic | Rochester, MN"},
            {"element": "Industry keywords in bullet context",
             "why":     "Embedding JD keywords in bullets (not just skills) doubles keyword hit rate",
             "impact":  10,
             "example": "Implemented HIPAA-compliant EHR workflow across 4 departments"},
        ],
        "remove_items": [
            {"element": "Passive / weak openers",
             "why":     "Drops ATS content quality score",
             "action":  "Replace 'Responsible for' → 'Managed'; 'Helped with' → 'Collaborated'"},
            {"element": "Irrelevant old jobs (15+ years)",
             "why":     "Pads length; ATS dilutes keyword density with unrelated content",
             "action":  "Summarise as 'Earlier Career: [Role] at [Company] (Year-Year)'"},
            {"element": "More than 8 bullets per role",
             "why":     "ATS truncates; recruiter attention drops after bullet 6",
             "action":  "Keep 4-6 strongest, most relevant bullets per role"},
            {"element": "Duties-focused language",
             "why":     "Describes what the role required, not what YOU delivered",
             "action":  "Flip from 'duties included X' to 'achieved/delivered X'"},
            {"element": "Unexplained acronyms",
             "why":     "ATS may not map acronym to keyword — use both forms",
             "action":  "Write: EHR (Electronic Health Records), not just EHR"},
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
            {"issue": "Wall of text (no bullets)",
             "example": "I was responsible for many things including...",
             "fix":     "Use 4-6 bullet points. Each bullet = one achievement with a result.",
             "impact":  8},
        ],
    },

    "education": {
        "add_items": [
            {"element": "Full degree name (not abbreviated)",
             "why":     "ATS matches 'Bachelor of Science in Nursing' — abbreviations may miss",
             "impact":  10,
             "example": "Bachelor of Science in Nursing (BSN), not just 'BSN'"},
            {"element": "Full institution name",
             "why":     "ATS may filter by institution; abbreviations cause mismatches",
             "impact":  8,
             "example": "University of California, Los Angeles — not 'UCLA'"},
            {"element": "Graduation year (or expected graduation)",
             "why":     "ATS calculates experience timeline from education dates",
             "impact":  8,
             "example": "May 2023 | Expected Dec 2024"},
            {"element": "GPA if 3.5 or above",
             "why":     "High GPA is a differentiator for early-career candidates",
             "impact":  5,
             "example": "GPA: 3.8/4.0 | Magna Cum Laude"},
            {"element": "Relevant coursework (for recent grads)",
             "why":     "Fills keyword gaps when work experience is limited",
             "impact":  6,
             "example": "Coursework: Financial Accounting, Tax Law, Corporate Finance, Audit"},
        ],
        "remove_items": [
            {"element": "GPA below 3.5",
             "why":     "Below-average GPA hurts more than it helps",
             "action":  "Remove unless the job posting specifically requests GPA"},
            {"element": "High school diploma (if you have a degree)",
             "why":     "Redundant once higher education is listed; wastes space",
             "action":  "Delete high school entry entirely"},
            {"element": "Irrelevant courses",
             "why":     "Dilutes keyword relevance in education section",
             "action":  "List only 4-6 courses directly relevant to the target role"},
        ],
        "quality_items": [
            {"issue": "Abbreviated degree name",
             "example": "BSN, UT Austin, 2020",
             "fix":     "Bachelor of Science in Nursing | University of Texas at Austin | May 2020",
             "impact":  8},
            {"issue": "No graduation year",
             "example": "B.A. English | State University",
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
             "example": "EHR (Electronic Health Records) | CRM (Customer Relationship Management)"},
            {"element": "Industry-specific hard skills",
             "why":     "Differentiates you from generic candidates",
             "impact":  12,
             "example": "Epic EHR | QuickBooks | AutoCAD | Salesforce | Bloomberg Terminal"},
            {"element": "Certifications as skills",
             "why":     "Cert names are high-value ATS keywords",
             "impact":  8,
             "example": "CPA | PMP | ACLS | AWS Solutions Architect"},
            {"element": "Tools and software",
             "why":     "Tool names are among the highest-frequency JD keywords",
             "impact":  10,
             "example": "Jira | Slack | HubSpot | Zoom | Microsoft Teams | SAP"},
        ],
        "remove_items": [
            {"element": "Proficiency bars / star ratings",
             "why":     "ATS cannot read visual elements; they contribute zero to keyword score",
             "action":  "Replace with text: 'Python (Advanced)' or 'Python — 5 years'"},
            {"element": "Soft skills without evidence",
             "why":     "Listed alone in skills, soft skills score near zero in ATS",
             "action":  "Move to summary or embed in experience bullets with examples"},
            {"element": "Outdated technologies (10+ years old)",
             "why":     "May signal outdated knowledge; dilutes keyword relevance",
             "action":  "Remove unless the target role explicitly lists them"},
            {"element": "Generic MS Office",
             "why":     "Too vague; ATS won't match to specific software queries",
             "action":  "Replace with: Microsoft Excel (Advanced), PowerPoint, Word, Outlook"},
            {"element": "Skills list exceeding 40 items",
             "why":     "Keyword dilution — ATS drops confidence score on over-stuffed lists",
             "action":  "Curate to 12-25 most relevant, role-specific skills"},
        ],
        "quality_items": [
            {"issue": "Vague skills",
             "example": "Good with computers, databases, cloud",
             "fix":     "Python • PostgreSQL • AWS EC2 • Docker • Kubernetes",
             "impact":  12},
            {"issue": "Missing tools from JD",
             "example": "(JD mentions Salesforce, resume doesn't list it)",
             "fix":     "Add 'Salesforce' to skills section if you have used it",
             "impact":  15},
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
             "example": "Patient Scheduling System | E-Commerce Analytics Dashboard"},
            {"element": "Technologies / tools used",
             "why":     "These are keyword-rich and directly match JD skill requirements",
             "impact":  10,
             "example": "Built using React, Node.js, PostgreSQL, and AWS Lambda"},
            {"element": "Quantified outcome",
             "why":     "Numbers differentiate a project from a class assignment",
             "impact":  10,
             "example": "Reduced scheduling conflicts by 70% | Served 10,000 monthly users"},
            {"element": "Link (GitHub / Live URL / Portfolio)",
             "why":     "Proves the work exists; recruiters click these",
             "impact":  6,
             "example": "github.com/name/project | projectname.com"},
        ],
        "remove_items": [
            {"element": "Class/academic projects labelled as professional",
             "why":     "Misrepresentation — label them clearly as 'Academic Project'",
             "action":  "Add '(Academic Project)' tag to university coursework"},
            {"element": "Projects older than 5 years (unless landmark)",
             "why":     "Old tech stack signals outdated skills",
             "action":  "Remove or consolidate under 'Earlier Projects'"},
        ],
        "quality_items": [
            {"issue": "Project with no outcome",
             "example": "Built a web app for task management",
             "fix":     "Built task management app (React, Firebase) adopted by 3 teams; "
                        "reduced missed deadlines by 45%",
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
             "example": "Issued by: PMI | SHRM | Amazon Web Services | American Heart Association"},
            {"element": "Date obtained and expiry (if applicable)",
             "why":     "ATS and recruiters check recency; expired certs raise red flags",
             "impact":  8,
             "example": "Obtained: Mar 2022 | Expires: Mar 2025"},
            {"element": "Licence number (if role-required)",
             "why":     "Mandatory for regulated professions: nursing, law, engineering",
             "impact":  10,
             "example": "RN Licence #123456 | State of California | Valid through 2026"},
        ],
        "remove_items": [
            {"element": "Expired certifications (without noting renewal plan)",
             "why":     "Raises questions about currency of knowledge",
             "action":  "Remove or note: '(Renewal in progress, expected Jan 2025)'"},
            {"element": "Irrelevant certifications",
             "why":     "Clutters the section; ATS may dilute score",
             "action":  "Keep only certs relevant to the target role"},
        ],
        "quality_items": [
            {"issue": "Missing issue date",
             "example": "PMP Certified",
             "fix":     "Project Management Professional (PMP) | PMI | Obtained Jan 2022",
             "impact":  8},
            {"issue": "Abbreviation only",
             "example": "ACLS, BLS, CNA",
             "fix":     "ACLS (Advanced Cardiac Life Support) | AHA | Valid 2024–2026",
             "impact":  6},
        ],
    },

    "languages": {
        "add_items": [
            {"element": "Proficiency level for every language",
             "why":     "ATS and recruiters need to know if you can actually do the job in that language",
             "impact":  8,
             "example": "Spanish (Professional Proficiency) | Mandarin (Conversational)"},
            {"element": "CEFR level for international roles",
             "why":     "Standard framework understood globally by ATS and hiring managers",
             "impact":  5,
             "example": "German (B2 – Upper Intermediate)"},
        ],
        "remove_items": [
            {"element": "Languages you cannot use professionally",
             "why":     "Listing 'Basic Portuguese' for a Portuguese-speaking role misleads",
             "action":  "Only list if you can hold professional conversations"},
        ],
        "quality_items": [
            {"issue": "Language without proficiency",
             "example": "French, German, Italian",
             "fix":     "French (Native) | German (Professional) | Italian (Conversational)",
             "impact":  8},
        ],
    },

    "volunteer": {
        "add_items": [
            {"element": "Organisation name and your role/title",
             "why":     "ATS treats volunteer work like paid work if formatted correctly",
             "impact":  6,
             "example": "Volunteer Nurse | Red Cross Blood Drive | Jan 2022 – Present"},
            {"element": "Impact metrics",
             "why":     "Numbers make volunteer experience credible and comparable to paid work",
             "impact":  8,
             "example": "Coordinated care for 150+ patients | Raised $28K for community health fund"},
            {"element": "Date range",
             "why":     "Fills employment gaps; shows ongoing commitment",
             "impact":  6,
             "example": "Jun 2020 – Present"},
        ],
        "remove_items": [
            {"element": "One-time events listed as ongoing roles",
             "why":     "Misrepresents commitment; recruiters verify",
             "action":  "Label clearly: '(One-time event, Mar 2023)'"},
        ],
        "quality_items": [
            {"issue": "No outcome or impact stated",
             "example": "Volunteered at a hospital during COVID",
             "fix":     "Volunteer Patient Liaison | City General Hospital | Mar–Dec 2020\n"
                        "• Supported 80+ patients weekly; coordinated with nurses to improve "
                        "discharge communication during peak COVID period.",
             "impact":  8},
        ],
    },

    "awards": {
        "add_items": [
            {"element": "Award name, issuing body, and year",
             "why":     "All three fields needed for ATS to categorise and date the achievement",
             "impact":  6,
             "example": "Employee of the Year | Acme Corp | 2022"},
            {"element": "Scope / significance statement",
             "why":     "Context turns a generic title into a credible differentiator",
             "impact":  6,
             "example": "Awarded to top 2% of 4,500 global employees"},
        ],
        "remove_items": [
            {"element": "Awards older than 10 years (unless extraordinary)",
             "why":     "Old awards suggest peak has passed",
             "action":  "Remove or consolidate under 'Early Career Recognition'"},
        ],
        "quality_items": [
            {"issue": "Award with no context",
             "example": "Top Performer Award, 2022",
             "fix":     "Top Performer Award | XYZ Company | Q3 2022\n"
                        "• Recognised for exceeding sales quota by 185% among 200 reps nationwide",
             "impact":  6},
        ],
    },

    "publications": {
        "add_items": [
            {"element": "Full citation (Authors, Title, Journal, Year)",
             "why":     "Standard academic format expected by ATS in research/academic roles",
             "impact":  7,
             "example": 'Smith, J., Lee, A. "Deep Learning in Medical Imaging." '
                        "Nature Medicine, 2023. DOI: 10.1038/xxx"},
            {"element": "DOI or URL",
             "why":     "Allows recruiters/ATS to verify; signals transparency",
             "impact":  5,
             "example": "doi.org/10.xxxx | arxiv.org/abs/xxxx"},
        ],
        "remove_items": [
            {"element": "Non-peer-reviewed blog posts (in publications section)",
             "why":     "Misrepresents publication quality",
             "action":  "Move to 'Content / Writing' or remove from Publications"},
        ],
        "quality_items": [
            {"issue": "Vague publication reference",
             "example": "Published research on AI in healthcare",
             "fix":     'Smith, J. "AI-Assisted Diagnosis in Radiology." '
                        "Radiology Today, Vol.12, 2023. doi:10.xxxx",
             "impact":  7},
        ],
    },

    "hobbies": {
        "add_items": [
            {"element": "Role-relevant activities only",
             "why":     "Hobbies are optional — only add if they signal relevant skills or personality",
             "impact":  3,
             "example": "Open-source contributor | Community coding bootcamp mentor | "
                        "Toastmasters public speaking"},
        ],
        "remove_items": [
            {"element": "Generic hobbies (reading, travel, music)",
             "why":     "Everyone lists these; zero differentiation",
             "action":  "Remove or replace with specific, verifiable activities"},
            {"element": "Politically or religiously sensitive activities",
             "why":     "Risk introducing unconscious bias",
             "action":  "Remove; keep hobbies professionally neutral"},
        ],
        "quality_items": [
            {"issue": "Only generic hobbies listed",
             "example": "Reading, travel, cooking",
             "fix":     "Consider omitting this section entirely OR replacing with: "
                        "'Marathon runner (3:42 PR) | Amateur photographer | "
                        "Open-source contributor (React ecosystem)'",
             "impact":  3},
        ],
    },

    "references": {
        "add_items": [
            {"element": "Full reference contact details (if requested by role)",
             "why":     "Saves recruiter time; signals professionalism",
             "impact":  3,
             "example": "Jane Smith | VP Marketing | Acme Corp | "
                        "jane@acme.com | +1 555-123-4567"},
        ],
        "remove_items": [
            {"element": '"References available upon request"',
             "why":     "Implied by every resume; wastes a line",
             "action":  "Delete this line entirely"},
            {"element": "References on the main resume (unless requested)",
             "why":     "Prepare a separate Reference Sheet; don't crowd your resume",
             "action":  "Remove and prepare a separate reference document"},
        ],
        "quality_items": [],
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL ATS-PASSING TACTICS
# ─────────────────────────────────────────────────────────────────────────────

GLOBAL_ATS_TACTICS = [
    "Submit as PDF (unless the ATS explicitly asks for DOCX) — PDF preserves formatting.",
    "Use standard section headings: 'Work Experience' not 'My Journey' — ATS searches for exact headings.",
    "Mirror keywords verbatim from the job description — ATS does exact-string matching.",
    "Place the most important keywords in the top third of the first page (highest ATS weight).",
    "Use both the spelled-out form AND abbreviation for every key term: EHR (Electronic Health Records).",
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
    Generates comprehensive, actionable ATS-passing feedback
    for all 13 resume sections across all industries.
    """

    SCORE_THRESHOLDS = {
        "excellent":        (85, 100),
        "good":             (70, 84),
        "needs_improvement":(50, 69),
        "critical":         (20, 49),
        "missing":          (0, 19),
    }

    ALL_SECTIONS = [
        "contact", "summary", "experience", "education", "skills",
        "projects", "certifications", "languages", "volunteer",
        "publications", "awards", "hobbies", "references",
    ]

    def generate_detailed_feedback(
        self,
        ats_score:     int,
        section_scores: Dict[str, int],
        resume:        Dict,
        ats_issues:    List,
        section_analyses: Dict = None,
    ) -> ComprehensiveFeedback:
        """
        Generate full feedback report.

        Args:
            ats_score:        Overall ATS score (0-100)
            section_scores:   Per-section scores
            resume:           Resume dict
            ats_issues:       Issue list from ATSRulesEngine
            section_analyses: SectionAnalysis objects from ATSRulesEngine
        """

        logger.info(f"Generating detailed feedback — overall score: {ats_score}")

        section_feedback: Dict[str, SectionFeedback] = {}

        for section in self.ALL_SECTIONS:
            score = section_scores.get(section, 0)
            section_data = resume.get(section)
            issues_for_section = [
                i for i in ats_issues if getattr(i, "section", "") == section
            ]
            analysis = (section_analyses or {}).get(section)

            feedback = self._build_section_feedback(
                section, score, section_data, issues_for_section, analysis
            )
            section_feedback[section] = feedback

        # Global insights
        top_3       = self._top_priorities(section_feedback)
        quick_wins  = self._quick_wins(section_feedback)
        strengths   = self._strengths(section_feedback)
        roadmap     = self._build_roadmap(section_feedback)
        total_pot   = sum(sf.impact_potential for sf in section_feedback.values())

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

    # ── PER-SECTION BUILDER ───────────────────────────────────────────────────

    def _build_section_feedback(
        self,
        section:      str,
        score:        int,
        data:         Any,
        issues:       List,
        analysis=None,
    ) -> SectionFeedback:

        template      = SECTION_TEMPLATES.get(section, {})
        is_present    = self._is_present(data)
        is_complete   = self._is_complete(section, data, analysis)

        add_items     = template.get("add_items", [])
        remove_items  = template.get("remove_items", [])
        quality_items = template.get("quality_items", [])

        # Detect which add_items are actually missing
        data_str = self._to_text(data).lower()
        missing_elements = [
            {**item, "type": "add"}
            for item in add_items
            if not is_present or len(data_str) < 10
               or any(
                    kw in data_str
                    for kw in ["no", "missing", "none"]
               ) is False and item["impact"] >= 8
        ] if not is_present else []

        # If section is present, check remove / quality
        to_remove = []
        quality_issues_found = []
        if is_present:
            # Quality checks from template
            for qi in quality_items:
                ex = qi.get("example", "").lower()
                if ex and len(ex) > 3 and ex in data_str:
                    quality_issues_found.append({**qi, "type": "quality"})

        # Pull from rule analysis if available
        rule_missing: List[str] = []
        rule_quality: List[str] = []
        rule_strengths: List[str] = []
        rule_tips: List[str] = []
        rule_rewrites: List[Dict] = []

        if analysis:
            rule_missing    = getattr(analysis, "missing_fields", [])
            rule_quality    = getattr(analysis, "quality_issues", [])
            rule_strengths  = getattr(analysis, "strengths", [])
            rule_tips       = getattr(analysis, "ats_tips", [])
            rule_rewrites   = getattr(analysis, "rewrite_examples", [])
            score           = getattr(analysis, "current_score", score)

        # ATS-passing tips: combine rule tips + template global tactics
        ats_tips = rule_tips[:5] if rule_tips else []

        # Build top priority fixes
        priorities = self._build_priorities(
            section, missing_elements, quality_issues_found,
            rule_missing, rule_quality
        )

        # Build quick wins
        quick_wins = self._build_quick_wins(quality_items, remove_items)

        # Detailed suggestions
        detailed = self._build_detailed(add_items, quality_items, remove_items)

        # Rewrite examples from template + rules
        rewrites = rule_rewrites or []
        if not rewrites:
            template_ex = {
                "summary": {"before": "Experienced professional with strong skills",
                            "after":  "[Title] with [X] years in [field]. Expert in [skill 1], [skill 2]. Delivered [metric achievement]."},
                "experience": {"before": "Responsible for managing team and handling projects",
                               "after":  "Led cross-functional team of [N], delivering [outcome] [X]% [better/faster/cheaper]"},
                "skills": {"before": "Good communication, teamwork, computers",
                           "after":  "[Specific Tool] • [Specific Tool] • [Industry Keyword] • [Certification]"},
                "education": {"before": "CS, MIT, 2020",
                              "after":  "Bachelor of Science in Computer Science | MIT | May 2020 | GPA: 3.9"},
            }
            if section in template_ex:
                rewrites = [template_ex[section]]

        # Impact potential
        missing_impact = sum(i.get("impact", 5) for i in missing_elements[:5])
        quality_impact = sum(i.get("impact", 4) for i in quality_issues_found[:5])
        impact_potential = min(missing_impact + quality_impact, 30)

        target_score = min(score + impact_potential, 95)
        status = self._status(score) if is_present else "missing"
        quality_level = "high" if score >= 80 else ("medium" if score >= 55 else "low")

        strengths = rule_strengths if rule_strengths else (
            [f"{section.replace('_', ' ').title()} section is present"] if is_present else []
        )

        return SectionFeedback(
            section_name=section,
            current_score=score,
            target_score=target_score,
            status=status,
            impact_potential=impact_potential,
            is_present=is_present,
            is_complete=is_complete,
            quality_level=quality_level,
            missing_elements=[
                {"element": m.get("element", m) if isinstance(m, dict) else m,
                 "why":     m.get("why", "Required for ATS compliance") if isinstance(m, dict) else "",
                 "impact":  m.get("impact", 5) if isinstance(m, dict) else 5}
                for m in (missing_elements or [{"element": r, "impact": 5} for r in rule_missing[:4]])
            ][:6],
            elements_to_remove=[
                {"element": r.get("element", ""), "why": r.get("why", ""),
                 "action":  r.get("action", "Remove")}
                for r in remove_items[:4]
            ],
            quality_issues=[
                {"issue": q.get("issue", q) if isinstance(q, dict) else q,
                 "example": q.get("example", "") if isinstance(q, dict) else "",
                 "fix":     q.get("fix", "") if isinstance(q, dict) else "",
                 "impact":  q.get("impact", 4) if isinstance(q, dict) else 4}
                for q in (quality_issues_found or [{"issue": rq, "impact": 4} for rq in rule_quality[:4]])
            ][:6],
            top_priority_fixes=priorities,
            quick_wins=quick_wins,
            detailed_suggestions=detailed,
            ats_passing_tips=ats_tips,
            rewrite_examples=rewrites[:3],
            strengths=strengths,
        )

    # ── HELPERS ───────────────────────────────────────────────────────────────

    def _build_priorities(self, section, missing, quality, rule_missing, rule_quality) -> List[Dict]:
        prios: List[Dict] = []

        # From missing elements (highest impact first)
        for item in sorted(missing, key=lambda x: x.get("impact", 0), reverse=True)[:2]:
            prios.append({
                "action":         f"Add: {item.get('element', 'missing element')}",
                "why":            item.get("why", "Required for ATS compliance"),
                "estimated_gain": item.get("impact", 5),
                "effort":         "easy",
                "time":           "5-10 minutes",
            })

        # From quality issues
        for item in quality[:2]:
            prios.append({
                "action":         f"Fix: {item.get('issue', 'quality issue')}",
                "current":        item.get("example", ""),
                "improved":       item.get("fix", ""),
                "estimated_gain": item.get("impact", 4),
                "effort":         "easy",
                "time":           "10-15 minutes",
            })

        # From rule engine
        for rm in rule_missing[:1]:
            if not any(rm in str(p) for p in prios):
                prios.append({
                    "action":         f"Add: {rm}",
                    "estimated_gain": 5,
                    "effort":         "easy",
                    "time":           "5 minutes",
                })

        return prios[:4]

    def _build_quick_wins(self, quality_items, remove_items) -> List[Dict]:
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

    def _build_detailed(self, add_items, quality_items, remove_items) -> List[Dict]:
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
                "type":            "improve",
                "issue":           qi.get("issue"),
                "current_example": qi.get("example"),
                "improved_example": qi.get("fix"),
                "impact":          qi.get("impact", 4),
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

    # ── OVERALL INSIGHTS ─────────────────────────────────────────────────────

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
                    "why_now":        f"Fixing {name} has the highest remaining impact potential "
                                      f"({fb.impact_potential} pts)",
                })
                step += 1
            if step > 8:
                break
        return roadmap

    # ── UTILS ─────────────────────────────────────────────────────────────────

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
    def _is_complete(section: str, data: Any, analysis) -> bool:
        if analysis:
            return getattr(analysis, "complete", False)
        if not DetailedFeedbackGenerator._is_present(data):
            return False
        required = {
            "experience": ["title", "company"],
            "education":  ["degree", "institution"],
        }
        if section in required and isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict):
                return any(f in first for f in required[section])
        return True

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
        return str(value)

    def _status(self, score: int) -> str:
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