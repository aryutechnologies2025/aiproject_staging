"""
Universal ATS Keyword Engine v3
─────────────────────────────────────────────────────────────────────────────
Covers ALL industries and ALL professions, not just tech.
Healthcare · Finance · Legal · Marketing · Sales · HR · Education · Design
Engineering · Construction · Hospitality · Creative · Science · Government

Features:
  - 800+ skills across 20+ industries and 50+ categories
  - Synonym/variant matching (e.g. "RN" == "registered nurse")
  - Industry auto-detection from resume content
  - Criticality weighting from job description context
  - Keyword density analysis
  - Missing-keyword recommendations with suggested placement
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SkillMatch:
    skill: str
    found_in_resume: bool
    frequency_in_jd: int
    matched_variants: List[str]
    confidence: float          # 0.0 – 1.0
    category: str
    industry: str
    criticality: float         # 0.5 – 2.0 (higher = more critical)
    suggested_section: str     # where to place if missing


@dataclass
class KeywordAnalysis:
    total_jd_keywords: int
    matched_keywords: int
    match_percentage: int
    matched_skills: List[SkillMatch]
    missing_critical_skills: List[str]          # required, not in resume
    missing_preferred_skills: List[str]         # preferred, not in resume
    found_strengths: List[str]
    keyword_density: float                      # 0.0 – 1.0
    detected_industry: str
    ats_keyword_gaps: List[Dict]                # actionable gap objects
    keyword_suggestions: List[Dict]             # where to add each keyword


# ─────────────────────────────────────────────────────────────────────────────
# UNIVERSAL SKILL DATABASE  (industry → category → skills)
# ─────────────────────────────────────────────────────────────────────────────

UNIVERSAL_SKILLS: Dict[str, Dict[str, Set[str]]] = {

    # ── INFORMATION TECHNOLOGY ───────────────────────────────────────────────
    "technology": {
        "programming_languages": {
            "python", "java", "javascript", "typescript", "c#", "c++", "c",
            "ruby", "php", "swift", "kotlin", "go", "rust", "scala", "r",
            "perl", "elixir", "haskell", "dart", "lua", "matlab", "groovy",
            "cobol", "fortran", "assembly", "shell", "bash", "powershell",
        },
        "web_frontend": {
            "react", "angular", "vue", "svelte", "next.js", "nuxt", "html",
            "css", "sass", "tailwind", "bootstrap", "jquery", "webpack",
            "vite", "redux", "graphql", "rest api", "pwa", "webassembly",
        },
        "web_backend": {
            "node.js", "express", "django", "flask", "fastapi", "spring boot",
            "rails", "laravel", "asp.net", "nestjs", "gin", "fiber",
            "phoenix", "actix", "grpc", "microservices",
        },
        "databases": {
            "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
            "oracle", "sql server", "dynamodb", "cassandra", "neo4j",
            "firebase", "supabase", "sqlite", "mariadb", "cockroachdb",
            "snowflake", "bigquery", "redshift", "clickhouse",
        },
        "cloud_devops": {
            "aws", "azure", "gcp", "docker", "kubernetes", "terraform",
            "ansible", "jenkins", "gitlab ci", "github actions", "circleci",
            "prometheus", "grafana", "datadog", "splunk", "elk stack",
            "linux", "nginx", "apache", "serverless", "lambda",
        },
        "data_ml": {
            "machine learning", "deep learning", "nlp", "computer vision",
            "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch",
            "keras", "spark", "hadoop", "kafka", "airflow", "dbt",
            "tableau", "power bi", "looker", "data engineering",
        },
        "security": {
            "cybersecurity", "penetration testing", "siem", "soc",
            "vulnerability assessment", "owasp", "encryption", "iam",
            "zero trust", "compliance", "gdpr", "iso 27001",
        },
        "methodologies": {
            "agile", "scrum", "kanban", "devops", "ci/cd", "tdd", "bdd",
            "microservices", "event-driven", "domain-driven design",
        },
    },

    # ── HEALTHCARE & NURSING ─────────────────────────────────────────────────
    "healthcare": {
        "clinical_skills": {
            "patient care", "clinical assessment", "vital signs",
            "medication administration", "wound care", "iv therapy",
            "phlebotomy", "venipuncture", "catheterization", "cpr",
            "bls", "acls", "pals", "airway management", "triage",
            "patient monitoring", "physical examination", "specimen collection",
        },
        "nursing_roles": {
            "registered nurse", "rn", "licensed practical nurse", "lpn",
            "nurse practitioner", "np", "clinical nurse specialist",
            "charge nurse", "nurse manager", "travel nurse",
            "surgical nurse", "icu nurse", "er nurse", "pediatric nurse",
        },
        "medical_specialties": {
            "emergency medicine", "intensive care", "icu", "oncology",
            "cardiology", "orthopedics", "obstetrics", "pediatrics",
            "geriatrics", "psychiatry", "neurology", "surgery",
            "radiology", "anesthesiology", "palliative care",
        },
        "healthcare_systems": {
            "ehr", "emr", "epic", "cerner", "meditech", "allscripts",
            "hipaa", "icd-10", "cpt codes", "electronic health records",
            "clinical documentation", "medical billing", "coding",
        },
        "certifications_medical": {
            "cna", "rn", "lpn", "np", "pa", "md", "do", "rrt",
            "emt", "paramedic", "phlebotomist", "medical assistant", "ma",
            "radiology technician", "pharmacy technician",
        },
        "soft_clinical": {
            "patient education", "care coordination", "interdisciplinary",
            "infection control", "patient advocacy", "compassionate care",
            "cultural competency", "critical thinking",
        },
    },

    # ── FINANCE & ACCOUNTING ─────────────────────────────────────────────────
    "finance": {
        "accounting": {
            "gaap", "ifrs", "financial reporting", "general ledger",
            "accounts payable", "accounts receivable", "reconciliation",
            "journal entries", "month-end close", "year-end close",
            "tax preparation", "tax compliance", "audit", "internal audit",
            "external audit", "sox compliance", "cost accounting",
            "managerial accounting", "forensic accounting",
        },
        "financial_analysis": {
            "financial modeling", "valuation", "dcf", "lbo",
            "comparable analysis", "financial forecasting", "budgeting",
            "variance analysis", "roi analysis", "kpi", "p&l management",
            "cash flow analysis", "balance sheet", "income statement",
            "investment analysis", "portfolio management",
        },
        "tools_finance": {
            "excel", "bloomberg", "quickbooks", "sap", "oracle financials",
            "sage", "xero", "netsuite", "workday", "hyperion",
            "anaplan", "tableau", "power bi", "sql",
        },
        "certifications_finance": {
            "cpa", "cfa", "cma", "cia", "frm", "cfp", "series 7",
            "series 63", "acca", "caia", "mba",
        },
        "banking_finance": {
            "credit analysis", "underwriting", "risk management",
            "compliance", "kyc", "aml", "trade finance",
            "corporate banking", "investment banking", "retail banking",
            "private equity", "venture capital", "hedge fund",
        },
    },

    # ── MARKETING & DIGITAL ──────────────────────────────────────────────────
    "marketing": {
        "digital_marketing": {
            "seo", "sem", "ppc", "google ads", "facebook ads",
            "instagram ads", "linkedin ads", "tiktok ads", "display advertising",
            "programmatic", "affiliate marketing", "email marketing",
            "marketing automation", "hubspot", "marketo", "mailchimp",
            "a/b testing", "conversion rate optimization", "cro",
        },
        "analytics": {
            "google analytics", "google tag manager", "mixpanel", "amplitude",
            "hotjar", "adobe analytics", "data studio", "looker",
            "tableau", "power bi", "sql", "excel", "kpi reporting",
        },
        "content_brand": {
            "content marketing", "content strategy", "copywriting",
            "brand strategy", "brand identity", "social media management",
            "community management", "influencer marketing", "pr",
            "public relations", "media relations", "press release",
            "editorial calendar", "seo writing", "ux writing",
        },
        "crm_sales_marketing": {
            "salesforce", "hubspot crm", "zoho", "pipedrive", "marketo",
            "customer journey", "lead generation", "demand generation",
            "account-based marketing", "abm", "inbound marketing",
            "customer segmentation", "market research", "persona development",
        },
        "creative_tools": {
            "adobe creative suite", "photoshop", "illustrator", "indesign",
            "canva", "figma", "sketch", "premier pro", "after effects",
            "final cut pro",
        },
    },

    # ── SALES ────────────────────────────────────────────────────────────────
    "sales": {
        "sales_skills": {
            "prospecting", "cold calling", "cold emailing", "outbound sales",
            "inbound sales", "quota attainment", "pipeline management",
            "deal closing", "contract negotiation", "territory management",
            "account management", "key account management", "upselling",
            "cross-selling", "solution selling", "consultative selling",
            "value selling", "spin selling", "challenger sale",
        },
        "sales_tools": {
            "salesforce", "hubspot", "pipedrive", "outreach", "salesloft",
            "linkedin sales navigator", "zoominfo", "apollo",
            "gong", "chorus", "clari", "excel",
        },
        "sales_types": {
            "b2b sales", "b2c sales", "saas sales", "enterprise sales",
            "smb sales", "channel sales", "inside sales", "field sales",
            "retail sales", "wholesale",
        },
    },

    # ── HUMAN RESOURCES ──────────────────────────────────────────────────────
    "hr": {
        "recruitment": {
            "talent acquisition", "recruiting", "sourcing", "headhunting",
            "applicant tracking", "ats", "job posting", "candidate screening",
            "behavioral interviewing", "competency-based interviewing",
            "offer negotiation", "onboarding", "employer branding",
        },
        "hr_operations": {
            "hris", "workday", "bamboohr", "adp", "successfactors",
            "payroll", "benefits administration", "compensation",
            "performance management", "360 feedback", "succession planning",
            "workforce planning", "headcount planning",
        },
        "employee_relations": {
            "employee engagement", "culture building", "diversity equity inclusion",
            "dei", "conflict resolution", "mediation", "disciplinary action",
            "policy development", "labor relations", "employment law",
        },
        "hr_certifications": {
            "phr", "sphr", "shrm-cp", "shrm-scp", "cipd",
        },
        "learning_development": {
            "l&d", "training development", "instructional design",
            "e-learning", "lms", "talent development",
        },
    },

    # ── LEGAL ────────────────────────────────────────────────────────────────
    "legal": {
        "legal_practice": {
            "litigation", "contract drafting", "contract review",
            "legal research", "due diligence", "discovery",
            "deposition", "motion practice", "brief writing",
            "legal writing", "case management", "trial preparation",
            "arbitration", "mediation", "negotiation",
        },
        "legal_areas": {
            "corporate law", "mergers and acquisitions", "m&a",
            "intellectual property", "ip", "trademark", "patent",
            "employment law", "labor law", "real estate law",
            "family law", "criminal law", "immigration law",
            "tax law", "environmental law", "compliance",
        },
        "legal_tools": {
            "westlaw", "lexisnexis", "clio", "mycase", "relativity",
            "ediscovery", "microsoft office", "document management",
        },
        "certifications_legal": {
            "jd", "llb", "llm", "bar admission", "paralegal certificate",
        },
    },

    # ── EDUCATION & TEACHING ─────────────────────────────────────────────────
    "education": {
        "teaching_skills": {
            "curriculum development", "lesson planning", "lesson delivery",
            "classroom management", "differentiated instruction",
            "formative assessment", "summative assessment", "grading",
            "rubric development", "project-based learning", "blended learning",
            "distance learning", "e-learning", "stem education",
            "special education", "iep", "504 plan",
        },
        "education_tools": {
            "google classroom", "canvas", "moodle", "blackboard",
            "zoom", "microsoft teams", "smartboard", "kahoot",
            "nearpod", "seesaw", "powerschool",
        },
        "education_roles": {
            "teacher", "lecturer", "professor", "adjunct",
            "curriculum coordinator", "instructional coach", "tutor",
            "teaching assistant", "principal", "vice principal",
            "dean", "superintendent",
        },
        "certifications_education": {
            "teaching certificate", "teaching credential", "state licensure",
            "tesol", "tefl", "celta", "national board certification",
        },
    },

    # ── DESIGN & CREATIVE ────────────────────────────────────────────────────
    "design": {
        "ux_ui": {
            "ux design", "ui design", "user research", "usability testing",
            "wireframing", "prototyping", "information architecture",
            "interaction design", "accessibility", "wcag",
            "user journey mapping", "persona creation", "design thinking",
            "a/b testing", "heuristic evaluation",
        },
        "design_tools": {
            "figma", "sketch", "adobe xd", "invision", "zeplin",
            "photoshop", "illustrator", "indesign", "after effects",
            "premiere pro", "blender", "cinema 4d", "canva",
            "principle", "framer", "webflow",
        },
        "graphic_design": {
            "brand identity", "logo design", "typography", "color theory",
            "print design", "packaging design", "illustration",
            "motion graphics", "video editing", "3d modeling",
        },
        "design_skills": {
            "visual design", "responsive design", "design systems",
            "atomic design", "style guide", "design tokens",
            "grid systems", "whitespace", "visual hierarchy",
        },
    },

    # ── ENGINEERING (NON-SOFTWARE) ───────────────────────────────────────────
    "engineering": {
        "engineering_tools": {
            "autocad", "solidworks", "catia", "ansys", "matlab",
            "labview", "pro/engineer", "inventor", "revit",
            "bim", "civil 3d", "staad pro", "etabs", "sap2000",
        },
        "engineering_skills": {
            "project management", "technical drawing", "fea", "cfd",
            "quality control", "root cause analysis", "fmea",
            "six sigma", "lean manufacturing", "kaizen",
            "iso 9001", "gd&t", "tolerance analysis",
        },
        "engineering_certifications": {
            "pe", "professional engineer", "pmp", "six sigma black belt",
            "six sigma green belt", "iso 9001 auditor",
        },
        "engineering_disciplines": {
            "mechanical engineering", "civil engineering",
            "electrical engineering", "chemical engineering",
            "structural engineering", "industrial engineering",
            "environmental engineering", "biomedical engineering",
        },
    },

    # ── PROJECT MANAGEMENT ───────────────────────────────────────────────────
    "project_management": {
        "pm_methodologies": {
            "agile", "scrum", "kanban", "waterfall", "prince2",
            "pmi", "pmp", "safe", "lean", "six sigma",
            "critical path method", "earned value management",
        },
        "pm_tools": {
            "jira", "asana", "monday.com", "trello", "microsoft project",
            "smartsheet", "basecamp", "notion", "confluence", "miro",
            "slack", "teams",
        },
        "pm_skills": {
            "stakeholder management", "risk management", "budget management",
            "resource allocation", "project planning", "requirements gathering",
            "scope management", "change management", "vendor management",
            "reporting", "executive communication",
        },
    },

    # ── CUSTOMER SERVICE ─────────────────────────────────────────────────────
    "customer_service": {
        "cs_skills": {
            "customer support", "technical support", "help desk",
            "ticketing systems", "zendesk", "freshdesk", "servicenow",
            "intercom", "salesforce service cloud", "live chat",
            "phone support", "email support", "escalation management",
            "customer satisfaction", "csat", "nps", "customer retention",
        },
        "cs_soft_skills": {
            "active listening", "empathy", "patience", "conflict resolution",
            "communication", "problem solving", "time management",
        },
    },

    # ── SUPPLY CHAIN & LOGISTICS ─────────────────────────────────────────────
    "supply_chain": {
        "logistics_skills": {
            "supply chain management", "inventory management",
            "procurement", "purchasing", "vendor management",
            "warehousing", "distribution", "fulfillment",
            "demand planning", "forecasting", "logistics coordination",
            "freight management", "import export", "customs compliance",
        },
        "logistics_tools": {
            "sap", "oracle scm", "erp", "warehouse management system",
            "wms", "transportation management system", "tms",
            "excel", "tableau", "sql",
        },
        "certifications_supply_chain": {
            "cpim", "cscp", "cltd", "apics", "six sigma",
        },
    },

    # ── CONSTRUCTION & TRADES ────────────────────────────────────────────────
    "construction": {
        "construction_skills": {
            "project management", "blueprint reading", "cost estimation",
            "scheduling", "quality control", "safety management",
            "osha", "site management", "subcontractor management",
            "procurement", "value engineering", "punch list",
        },
        "construction_tools": {
            "autocad", "revit", "bluebeam", "procore", "primavera",
            "ms project", "planswift", "sage 300",
        },
        "trades": {
            "electrical", "plumbing", "hvac", "carpentry", "masonry",
            "welding", "ironwork", "roofing", "painting",
        },
    },

    # ── HOSPITALITY & TOURISM ────────────────────────────────────────────────
    "hospitality": {
        "hospitality_skills": {
            "guest relations", "front desk operations", "housekeeping",
            "food and beverage", "f&b", "event management",
            "banquet management", "revenue management", "yield management",
            "property management", "customer service", "concierge",
        },
        "hospitality_tools": {
            "opera", "micros", "infrasys", "hotelogix", "roommaster",
            "pos systems", "point of sale",
        },
        "certifications_hospitality": {
            "servsafe", "tips certified", "cmp", "certified meeting planner",
        },
    },

    # ── SCIENCE & RESEARCH ───────────────────────────────────────────────────
    "science": {
        "research_skills": {
            "data collection", "data analysis", "statistical analysis",
            "spss", "r", "sas", "literature review", "grant writing",
            "peer review", "lab management", "experimental design",
            "hypothesis testing", "qualitative research", "quantitative research",
        },
        "laboratory": {
            "pcr", "elisa", "western blot", "cell culture", "microscopy",
            "chromatography", "spectroscopy", "flow cytometry",
            "gel electrophoresis", "mass spectrometry", "gmp",
        },
        "science_certifications": {
            "phd", "postdoc", "md", "ms", "bs",
        },
    },

    # ── GOVERNMENT & NON-PROFIT ──────────────────────────────────────────────
    "government": {
        "gov_skills": {
            "policy analysis", "policy development", "regulatory affairs",
            "grant management", "grant writing", "budget management",
            "program management", "stakeholder engagement",
            "public administration", "compliance", "procurement",
            "government contracting", "federal acquisition regulation", "far",
        },
        "gov_tools": {
            "grants.gov", "sam.gov", "fpds", "tableau", "excel",
            "arcgis", "salesforce",
        },
    },

    # ── UNIVERSAL SOFT SKILLS ────────────────────────────────────────────────
    "universal": {
        "leadership": {
            "leadership", "team leadership", "people management",
            "cross-functional collaboration", "mentoring", "coaching",
            "executive presence", "decision making", "strategic thinking",
            "organizational leadership", "servant leadership",
        },
        "communication": {
            "communication", "presentation skills", "public speaking",
            "written communication", "technical writing", "storytelling",
            "active listening", "stakeholder communication",
            "executive communication", "facilitation",
        },
        "problem_solving": {
            "problem solving", "critical thinking", "analytical thinking",
            "root cause analysis", "data-driven decision making",
            "innovation", "creativity", "design thinking",
        },
        "project_org": {
            "time management", "multitasking", "prioritization",
            "attention to detail", "deadline management",
            "self-starter", "adaptability", "resilience",
        },
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# SYNONYM MAP  (canonical → set of variants)
# ─────────────────────────────────────────────────────────────────────────────

SKILL_SYNONYMS: Dict[str, Set[str]] = {
    # Tech
    "javascript":       {"js", "ecmascript", "es6", "es2015"},
    "typescript":       {"ts"},
    "python":           {"python3", "python 3", "py"},
    "c#":               {"csharp", "c sharp", "dotnet", ".net"},
    "c++":              {"cpp", "c plus plus"},
    "node.js":          {"node", "nodejs"},
    "react":            {"reactjs", "react.js"},
    "vue":              {"vuejs", "vue.js"},
    "angular":          {"angularjs"},
    "next.js":          {"nextjs"},
    "postgresql":       {"postgres"},
    "microsoft sql server": {"mssql", "sql server", "ms sql"},
    "gcp":              {"google cloud", "google cloud platform"},
    "aws":              {"amazon web services"},
    "azure":            {"microsoft azure"},
    "kubernetes":       {"k8s"},
    "machine learning": {"ml"},
    "artificial intelligence": {"ai"},
    "natural language processing": {"nlp"},
    "computer vision":  {"cv"},
    "ci/cd":            {"continuous integration", "continuous deployment"},

    # Healthcare
    "registered nurse": {"rn"},
    "licensed practical nurse": {"lpn"},
    "nurse practitioner": {"np", "aprn"},
    "electronic health records": {"ehr", "emr"},
    "basic life support":  {"bls"},
    "advanced cardiac life support": {"acls"},
    "cardiopulmonary resuscitation": {"cpr"},
    "emergency medical technician": {"emt"},
    "certified nursing assistant": {"cna"},

    # Finance
    "certified public accountant": {"cpa"},
    "chartered financial analyst": {"cfa"},
    "generally accepted accounting principles": {"gaap"},
    "international financial reporting standards": {"ifrs"},
    "mergers and acquisitions": {"m&a"},
    "profit and loss": {"p&l"},
    "return on investment": {"roi"},
    "key performance indicators": {"kpi", "kpis"},

    # Marketing
    "search engine optimization": {"seo"},
    "search engine marketing": {"sem"},
    "pay per click": {"ppc"},
    "customer relationship management": {"crm"},
    "conversion rate optimization": {"cro"},
    "account-based marketing": {"abm"},

    # Legal
    "juris doctor": {"jd"},
    "intellectual property": {"ip"},
    "mergers and acquisitions": {"m&a"},

    # HR
    "human resources information system": {"hris"},
    "diversity equity inclusion": {"dei"},
    "learning and development": {"l&d"},
    "society for human resource management": {"shrm"},

    # PM
    "project management professional": {"pmp"},
    "program evaluation and review technique": {"pert"},

    # Supply chain
    "enterprise resource planning": {"erp"},
    "warehouse management system": {"wms"},
    "transportation management system": {"tms"},

    # Government
    "federal acquisition regulation": {"far"},

    # Universal
    "profit and loss":  {"p&l"},
}

# ─────────────────────────────────────────────────────────────────────────────
# INDUSTRY DETECTION SIGNALS
# ─────────────────────────────────────────────────────────────────────────────

INDUSTRY_SIGNALS: Dict[str, List[str]] = {
    "technology":    ["python", "java", "developer", "engineer", "software", "api", "database", "cloud"],
    "healthcare":    ["patient", "clinical", "nurse", "doctor", "hospital", "medical", "rn", "ehr", "hipaa"],
    "finance":       ["accounting", "audit", "gaap", "cpa", "financial", "tax", "budget", "investment"],
    "marketing":     ["seo", "campaign", "marketing", "brand", "digital", "content", "analytics"],
    "sales":         ["quota", "pipeline", "sales", "revenue", "account executive", "prospecting", "crm"],
    "hr":            ["recruitment", "talent", "onboarding", "hris", "payroll", "hr business partner"],
    "legal":         ["litigation", "attorney", "lawyer", "contract", "compliance", "counsel", "paralegal"],
    "education":     ["teaching", "curriculum", "classroom", "students", "school", "educator", "professor"],
    "design":        ["ux", "ui", "figma", "sketch", "typography", "wireframe", "visual design"],
    "engineering":   ["autocad", "solidworks", "mechanical", "civil", "electrical", "structural", "pe"],
    "supply_chain":  ["procurement", "inventory", "logistics", "warehousing", "supply chain", "forecasting"],
    "construction":  ["construction", "site management", "osha", "contractor", "blueprint", "procore"],
    "hospitality":   ["hotel", "restaurant", "guest", "front desk", "hospitality", "food service"],
    "science":       ["research", "laboratory", "phd", "clinical trial", "pcr", "data analysis"],
    "government":    ["policy", "government", "federal", "grant", "public sector", "regulatory"],
}

# Where to suggest adding keywords
SECTION_PLACEMENT_HINTS: Dict[str, str] = {
    "tool":         "skills",
    "language":     "skills",
    "certification":"certifications",
    "soft_skill":   "summary or experience bullets",
    "methodology":  "skills or experience bullets",
    "achievement":  "experience bullets",
    "role":         "job title or summary",
    "industry_keyword": "summary or experience bullets",
}

# JD criticality markers
CRITICALITY_MARKERS = {
    "must have":        2.0,
    "required":         2.0,
    "essential":        1.8,
    "mandatory":        1.8,
    "must":             1.6,
    "key requirement":  1.6,
    "key":              1.5,
    "preferred":        1.0,
    "nice to have":     0.6,
    "bonus":            0.5,
    "desirable":        0.6,
    "plus":             0.5,
}


# ─────────────────────────────────────────────────────────────────────────────
# KEYWORD ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class KeywordEngine:
    """Universal keyword extraction and matching for all industries."""

    def __init__(self) -> None:
        # Flat skill lookup: skill_text → (industry, category)
        self._skill_db: Dict[str, Tuple[str, str]] = {}
        # Reverse synonym lookup: variant → canonical
        self._synonym_reverse: Dict[str, str] = {}
        self._build_indexes()

    # ── INDEX BUILD ──────────────────────────────────────────────────────────

    def _build_indexes(self) -> None:
        for industry, categories in UNIVERSAL_SKILLS.items():
            for category, skills in categories.items():
                for skill in skills:
                    self._skill_db[skill.lower()] = (industry, category)

        for canonical, variants in SKILL_SYNONYMS.items():
            c_lower = canonical.lower()
            for v in variants:
                self._synonym_reverse[v.lower()] = c_lower

    # ── INDUSTRY DETECTION ───────────────────────────────────────────────────

    def detect_industry(self, resume: Dict) -> str:
        """Detect dominant industry from resume content."""
        text = self._resume_to_text(resume).lower()
        scores: Dict[str, int] = {}
        for industry, signals in INDUSTRY_SIGNALS.items():
            scores[industry] = sum(1 for s in signals if s in text)
        if not any(scores.values()):
            return "general"
        return max(scores, key=lambda k: scores[k])

    # ── JD PARSING ───────────────────────────────────────────────────────────

    def extract_jd_keywords(self, jd: str) -> Dict:
        """Extract structured keywords from job description."""
        if not jd:
            return {"required": [], "preferred": [], "all": [], "frequency": {}, "criticality": {}}

        jd_lower = jd.lower()
        all_found = self._find_skills_in_text(jd_lower)
        frequency = Counter(all_found)

        required = self._extract_by_context(jd, ["must have", "required", "essential", "mandatory"])
        preferred = self._extract_by_context(jd, ["preferred", "nice to have", "bonus", "desirable"])
        criticality = self._score_criticality(jd_lower, list(frequency.keys()))

        return {
            "required":    list(set(required)),
            "preferred":   list(set(preferred)),
            "all":         list(frequency.keys()),
            "frequency":   dict(frequency),
            "criticality": criticality,
        }

    def _extract_by_context(self, text: str, markers: List[str]) -> List[str]:
        skills: List[str] = []
        for marker in markers:
            # Grab up to 200 chars after marker
            for m in re.finditer(re.escape(marker), text, re.IGNORECASE):
                snippet = text[m.end(): m.end() + 200]
                skills.extend(self._find_skills_in_text(snippet.lower()))
        return skills

    def _score_criticality(self, jd_lower: str, skills: List[str]) -> Dict[str, float]:
        criticality: Dict[str, float] = {}
        for skill in skills:
            score = 1.0
            for marker, weight in CRITICALITY_MARKERS.items():
                pattern = rf"{re.escape(marker)}[^.!?\n]{{0,120}}\b{re.escape(skill)}\b"
                if re.search(pattern, jd_lower):
                    score = max(score, weight)
            # Boost for repetition
            count = jd_lower.count(skill)
            if count >= 4:
                score = min(score * 1.5, 2.0)
            elif count >= 2:
                score = min(score * 1.2, 2.0)
            criticality[skill] = round(score, 2)
        return criticality

    # ── RESUME PARSING ───────────────────────────────────────────────────────

    def extract_resume_skills(self, resume: Dict) -> List[str]:
        """Extract all skills mentioned anywhere in the resume."""
        skills: Set[str] = set()

        # Explicit skills section
        for s in (resume.get("skills") or []):
            normalized = str(s).lower().strip()
            skills.add(normalized)
            # Also add canonical form if this is a synonym
            canonical = self._synonym_reverse.get(normalized)
            if canonical:
                skills.add(canonical)

        # Other sections
        sections_to_scan = ["summary", "experience", "projects",
                            "certifications", "education", "publications"]
        for section in sections_to_scan:
            content = resume.get(section)
            if not content:
                continue
            text = self._section_to_text(content).lower()
            found = self._find_skills_in_text(text)
            skills.update(found)

        return list(skills)

    # ── MATCHING ─────────────────────────────────────────────────────────────

    def match_skills(self, resume: Dict, job_description: str) -> KeywordAnalysis:
        """Full keyword match between resume and JD."""
        if not job_description:
            detected = self.detect_industry(resume)
            return KeywordAnalysis(
                total_jd_keywords=0, matched_keywords=0, match_percentage=0,
                matched_skills=[], missing_critical_skills=[], missing_preferred_skills=[],
                found_strengths=[], keyword_density=0.0,
                detected_industry=detected, ats_keyword_gaps=[], keyword_suggestions=[],
            )

        jd_data = self.extract_jd_keywords(job_description)
        resume_skills = set(self.extract_resume_skills(resume))
        detected_industry = self.detect_industry(resume)

        matched_skills: List[SkillMatch] = []
        matched_set: Set[str] = set()
        missing_critical: List[str] = []
        missing_preferred: List[str] = []
        ats_gaps: List[Dict] = []
        suggestions: List[Dict] = []

        for jd_skill in jd_data["all"]:
            canonical = self._synonym_reverse.get(jd_skill, jd_skill)
            found, variant = self._check_match(canonical, jd_skill, resume_skills)
            industry, category = self._skill_db.get(canonical, ("general", "general"))
            criticality = jd_data["criticality"].get(jd_skill, 1.0)
            freq = jd_data["frequency"].get(jd_skill, 1)

            if found:
                matched_set.add(jd_skill)
                matched_skills.append(SkillMatch(
                    skill=jd_skill, found_in_resume=True,
                    frequency_in_jd=freq, matched_variants=[variant or jd_skill],
                    confidence=0.95 if variant == jd_skill else 0.80,
                    category=category, industry=industry,
                    criticality=criticality,
                    suggested_section="skills",
                ))
            else:
                # Build gap entry
                gap = {
                    "keyword": jd_skill,
                    "criticality": criticality,
                    "frequency_in_jd": freq,
                    "industry": industry,
                    "category": category,
                    "is_required": jd_skill in jd_data["required"],
                    "suggested_section": self._suggest_section(category, jd_skill),
                    "how_to_add": self._generate_add_tip(jd_skill, category),
                }
                ats_gaps.append(gap)
                suggestions.append({
                    "keyword": jd_skill,
                    "add_to": self._suggest_section(category, jd_skill),
                    "example": self._generate_add_tip(jd_skill, category),
                })

                if jd_skill in jd_data["required"] or criticality >= 1.8:
                    missing_critical.append(jd_skill)
                elif jd_skill in jd_data["preferred"] or criticality >= 0.9:
                    missing_preferred.append(jd_skill)

        # Sort gaps by criticality desc
        ats_gaps.sort(key=lambda x: x["criticality"], reverse=True)

        total = len(jd_data["all"])
        matched_count = len(matched_set)
        match_pct = int((matched_count / total) * 100) if total > 0 else 0

        # Keyword density
        total_words = self._count_resume_words(resume)
        skill_count = len(resume_skills)
        density = min(skill_count / total_words, 1.0) if total_words > 0 else 0.0

        strengths = [s.skill for s in matched_skills
                     if s.criticality >= 1.5][:8]

        return KeywordAnalysis(
            total_jd_keywords=total,
            matched_keywords=matched_count,
            match_percentage=match_pct,
            matched_skills=matched_skills,
            missing_critical_skills=missing_critical,
            missing_preferred_skills=missing_preferred,
            found_strengths=strengths,
            keyword_density=round(density, 3),
            detected_industry=detected_industry,
            ats_keyword_gaps=ats_gaps,
            keyword_suggestions=suggestions,
        )

    def calculate_keyword_score(self, analysis: KeywordAnalysis) -> int:
        """Weighted keyword score 0-100."""
        if analysis.total_jd_keywords == 0:
            return 100  # No JD → perfect by default

        base = analysis.match_percentage

        # Penalise for missing critical keywords
        critical_penalty = len(analysis.missing_critical_skills) * 8
        base = max(base - critical_penalty, 0)

        # Small bonus for healthy keyword density
        if analysis.keyword_density > 0.07:
            base = min(base + 5, 100)

        # Bonus for hitting all required
        if not analysis.missing_critical_skills:
            base = min(base + 10, 100)

        return max(int(base), 0)

    # ── HELPERS ──────────────────────────────────────────────────────────────

    def _find_skills_in_text(self, text: str) -> List[str]:
        found: List[str] = []
        # Direct database match
        for skill in self._skill_db:
            if re.search(rf"\b{re.escape(skill)}\b", text):
                found.append(skill)
        # Synonym match
        for variant, canonical in self._synonym_reverse.items():
            if re.search(rf"\b{re.escape(variant)}\b", text):
                if canonical not in found:
                    found.append(canonical)
        return found

    def _check_match(self, canonical: str, original: str,
                     resume_skills: Set[str]) -> Tuple[bool, Optional[str]]:
        # Direct
        if canonical in resume_skills or original in resume_skills:
            return True, original
        # Synonym variants
        for variant in SKILL_SYNONYMS.get(canonical, set()):
            if variant.lower() in resume_skills:
                return True, variant
        # Partial (substring, min 4 chars)
        if len(canonical) >= 4:
            for rs in resume_skills:
                if canonical in rs or rs in canonical:
                    return True, rs
        return False, None

    def _suggest_section(self, category: str, skill: str) -> str:
        if any(x in category for x in ["certification", "license"]):
            return "certifications"
        if any(x in category for x in ["tool", "language", "database", "platform", "framework"]):
            return "skills"
        if any(x in category for x in ["soft", "leadership", "communication"]):
            return "summary or experience bullets"
        return "skills or experience bullets"

    def _generate_add_tip(self, skill: str, category: str) -> str:
        if "tool" in category or "language" in category:
            return f'Add "{skill}" to your Skills section directly.'
        if "certification" in category:
            return f'Add "{skill}" to a Certifications section with date obtained.'
        if "soft" in category or "leadership" in category:
            return f'Weave "{skill}" into a summary statement or experience bullet (e.g., "Led cross-functional team demonstrating {skill}").'
        return f'Include "{skill}" in a relevant bullet point under Work Experience or Projects.'

    def _section_to_text(self, content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(" ".join(str(v) for v in item.values() if v))
                else:
                    parts.append(str(item))
            return " ".join(parts)
        if isinstance(content, dict):
            return " ".join(str(v) for v in content.values() if v)
        return str(content)

    def _resume_to_text(self, resume: Dict) -> str:
        parts: List[str] = []
        for key in ["summary", "experience", "skills", "education",
                    "projects", "certifications", "awards", "publications"]:
            val = resume.get(key)
            if val:
                parts.append(self._section_to_text(val))
        return " ".join(parts)

    def _count_resume_words(self, resume: Dict) -> int:
        return len(self._resume_to_text(resume).split())