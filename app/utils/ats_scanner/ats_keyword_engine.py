# /home/aryu_user/Arun/aiproject_staging/app/utils/ats_keyword_engine.py
"""
Advanced Keyword Extraction and Semantic Matching
Implements smart skill extraction, job description parsing, and relevance scoring
"""

import re
from typing import Dict, List, Set, Tuple
from collections import Counter
from dataclasses import dataclass
import json

# =====================================================
# DATA CLASSES
# =====================================================

@dataclass
class SkillMatch:
    """Represents a skill match between resume and job description"""
    skill: str
    found_in_resume: bool
    frequency_in_jd: int
    matched_variants: List[str]
    confidence: float  # 0-1
    category: str  # 'hard_skill', 'soft_skill', 'tool', 'platform'


@dataclass
class KeywordAnalysis:
    """Complete keyword analysis results"""
    total_jd_keywords: int
    matched_keywords: int
    match_percentage: int
    matched_skills: List[SkillMatch]
    missing_critical_skills: List[str]
    found_strengths: List[str]
    keyword_density: float  # 0-1


# =====================================================
# SKILL CATEGORIES & DATABASES
# =====================================================

SKILL_CATEGORIES = {
    # Programming Languages
    "programming_languages": {
        "python", "java", "javascript", "typescript", "csharp", "c#", "c++",
        "ruby", "php", "swift", "kotlin", "go", "rust", "scala", "r",
        "perl", "elixir", "clojure", "haskell"
    },
    
    # Web Frameworks
    "web_frameworks": {
        "react", "angular", "vue", "django", "flask", "fastapi", "spring",
        "rails", "laravel", "aspnet", "node", "express", "nest", "next",
        "nuxt", "svelte", "ember", "backbone"
    },
    
    # Databases
    "databases": {
        "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
        "oracle", "mssql", "dynamodb", "cassandra", "neo4j", "firebase",
        "supabase", "cockroachdb", "mariadb", "sqlite"
    },
    
    # Cloud Platforms
    "cloud_platforms": {
        "aws", "azure", "gcp", "google cloud", "heroku", "digitalocean",
        "linode", "vultr", "aws ec2", "aws lambda", "azure vm"
    },
    
    # DevOps & Tools
    "devops_tools": {
        "docker", "kubernetes", "jenkins", "gitlab", "github", "git",
        "terraform", "ansible", "prometheus", "grafana", "elk", "datadog",
        "new relic", "splunk", "circleci", "travis"
    },
    
    # Data & Analytics
    "data_tools": {
        "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch",
        "spark", "hadoop", "kafka", "tableau", "powerbi", "looker",
        "sql", "dbt", "airflow", "snowflake"
    },
    
    # APIs & Protocols
    "api_tools": {
        "rest", "graphql", "grpc", "soap", "mqtt", "amqp",
        "openapi", "swagger", "postman"
    },
    
    # Message Queues
    "message_queues": {
        "kafka", "rabbitmq", "redis", "activemq", "zeromq", "nats"
    },
    
    # Soft Skills
    "soft_skills": {
        "communication", "leadership", "teamwork", "problem solving",
        "project management", "agile", "scrum", "kanban", "critical thinking",
        "time management", "collaboration", "mentoring", "stakeholder management"
    },
    
    # Methodologies
    "methodologies": {
        "agile", "scrum", "kanban", "waterfall", "ci/cd", "tdd",
        "bdd", "microservices", "soa", "event-driven"
    }
}

# Skill variants and synonyms
SKILL_SYNONYMS = {
    "csharp": {"c#", "c sharp"},
    "cpp": {"c++", "cplusplus"},
    "js": {"javascript", "js"},
    "ts": {"typescript", "ts"},
    "python3": {"python 3", "python3", "python"},
    "nodejs": {"node.js", "node", "nodejs"},
    "expressjs": {"express.js", "express", "expressjs"},
    "dotnet": {"dot net", ".net", "dotnet", "asp.net"},
    "aspnet": {"asp.net", "aspnet", "asp .net"},
    "mssql": {"mssqlserver", "sql server", "ms sql", "mssql"},
    "gcp": {"google cloud", "google cloud platform"},
    "ml": {"machine learning", "ml"},
    "nlp": {"natural language processing", "nlp"},
    "llm": {"large language model", "llm"},
    "ai": {"artificial intelligence", "ai"},
    "cv": {"computer vision", "cv"},
    "db": {"database", "db"},
    "api": {"application programming interface", "api"},
    "qa": {"quality assurance", "qa"},
    "cicd": {"ci/cd", "continuous integration", "continuous deployment"},
    "pb": {"product breakdown"},
    "jira": {"jira", "atlassian jira"}
}

CRITICAL_SKILL_INDICATORS = {
    "must have": 2.0,
    "required": 2.0,
    "essential": 1.8,
    "key": 1.5,
    "preferred": 1.0,
    "nice to have": 0.5
}


# =====================================================
# MAIN KEYWORD ENGINE
# =====================================================

class KeywordEngine:
    """Advanced keyword extraction and matching"""
    
    def __init__(self):
        self.all_skills = self._build_skill_database()
    
    def _build_skill_database(self) -> Dict[str, str]:
        """Build comprehensive skill database with categories"""
        skill_db = {}
        for category, skills in SKILL_CATEGORIES.items():
            for skill in skills:
                skill_db[skill.lower()] = category
        return skill_db
    
    # =========== JOB DESCRIPTION PARSING ===========
    
    def extract_job_keywords(self, job_description: str) -> Dict:
        """
        Extract and prioritize keywords from job description
        """
        if not job_description:
            return {
                "required_skills": [],
                "preferred_skills": [],
                "all_skills": [],
                "skill_frequency": {},
                "criticality_score": {}
            }
        
        jd_lower = job_description.lower()
        
        # Extract required vs preferred
        required = self._extract_critical_skills(job_description, "required")
        preferred = self._extract_critical_skills(job_description, "preferred")
        
        # Find all skill mentions
        all_mentioned = self._find_all_skills(jd_lower)
        
        # Calculate frequency
        skill_frequency = Counter(all_mentioned)
        
        # Score criticality
        criticality = self._score_criticality(job_description, all_mentioned)
        
        return {
            "required_skills": required,
            "preferred_skills": preferred,
            "all_skills": list(skill_frequency.keys()),
            "skill_frequency": dict(skill_frequency),
            "criticality_score": criticality
        }
    
    def _extract_critical_skills(self, text: str, level: str) -> List[str]:
        """Extract skills marked as required/preferred"""
        patterns = {
            "required": [
                r"required[:\s]+([^.\n]+)",
                r"must have[:\s]+([^.\n]+)",
                r"essential[:\s]+([^.\n]+)"
            ],
            "preferred": [
                r"preferred[:\s]+([^.\n]+)",
                r"nice to have[:\s]+([^.\n]+)",
                r"bonus[:\s]+([^.\n]+)"
            ]
        }
        
        skills = set()
        for pattern in patterns.get(level, []):
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                extracted = self._parse_skill_list(match)
                skills.update(extracted)
        
        return list(skills)
    
    def _find_all_skills(self, text: str) -> List[str]:
        """Find all skill mentions in text"""
        found_skills = []
        text_lower = text.lower()
        
        for skill, category in self.all_skills.items():
            if skill in text_lower:
                found_skills.append(skill)
                
                # Check for variants
                if skill in SKILL_SYNONYMS:
                    for variant in SKILL_SYNONYMS[skill]:
                        if variant.lower() in text_lower:
                            found_skills.append(variant.lower())
        
        return list(set(found_skills))
    
    def _parse_skill_list(self, text: str) -> List[str]:
        """Parse comma/slash separated skill list"""
        separators = [",", "/", "and", "or", ";"]
        skills = [text]
        
        for sep in separators:
            new_skills = []
            for skill in skills:
                new_skills.extend(skill.split(sep))
            skills = new_skills
        
        return [s.strip().lower() for s in skills if s.strip()]
    
    def _score_criticality(self, job_desc: str, skills: List[str]) -> Dict[str, float]:
        """Score how critical each skill is based on context"""
        criticality = {}
        jd_lower = job_desc.lower()
        
        for skill in set(skills):
            score = 1.0
            
            # Check surrounding context
            for indicator, multiplier in CRITICAL_SKILL_INDICATORS.items():
                pattern = rf"{indicator}[^.]*\b{skill}\b"
                if re.search(pattern, jd_lower, re.IGNORECASE):
                    score = multiplier
                    break
            
            # Boost score based on repetition
            count = jd_lower.count(skill.lower())
            if count > 3:
                score *= 1.5
            elif count > 1:
                score *= 1.2
            
            criticality[skill] = min(score, 2.0)  # Cap at 2.0
        
        return criticality
    
    # =========== RESUME ANALYSIS ===========
    
    def extract_resume_skills(self, resume: Dict) -> List[str]:
        """Extract all skills from resume"""
        skills = set()
        
        # From skills section
        if "skills" in resume:
            skills.update([s.lower() for s in resume["skills"]])
        
        # From experience bullets (skill mentions)
        if "experience" in resume:
            for exp in resume["experience"]:
                for bullet in exp.get("bullets", []):
                    found = self._find_all_skills(bullet.lower())
                    skills.update(found)
        
        # From education
        if "education" in resume:
            for edu in resume["education"]:
                text = " ".join([
                    edu.get("degree", ""),
                    " ".join(edu.get("educationDescription", []))
                ])
                found = self._find_all_skills(text.lower())
                skills.update(found)
        
        return list(skills)
    
    # =========== MATCHING & ANALYSIS ===========
    
    def match_skills(self, resume: Dict, job_description: str) -> KeywordAnalysis:
        """
        Match resume skills against job description
        """
        if not job_description:
            return KeywordAnalysis(
                total_jd_keywords=0,
                matched_keywords=0,
                match_percentage=0,
                matched_skills=[],
                missing_critical_skills=[],
                found_strengths=[],
                keyword_density=0.0
            )
        
        # Extract keywords
        jd_keywords = self.extract_job_keywords(job_description)
        resume_skills = self.extract_resume_skills(resume)
        
        # Match skills
        matched_skills = []
        matched_count = 0
        
        all_jd_skills = jd_keywords["all_skills"]
        resume_lower = set([s.lower() for s in resume_skills])
        
        for jd_skill in all_jd_skills:
            # Direct match
            if jd_skill in resume_lower:
                matched_count += 1
                matched_skills.append(SkillMatch(
                    skill=jd_skill,
                    found_in_resume=True,
                    frequency_in_jd=jd_keywords["skill_frequency"].get(jd_skill, 1),
                    matched_variants=[jd_skill],
                    confidence=0.95,
                    category=self.all_skills.get(jd_skill, "unknown")
                ))
            else:
                # Check for variants
                matched_variant = self._find_variant_match(jd_skill, resume_lower)
                if matched_variant:
                    matched_count += 1
                    matched_skills.append(SkillMatch(
                        skill=jd_skill,
                        found_in_resume=True,
                        frequency_in_jd=jd_keywords["skill_frequency"].get(jd_skill, 1),
                        matched_variants=[matched_variant],
                        confidence=0.80,
                        category=self.all_skills.get(jd_skill, "unknown")
                    ))
        
        # Find missing critical skills
        missing_critical = []
        for skill in jd_keywords["required_skills"]:
            if skill not in resume_lower:
                if not self._find_variant_match(skill, resume_lower):
                    missing_critical.append(skill)
        
        # Calculate metrics
        total_jd = len(all_jd_skills)
        match_percentage = int((matched_count / total_jd) * 100) if total_jd > 0 else 0
        
        # Calculate keyword density
        total_words = self._count_resume_words(resume)
        skill_mentions = len(resume_skills)
        keyword_density = skill_mentions / total_words if total_words > 0 else 0
        
        return KeywordAnalysis(
            total_jd_keywords=total_jd,
            matched_keywords=matched_count,
            match_percentage=match_percentage,
            matched_skills=matched_skills,
            missing_critical_skills=missing_critical,
            found_strengths=[s.skill for s in matched_skills[:5]],
            keyword_density=min(keyword_density, 1.0)
        )
    
    def _find_variant_match(self, skill: str, resume_skills: Set[str]) -> str:
        """Check if skill has a variant match in resume"""
        skill_lower = skill.lower()
        
        # Check synonyms
        if skill_lower in SKILL_SYNONYMS:
            for variant in SKILL_SYNONYMS[skill_lower]:
                if variant.lower() in resume_skills:
                    return variant.lower()
        
        # Fuzzy matching for partial matches
        for resume_skill in resume_skills:
            if len(skill_lower) > 3:
                # Check if skill is substring
                if skill_lower in resume_skill or resume_skill in skill_lower:
                    return resume_skill
        
        return None
    
    def _count_resume_words(self, resume: Dict) -> int:
        """Count total words in resume"""
        word_count = 0
        
        for key in ["summary", "skills", "experience", "education"]:
            if key == "skills":
                word_count += len(resume.get(key, []))
            elif key in ["experience", "education"]:
                for item in resume.get(key, []):
                    if isinstance(item, dict):
                        text_parts = []

                        for value in item.values():
                            if isinstance(value, str):
                                text_parts.append(value)
                            elif isinstance(value, list):
                                text_parts.extend([str(v) for v in value])
                            elif value is not None:
                                text_parts.append(str(value))

                        word_count += len(" ".join(text_parts).split())
            else:
                word_count += len(resume.get(key, "").split())
        
        return word_count
    
    # =========== SCORE CALCULATION ===========
    
    def calculate_keyword_score(self, analysis: KeywordAnalysis,
                               jd_criticality: Dict[str, float] = None) -> int:
        """
        Calculate keyword match score (0-100)
        Weighs critical skills more heavily
        """
        if analysis.total_jd_keywords == 0:
            return 100  # No JD provided
        
        # Base score on match percentage
        base_score = analysis.match_percentage
        
        # Bonus for matching critical skills
        if analysis.missing_critical_skills:
            critical_penalty = len(analysis.missing_critical_skills) * 10
            base_score -= min(critical_penalty, base_score)
        else:
            base_score = min(base_score + 10, 100)
        
        # Bonus for high keyword density
        if analysis.keyword_density > 0.08:  # Healthy keyword density
            base_score = min(base_score + 5, 100)
        
        return max(int(base_score), 0)