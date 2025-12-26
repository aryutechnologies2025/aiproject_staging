COURSE_KEYWORDS = [
    "frontend",
    "front end",
    "ui",
    "ui ux",
    "react",
    "react js",
    "full stack",
    "mern",
    "python fullstack",
    "php",
    "laravel",
    "software testing",
    "manual testing",
    "wordpress",
    "webflow",
    "shopify",
]

BUSINESS_KEYWORDS = [
    "course",
    "training",
    "classes",
    "fee",
    "fees",
    "pricing",
    "duration",
    "syllabus",
    "demo",
    "enroll",
    "admission",
    "join",
    "batch",
]

def detect_lead_intent(message: str) -> bool:
    msg = message.lower()

    has_course = any(k in msg for k in COURSE_KEYWORDS)
    has_business = any(k in msg for k in BUSINESS_KEYWORDS)

    return has_course and has_business
