CONTACT_KEYWORDS = [
    "contact",
    "call",
    "phone",
    "number",
    "whatsapp",
    "email",
    "mail",
    "reach",
    "talk",
    "support",
]

def detect_contact_intent(message: str) -> bool:
    msg = message.lower()
    return any(k in msg for k in CONTACT_KEYWORDS)

def detect_specific_contact(message: str) -> str:
    msg = message.lower()
    if "whatsapp" in msg:
        return "whatsapp"
    if "email" in msg or "mail" in msg:
        return "email"
    if "call" in msg or "phone" in msg:
        return "call"
    return "general"
