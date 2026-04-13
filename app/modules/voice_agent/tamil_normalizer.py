import re

ABBR_MAP = {
    "BCA": "பி.சி.ஏ",
    "MBA": "எம்.பி.ஏ",
    "MCA": "எம்.சி.ஏ",
    "BSC": "பி.எஸ்.சி",
    "BE": "பி.இ",
    "BTech": "பி.டெக்",
    "MTech": "எம்.டெக்",
    "BA": "பி.ஏ",
    "MA": "எம்.ஏ",
    "HR": "எச்.ஆர்",
    "IT": "ஐ.டி",
    "BPO": "பி.பி.ஓ",
    "MNC": "எம்.என்.சி",
    "LPA": "எல்.பி.ஏ",
    "CTC": "சி.டி.சி",
    "WFH": "வொர்க் ஃப்ரம் ஹோம்",
    "WFO": "வொர்க் ஃப்ரம் ஆபீஸ்",
    "PM": "மாலை",
    "AM": "காலை",
}

ONES_TA = [
    "", "ஒன்று", "இரண்டு", "மூன்று", "நான்கு", "ஐந்து",
    "ஆறு", "ஏழு", "எட்டு", "ஒன்பது", "பத்து",
    "பதினொன்று", "பன்னிரண்டு", "பதிமூன்று", "பதினான்கு", "பதினைந்து",
    "பதினாறு", "பதினேழு", "பதினெட்டு", "பத்தொன்பது",
]

TENS_TA = [
    "", "", "இருபது", "முப்பது", "நாற்பது", "ஐம்பது",
    "அறுபது", "எழுபது", "எண்பது", "தொண்ணூறு",
]


def _int_to_tamil(n: int) -> str:
    if n < 0:
        return "கழித்தல் " + _int_to_tamil(-n)
    if n == 0:
        return "பூஜ்யம்"
    if n < 20:
        return ONES_TA[n]
    if n < 100:
        return (TENS_TA[n // 10] + " " + ONES_TA[n % 10]).strip()
    if n < 1000:
        rest = _int_to_tamil(n % 100) if n % 100 != 0 else ""
        return (ONES_TA[n // 100] + " நூறு " + rest).strip()
    if n < 100000:
        rest = _int_to_tamil(n % 1000) if n % 1000 != 0 else ""
        return (_int_to_tamil(n // 1000) + " ஆயிரம் " + rest).strip()
    if n < 10000000:
        rest = _int_to_tamil(n % 100000) if n % 100000 != 0 else ""
        return (_int_to_tamil(n // 100000) + " லட்சம் " + rest).strip()
    rest = _int_to_tamil(n % 10000000) if n % 10000000 != 0 else ""
    return (_int_to_tamil(n // 10000000) + " கோடி " + rest).strip()


def _replace_number(match: re.Match) -> str:
    try:
        return _int_to_tamil(int(match.group()))
    except Exception:
        return match.group()


def normalize(text: str) -> str:
    for abbr, expansion in ABBR_MAP.items():
        text = re.sub(rf'\b{re.escape(abbr)}\b', expansion, text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d{1,8}\b', _replace_number, text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text