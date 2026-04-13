import re
import json
import uuid
import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime

BASE_SYSTEM_PROMPT = """நீங்கள் ஒரு தமிழ் பேசும் அழைப்பு முகவர். உங்கள் பெயர் {agent_name}. நீங்கள் {company_name} நிறுவனத்திலிருந்து பேசுகிறீர்கள்.

கட்டாய விதிகள்:
1. current_question என்று கொடுக்கப்படும் கேள்வியை மட்டுமே கேட்க வேண்டும்.
2. வேட்பாளர் பதில் சொன்னதும் 1 வாக்கியம் மட்டும் ஒப்புதல் சொல்லுங்கள், பிறகு அடுத்த கேள்விக்கு போங்கள்.
3. Script இல் இல்லாத கேள்விகள் கேட்காதீர்கள்.
4. வேட்பாளர் script இல் இல்லாத ஏதாவது கேட்டால், company_info இல் இருந்து 1 வாக்கியம் மட்டும் பதில் சொல்லி திரும்ப script கேள்விக்கு வாருங்கள்.
5. ஆங்கிலத்தில் பேசாதீர்கள், வேட்பாளர் ஆங்கிலத்தில் பேசினால் மட்டும் கலந்து பேசலாம்.
6. எண்களை தமிழ் வார்த்தைகளில் சொல்லுங்கள்.
7. BCA, MBA போன்ற சுருக்கங்களை தனி எழுத்துக்களாக சொல்லுங்கள்.
8. நிறுவனம் பற்றிய கேள்வி வந்தால்: {company_info_snippet}

{extra_instructions}

Response format (JSON மட்டும்):
{{
  "speech": "வேட்பாளரிடம் சொல்ல வேண்டிய வாக்கியம்",
  "lead_score": "hot அல்லது warm அல்லது cold",
  "score_confidence": 0-100,
  "intent_flags": ["interview_requested", "callback_requested", "not_interested", "interested", "busy"],
  "advance_script": true அல்லது false,
  "should_end_call": false
}}

lead_score criteria:
- hot: வேட்பாளர் உடனே வர தயார், மிகவும் ஆர்வமாக இருக்கிறார்
- warm: ஆர்வம் இருக்கிறது ஆனால் உறுதி இல்லை
- cold: ஆர்வம் இல்லை அல்லது பதில் சொல்லவில்லை"""

DEFAULT_OBJECTIONS = {
    "busy": "சரி, உங்கள் நேரத்தை மதிக்கிறோம். நாளை காலை பேசலாமா?",
    "not_interested": "புரிகிறது. ஆனால் இது மிகவும் நல்ல வாய்ப்பு. சுருக்கமாக ஒரு நிமிடம் சொல்லட்டுமா?",
    "call_later": "நிச்சயமாக. எந்த நேரத்தில் திரும்ப அழைக்கட்டும்?",
    "no_answer": "சரி, நன்றி. வேறு நேரத்தில் தொடர்பு கொள்கிறோம்.",
}

DEFAULT_CLOSING_HOT = "மிகவும் நன்றி {name}. நேர்காணல் உறுதி செய்யப்பட்டது. SMS மூலம் விவரங்கள் அனுப்புகிறோம். வாழ்த்துக்கள்!"
DEFAULT_CLOSING_WARM = "நன்றி {name}. உங்கள் விவரங்கள் பதிவு செய்துவிட்டோம். விரைவில் திரும்ப தொடர்பு கொள்கிறோம்."
DEFAULT_CLOSING_COLD = "நன்றி {name}. உங்கள் நேரத்திற்கு நன்றி. நல்ல வாய்ப்பு வந்தால் தெரிவிக்கிறோம்."


def build_system_prompt(company_name: str, agent_name: str,
                         company_info_snippet: str = "",
                         extra_instructions: str = "") -> str:
    return BASE_SYSTEM_PROMPT.format(
        agent_name=agent_name,
        company_name=company_name,
        company_info_snippet=company_info_snippet or f"{company_name} நிறுவனம் பற்றி விரைவில் தெரிவிக்கிறோம்.",
        extra_instructions=extra_instructions,
    )


def _extract_questions_from_text(text: str) -> List[Dict]:
    steps = []
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    question_patterns = [
        r'^\d+[\.\)]\s*(.+)',
        r'^Q\d*[\.\):\s]+(.+)',
        r'^கேள்வி\s*\d*[\.\):\s]*(.+)',
        r'^Step\s*\d+[\.\):\s]+(.+)',
    ]

    collected = []
    for line in lines:
        for pattern in question_patterns:
            m = re.match(pattern, line, re.IGNORECASE)
            if m:
                collected.append(m.group(1).strip())
                break
        else:
            if len(line) > 20 and line.endswith('?'):
                collected.append(line)

    if not collected:
        collected = [l for l in lines if len(l) > 15 and (l.endswith('?') or 'வேண்டும்' in l or 'இருக்கிறீர்களா' in l)]

    state_map = {0: "greeting", len(collected) - 1: "scheduling"}
    for i, q in enumerate(collected):
        state = state_map.get(i, "qualifying")
        steps.append({
            "id": i,
            "state": state,
            "question": q,
            "fallback": q,
        })

    return steps


def _extract_from_pdf(file_path: str) -> Tuple[List[Dict], Dict, str, str, str, str]:
    try:
        import pdfplumber
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception:
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += (page.extract_text() or "") + "\n"
        except Exception as e:
            raise ValueError(f"Could not extract text from PDF: {e}")

    return _parse_script_text(text)


def _parse_script_text(text: str) -> Tuple[List[Dict], Dict, str, str, str, str]:
    steps = _extract_questions_from_text(text)

    objections = dict(DEFAULT_OBJECTIONS)

    objection_section = re.search(
        r'(objection|ஆட்சேர்ப்பு\s*மறுப்பு|எதிர்ப்பு|response|பதில்).*?(?=\n\n|\Z)',
        text, re.IGNORECASE | re.DOTALL
    )
    if objection_section:
        obj_text = objection_section.group(0)
        busy_m = re.search(r'busy[:\s]+(.+)', obj_text, re.IGNORECASE)
        not_int_m = re.search(r'not.interested[:\s]+(.+)', obj_text, re.IGNORECASE)
        if busy_m:
            objections["busy"] = busy_m.group(1).strip()
        if not_int_m:
            objections["not_interested"] = not_int_m.group(1).strip()

    closing_hot = DEFAULT_CLOSING_HOT
    closing_warm = DEFAULT_CLOSING_WARM
    closing_cold = DEFAULT_CLOSING_COLD

    hot_m = re.search(r'hot[:\s]+(.+)', text, re.IGNORECASE)
    warm_m = re.search(r'warm[:\s]+(.+)', text, re.IGNORECASE)
    cold_m = re.search(r'cold[:\s]+(.+)', text, re.IGNORECASE)

    if hot_m:
        closing_hot = hot_m.group(1).strip()
    if warm_m:
        closing_warm = warm_m.group(1).strip()
    if cold_m:
        closing_cold = cold_m.group(1).strip()

    extra = ""
    extra_m = re.search(r'(additional\s*instructions?|கூடுதல்\s*விதிகள்)[:\s]+(.+?)(?=\n\n|\Z)', text, re.IGNORECASE | re.DOTALL)
    if extra_m:
        extra = extra_m.group(2).strip()

    return steps, objections, closing_hot, closing_warm, closing_cold, extra


def parse_uploaded_script(file_path: str, filename: str) -> Dict:
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        steps = data.get("steps", [])
        for i, step in enumerate(steps):
            if "id" not in step:
                step["id"] = i
            if "state" not in step:
                step["state"] = "qualifying"
            if "fallback" not in step:
                step["fallback"] = step.get("question", "")
        return {
            "steps": steps,
            "objection_responses": data.get("objection_responses", DEFAULT_OBJECTIONS),
            "closing_hot": data.get("closing_hot", DEFAULT_CLOSING_HOT),
            "closing_warm": data.get("closing_warm", DEFAULT_CLOSING_WARM),
            "closing_cold": data.get("closing_cold", DEFAULT_CLOSING_COLD),
            "system_prompt_extra": data.get("system_prompt_extra", ""),
        }

    elif ext == ".pdf":
        steps, objections, closing_hot, closing_warm, closing_cold, extra = _extract_from_pdf(file_path)

    elif ext in (".txt", ".md"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        steps, objections, closing_hot, closing_warm, closing_cold, extra = _parse_script_text(text)

    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .json, .pdf, or .txt")

    if not steps:
        raise ValueError("No script questions found in the uploaded file. Use the JSON template for precise control.")

    return {
        "steps": steps,
        "objection_responses": objections,
        "closing_hot": closing_hot,
        "closing_warm": closing_warm,
        "closing_cold": closing_cold,
        "system_prompt_extra": extra,
    }


def get_script_json_template(company_name: str = "YourCompany", agent_name: str = "பிரியா") -> Dict:
    return {
        "steps": [
            {
                "id": 0,
                "state": "greeting",
                "question": f"வணக்கம் {{name}} அவர்களா? நான் {company_name} நிறுவனத்திலிருந்து பேசுகிறேன். இப்போது சற்று பேசலாமா?",
                "fallback": f"வணக்கம், நான் {company_name} இல் இருந்து பேசுகிறேன். இப்போது நேரம் இருக்கிறதா?",
            },
            {
                "id": 1,
                "state": "qualifying",
                "question": "நீங்கள் தற்போது வேலை தேடுகிறீர்களா?",
                "fallback": "தற்போது வேலை வாய்ப்பில் ஆர்வம் இருக்கிறதா?",
            },
            {
                "id": 2,
                "state": "qualifying",
                "question": "உங்கள் படிப்பு தகுதி என்ன?",
                "fallback": "உங்கள் கல்வி தகுதி சொல்லுங்கள்.",
            },
            {
                "id": 3,
                "state": "scheduling",
                "question": "நேர்காணலுக்கு எப்போது வர முடியும்?",
                "fallback": "நேர்காணலுக்கு வசதியான நேரம் சொல்லுங்கள்.",
            },
        ],
        "objection_responses": DEFAULT_OBJECTIONS,
        "closing_hot": DEFAULT_CLOSING_HOT,
        "closing_warm": DEFAULT_CLOSING_WARM,
        "closing_cold": DEFAULT_CLOSING_COLD,
        "system_prompt_extra": f"நீங்கள் {company_name} நிறுவனம் பற்றிய கேள்விகளுக்கு சுருக்கமாக பதில் சொல்லுங்கள்.",
    }