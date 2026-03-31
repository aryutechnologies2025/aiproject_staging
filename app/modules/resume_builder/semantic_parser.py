import re


def clean(text):
    return re.sub(r"\s+", " ", text).strip()


# =========================
# SECTION DETECTOR
# =========================
def get_section(name: str):
    n = name.lower()

    if "experience" in n:
        return "experience"
    if "education" in n:
        return "education"
    if "project" in n:
        return "projects"
    if "skill" in n or "technical" in n:
        return "skills"
    if "summary" in n:
        return "summary"
    if "language" in n:
        return "languages"

    return None


# =========================
# EXPERIENCE
# =========================
def parse_experience(blocks):
    results = []
    current = None
    active = False

    for b in blocks:

        # =========================
        # SECTION CONTROL
        # =========================
        if b["type"] == "heading" and b["level"] == 2:
            active = get_section(b["value"]) == "experience"
            continue

        if not active:
            continue

        # =========================
        # ROLE
        # =========================
        if b["type"] == "heading" and b["level"] == 3:
            if current:
                results.append(current)

            current = {
                "role": clean(b["value"]),
                "company": "",
                "duration": "",
                "location": "",
                "description": []
            }

        # =========================
        # COMPANY FORMAT 1 (with |)
        # =========================
        elif b["type"] == "text" and "|" in b["value"] and current:
            text = b["value"].replace("**", "")
            parts = [clean(p) for p in text.split("|")]

            if len(parts) >= 1:
                current["company"] = parts[0]
            if len(parts) >= 2:
                current["duration"] = parts[1]
            if len(parts) >= 3:
                current["location"] = parts[2]

        # =========================
        # COMPANY FORMAT 2 (ONLY COMPANY NAME)
        # =========================
        elif b["type"] == "text" and "**" in b["value"] and current and not current["company"]:
            text = clean(b["value"].replace("**", ""))

            # avoid picking wrong lines
            if "|" not in text and len(text.split()) <= 8:
                current["company"] = text

        # =========================
        # DATE + LOCATION (NO ICON DEPENDENCY)
        # =========================
        elif b["type"] == "text" and current:

            text = b["value"]

            # case 1: 📅 📍 format
            if "📅" in text:
                parts = text.split("📍")

                current["duration"] = clean(parts[0].replace("📅", ""))

                if len(parts) > 1:
                    current["location"] = clean(parts[1])

            # case 2: normal format (no icons)
            elif re.search(r"\d{2}/\d{4}|\w+\s\d{4}", text):
                # Example: "Jan 2023 - Present Chennai"
                
                parts = text.split(",")

                if len(parts) >= 1:
                    current["duration"] = clean(parts[0])

                if len(parts) >= 2:
                    current["location"] = clean(parts[1])

        # =========================
        # BULLETS
        # =========================
        elif b["type"] == "bullet" and current:
            current["description"].append(clean(b["value"]))

    if current:
        results.append(current)

    return results


# =========================
# EDUCATION
# =========================
def parse_education(blocks):
    results = []
    current = None
    active = False

    for b in blocks:

        if b["type"] == "heading" and b["level"] == 2:
            active = get_section(b["value"]) == "education"
            continue

        if not active:
            continue

        if b["type"] == "heading" and b["level"] == 3:
            if current:
                results.append(current)

            current = {
                "degree": clean(b["value"]),
                "institution": "",
                "duration": "",
                "location": "",
                "cgpa": ""
            }

        elif b["type"] == "text" and "**" in b["value"] and current:
            text = b["value"].replace("**", "")

            if "cgpa" in text.lower():
                current["cgpa"] = clean(text.split(":")[-1])
            else:
                current["institution"] = clean(text)

        elif b["type"] == "text" and "📅" in b["value"] and current:
            parts = b["value"].split("📍")
            current["duration"] = clean(parts[0].replace("📅", ""))

            if len(parts) > 1:
                current["location"] = clean(parts[1])

    if current:
        results.append(current)

    return results


# =========================
# SKILLS
# =========================
def parse_skills(blocks):
    skills = []
    active = False

    for b in blocks:

        if b["type"] == "heading" and b["level"] == 2:
            active = get_section(b["value"]) == "skills"
            continue

        if not active:
            continue

        if b["type"] == "bullet":
            parts = re.split(r",|/|\|", b["value"])

            for p in parts:
                p = clean(p)

                if len(p) > 1 and not p.lower().startswith("linear"):
                    skills.append(p)

    return list(set(skills))


# =========================
# PROJECTS
# =========================
def parse_projects(blocks):
    results = []
    current = None
    active = False

    for b in blocks:

        if b["type"] == "heading" and b["level"] == 2:
            active = get_section(b["value"]) == "projects"
            continue

        if not active:
            continue

        # PROJECT NAME
        if b["type"] == "text" and b["value"].startswith("**"):
            if current:
                results.append(current)

            current = {
                "name": clean(b["value"].replace("**", "")),
                "description": [],
                "tech_stack": []
            }

        elif b["type"] == "bullet" and current:
            text = clean(b["value"])

            if "tech" in text.lower():
                current["tech_stack"] = [
                    clean(t) for t in text.split(":")[-1].split(",")
                ]
            else:
                current["description"].append(text)

    if current:
        results.append(current)

    return results

def extract_personal_info(blocks):
    data = {
        "name": "",
        "title": "",
        "phone": "",
        "email": "",
        "location": "",
        "links": []
    }

    header_done = False

    for b in blocks:

        # STOP when sections start
        if b["type"] == "heading" and b["level"] == 2:
            break

        # NAME
        if b["type"] == "heading" and b["level"] == 1:
            data["name"] = b["value"]

        # TITLE
        elif b["type"] == "heading" and b["level"] == 3:
            data["title"] = b["value"]

        # CONTACT LINE
        elif b["type"] == "text":

            text = b["value"]

            # EMAIL
            email = re.search(r'[\w\.-]+@[\w\.-]+', text)
            if email:
                data["email"] = email.group()

            # PHONE
            phone = re.search(r'(\+?\d[\d\s]{8,})', text)
            if phone:
                data["phone"] = phone.group().strip()

            # LOCATION (split by |)
            if "|" in text:
                parts = [p.strip() for p in text.split("|")]

                if len(parts) >= 3:
                    data["location"] = parts[2]

                # links
                for p in parts:
                    if "http" in p.lower() or "github" in p.lower() or "portfolio" in p.lower():
                        data["links"].append(p)

    return data


# =========================
# MAIN
# =========================
def parse_resume(blocks):
    return {
        "personal_info": extract_personal_info(blocks),
        "experience": parse_experience(blocks),
        "education": parse_education(blocks),
        "skills": parse_skills(blocks),
        "projects": parse_projects(blocks)
    }

