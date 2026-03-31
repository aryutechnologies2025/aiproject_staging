import re


def clean(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def is_bold_line(line: str):
    return line.strip().startswith("**") and line.strip().endswith("**")


def strip_bold(line: str):
    return clean(line.replace("**", ""))


def markdown_to_structured_json(markdown: str):

    lines = markdown.split("\n")

    root = {}
    stack = [(0, root)]

    for line in lines:
        raw = line
        line = line.rstrip()

        if not line.strip():
            continue

        # =====================================================
        # 1. HEADINGS (# only, ignore ### as section)
        # =====================================================
        if line.startswith("## "):   # ONLY major sections

            level = 2
            key = clean(line.replace("##", ""))

            node = {}

            while stack and stack[-1][0] >= level:
                stack.pop()

            parent = stack[-1][1]
            parent[key] = node

            stack.append((level, node))

        # =====================================================
        # 2. SUB-HEADINGS (### → store as title, not section)
        # =====================================================
        elif line.startswith("###"):

            parent = stack[-1][1]

            parent["__subsection__"] = clean(line.replace("###", ""))

        # =====================================================
        # 3. BOLD BLOCKS (**...**) → IMPORTANT FIX
        # =====================================================
        elif is_bold_line(line):

            parent = stack[-1][1]
            value = strip_bold(line)

            if "__titles__" not in parent:
                parent["__titles__"] = []

            parent["__titles__"].append(value)

        # =====================================================
        # 4. BULLETS
        # =====================================================
        elif line.strip().startswith(("*", "-", "•")):

            parent = stack[-1][1]

            if "__list__" not in parent:
                parent["__list__"] = []

            parent["__list__"].append(clean(line.lstrip("*-• ")))

        # =====================================================
        # 5. NORMAL TEXT
        # =====================================================
        else:

            parent = stack[-1][1]

            if "__text__" not in parent:
                parent["__text__"] = []

            parent["__text__"].append(clean(raw))

    return root