from markdown_it import MarkdownIt

md = MarkdownIt()


def markdown_to_blocks(markdown: str):
    tokens = md.parse(markdown)

    blocks = []
    i = 0

    while i < len(tokens):
        t = tokens[i]

        # HEADINGS
        if t.type == "heading_open":
            level = int(t.tag[1])
            content = tokens[i + 1].content

            blocks.append({
                "type": "heading",
                "level": level,
                "value": content.strip()
            })
            i += 2

        # INLINE TEXT
        elif t.type == "inline" and t.content.strip():
            text = t.content.strip()

            blocks.append({
                "type": "text",
                "value": text
            })

        # BULLETS
        elif t.type == "list_item_open":
            content = tokens[i + 2].content

            blocks.append({
                "type": "bullet",
                "value": content.strip()
            })
            i += 4

        i += 1

    return blocks