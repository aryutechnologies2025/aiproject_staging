# /home/aryu_user/Arun/aiproject_staging/app/utils/text_extraction.py
import pdfplumber
import docx
from fastapi import UploadFile


async def extract_text(file: UploadFile) -> str:
    filename = file.filename.lower()

    if filename.endswith(".pdf"):
        return await extract_pdf_text(file)

    if filename.endswith(".docx"):
        return await extract_docx_text(file)

    raise ValueError("Unsupported file type")


async def extract_pdf_text(file: UploadFile) -> str:
    text = []

    with pdfplumber.open(file.file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)

    return "\n".join(text)


async def extract_docx_text(file: UploadFile) -> str:
    doc = docx.Document(file.file)
    text = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(text)