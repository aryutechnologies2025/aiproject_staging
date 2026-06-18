import os
from fastapi import UploadFile, HTTPException, File

# Allowed byte signatures for standard document uploads
VALID_MAGIC_SIGNATURES = {
    b"%PDF": "application/pdf",
    b"PK\x03\x04": "application/vnd.openxmlformats-officedocument.wordprocessingml.document", # Genuine DOCX
    b"\xd0\xcf\x11\xe0": "application/msword" # Old DOC binary formats
}

async def validate_file_security(
    file: UploadFile = File(...)
) -> UploadFile:
    """
    Verifies the actual inner contents of uploaded resumes.
    Blocks extension spoofing used to deliver executable malware scripts or viruses.
    """

    ext = os.path.splitext(file.filename)[-1].lower()
    if ext not in [".pdf", ".docx", ".doc"]:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file extension. Only valid PDF, DOCX, and DOC formats are accepted."
        )

    header = await file.read(4)
    await file.seek(0)

    matched_format = False
    for signature in VALID_MAGIC_SIGNATURES:
        if header.startswith(signature):
            matched_format = True
            break

    if not matched_format:
        raise HTTPException(
            status_code=400,
            detail="Malicious or corrupted file structure detected. The file type does not match its extension."
        )

    return file

