import os
from fastapi import UploadFile, HTTPException

# Allowed byte signatures for standard document uploads
VALID_MAGIC_SIGNATURES = {
    b"%PDF": "application/pdf",
    b"PK\x03\x04": "application/vnd.openxmlformats-officedocument.wordprocessingml.document", # Genuine DOCX
    b"\xd0\xcf\x11\xe0": "application/msword" # Old DOC binary formats
}

async def validate_file_security(file: UploadFile) -> UploadFile:
    """
    Verifies the actual inner contents of uploaded resumes.
    Blocks extension spoofing used to deliver executable malware scripts or viruses.
    """
    # 1. Check extension name as initial layer (requires imported 'os' module)
    ext = os.path.splitext(file.filename)[-1].lower()
    if ext not in [".pdf", ".docx", ".doc"]:
        raise HTTPException(
            status_code=400, 
            detail="Unsupported file extension. Only valid PDF, DOCX, and DOC formats are accepted."
        )

    # 2. Inspect inner Magic Bytes (Reads first 4 bytes)
    header = await file.read(4)
    await file.seek(0) # Essential reset so downstream parsers can read the file stream from the start
    
    # Check if the signature matches our verified whitelist
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