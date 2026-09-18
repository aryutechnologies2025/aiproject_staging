import os
import secrets
import uuid
import logging
from dataclasses import dataclass
from typing import Optional
from fastapi import UploadFile, HTTPException, File, Request, status
from app.core.settings import settings

logger = logging.getLogger("app.core.security")

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


@dataclass
class RequestContext:
    user_id: str
    request_id: str
    operation: Optional[str] = None
    is_trusted: bool = False


async def get_trusted_request_context(request: Request) -> RequestContext:
    """
    Validates internal service credentials forwarded from Django or trusted microservices.
    Ensures that X-User-ID and X-Operation are only trusted when X-Internal-Service-Key matches.
    """
    service_key = request.headers.get("X-Internal-Service-Key")
    user_id_header = request.headers.get("X-User-ID")
    request_id_header = request.headers.get("X-Request-ID")
    operation_header = request.headers.get("X-Operation")

    configured_secret = settings.INTERNAL_SERVICE_SECRET or os.getenv("INTERNAL_SERVICE_SECRET")

    is_trusted = False
    if configured_secret:
        if not service_key or not secrets.compare_digest(service_key, configured_secret):
            client_ip = request.client.host if request.client else "unknown"
            logger.warning(f"[Security] Unauthorized request to {request.url.path} from {client_ip}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing internal service key."
            )
        is_trusted = True
    else:
        # When no secret is configured in environment (e.g. testing / local dev)
        if service_key:
            is_trusted = True

    resolved_user_id = user_id_header if is_trusted and user_id_header else "anonymous"
    resolved_request_id = request_id_header.strip() if request_id_header and request_id_header.strip() else f"req-{uuid.uuid4()}"
    resolved_operation = operation_header.strip() if is_trusted and operation_header and operation_header.strip() else None

    return RequestContext(
        user_id=resolved_user_id,
        request_id=resolved_request_id,
        operation=resolved_operation,
        is_trusted=is_trusted,
    )


