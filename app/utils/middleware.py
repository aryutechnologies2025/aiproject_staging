from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response
from app.utils.crypto_utils import encrypt_json, decrypt_json
import json

app = FastAPI()

async def encrypt_response_middleware(request: Request, call_next):
    response = await call_next(request)

    body = b""
    async for chunk in response.body_iterator:
        body += chunk

    try:
        data = json.loads(body)
    except Exception:
        data = body.decode()

    encrypted = encrypt_json(data)

    return Response(
        content=encrypted,
        media_type="text/plain"
    )

async def get_decrypted_body(request: Request):
    try:
        raw_body = await request.body()

        if not raw_body:
            raise HTTPException(status_code=400, detail="Empty body")

        encrypted_payload = raw_body.decode()

        return decrypt_json(encrypted_payload)

    except Exception:
        raise HTTPException(status_code=400, detail="Invalid encrypted request")
    
