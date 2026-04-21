import json, base64, os
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad

SECRET_KEY = b'78156913478569134785923728102000'  # 32 bytes

def encrypt_json(data: dict) -> str:
    iv = os.urandom(16)
    cipher = AES.new(SECRET_KEY, AES.MODE_CBC, iv)

    raw = json.dumps(data).encode()
    encrypted = cipher.encrypt(pad(raw, AES.block_size))

    return base64.b64encode(iv + encrypted).decode()


def decrypt_json(payload: str) -> dict:
    raw = base64.b64decode(payload)

    iv = raw[:16]
    encrypted = raw[16:]

    cipher = AES.new(SECRET_KEY, AES.MODE_CBC, iv)
    decrypted = unpad(cipher.decrypt(encrypted), AES.block_size)

    return json.loads(decrypted.decode())