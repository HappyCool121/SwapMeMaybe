from __future__ import annotations

import base64
import hashlib
import os
import time
import uuid
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey


def create_key(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    key = Ed25519PrivateKey.generate()
    data = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as file:
        file.write(data)
    return public_key(key)


def load_key(path: Path) -> Ed25519PrivateKey:
    key = serialization.load_pem_private_key(path.read_bytes(), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError("expected an Ed25519 signing key")
    return key


def public_key(key: Ed25519PrivateKey) -> str:
    return base64.b64encode(key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)).decode()


def message(method: str, path: str, timestamp: str, nonce: str, body: bytes) -> bytes:
    return f"{method.upper()}\n{path}\n{timestamp}\n{nonce}\n{hashlib.sha256(body).hexdigest()}".encode()


def sign(key: Ed25519PrivateKey, method: str, path: str, body: bytes = b"") -> dict[str, str]:
    timestamp, nonce = str(int(time.time())), uuid.uuid4().hex
    signature = key.sign(message(method, path, timestamp, nonce, body))
    return {"x-liveswap-time": timestamp, "x-liveswap-nonce": nonce, "x-liveswap-signature": base64.b64encode(signature).decode()}


class Verifier:
    def __init__(self, public: str, *, tolerance: int = 60):
        self.key = Ed25519PublicKey.from_public_bytes(base64.b64decode(public, validate=True))
        self.tolerance = tolerance
        self.used: dict[str, int] = {}

    def verify(self, method: str, path: str, body: bytes, headers, *, now: int | None = None) -> None:
        now = int(time.time()) if now is None else now
        try:
            timestamp = headers["x-liveswap-time"]
            nonce = headers["x-liveswap-nonce"]
            signature = base64.b64decode(headers["x-liveswap-signature"], validate=True)
            if abs(now - int(timestamp)) > self.tolerance or len(nonce) != 32:
                raise ValueError("expired or invalid signature timestamp")
            self.used = {key: expires for key, expires in self.used.items() if expires >= now}
            if nonce in self.used or len(self.used) >= 20_000:
                raise ValueError("replayed request or nonce capacity reached")
            self.key.verify(signature, message(method, path, timestamp, nonce, body))
            self.used[nonce] = now + self.tolerance
        except (KeyError, ValueError, InvalidSignature) as exc:
            raise ValueError("invalid or replayed worker request") from exc
