"""Host-key encryption boundary for secret persistence."""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import Protocol

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class SecretCipherError(RuntimeError):
    """Ciphertext cannot be safely encrypted or decrypted."""


@dataclass(frozen=True, slots=True)
class EncryptedSecret:
    key_id: str
    nonce: bytes
    ciphertext: bytes


class HostKeySecretCipher(Protocol):
    """Encryption implementation backed by a key that is never stored in SQLite."""

    def encrypt(self, plaintext: bytes, *, associated_data: bytes) -> EncryptedSecret: ...

    def decrypt(self, value: EncryptedSecret, *, associated_data: bytes) -> bytes: ...


class AesGcmHostKeyCipher:
    """AES-256-GCM cipher using a host-provided key and per-revision nonce."""

    def __init__(self, key: bytes, *, key_id: str = "host-v1") -> None:
        if len(key) != 32:
            raise ValueError("secret vault host key must contain exactly 32 bytes")
        safe_key_id = str(key_id or "").strip()
        if not safe_key_id or len(safe_key_id) > 100:
            raise ValueError("secret vault key_id is invalid")
        self._cipher = AESGCM(bytes(key))
        self._key_id = safe_key_id

    @classmethod
    def from_base64(cls, value: str, *, key_id: str = "host-v1") -> AesGcmHostKeyCipher:
        try:
            raw = str(value or "").strip().encode("ascii")
            key = base64.b64decode(raw, altchars=b"-_", validate=True)
        except (ValueError, UnicodeEncodeError) as exc:
            raise ValueError("secret vault host key must be URL-safe base64") from exc
        return cls(key, key_id=key_id)

    def encrypt(self, plaintext: bytes, *, associated_data: bytes) -> EncryptedSecret:
        nonce = os.urandom(12)
        return EncryptedSecret(
            key_id=self._key_id,
            nonce=nonce,
            ciphertext=self._cipher.encrypt(nonce, plaintext, associated_data),
        )

    def decrypt(self, value: EncryptedSecret, *, associated_data: bytes) -> bytes:
        if value.key_id != self._key_id:
            raise SecretCipherError("secret was encrypted by an unavailable host key")
        try:
            return self._cipher.decrypt(value.nonce, value.ciphertext, associated_data)
        except InvalidTag as exc:
            raise SecretCipherError("secret ciphertext failed authentication") from exc


__all__ = [
    "AesGcmHostKeyCipher",
    "EncryptedSecret",
    "HostKeySecretCipher",
    "SecretCipherError",
]
