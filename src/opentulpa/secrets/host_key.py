"""Load the host-only root key used to encrypt persisted credentials."""

from __future__ import annotations

import os
from pathlib import Path

from opentulpa.secrets.cipher import AesGcmHostKeyCipher


def load_or_create_host_cipher(data_root: Path) -> AesGcmHostKeyCipher:
    """Load a configured key or create a private, durable host key."""

    configured = str(os.environ.get("OPENTULPA_SECRET_VAULT_KEY", "") or "").strip()
    if configured:
        return AesGcmHostKeyCipher.from_base64(configured)

    key_path = data_root.resolve() / "bootstrap" / "secret-vault.key"
    key_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if key_path.parent.is_symlink():
        raise RuntimeError("secret vault key directory cannot be a symlink")
    if not key_path.exists():
        descriptor = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(os.urandom(32))
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            key_path.unlink(missing_ok=True)
            raise
    if key_path.is_symlink() or not key_path.is_file():
        raise RuntimeError("secret vault key must be a regular file")
    if key_path.stat().st_mode & 0o077:
        raise RuntimeError("secret vault key permissions must be 0600")
    key = key_path.read_bytes()
    if len(key) != 32:
        raise RuntimeError("secret vault key must contain exactly 32 bytes")
    return AesGcmHostKeyCipher(key)


__all__ = ["load_or_create_host_cipher"]
