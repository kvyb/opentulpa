"""Host-key encrypted secret storage for mutable capabilities."""

from opentulpa.secrets.capability_resolver import (
    CapabilityCredentialLifecycle,
    VaultCapabilitySecretResolver,
)
from opentulpa.secrets.cipher import (
    AesGcmHostKeyCipher,
    EncryptedSecret,
    HostKeySecretCipher,
    SecretCipherError,
)
from opentulpa.secrets.ingress import (
    SecretIngressHook,
    SecretIngressResult,
    SecretIngressService,
)
from opentulpa.secrets.models import (
    IssuedSecretGrant,
    SecretGrantReceipt,
    SecretHandle,
    SecretMaterial,
    SecretState,
)
from opentulpa.secrets.service import SecretChangeListener, SecretVaultService
from opentulpa.secrets.vault import (
    SecretGrantError,
    SecretVault,
    SecretVaultConflictError,
    SecretVaultError,
    SecretVaultNotFoundError,
)

__all__ = [
    "AesGcmHostKeyCipher",
    "CapabilityCredentialLifecycle",
    "EncryptedSecret",
    "HostKeySecretCipher",
    "IssuedSecretGrant",
    "SecretCipherError",
    "SecretChangeListener",
    "SecretGrantError",
    "SecretGrantReceipt",
    "SecretHandle",
    "SecretIngressHook",
    "SecretIngressResult",
    "SecretIngressService",
    "SecretMaterial",
    "SecretState",
    "SecretVault",
    "SecretVaultConflictError",
    "SecretVaultError",
    "SecretVaultNotFoundError",
    "SecretVaultService",
    "VaultCapabilitySecretResolver",
]
