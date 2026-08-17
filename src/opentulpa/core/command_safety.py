"""Command guards for host-owned execution boundaries."""

from __future__ import annotations

import re

_HOST_LIFECYCLE_RE = re.compile(
    r"\b(?:docker(?:-compose)?|podman(?:-compose)?)\b[^;&|\n]*\b"
    r"(?:create|down|kill|pause|restart|rm|run|start|stop|unpause|up)\b"
    r"|\bsystemctl\b[^;&|\n]*\b"
    r"(?:disable|enable|halt|poweroff|reboot|restart|start|stop)\b"
    r"|\bservice\s+\S+\s+(?:restart|start|stop)\b"
    r"|\bkill\s+(?:-[A-Za-z0-9]+\s+)*(?:1|\$\$|\$\{?PPID\}?)\b"
    r"|\b(?:halt|poweroff|reboot|shutdown)\b",
    re.IGNORECASE,
)


def contains_host_lifecycle_command(command: str) -> bool:
    return _HOST_LIFECYCLE_RE.search(str(command or "")) is not None


__all__ = ["contains_host_lifecycle_command"]
