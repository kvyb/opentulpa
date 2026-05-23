"""Telegram chat access and command routing policy."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TelegramAccessDecision:
    normal_allowed: bool
    support_allowed: bool

    @classmethod
    def from_flags(
        cls,
        *,
        normal_allowed: bool,
        support_allowed: bool,
    ) -> TelegramAccessDecision:
        decision = cls(normal_allowed=bool(normal_allowed), support_allowed=bool(support_allowed))
        assert decision.allowed == (decision.normal_allowed or decision.support_allowed)
        return decision

    @property
    def allowed(self) -> bool:
        return self.normal_allowed or self.support_allowed

    @property
    def restricted_reply(self) -> str | None:
        if self.allowed:
            return None
        return "This bot is restricted and your Telegram account is not allowed."

    @property
    def should_auto_bind_allowed_username(self) -> bool:
        return self.normal_allowed and not self.support_allowed

    @property
    def should_configure_support_commands(self) -> bool:
        return self.support_allowed


@dataclass(frozen=True)
class TelegramCommandRoute:
    command_name: str
    kind: str

    @classmethod
    def from_command(
        cls,
        *,
        command_name: str,
        support_allowed: bool,
        is_support_command: bool,
    ) -> TelegramCommandRoute:
        safe_command = str(command_name or "").strip().lower()
        assert not safe_command or safe_command.startswith("/")
        if is_support_command:
            kind = "support_command" if support_allowed else "restricted_support_command"
        elif safe_command in {"/start", "/help"}:
            kind = "start_help"
        elif safe_command == "/status":
            kind = "status"
        elif safe_command == "/fresh":
            kind = "fresh"
        elif safe_command == "/debug_logs":
            kind = "debug_logs"
        else:
            kind = "chat"
        return cls(command_name=safe_command, kind=kind)

    @property
    def is_chat(self) -> bool:
        return self.kind == "chat"
