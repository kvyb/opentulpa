from __future__ import annotations

from opentulpa.interfaces.telegram.chat_routing import (
    TelegramAccessDecision,
    TelegramCommandRoute,
)


def test_telegram_access_decision_separates_owner_and_support_permissions() -> None:
    owner = TelegramAccessDecision.from_flags(normal_allowed=True, support_allowed=False)
    support = TelegramAccessDecision.from_flags(normal_allowed=False, support_allowed=True)
    blocked = TelegramAccessDecision.from_flags(normal_allowed=False, support_allowed=False)

    assert owner.restricted_reply is None
    assert owner.should_auto_bind_allowed_username is True
    assert owner.should_configure_support_commands is False
    assert support.restricted_reply is None
    assert support.should_auto_bind_allowed_username is False
    assert support.should_configure_support_commands is True
    assert blocked.restricted_reply == "This bot is restricted and your Telegram account is not allowed."


def test_telegram_command_route_classifies_commands_without_transport_state() -> None:
    assert (
        TelegramCommandRoute.from_command(
            command_name="/support_bind",
            support_allowed=False,
            is_support_command=True,
        ).kind
        == "restricted_support_command"
    )
    assert (
        TelegramCommandRoute.from_command(
            command_name="/support_bind",
            support_allowed=True,
            is_support_command=True,
        ).kind
        == "support_command"
    )
    assert (
        TelegramCommandRoute.from_command(
            command_name="/fresh",
            support_allowed=False,
            is_support_command=False,
        ).kind
        == "fresh"
    )
    assert TelegramCommandRoute.from_command(
        command_name="",
        support_allowed=False,
        is_support_command=False,
    ).is_chat
