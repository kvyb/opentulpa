"""Shared helpers for domain-specific tool modules."""

from __future__ import annotations

from typing import Any


def require_customer_id(runtime: Any) -> str:
    getter = getattr(runtime, "get_active_customer_id", None)
    customer_id = ""
    if callable(getter):
        customer_id = str(getter() or "").strip()
    if not customer_id:
        customer_id = str(getattr(runtime, "_active_customer_id", "") or "").strip()
    if not customer_id:
        raise RuntimeError("customer_id is missing in runtime context")
    return customer_id


def require_thread_id(runtime: Any) -> str:
    getter = getattr(runtime, "get_active_thread_id", None)
    thread_id = ""
    if callable(getter):
        thread_id = str(getter() or "").strip()
    if not thread_id:
        thread_id = str(getattr(runtime, "_active_thread_id", "") or "").strip()
    if not thread_id:
        raise RuntimeError("thread_id is missing in runtime context")
    return thread_id
