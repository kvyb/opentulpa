from __future__ import annotations

import os

import pytest

from opentulpa.capability_workers.telegram_worker import (
    WorkerConfigurationError,
    read_secret,
)


def test_secret_is_consumed_from_environment_without_remaining_in_child_env() -> None:
    environ = {"SECRET": " private-value "}

    assert read_secret("SECRET", "SECRET_FD", environ=environ) == "private-value"
    assert environ == {}


def test_secret_can_be_read_from_inherited_file_descriptor() -> None:
    read_fd, write_fd = os.pipe()
    os.write(write_fd, b"fd-private-value\n")
    os.close(write_fd)
    environ = {"SECRET_FD": str(read_fd)}

    assert read_secret("SECRET", "SECRET_FD", environ=environ) == "fd-private-value"
    assert environ == {}
    with pytest.raises(OSError):
        os.read(read_fd, 1)


def test_secret_sources_are_mutually_exclusive_and_errors_do_not_include_values() -> None:
    environ = {"SECRET": "do-not-leak", "SECRET_FD": "123"}

    with pytest.raises(WorkerConfigurationError) as captured:
        read_secret("SECRET", "SECRET_FD", environ=environ)

    assert "do-not-leak" not in str(captured.value)
    assert environ == {}


def test_optional_secret_can_be_absent() -> None:
    assert read_secret("SECRET", "SECRET_FD", environ={}, required=False) is None
