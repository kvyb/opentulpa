from __future__ import annotations

import pytest

from opentulpa.deep_agent.shell_policy import (
    ShellCommandDisposition,
    classify_shell_command,
)


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf build",
        'rm -rf "$TARGET"',
        'rm "$TARGET" -rf',
        "rm -fr build",
        "rm -r -f build",
        "/bin/rm --recursive --force build",
        "rm --recurs --forc build",
        "rm \\\n-rf build",
        "r\\\nm -rf build",
        "bash -c 'rm -rf build'",
        'echo "$(rm -rf build)"',
        "find . -exec rm -rf {} +",
        "printf '%s\\0' build | xargs -0 rm -rf",
        "sudo /bin/rm --recursive --force build",
        "sudo -u root rm -rf build",
        "env -i rm -rf build",
        "exec -a cleanup rm -rf build",
        "timeout 10 rm -rf build",
        "nice -n 5 rm -rf build",
    ],
)
def test_recursive_forced_removal_requires_approval(command: str) -> None:
    assert classify_shell_command(command) is ShellCommandDisposition.REQUIRE_APPROVAL


@pytest.mark.parametrize(
    "command",
    [
        "rm build.log",
        "rm -r build",
        "rm -f build.log",
        "rm -force build.log",
        "rm -- -rf",
        "rm -- \"$FILE\"",
        "grep -R 'rm -rf' .",
        "printf '%s\\n' 'rm -rf build'",
        "echo rm -rf build",
        "command -v rm -rf",
        "# rm -rf build\nprintf done",
        "cat <<'EOF'\nrm -rf build\nEOF\n",
        "bash scripts/check.sh",
    ],
)
def test_non_destructive_references_do_not_require_approval(command: str) -> None:
    assert classify_shell_command(command) is ShellCommandDisposition.ALLOW


@pytest.mark.parametrize(
    "command",
    [
        "rm${IFS}-rf build",
        "$COMMAND -rf build",
        "rm -r${FORCE} build",
        "bash -c \"$COMMAND\"",
        "printf 'rm -rf build' | sh",
        "sh -s",
        "rm 'unterminated",
        "\x00rm -rf build",
    ],
)
def test_ambiguous_shell_construction_is_rejected(command: str) -> None:
    assert classify_shell_command(command) is ShellCommandDisposition.REJECT
