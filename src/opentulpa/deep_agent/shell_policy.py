"""Approval policy for model-generated shell commands."""

from __future__ import annotations

import shlex
from enum import StrEnum
from pathlib import PurePosixPath

import tree_sitter_bash
from tree_sitter import Language, Node, Parser

_BASH_LANGUAGE = Language(tree_sitter_bash.language())
_DYNAMIC_WORD_NODES = frozenset(
    {
        "arithmetic_expansion",
        "command_substitution",
        "expansion",
        "process_substitution",
        "simple_expansion",
    }
)
_SHELL_COMMANDS = frozenset({"bash", "dash", "ksh", "sh", "zsh"})
_SIMPLE_WRAPPERS = frozenset({"builtin", "command", "exec", "nohup"})
_RM_SHORT_OPTIONS = frozenset({"d", "f", "i", "I", "r", "R", "v", "W"})
_ENV_SHORT_OPTIONS = frozenset({"0", "i", "v"})
_ENV_SHORT_OPTIONS_WITH_VALUE = frozenset({"C", "S", "u"})
_ENV_LONG_OPTIONS = frozenset({"debug", "ignore-environment", "null"})
_ENV_LONG_OPTIONS_WITH_VALUE = frozenset({"chdir", "split-string", "unset"})
_SUDO_SHORT_OPTIONS = frozenset({"A", "b", "E", "e", "H", "K", "k", "n", "P", "S", "V", "v"})
_SUDO_SHORT_OPTIONS_WITH_VALUE = frozenset(
    {"C", "D", "g", "h", "p", "R", "r", "T", "t", "U", "u"}
)
_SUDO_LONG_OPTIONS = frozenset(
    {
        "askpass",
        "background",
        "bell",
        "edit",
        "help",
        "host",
        "login",
        "non-interactive",
        "preserve-env",
        "remove-timestamp",
        "reset-timestamp",
        "set-home",
        "stdin",
        "validate",
        "version",
    }
)
_SUDO_LONG_OPTIONS_WITH_VALUE = frozenset(
    {
        "chdir",
        "close-from",
        "group",
        "other-user",
        "prompt",
        "role",
        "type",
        "user",
    }
)
_XARGS_OPTIONS_WITH_VALUE = frozenset(
    {
        "-a",
        "-d",
        "-E",
        "-I",
        "-L",
        "-n",
        "-P",
        "-s",
        "--arg-file",
        "--delimiter",
        "--eof",
        "--max-args",
        "--max-chars",
        "--max-lines",
        "--max-procs",
        "--replace",
    }
)


class ShellCommandDisposition(StrEnum):
    """How OpenTulpa may handle one model-generated shell command."""

    ALLOW = "allow"
    REQUIRE_APPROVAL = "require_approval"
    REJECT = "reject"


def classify_shell_command(command: str) -> ShellCommandDisposition:
    """Classify recursive forced removal without treating quoted text as execution."""

    if not isinstance(command, str) or not command.strip() or "\x00" in command:
        return ShellCommandDisposition.REJECT
    source = _remove_line_continuations(command).encode()
    tree = Parser(_BASH_LANGUAGE).parse(source)
    if tree.root_node.has_error:
        return ShellCommandDisposition.REJECT

    disposition = ShellCommandDisposition.ALLOW
    for node in _walk(tree.root_node):
        if node.type != "command":
            continue
        current = _classify_command(node, source)
        if current is ShellCommandDisposition.REJECT:
            return current
        if current is ShellCommandDisposition.REQUIRE_APPROVAL:
            disposition = current
    return disposition


def _classify_command(node: Node, source: bytes) -> ShellCommandDisposition:
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return ShellCommandDisposition.ALLOW
    name = _literal_word(name_node, source)
    if name is None:
        return ShellCommandDisposition.REJECT

    arguments = [
        _literal_word(argument, source)
        for argument in node.children_by_field_name("argument")
    ]
    return _classify_words(name, arguments)


def _classify_words(
    name: str,
    arguments: list[str | None],
) -> ShellCommandDisposition:
    executable = PurePosixPath(name).name
    if executable == "rm":
        return _classify_rm_arguments(arguments)
    if executable in _SIMPLE_WRAPPERS:
        return _classify_wrapped(executable, arguments)
    if executable in {"env", "sudo"}:
        return _classify_env_or_sudo(executable, arguments)
    if executable in _SHELL_COMMANDS:
        return _classify_shell_script(arguments)
    if executable == "eval":
        return _classify_literal_script(arguments)
    if executable in {"xargs", "parallel"}:
        return _classify_xargs(arguments)
    if executable == "find":
        return _classify_find(arguments)
    if executable == "busybox":
        return _classify_wrapped(executable, arguments)
    return ShellCommandDisposition.ALLOW


def _classify_rm_arguments(arguments: list[str | None]) -> ShellCommandDisposition:
    recursive = False
    force = False
    for argument in arguments:
        if argument is None:
            return ShellCommandDisposition.REJECT
        if argument == "--":
            break
        if len(argument) > 2 and "--recursive".startswith(argument):
            recursive = True
        elif len(argument) > 2 and "--force".startswith(argument):
            force = True
        elif argument.startswith("-") and not argument.startswith("--"):
            flags = argument[1:]
            if set(flags) <= _RM_SHORT_OPTIONS:
                recursive = recursive or bool({"r", "R"}.intersection(flags))
                force = force or "f" in flags
    if recursive and force:
        return ShellCommandDisposition.REQUIRE_APPROVAL
    return ShellCommandDisposition.ALLOW


def _classify_wrapped(
    executable: str,
    arguments: list[str | None],
) -> ShellCommandDisposition:
    remaining = list(arguments)
    if executable == "command" and any(argument in {"-V", "-v"} for argument in remaining):
        return ShellCommandDisposition.ALLOW
    if executable == "exec":
        remaining = _consume_options(
            remaining,
            short_options=frozenset({"c", "l"}),
            short_options_with_value=frozenset({"a"}),
            long_options=frozenset(),
            long_options_with_value=frozenset(),
        )
        if remaining is None:
            return ShellCommandDisposition.REJECT
    else:
        while remaining and remaining[0] is not None and remaining[0].startswith("-"):
            remaining.pop(0)
    if not remaining:
        return ShellCommandDisposition.ALLOW
    name = remaining.pop(0)
    if name is None:
        return ShellCommandDisposition.REJECT
    return _classify_words(name, remaining)


def _classify_env_or_sudo(
    executable: str,
    arguments: list[str | None],
) -> ShellCommandDisposition:
    if executable == "env":
        remaining = _consume_options(
            arguments,
            short_options=_ENV_SHORT_OPTIONS,
            short_options_with_value=_ENV_SHORT_OPTIONS_WITH_VALUE,
            long_options=_ENV_LONG_OPTIONS,
            long_options_with_value=_ENV_LONG_OPTIONS_WITH_VALUE,
            assignments=True,
        )
    else:
        remaining = _consume_options(
            arguments,
            short_options=_SUDO_SHORT_OPTIONS,
            short_options_with_value=_SUDO_SHORT_OPTIONS_WITH_VALUE,
            long_options=_SUDO_LONG_OPTIONS,
            long_options_with_value=_SUDO_LONG_OPTIONS_WITH_VALUE,
            assignments=True,
        )
    if remaining is None:
        return ShellCommandDisposition.REJECT
    if not remaining:
        return ShellCommandDisposition.ALLOW
    name = remaining.pop(0)
    if name is None:
        return ShellCommandDisposition.REJECT
    return _classify_words(name, remaining)


def _classify_shell_script(arguments: list[str | None]) -> ShellCommandDisposition:
    for index, argument in enumerate(arguments):
        if argument == "-c" or (
            argument is not None
            and argument.startswith("-")
            and "c" in argument[1:]
        ):
            if index + 1 >= len(arguments) or arguments[index + 1] is None:
                return ShellCommandDisposition.REJECT
            return classify_shell_command(arguments[index + 1])
    literal_arguments = [argument for argument in arguments if argument is not None]
    script_paths = [argument for argument in literal_arguments if not argument.startswith("-")]
    if any(path not in {"-", "/dev/stdin", "/proc/self/fd/0"} for path in script_paths):
        return ShellCommandDisposition.ALLOW
    return ShellCommandDisposition.REJECT


def _classify_literal_script(arguments: list[str | None]) -> ShellCommandDisposition:
    if any(argument is None for argument in arguments):
        return ShellCommandDisposition.REJECT
    script = " ".join(argument for argument in arguments if argument)
    return (
        classify_shell_command(script)
        if script
        else ShellCommandDisposition.ALLOW
    )


def _consume_options(
    arguments: list[str | None],
    *,
    short_options: frozenset[str],
    short_options_with_value: frozenset[str],
    long_options: frozenset[str],
    long_options_with_value: frozenset[str],
    assignments: bool = False,
) -> list[str | None] | None:
    remaining = list(arguments)
    while remaining:
        argument = remaining[0]
        if argument is None:
            return None
        if argument == "--":
            return remaining[1:]
        if assignments and "=" in argument and not argument.startswith("-"):
            remaining.pop(0)
            continue
        if not argument.startswith("-") or argument == "-":
            return remaining
        remaining.pop(0)
        if argument.startswith("--"):
            option, separator, _ = argument[2:].partition("=")
            if option in long_options:
                if separator:
                    return None
                continue
            if option not in long_options_with_value:
                return None
            if not separator and (not remaining or remaining.pop(0) is None):
                return None
            continue
        flags = argument[1:]
        for index, flag in enumerate(flags):
            if flag in short_options:
                continue
            if flag not in short_options_with_value:
                return None
            if index == len(flags) - 1 and (
                not remaining or remaining.pop(0) is None
            ):
                return None
            break
    return remaining


def _classify_xargs(arguments: list[str | None]) -> ShellCommandDisposition:
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument is None:
            return ShellCommandDisposition.REJECT
        if argument == "--":
            index += 1
            break
        option = argument.split("=", 1)[0]
        if not argument.startswith("-"):
            break
        index += 1
        if option in _XARGS_OPTIONS_WITH_VALUE and "=" not in argument:
            index += 1
    if index >= len(arguments):
        return ShellCommandDisposition.ALLOW
    name = arguments[index]
    if name is None:
        return ShellCommandDisposition.REJECT
    return _classify_words(name, arguments[index + 1 :])


def _classify_find(arguments: list[str | None]) -> ShellCommandDisposition:
    disposition = ShellCommandDisposition.ALLOW
    for index, argument in enumerate(arguments):
        if argument not in {"-exec", "-execdir"}:
            continue
        command: list[str | None] = []
        for value in arguments[index + 1 :]:
            if value in {";", "+"}:
                break
            command.append(value)
        if not command:
            return ShellCommandDisposition.REJECT
        name = command.pop(0)
        if name is None:
            return ShellCommandDisposition.REJECT
        current = _classify_words(name, command)
        if current is ShellCommandDisposition.REJECT:
            return current
        if current is ShellCommandDisposition.REQUIRE_APPROVAL:
            disposition = current
    return disposition


def _literal_word(node: Node, source: bytes) -> str | None:
    if any(
        descendant.type in _DYNAMIC_WORD_NODES
        for descendant in _walk(node)
    ):
        return None
    raw = source[node.start_byte : node.end_byte].decode().replace("\\\n", "")
    try:
        values = shlex.split(raw, comments=False, posix=True)
    except ValueError:
        return None
    if len(values) != 1:
        return None
    return values[0]


def _remove_line_continuations(command: str) -> str:
    output: list[str] = []
    quote: str | None = None
    index = 0
    while index < len(command):
        char = command[index]
        if (
            quote != "'"
            and char == "\\"
            and index + 1 < len(command)
            and command[index + 1] == "\n"
        ):
            index += 2
            continue
        if char == "\\" and quote != "'" and index + 1 < len(command):
            output.extend((char, command[index + 1]))
            index += 2
            continue
        if char in {"'", '"'}:
            if quote is None:
                quote = char
            elif quote == char:
                quote = None
        output.append(char)
        index += 1
    return "".join(output)


def _walk(node: Node):
    yield node
    for child in node.children:
        yield from _walk(child)


__all__ = ["ShellCommandDisposition", "classify_shell_command"]
