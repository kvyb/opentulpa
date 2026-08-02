"""Approval policy for model-generated shell commands."""

from __future__ import annotations

import shlex
from collections.abc import Iterator
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
_COMMAND_STRING_WRAPPERS = frozenset({"script", "su"})
_LITERAL_DATA_COMMANDS = frozenset({"echo", "grep", "printf", "rg"})
_RM_SHORT_OPTIONS = frozenset({"d", "f", "i", "I", "r", "R", "v", "W"})
_ENV_SHORT_OPTIONS = frozenset({"0", "i", "v"})
_ENV_SHORT_OPTIONS_WITH_VALUE = frozenset({"C", "S", "u"})
_ENV_LONG_OPTIONS = frozenset({"debug", "ignore-environment", "null"})
_ENV_LONG_OPTIONS_WITH_VALUE = frozenset({"chdir", "split-string", "unset"})
_DOCKER_SHORT_OPTIONS = frozenset({"D", "v"})
_DOCKER_SHORT_OPTIONS_WITH_VALUE = frozenset({"c", "H", "l"})
_DOCKER_LONG_OPTIONS = frozenset({"debug", "help", "tls", "tlsverify", "version"})
_DOCKER_LONG_OPTIONS_WITH_VALUE = frozenset(
    {"config", "context", "host", "log-level", "tlscacert", "tlscert", "tlskey"}
)
_COMPOSE_SHORT_OPTIONS: frozenset[str] = frozenset()
_COMPOSE_SHORT_OPTIONS_WITH_VALUE = frozenset({"f", "p"})
_COMPOSE_LONG_OPTIONS = frozenset({"compatibility", "dry-run", "help", "verbose", "version"})
_COMPOSE_LONG_OPTIONS_WITH_VALUE = frozenset(
    {
        "ansi",
        "env-file",
        "file",
        "parallel",
        "profile",
        "progress",
        "project-directory",
        "project-name",
    }
)
_COMPOSE_RESTART_COMMANDS = frozenset(
    {"down", "kill", "pause", "restart", "rm", "start", "stop", "unpause", "up"}
)
_DOCKER_LIFECYCLE_COMMANDS = frozenset(
    {"kill", "pause", "restart", "rm", "start", "stop", "unpause"}
)
_TIMEOUT_SHORT_OPTIONS = frozenset({"f", "p", "v"})
_TIMEOUT_SHORT_OPTIONS_WITH_VALUE = frozenset({"k", "s"})
_TIMEOUT_LONG_OPTIONS = frozenset({"foreground", "preserve-status", "verbose"})
_TIMEOUT_LONG_OPTIONS_WITH_VALUE = frozenset({"kill-after", "signal"})
_WATCH_SHORT_OPTIONS = frozenset({"b", "d", "e", "g", "p", "q", "t", "x"})
_WATCH_SHORT_OPTIONS_WITH_VALUE = frozenset({"n"})
_WATCH_LONG_OPTIONS = frozenset(
    {"beep", "chgexit", "differences", "errexit", "exec", "equexit", "no-title", "precise"}
)
_WATCH_LONG_OPTIONS_WITH_VALUE = frozenset({"interval"})
_SETSID_SHORT_OPTIONS = frozenset({"c", "f", "w"})
_SETSID_SHORT_OPTIONS_WITH_VALUE: frozenset[str] = frozenset()
_SETSID_LONG_OPTIONS = frozenset({"ctty", "fork", "wait"})
_SETSID_LONG_OPTIONS_WITH_VALUE: frozenset[str] = frozenset()
_STDBUF_SHORT_OPTIONS: frozenset[str] = frozenset()
_STDBUF_SHORT_OPTIONS_WITH_VALUE = frozenset({"e", "i", "o"})
_STDBUF_LONG_OPTIONS: frozenset[str] = frozenset()
_STDBUF_LONG_OPTIONS_WITH_VALUE = frozenset({"error", "input", "output"})
_NICE_SHORT_OPTIONS: frozenset[str] = frozenset()
_NICE_SHORT_OPTIONS_WITH_VALUE = frozenset({"n"})
_NICE_LONG_OPTIONS: frozenset[str] = frozenset()
_NICE_LONG_OPTIONS_WITH_VALUE = frozenset({"adjustment"})
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
_XARGS_SHORT_OPTIONS = frozenset({"0", "o", "p", "r", "t", "x"})
_XARGS_SHORT_OPTIONS_WITH_VALUE = frozenset({"E", "I", "L", "P", "a", "d", "n", "s"})
_XARGS_LONG_OPTIONS = frozenset(
    {"exit", "interactive", "no-run-if-empty", "null", "open-tty", "verbose"}
)
_XARGS_LONG_OPTIONS_WITH_VALUE = frozenset(
    {
        "arg-file",
        "delimiter",
        "eof",
        "max-args",
        "max-chars",
        "max-lines",
        "max-procs",
        "replace",
    }
)


class ShellCommandDisposition(StrEnum):
    """How OpenTulpa may handle one model-generated shell command."""

    ALLOW = "allow"
    REQUIRE_APPROVAL = "require_approval"
    REJECT = "reject"


def classify_shell_command(command: str) -> ShellCommandDisposition:
    """Classify risky shell actions without treating quoted text as execution."""

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
    if executable in {"docker", "docker-compose"}:
        return _classify_docker_command(executable, arguments)
    if executable == "timeout":
        return _classify_timeout(arguments)
    if executable == "watch":
        return _classify_argv_wrapper(
            arguments,
            short_options=_WATCH_SHORT_OPTIONS,
            short_options_with_value=_WATCH_SHORT_OPTIONS_WITH_VALUE,
            long_options=_WATCH_LONG_OPTIONS,
            long_options_with_value=_WATCH_LONG_OPTIONS_WITH_VALUE,
        )
    if executable == "setsid":
        return _classify_argv_wrapper(
            arguments,
            short_options=_SETSID_SHORT_OPTIONS,
            short_options_with_value=_SETSID_SHORT_OPTIONS_WITH_VALUE,
            long_options=_SETSID_LONG_OPTIONS,
            long_options_with_value=_SETSID_LONG_OPTIONS_WITH_VALUE,
        )
    if executable == "stdbuf":
        return _classify_argv_wrapper(
            arguments,
            short_options=_STDBUF_SHORT_OPTIONS,
            short_options_with_value=_STDBUF_SHORT_OPTIONS_WITH_VALUE,
            long_options=_STDBUF_LONG_OPTIONS,
            long_options_with_value=_STDBUF_LONG_OPTIONS_WITH_VALUE,
        )
    if executable == "time":
        return _classify_time(arguments)
    if executable == "nice":
        return _classify_argv_wrapper(
            arguments,
            short_options=_NICE_SHORT_OPTIONS,
            short_options_with_value=_NICE_SHORT_OPTIONS_WITH_VALUE,
            long_options=_NICE_LONG_OPTIONS,
            long_options_with_value=_NICE_LONG_OPTIONS_WITH_VALUE,
        )
    if executable == "flock":
        return _classify_flock(arguments)
    if executable in _SHELL_COMMANDS:
        return _classify_shell_script(arguments)
    if executable == "eval":
        return _classify_literal_script(arguments)
    if executable == "xargs":
        return _classify_xargs(arguments)
    if executable == "find":
        return _classify_find(arguments)
    if executable == "busybox":
        return _classify_wrapped(executable, arguments)
    if executable in _COMMAND_STRING_WRAPPERS:
        return _classify_command_string_wrapper(arguments)
    if executable in _LITERAL_DATA_COMMANDS:
        return ShellCommandDisposition.ALLOW
    return ShellCommandDisposition.ALLOW


def _classify_rm_arguments(arguments: list[str | None]) -> ShellCommandDisposition:
    recursive = False
    force = False
    ambiguous_option = False
    for argument in arguments:
        if argument is None:
            ambiguous_option = True
            continue
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
    if ambiguous_option:
        return ShellCommandDisposition.REJECT
    return ShellCommandDisposition.ALLOW


def _classify_wrapped(
    executable: str,
    arguments: list[str | None],
) -> ShellCommandDisposition:
    remaining = list(arguments)
    if executable == "command" and any(argument in {"-V", "-v"} for argument in remaining):
        return ShellCommandDisposition.ALLOW
    if executable == "exec":
        parsed = _consume_options(
            remaining,
            short_options=frozenset({"c", "l"}),
            short_options_with_value=frozenset({"a"}),
            long_options=frozenset(),
            long_options_with_value=frozenset(),
        )
        if parsed is None:
            return ShellCommandDisposition.REJECT
        remaining = parsed
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
        return _classify_env_arguments(arguments)
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


def _classify_env_arguments(arguments: list[str | None]) -> ShellCommandDisposition:
    remaining = list(arguments)
    while remaining:
        argument = remaining.pop(0)
        if argument is None:
            return ShellCommandDisposition.REJECT
        if argument == "--":
            break
        if "=" in argument and not argument.startswith("-"):
            continue
        if not argument.startswith("-") or argument == "-":
            return _classify_words(argument, remaining)
        split_string: str | None = None
        if argument in {"-S", "--split-string"}:
            if not remaining or remaining[0] is None:
                return ShellCommandDisposition.REJECT
            split_string = remaining.pop(0)
        elif argument.startswith("--"):
            option, separator, option_value = argument[2:].partition("=")
            if option == "split-string":
                if separator:
                    split_string = option_value
                elif remaining and remaining[0] is not None:
                    split_string = remaining.pop(0)
                else:
                    return ShellCommandDisposition.REJECT
            elif option in _ENV_LONG_OPTIONS:
                if separator:
                    return ShellCommandDisposition.REJECT
                continue
            elif option in _ENV_LONG_OPTIONS_WITH_VALUE:
                if not separator and (not remaining or remaining.pop(0) is None):
                    return ShellCommandDisposition.REJECT
                continue
            else:
                return ShellCommandDisposition.REJECT
        else:
            flags = argument[1:]
            for index, flag in enumerate(flags):
                if flag in _ENV_SHORT_OPTIONS:
                    continue
                if flag not in _ENV_SHORT_OPTIONS_WITH_VALUE:
                    return ShellCommandDisposition.REJECT
                if index != len(flags) - 1:
                    return ShellCommandDisposition.REJECT
                if not remaining or remaining[0] is None:
                    return ShellCommandDisposition.REJECT
                value = remaining.pop(0)
                if value is None:
                    return ShellCommandDisposition.REJECT
                if flag == "S":
                    split_string = value
                break
        if split_string is None:
            continue
        split_arguments = _split_shell_words(split_string)
        if split_arguments is None:
            return ShellCommandDisposition.REJECT
        remaining = [*split_arguments, *remaining]
    if not remaining:
        return ShellCommandDisposition.ALLOW
    name = remaining.pop(0)
    if name is None:
        return ShellCommandDisposition.REJECT
    return _classify_words(name, remaining)


def _classify_docker_command(
    executable: str,
    arguments: list[str | None],
) -> ShellCommandDisposition:
    if executable == "docker-compose":
        return _classify_docker_compose_arguments(arguments)
    remaining = _consume_options(
        arguments,
        short_options=_DOCKER_SHORT_OPTIONS,
        short_options_with_value=_DOCKER_SHORT_OPTIONS_WITH_VALUE,
        long_options=_DOCKER_LONG_OPTIONS,
        long_options_with_value=_DOCKER_LONG_OPTIONS_WITH_VALUE,
    )
    if remaining is None:
        return ShellCommandDisposition.REJECT
    if not remaining:
        return ShellCommandDisposition.ALLOW
    subcommand = remaining.pop(0)
    if subcommand is None:
        return ShellCommandDisposition.REJECT
    if subcommand == "{}":
        return ShellCommandDisposition.REJECT
    docker_command = PurePosixPath(subcommand).name.casefold()
    if docker_command in _DOCKER_LIFECYCLE_COMMANDS:
        return ShellCommandDisposition.REJECT
    if docker_command == "container":
        return _classify_docker_container_arguments(remaining)
    if docker_command != "compose":
        return ShellCommandDisposition.ALLOW
    return _classify_docker_compose_arguments(remaining)


def _classify_docker_compose_arguments(
    arguments: list[str | None],
) -> ShellCommandDisposition:
    remaining = _consume_options(
        arguments,
        short_options=_COMPOSE_SHORT_OPTIONS,
        short_options_with_value=_COMPOSE_SHORT_OPTIONS_WITH_VALUE,
        long_options=_COMPOSE_LONG_OPTIONS,
        long_options_with_value=_COMPOSE_LONG_OPTIONS_WITH_VALUE,
    )
    if remaining is None:
        return ShellCommandDisposition.REJECT
    if not remaining:
        return ShellCommandDisposition.ALLOW
    command = remaining[0]
    if command is None:
        return ShellCommandDisposition.REJECT
    if command == "{}":
        return ShellCommandDisposition.REJECT
    if command.casefold() in _COMPOSE_RESTART_COMMANDS:
        return ShellCommandDisposition.REJECT
    return ShellCommandDisposition.ALLOW


def _classify_docker_container_arguments(
    arguments: list[str | None],
) -> ShellCommandDisposition:
    if not arguments:
        return ShellCommandDisposition.ALLOW
    command = arguments[0]
    if command is None or command == "{}":
        return ShellCommandDisposition.REJECT
    if command.casefold() in _DOCKER_LIFECYCLE_COMMANDS:
        return ShellCommandDisposition.REJECT
    return ShellCommandDisposition.ALLOW


def _docker_command_prefix_complete(executable: str, arguments: list[str | None]) -> bool:
    if executable == "docker-compose":
        remaining = _consume_options(
            arguments,
            short_options=_COMPOSE_SHORT_OPTIONS,
            short_options_with_value=_COMPOSE_SHORT_OPTIONS_WITH_VALUE,
            long_options=_COMPOSE_LONG_OPTIONS,
            long_options_with_value=_COMPOSE_LONG_OPTIONS_WITH_VALUE,
        )
        return bool(remaining)
    remaining = _consume_options(
        arguments,
        short_options=_DOCKER_SHORT_OPTIONS,
        short_options_with_value=_DOCKER_SHORT_OPTIONS_WITH_VALUE,
        long_options=_DOCKER_LONG_OPTIONS,
        long_options_with_value=_DOCKER_LONG_OPTIONS_WITH_VALUE,
    )
    if not remaining:
        return False
    if remaining[0] is None or PurePosixPath(remaining[0]).name != "compose":
        return True
    compose = _consume_options(
        remaining[1:],
        short_options=_COMPOSE_SHORT_OPTIONS,
        short_options_with_value=_COMPOSE_SHORT_OPTIONS_WITH_VALUE,
        long_options=_COMPOSE_LONG_OPTIONS,
        long_options_with_value=_COMPOSE_LONG_OPTIONS_WITH_VALUE,
    )
    return bool(compose)


def _classify_command_string_wrapper(arguments: list[str | None]) -> ShellCommandDisposition:
    handled, disposition = _classify_command_string_options(arguments)
    return disposition if handled else ShellCommandDisposition.ALLOW


def _classify_command_string_options(
    arguments: list[str | None],
) -> tuple[bool, ShellCommandDisposition]:
    handled = False
    disposition = ShellCommandDisposition.ALLOW
    for index, argument in enumerate(arguments):
        if argument is None:
            return True, ShellCommandDisposition.REJECT
        command = ""
        if argument in {"-c", "--command", "--session-command"}:
            if index + 1 >= len(arguments) or arguments[index + 1] is None:
                return True, ShellCommandDisposition.REJECT
            next_argument = arguments[index + 1]
            if next_argument is None:
                return True, ShellCommandDisposition.REJECT
            command = next_argument
        elif argument.startswith(("--command=", "--session-command=")):
            command = argument.partition("=")[2]
        elif argument.startswith("-") and not argument.startswith("--") and "c" in argument[1:]:
            if argument != "-c":
                return True, ShellCommandDisposition.REJECT
            if index + 1 >= len(arguments) or arguments[index + 1] is None:
                return True, ShellCommandDisposition.REJECT
            next_argument = arguments[index + 1]
            if next_argument is None:
                return True, ShellCommandDisposition.REJECT
            command = next_argument
        if not command:
            continue
        handled = True
        current = classify_shell_command(command)
        if current is ShellCommandDisposition.REJECT:
            return True, current
        if current is ShellCommandDisposition.REQUIRE_APPROVAL:
            disposition = current
    return handled, disposition


def _classify_timeout(arguments: list[str | None]) -> ShellCommandDisposition:
    return _classify_argv_wrapper(
        arguments,
        short_options=_TIMEOUT_SHORT_OPTIONS,
        short_options_with_value=_TIMEOUT_SHORT_OPTIONS_WITH_VALUE,
        long_options=_TIMEOUT_LONG_OPTIONS,
        long_options_with_value=_TIMEOUT_LONG_OPTIONS_WITH_VALUE,
        leading_operands=1,
    )


def _classify_argv_wrapper(
    arguments: list[str | None],
    *,
    short_options: frozenset[str],
    short_options_with_value: frozenset[str],
    long_options: frozenset[str],
    long_options_with_value: frozenset[str],
    leading_operands: int = 0,
) -> ShellCommandDisposition:
    remaining = _consume_options(
        arguments,
        short_options=short_options,
        short_options_with_value=short_options_with_value,
        long_options=long_options,
        long_options_with_value=long_options_with_value,
    )
    if remaining is None:
        return ShellCommandDisposition.REJECT
    for _ in range(max(0, leading_operands)):
        if not remaining:
            return ShellCommandDisposition.ALLOW
        if remaining.pop(0) is None:
            return ShellCommandDisposition.REJECT
    if not remaining:
        return ShellCommandDisposition.ALLOW
    name = remaining.pop(0)
    if name is None:
        return ShellCommandDisposition.REJECT
    return _classify_words(name, remaining)


def _classify_time(arguments: list[str | None]) -> ShellCommandDisposition:
    remaining = list(arguments)
    while remaining:
        argument = remaining[0]
        if argument is None:
            return ShellCommandDisposition.REJECT
        if argument == "--":
            remaining.pop(0)
            break
        if not argument.startswith("-") or argument == "-":
            break
        remaining.pop(0)
        if argument.startswith("--"):
            option, separator, _ = argument[2:].partition("=")
            if option in {"format", "output"} and not separator and (
                not remaining or remaining.pop(0) is None
            ):
                return ShellCommandDisposition.REJECT
            continue
        flags = argument[1:]
        if flags[:1] in {"f", "o"} and len(flags) == 1 and (
            not remaining or remaining.pop(0) is None
        ):
            return ShellCommandDisposition.REJECT
    if not remaining:
        return ShellCommandDisposition.ALLOW
    name = remaining.pop(0)
    if name is None:
        return ShellCommandDisposition.REJECT
    return _classify_words(name, remaining)


def _classify_flock(arguments: list[str | None]) -> ShellCommandDisposition:
    handled, disposition = _classify_command_string_options(arguments)
    if handled:
        return disposition
    remaining = _consume_flock_options(arguments)
    if remaining is None:
        return ShellCommandDisposition.REJECT
    if not remaining:
        return ShellCommandDisposition.ALLOW
    if remaining.pop(0) is None:
        return ShellCommandDisposition.REJECT
    if not remaining:
        return ShellCommandDisposition.ALLOW
    name = remaining.pop(0)
    if name is None:
        return ShellCommandDisposition.REJECT
    return _classify_words(name, remaining)


def _consume_flock_options(arguments: list[str | None]) -> list[str | None] | None:
    remaining = list(arguments)
    while remaining:
        argument = remaining[0]
        if argument is None:
            return None
        if argument == "--":
            return remaining[1:]
        if not argument.startswith("-") or argument == "-":
            return remaining
        remaining.pop(0)
        if argument in {"-E", "-w", "--conflict-exit-code", "--timeout"}:
            if not remaining or remaining.pop(0) is None:
                return None
            continue
        if argument.startswith(("--conflict-exit-code=", "--timeout=")):
            continue
        if argument.startswith("--"):
            continue
        if any(flag in {"E", "w"} for flag in argument[1:]) and argument not in {"-E", "-w"}:
            return None
    return remaining


def _classify_shell_script(arguments: list[str | None]) -> ShellCommandDisposition:
    for index, argument in enumerate(arguments):
        try:
            uses_command_string = _shell_option_uses_command_string(argument)
        except ValueError:
            return ShellCommandDisposition.REJECT
        if argument == "-c" or uses_command_string:
            if index + 1 >= len(arguments):
                return ShellCommandDisposition.REJECT
            script = arguments[index + 1]
            if script is None:
                return ShellCommandDisposition.REJECT
            return classify_shell_command(script)
    literal_arguments = [argument for argument in arguments if argument is not None]
    script_paths = [argument for argument in literal_arguments if not argument.startswith("-")]
    if any(path not in {"-", "/dev/stdin", "/proc/self/fd/0"} for path in script_paths):
        return ShellCommandDisposition.ALLOW
    return ShellCommandDisposition.REJECT


def _shell_option_uses_command_string(argument: str | None) -> bool:
    if argument is None or not argument.startswith("-") or argument.startswith("--"):
        return False
    flags = argument[1:]
    if "c" not in flags:
        return False
    if not flags.endswith("c"):
        raise ValueError("shell -c command string must be a separate argument")
    return True


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
    if _xargs_uses_replacement(arguments):
        return ShellCommandDisposition.REJECT
    remaining = _consume_options(
        arguments,
        short_options=_XARGS_SHORT_OPTIONS,
        short_options_with_value=_XARGS_SHORT_OPTIONS_WITH_VALUE,
        long_options=_XARGS_LONG_OPTIONS,
        long_options_with_value=_XARGS_LONG_OPTIONS_WITH_VALUE,
    )
    if remaining is None:
        return ShellCommandDisposition.REJECT
    if not remaining:
        return ShellCommandDisposition.ALLOW
    name = remaining.pop(0)
    if name is None:
        return ShellCommandDisposition.REJECT
    command_words = [name, *remaining]
    rm_disposition = _xargs_rm_disposition(command_words)
    if rm_disposition is not ShellCommandDisposition.ALLOW:
        return rm_disposition
    if _contains_incomplete_docker_prefix(command_words):
        return ShellCommandDisposition.REJECT
    return _classify_words(name, remaining)


def _xargs_rm_disposition(arguments: list[str | None]) -> ShellCommandDisposition:
    if arguments and arguments[0] is not None:
        first = PurePosixPath(arguments[0]).name
        if first in _LITERAL_DATA_COMMANDS:
            return ShellCommandDisposition.ALLOW
    for index, argument in enumerate(arguments):
        if argument is None or PurePosixPath(argument).name != "rm":
            continue
        rm_arguments = arguments[index + 1 :]
        disposition = _classify_rm_arguments(rm_arguments)
        if disposition is ShellCommandDisposition.REQUIRE_APPROVAL:
            return disposition
        if "--" not in rm_arguments:
            return ShellCommandDisposition.REQUIRE_APPROVAL
    return ShellCommandDisposition.ALLOW


def _contains_incomplete_docker_prefix(arguments: list[str | None]) -> bool:
    for index, argument in enumerate(arguments):
        if argument is None:
            continue
        executable = PurePosixPath(argument).name
        if executable in {"docker", "docker-compose"} and not _docker_command_prefix_complete(
            executable,
            arguments[index + 1 :],
        ):
            return True
    return False


def _xargs_uses_replacement(arguments: list[str | None]) -> bool:
    for argument in arguments:
        if argument is None:
            return True
        if argument == "--":
            return False
        if not argument.startswith("-") or argument == "-":
            return False
        if argument.startswith("--"):
            option = argument[2:].partition("=")[0]
            if option == "replace":
                return True
            continue
        flags = argument[1:]
        if "I" in flags or "i" in flags:
            return True
    return False


def _classify_find(arguments: list[str | None]) -> ShellCommandDisposition:
    disposition = ShellCommandDisposition.ALLOW
    for index, argument in enumerate(arguments):
        if argument not in {"-exec", "-execdir", "-ok", "-okdir"}:
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


def _split_shell_words(value: str) -> list[str] | None:
    try:
        return shlex.split(value, comments=False, posix=True)
    except ValueError:
        return None


def _literal_word(node: Node, source: bytes) -> str | None:
    if any(
        descendant.type in _DYNAMIC_WORD_NODES
        for descendant in _walk(node)
    ):
        return None
    raw = source[node.start_byte : node.end_byte].decode().replace("\\\n", "")
    if "$'" in raw or '$"' in raw:
        return None
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


def _walk(node: Node) -> Iterator[Node]:
    yield node
    for child in node.children:
        yield from _walk(child)


__all__ = ["ShellCommandDisposition", "classify_shell_command"]
