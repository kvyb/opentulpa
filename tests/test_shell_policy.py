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
        "time rm -rf build",
        "/usr/bin/time -l rm -rf build",
        "printf '%s\n' -rf build | xargs rm",
        "nice -n 5 rm -rf build",
        "watch rm -rf build",
        "setsid rm -rf build",
        "stdbuf -oL rm -rf build",
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


@pytest.mark.parametrize(
    "command",
    [
        "docker compose up -d",
        "docker compose stop opentulpa",
        "docker compose start opentulpa",
        "docker compose kill opentulpa",
        "docker compose rm opentulpa",
        "docker restart opentulpa",
        "docker stop opentulpa",
        "docker kill opentulpa",
        "docker rm opentulpa",
        "docker container restart opentulpa",
        "docker compose $'up' -d",
        "docker $'compose' up -d",
        "$'docker' compose restart opentulpa",
        "cd /opt/opentulpa && docker compose up -d",
        "docker compose --profile app restart",
        "docker --context prod compose -f docker-compose.yml down",
        "docker-compose up -d",
        "sudo docker compose restart opentulpa",
        "env -S 'docker compose up -d'",
        "bash -c 'docker compose restart opentulpa'",
        "bash -lc 'docker compose restart opentulpa'",
        "bash -cdocker\\ compose\\ up\\ -d",
        "timeout 10 bash -c 'docker compose up -d'",
        "time docker compose up -d",
        "/usr/bin/time -- docker compose up -d",
        "sudo timeout 10 bash -c 'docker compose down'",
        "timeout 10 env -S 'docker compose up -d'",
        "printf '%s\n' opentulpa | xargs docker compose restart",
        "printf '%s\n' 'up -d' | xargs docker compose",
        "printf '%s\n' 'compose up -d' | xargs docker",
        "printf '%s\n' 'up -d' | xargs sudo docker compose",
        "printf '%s\n' 'up -d' | xargs env docker compose",
        "printf '%s\n' 'docker compose up -d' | xargs -I{} sh -c '{}'",
        "printf '%s\n' 'up -d' | xargs -I{} docker compose {}",
        "find . -exec docker compose up -d {} +",
        "find . -ok docker compose up -d {} ';'",
        "find up -maxdepth 0 -exec docker compose {} -d ';'",
        "find compose -maxdepth 0 -exec docker {} up -d ';'",
        "timeout 10 docker compose up -d",
        "watch docker compose up -d",
        "setsid docker compose down",
        "stdbuf -oL docker compose restart opentulpa",
        "flock /tmp/rm docker compose up -d",
        "flock /tmp/lock -c 'docker compose up -d'",
        "su -c 'docker compose down'",
        "su --session-command='docker compose down'",
        "script -q /dev/null -c 'docker compose restart opentulpa'",
        "script -cdocker\\ compose\\ up\\ -d /dev/null",
    ],
)
def test_docker_compose_lifecycle_commands_are_rejected(command: str) -> None:
    assert classify_shell_command(command) is ShellCommandDisposition.REJECT


@pytest.mark.parametrize(
    "command",
    [
        "docker ps",
        "docker logs opentulpa",
        "docker compose ps",
        "docker compose logs opentulpa",
        "docker compose config",
        "docker compose build opentulpa",
        "docker-compose logs opentulpa",
        "watch docker compose ps",
        "setsid docker ps",
        "stdbuf -oL docker logs opentulpa",
        "bash -lc 'docker compose ps'",
        "git -C docker compose up -d",
        "make docker compose up -d",
        "my-tool docker compose up -d",
        "grep -R 'docker compose up -d' .",
        "printf '%s\n' 'docker compose restart opentulpa'",
        "echo docker compose up -d",
    ],
)
def test_non_lifecycle_docker_compose_commands_are_allowed(command: str) -> None:
    assert classify_shell_command(command) is ShellCommandDisposition.ALLOW
