from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_installer_uses_editable_source_and_creates_global_command(tmp_path: Path) -> None:
    (tmp_path / "home").mkdir()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    tool_bin = tmp_path / "tools"
    tool_bin.mkdir()
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "pyproject.toml").write_text("[project]\nname='opentulpa'\n", encoding="utf-8")
    calls = tmp_path / "uv-calls"
    uv = fake_bin / "uv"
    uv.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$*\" >> '{calls}'\n"
        "if [ \"$1\" = 'sync' ]; then\n"
        "  mkdir -p \"$5/.venv/bin\"\n"
        "  printf '#!/bin/sh\\n' > \"$5/.venv/bin/opentulpa\"\n"
        "  chmod +x \"$5/.venv/bin/opentulpa\"\n"
        "fi\n",
        encoding="utf-8",
    )
    uv.chmod(0o755)

    result = subprocess.run(
        ["sh", str(REPO_ROOT / "install.sh")],
        env={
            **os.environ,
            "HOME": str(tmp_path / "home"),
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "SHELL": "/bin/sh",
            "OPENTULPA_INSTALL_SOURCE": str(source_root),
            "OPENTULPA_BIN_DIR": str(tool_bin),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert f"sync --locked --no-dev --project {source_root}" in calls.read_text(encoding="utf-8")
    assert "OpenTulpa installed" in result.stdout
    assert "opentulpa" in result.stdout
