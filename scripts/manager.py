import asyncio
import os
import re
import secrets
import signal
import subprocess
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = REPO_ROOT / ".opentulpa" / "logs"
APP_LOG = LOG_DIR / "app.log"
TUNNEL_LOG = LOG_DIR / "cloudflared.log"

# Hardcoded runtime policy.
STARTUP_WAIT_SECONDS = 180
WEBHOOK_SYNC_ATTEMPTS = 12
TUNNEL_URL_POLL_ATTEMPTS = 60
TUNNEL_RECOVER_ATTEMPTS = 3
TUNNEL_DNS_WARMUP_SECONDS = 20


def load_dotenv(repo_root: Path) -> None:
    env_path = repo_root / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip().strip('"').strip("'")


class TulpaManager:
    def __init__(self) -> None:
        self.app_proc: subprocess.Popen | None = None
        self.tunnel_proc: subprocess.Popen | None = None
        self.stopping = False
        self.app_port = int(os.environ.get("PORT", "8000"))
        self.app_host = str(os.environ.get("HOST", "127.0.0.1")).strip() or "127.0.0.1"

    def log(self, msg: str) -> None:
        print(f"[manager] {msg}")

    def error(self, msg: str) -> None:
        print(f"[error] {msg}", file=sys.stderr)

    @property
    def health_url(self) -> str:
        return f"http://127.0.0.1:{self.app_port}/healthz"

    @property
    def tunnel_target_url(self) -> str:
        return f"http://localhost:{self.app_port}"

    def cleanup_stale_processes(self) -> None:
        self.log("cleaning up stale processes...")
        try:
            result = subprocess.run(
                ["lsof", "-t", f"-iTCP:{self.app_port}", "-sTCP:LISTEN"],
                capture_output=True,
                text=True,
            )
            for pid in result.stdout.strip().split():
                if not pid:
                    continue
                self.log(f"killing process {pid} on port {self.app_port}")
                subprocess.run(["kill", "-9", pid], check=False)
        except Exception as exc:
            self.log(f"could not check port {self.app_port}: {exc}")

    def rotate_logs(self) -> None:
        self.log("rotating logs...")
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        for log_path in (APP_LOG, TUNNEL_LOG):
            if log_path.exists():
                log_path.replace(log_path.with_suffix(".log.old"))

    def build_app_env(self) -> dict[str, str]:
        app_env = os.environ.copy()
        if str(app_env.get("TELEGRAM_BOT_TOKEN", "")).strip() and not str(
            app_env.get("TELEGRAM_WEBHOOK_SECRET", "")
        ).strip():
            app_env["TELEGRAM_WEBHOOK_SECRET"] = secrets.token_urlsafe(24)
            self.log("generated ephemeral TELEGRAM_WEBHOOK_SECRET for this run.")
        if not str(app_env.get("HOST", "")).strip():
            app_env["HOST"] = self.app_host
            self.log(f"defaulted HOST={self.app_host} for local-only app binding.")

        src_dir = str((REPO_ROOT / "src").resolve())
        existing_pythonpath = app_env.get("PYTHONPATH", "")
        app_env["PYTHONPATH"] = (
            f"{src_dir}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else src_dir
        )

        app_env["OPENAI_MAX_RETRIES"] = "10"
        app_env["HTTPX_TIMEOUT"] = "120.0"
        return app_env

    def launch_app(self, app_env: dict[str, str]) -> None:
        self.log("launching OpenTulpa app...")
        with APP_LOG.open("w") as app_log_file:
            self.app_proc = subprocess.Popen(
                [sys.executable, "-m", "opentulpa"],
                env=app_env,
                stdout=app_log_file,
                stderr=subprocess.STDOUT,
                cwd=REPO_ROOT,
            )

    async def wait_for_app_health(self) -> bool:
        self.log("waiting for app to be healthy...")
        for _ in range(STARTUP_WAIT_SECONDS):
            if self.app_proc is not None and self.app_proc.poll() is not None:
                self.error("app exited early. check app.log")
                return False
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(self.health_url)
                    if response.status_code == 200:
                        return True
            except Exception:
                pass
            await asyncio.sleep(1)
        self.error("app health check timed out.")
        return False

    def launch_tunnel(self) -> None:
        self.log("launching Cloudflare tunnel...")
        with TUNNEL_LOG.open("w") as tunnel_log_file:
            self.tunnel_proc = subprocess.Popen(
                ["cloudflared", "tunnel", "--url", self.tunnel_target_url],
                stdout=tunnel_log_file,
                stderr=subprocess.STDOUT,
            )

    async def extract_tunnel_url(self) -> str | None:
        self.log("extracting tunnel URL...")
        pattern = re.compile(r"https://[a-zA-Z0-9.-]+\.trycloudflare\.com")
        for _ in range(TUNNEL_URL_POLL_ATTEMPTS):
            if self.tunnel_proc is not None and self.tunnel_proc.poll() is not None:
                self.error("tunnel exited early. check cloudflared.log")
                return None
            if TUNNEL_LOG.exists():
                content = TUNNEL_LOG.read_text()
                match = pattern.search(content)
                if match:
                    return match.group(0)
            await asyncio.sleep(0.5)
        return None

    @staticmethod
    def _safe_json(response: Any) -> dict[str, Any]:
        try:
            return response.json() if getattr(response, "content", b"") else {}
        except Exception:
            return {}

    async def sync_webhook(self, *, bot_token: str, secret: str | None, tunnel_url: str) -> bool:
        webhook_url = f"{tunnel_url}/webhook/telegram"
        self.log(f"syncing telegram webhook to {webhook_url}...")
        base = f"https://api.telegram.org/bot{bot_token}"

        async with httpx.AsyncClient() as client:
            for attempt in range(1, WEBHOOK_SYNC_ATTEMPTS + 1):
                data: dict[str, str] = {"url": webhook_url}
                if secret:
                    data["secret_token"] = secret

                try:
                    set_resp = await client.post(f"{base}/setWebhook", data=data, timeout=15.0)
                    set_payload = self._safe_json(set_resp)
                except Exception as exc:
                    self.error(f"setWebhook attempt {attempt}/{WEBHOOK_SYNC_ATTEMPTS} failed: {exc}")
                    await asyncio.sleep(min(2.0 * attempt, 10.0))
                    continue

                if not bool(set_payload.get("ok")):
                    description = str(set_payload.get("description", "")).strip()
                    self.error(
                        f"setWebhook attempt {attempt}/{WEBHOOK_SYNC_ATTEMPTS} returned error: "
                        f"{str(set_payload)[:200]}"
                    )
                    if "Failed to resolve host" in description:
                        await asyncio.sleep(min(4.0 * attempt, 20.0))
                    else:
                        await asyncio.sleep(min(2.0 * attempt, 10.0))
                    continue

                try:
                    info_resp = await client.get(f"{base}/getWebhookInfo", timeout=15.0)
                    info_payload = self._safe_json(info_resp)
                except Exception as exc:
                    self.error(
                        f"getWebhookInfo attempt {attempt}/{WEBHOOK_SYNC_ATTEMPTS} failed: {exc}"
                    )
                    await asyncio.sleep(min(2.0 * attempt, 10.0))
                    continue

                result = info_payload.get("result", {}) if isinstance(info_payload, dict) else {}
                live_url = str(result.get("url", "")).strip()
                last_error = str(result.get("last_error_message", "")).strip()
                pending = result.get("pending_update_count")

                if live_url == webhook_url and not last_error:
                    self.log(f"webhook synced (pending updates: {pending}).")
                    return True

                self.error(
                    "webhook verification failed "
                    f"attempt {attempt}/{WEBHOOK_SYNC_ATTEMPTS}: "
                    f"url={live_url or '<empty>'}, "
                    f"last_error={last_error or '<none>'}, pending={pending}"
                )
                await asyncio.sleep(min(2.0 * attempt, 10.0))

        return False

    async def recover_tunnel_and_webhook(self, app_env: dict[str, str]) -> str | None:
        bot_token = str(app_env.get("TELEGRAM_BOT_TOKEN", "")).strip()
        secret = str(app_env.get("TELEGRAM_WEBHOOK_SECRET", "")).strip() or None
        tunnel_url: str | None = None

        for attempt in range(1, TUNNEL_RECOVER_ATTEMPTS + 1):
            self.log(f"tunnel recovery attempt {attempt}/{TUNNEL_RECOVER_ATTEMPTS}...")
            needs_fresh_tunnel = self.tunnel_proc is None or self.tunnel_proc.poll() is not None
            if needs_fresh_tunnel or not tunnel_url:
                if self.tunnel_proc is not None and self.tunnel_proc.poll() is None:
                    with suppress(Exception):
                        self.tunnel_proc.terminate()
                    with suppress(Exception):
                        self.tunnel_proc.wait(timeout=3)
                self.tunnel_proc = None
                self.launch_tunnel()
                tunnel_url = await self.extract_tunnel_url()
                if not tunnel_url:
                    self.error("recovery: could not detect tunnel URL.")
                    continue
                self.log(f"recovery: tunnel live: {tunnel_url}")
                self.log(f"waiting {TUNNEL_DNS_WARMUP_SECONDS}s for tunnel DNS propagation...")
                await asyncio.sleep(TUNNEL_DNS_WARMUP_SECONDS)

            if not bot_token:
                return tunnel_url

            if await self.sync_webhook(bot_token=bot_token, secret=secret, tunnel_url=tunnel_url):
                return tunnel_url

            self.error("recovery: webhook sync failed.")
            await asyncio.sleep(min(3.0 * attempt, 10.0))

        return None

    async def run(self) -> None:
        self.cleanup_stale_processes()
        self.rotate_logs()

        app_env = self.build_app_env()
        self.launch_app(app_env)

        if not await self.wait_for_app_health():
            self.stop()
            return

        tunnel_url = await self.recover_tunnel_and_webhook(app_env)
        if not tunnel_url:
            self.error("failed to establish healthy tunnel+webhook state.")
            self.stop()
            return

        self.log("--- OpenTulpa is live ---")
        self.log(f"Tunnel URL: {tunnel_url}")
        self.log("Press Ctrl+C to shutdown.")

        while not self.stopping:
            if self.app_proc is not None and self.app_proc.poll() is not None:
                self.error("app process died.")
                break
            if self.tunnel_proc is not None and self.tunnel_proc.poll() is not None:
                self.error("tunnel process died; attempting recovery.")
                recovered_url = await self.recover_tunnel_and_webhook(app_env)
                if not recovered_url:
                    self.error("tunnel recovery failed.")
                    break
                tunnel_url = recovered_url
                self.log(f"tunnel recovered: {tunnel_url}")
            await asyncio.sleep(5)

        self.stop()

    def stop(self) -> None:
        if self.stopping:
            return
        self.stopping = True
        self.log("shutting down processes...")
        for proc in (self.tunnel_proc, self.app_proc):
            if proc is None or proc.poll() is not None:
                continue
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


def main() -> None:
    load_dotenv(REPO_ROOT)
    manager = TulpaManager()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: manager.stop())

    try:
        loop.run_until_complete(manager.run())
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"[fatal] {exc}")
    finally:
        manager.stop()
        loop.close()


if __name__ == "__main__":
    main()
