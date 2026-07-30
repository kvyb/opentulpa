import { Buffer } from "node:buffer";
import process from "node:process";

import { Sandbox } from "railway";
import WebSocket from "ws";

const INPUT_ARCHIVE = "/tmp/opentulpa-workspace-in.tar.gz";
const OUTPUT_ARCHIVE = "/tmp/opentulpa-workspace-out.tar.gz";

async function readRequest() {
  const chunks = [];
  for await (const chunk of process.stdin) {
    chunks.push(chunk);
  }
  return JSON.parse(Buffer.concat(chunks).toString("utf8"));
}

function clientOptions(request) {
  return {
    environmentId: request.environmentId,
    webSocketImpl: WebSocket,
  };
}

async function sandboxFor(request) {
  if (request.sandboxId) {
    try {
      const existing = await Sandbox.connect(request.sandboxId, clientOptions(request));
      if (existing.status === "RUNNING") {
        return existing;
      }
    } catch {
      // Idle sandboxes are expected to disappear. Recreate them below.
    }
  }
  return Sandbox.create({
    ...clientOptions(request),
    idleTimeoutMinutes: request.idleTimeoutMinutes,
    networkIsolation: "ISOLATED",
  });
}

async function checkedExec(
  sandbox,
  command,
  timeoutSec,
  cwd = "/",
  failureMessage = "Railway sandbox workspace synchronization failed",
) {
  const result = await sandbox.exec(command, { cwd, timeoutSec });
  if (result.timedOut || result.exitCode !== 0) {
    throw new Error(failureMessage);
  }
}

async function ensureBaselineTools(sandbox) {
  await checkedExec(
    sandbox,
    [
      "if command -v ssh >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1; then exit 0; fi",
      "apt-get update -qq",
      "DEBIAN_FRONTEND=noninteractive apt-get install -y -qq openssh-client python3 >/dev/null",
    ].join(" && "),
    120,
    "/",
    "Railway sandbox baseline tools could not be installed",
  );
}

function boundedOutput(result, limit) {
  const combined = Buffer.from(`${result.stdout}${result.stderr}`, "utf8");
  if (result.timedOut) {
    const message = `command timed out after ${limit.timeoutSec}s`;
    const keep = Math.max(0, limit.bytes - Buffer.byteLength(message) - 1);
    return {
      output: `${combined.subarray(0, keep).toString("utf8").trim()}\n${message}`.trim(),
      exitCode: 124,
      truncated: combined.length > keep,
    };
  }
  if (result.truncated || combined.length > limit.bytes) {
    const message = "sandbox output exceeded its limit";
    const keep = Math.max(0, limit.bytes - Buffer.byteLength(message) - 1);
    return {
      output: `${combined.subarray(0, keep).toString("utf8").trim()}\n${message}`.trim(),
      exitCode: 125,
      truncated: true,
    };
  }
  return {
    output: combined.toString("utf8").trim(),
    exitCode: result.exitCode ?? 127,
    truncated: false,
  };
}

async function main() {
  const request = await readRequest();
  const sandbox = await sandboxFor(request);
  await ensureBaselineTools(sandbox);
  if (request.workspaceArchive) {
    await sandbox.files.write(INPUT_ARCHIVE, Buffer.from(request.workspaceArchive, "base64"), {
      mode: 0o600,
    });
    await checkedExec(
      sandbox,
      `rm -rf /workspace && mkdir -p /workspace && tar -xzf ${INPUT_ARCHIVE} -C /workspace && rm -f ${INPUT_ARCHIVE}`,
      60,
    );
  }

  const result = await sandbox.exec(request.command, {
    cwd: request.workspaceArchive ? "/workspace" : "/",
    timeoutSec: request.timeoutSec,
  });
  const bounded = boundedOutput(result, {
    bytes: request.maxOutputBytes,
    timeoutSec: request.timeoutSec,
  });

  let workspaceArchive = null;
  const workspaceSynchronized = Boolean(request.workspaceArchive)
    && !result.timedOut
    && !result.truncated
    && Buffer.byteLength(`${result.stdout}${result.stderr}`, "utf8") <= request.maxOutputBytes;
  if (workspaceSynchronized) {
    await checkedExec(
      sandbox,
      `rm -f ${OUTPUT_ARCHIVE} && tar -czf ${OUTPUT_ARCHIVE} -C /workspace .`,
      60,
    );
    const archiveEntry = await sandbox.files.stat(OUTPUT_ARCHIVE);
    if (archiveEntry.size > request.maxWorkspaceArchiveBytes) {
      await sandbox.files.remove(OUTPUT_ARCHIVE);
      throw new Error("Railway sandbox workspace archive exceeded its limit");
    }
    const archive = Buffer.from(
      await sandbox.files.read(OUTPUT_ARCHIVE, { format: "bytes" }),
    );
    await sandbox.files.remove(OUTPUT_ARCHIVE);
    workspaceArchive = archive.toString("base64");
  }

  process.stdout.write(
    JSON.stringify({
      ok: true,
      sandboxId: sandbox.id,
      workspaceArchive,
      workspaceSynchronized,
      ...bounded,
    }),
  );
}

main().catch((error) => {
  process.stdout.write(
    JSON.stringify({
      ok: false,
      error: error instanceof Error ? error.message : "Railway sandbox bridge failed",
    }),
  );
  process.exitCode = 1;
});
