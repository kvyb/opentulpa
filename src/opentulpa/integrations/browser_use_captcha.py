"""Optional Browser Use CAPTCHA action backed by CapSolver."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from opentulpa.integrations.capsolver import CapSolverClient, CapSolverError

if TYPE_CHECKING:
    from browser_use import ActionResult, Controller


@dataclass(frozen=True, slots=True)
class BrowserCaptchaChallenge:
    captcha_type: str
    website_url: str
    website_key: str
    marker: str
    page_action: str | None = None


_DETECT_CAPTCHA_SCRIPT = """() => {
  const currentUrl = String(window.location.href || '');
  const attr = (element, name) => element ? String(element.getAttribute(name) || '').trim() : '';
  const parseParam = (rawUrl, names) => {
    try {
      const url = new URL(rawUrl, currentUrl);
      for (const name of names) {
        const value = String(url.searchParams.get(name) || '').trim();
        if (value) return value;
      }
    } catch (_) {}
    return '';
  };
  const scriptTexts = Array.from(document.scripts)
    .map((script) => String(script.textContent || ''))
    .join('\\n');
  const scriptSrcs = Array.from(document.scripts)
    .map((script) => String(script.getAttribute('src') || ''))
    .filter(Boolean);
  const pageSource = scriptTexts + '\\n' + scriptSrcs.join('\\n');

  const recaptchaElement = document.querySelector(
    '.g-recaptcha[data-sitekey], [data-sitekey][class*="g-recaptcha"]'
  );
  const recaptchaKey = attr(recaptchaElement, 'data-sitekey');
  if (recaptchaKey) {
    return {
      captchaType: 'recaptcha_v2',
      websiteUrl: currentUrl,
      websiteKey: recaptchaKey,
      marker: 'g-recaptcha'
    };
  }

  const renderScript = scriptSrcs.find((src) =>
    src.includes('recaptcha') && parseParam(src, ['render'])
  );
  const renderKey = renderScript ? parseParam(renderScript, ['render']) : '';
  const executeKeyMatch = pageSource.match(
    /grecaptcha(?:\\.enterprise)?\\.execute\\s*\\(\\s*['"]([^'"]+)['"]/s
  );
  const actionMatch = pageSource.match(
    /grecaptcha(?:\\.enterprise)?\\.execute\\s*\\([^)]*?action\\s*:\\s*['"]([^'"]+)['"]/s
  );
  const recaptchaV3Key = String((executeKeyMatch && executeKeyMatch[1]) || renderKey || '').trim();
  const recaptchaV3Action = String((actionMatch && actionMatch[1]) || '').trim();
  if (recaptchaV3Key && recaptchaV3Key !== 'explicit') {
    return {
      captchaType: 'recaptcha_v3',
      websiteUrl: currentUrl,
      websiteKey: recaptchaV3Key,
      pageAction: recaptchaV3Action,
      marker: 'recaptcha v3'
    };
  }

  const recaptchaFrame = Array.from(document.querySelectorAll('iframe[src*="recaptcha"]'))
    .find((frame) => parseParam(frame.getAttribute('src') || '', ['k', 'render']));
  const frameRecaptchaKey = recaptchaFrame
    ? parseParam(recaptchaFrame.getAttribute('src') || '', ['k', 'render'])
    : '';
  if (frameRecaptchaKey && frameRecaptchaKey !== 'explicit') {
    return {
      captchaType: 'recaptcha_v2',
      websiteUrl: currentUrl,
      websiteKey: frameRecaptchaKey,
      marker: 'recaptcha iframe'
    };
  }

  const turnstileElement = document.querySelector(
    '.cf-turnstile[data-sitekey], [data-sitekey][class*="turnstile"]'
  );
  const turnstileKey = attr(turnstileElement, 'data-sitekey');
  if (turnstileKey) {
    return {
      captchaType: 'turnstile',
      websiteUrl: currentUrl,
      websiteKey: turnstileKey,
      marker: 'cf-turnstile'
    };
  }

  const turnstileFrame = Array.from(document.querySelectorAll('iframe[src*="challenges.cloudflare.com"]'))
    .find((frame) => parseParam(frame.getAttribute('src') || '', ['sitekey', 'k']));
  const frameTurnstileKey = turnstileFrame
    ? parseParam(turnstileFrame.getAttribute('src') || '', ['sitekey', 'k'])
    : '';
  if (frameTurnstileKey) {
    return {
      captchaType: 'turnstile',
      websiteUrl: currentUrl,
      websiteKey: frameTurnstileKey,
      marker: 'turnstile iframe'
    };
  }

  return null;
}"""


_INJECT_CAPTCHA_TOKEN_SCRIPT = """(captchaType, token) => {
  const dispatchValueEvents = (element) => {
    for (const name of ['input', 'change']) {
      element.dispatchEvent(new Event(name, { bubbles: true }));
    }
  };
  const setElementValue = (element, value) => {
    if (!element) return false;
    element.value = value;
    element.innerHTML = value;
    dispatchValueEvents(element);
    return true;
  };
  const ensureHiddenField = (name, id) => {
    let element = document.querySelector(`textarea[name="${name}"], input[name="${name}"], #${id}`);
    if (!element) {
      element = document.createElement('textarea');
      element.name = name;
      element.id = id;
      element.style.display = 'none';
      document.body.appendChild(element);
    }
    return element;
  };
  const callNamedCallback = (callbackName) => {
    if (!callbackName) return false;
    const callback = callbackName.split('.').reduce((obj, key) => obj && obj[key], window);
    if (typeof callback !== 'function') return false;
    callback(token);
    return true;
  };

  let fieldsUpdated = 0;
  let callbacksCalled = 0;
  if (captchaType === 'recaptcha_v2' || captchaType === 'recaptcha_v3') {
    const response = ensureHiddenField('g-recaptcha-response', 'g-recaptcha-response');
    if (setElementValue(response, token)) fieldsUpdated += 1;
    const callbackName = document.querySelector('.g-recaptcha[data-callback], [data-callback]')
      ?.getAttribute('data-callback');
    if (callNamedCallback(callbackName)) callbacksCalled += 1;

    const clients = window.___grecaptcha_cfg && window.___grecaptcha_cfg.clients;
    const seen = new Set();
    const walk = (value) => {
      if (!value || typeof value !== 'object' || seen.has(value)) return;
      seen.add(value);
      for (const [key, child] of Object.entries(value)) {
        if (key === 'callback' && typeof child === 'function') {
          child(token);
          callbacksCalled += 1;
        } else if (child && typeof child === 'object') {
          walk(child);
        }
      }
    };
    if (clients && typeof clients === 'object') walk(clients);
  } else if (captchaType === 'turnstile') {
    const response = ensureHiddenField('cf-turnstile-response', 'cf-turnstile-response');
    if (setElementValue(response, token)) fieldsUpdated += 1;
    const callbackName = document.querySelector('.cf-turnstile[data-callback], [data-callback]')
      ?.getAttribute('data-callback');
    if (callNamedCallback(callbackName)) callbacksCalled += 1;
  }

  return {
    ok: fieldsUpdated > 0 || callbacksCalled > 0,
    fieldsUpdated,
    callbacksCalled
  };
}"""


def register_capsolver_action(controller: Controller, capsolver: CapSolverClient) -> Controller:
    """Register the CapSolver action on an existing Browser Use controller."""
    from browser_use import ActionResult

    @controller.action(
        "Solve a reCAPTCHA v2, reCAPTCHA v3, or Cloudflare Turnstile challenge using CapSolver. "
        "Use only when a CAPTCHA is blocking the browser task.",
        domains=["*"],
    )
    async def solve_captcha_with_capsolver(browser_session) -> ActionResult:
        page = await browser_session.get_current_page()
        if page is None:
            return ActionResult(success=False, error="No active browser page for CAPTCHA solving")

        try:
            challenge = await detect_browser_captcha(page)
            if challenge is None:
                return ActionResult(
                    success=False,
                    extracted_content="No supported CAPTCHA was detected on the current page.",
                )
            if challenge.captcha_type == "recaptcha_v2":
                result = await capsolver.solve_recaptcha_v2(
                    website_url=challenge.website_url,
                    website_key=challenge.website_key,
                )
            elif challenge.captcha_type == "recaptcha_v3":
                result = await capsolver.solve_recaptcha_v3(
                    website_url=challenge.website_url,
                    website_key=challenge.website_key,
                    page_action=challenge.page_action,
                )
            elif challenge.captcha_type == "turnstile":
                result = await capsolver.solve_turnstile(
                    website_url=challenge.website_url,
                    website_key=challenge.website_key,
                )
            else:
                return ActionResult(
                    success=False,
                    error=f"Unsupported CAPTCHA type: {challenge.captcha_type}",
                )
            injected = await inject_browser_captcha_token(
                page,
                captcha_type=challenge.captcha_type,
                token=result.token,
            )
            if not injected.get("ok"):
                return ActionResult(
                    success=False,
                    error="CapSolver returned a token but OpenTulpa could not inject it into the page.",
                )
            return ActionResult(
                extracted_content=f"CAPTCHA solved with CapSolver ({challenge.captcha_type}).",
            )
        except (CapSolverError, ValueError) as exc:
            return ActionResult(success=False, error=str(exc))

    return controller


def build_capsolver_controller(capsolver: CapSolverClient) -> Controller:
    """Build a Browser Use controller with a CapSolver action."""
    from browser_use import Controller

    return register_capsolver_action(Controller(), capsolver)


async def detect_browser_captcha(page: Any) -> BrowserCaptchaChallenge | None:
    raw = await page.evaluate(_DETECT_CAPTCHA_SCRIPT)
    data = _coerce_json_object(raw)
    if not data:
        return None
    captcha_type = str(data.get("captchaType") or "").strip()
    website_url = str(data.get("websiteUrl") or "").strip()
    website_key = str(data.get("websiteKey") or "").strip()
    marker = str(data.get("marker") or "").strip()
    page_action = str(data.get("pageAction") or "").strip() or None
    if captcha_type not in {"recaptcha_v2", "recaptcha_v3", "turnstile"}:
        return None
    if not website_url:
        raise ValueError("Detected CAPTCHA without a page URL")
    if not website_key:
        raise ValueError("Detected CAPTCHA without a site key")
    return BrowserCaptchaChallenge(
        captcha_type=captcha_type,
        website_url=website_url,
        website_key=website_key,
        marker=marker,
        page_action=page_action,
    )


async def inject_browser_captcha_token(
    page: Any,
    *,
    captcha_type: str,
    token: str,
) -> dict[str, Any]:
    safe_type = str(captcha_type or "").strip()
    safe_token = str(token or "").strip()
    if safe_type not in {"recaptcha_v2", "recaptcha_v3", "turnstile"}:
        raise ValueError(f"Unsupported CAPTCHA type: {safe_type}")
    if not safe_token:
        raise ValueError("CAPTCHA solution token is empty")
    raw = await page.evaluate(_INJECT_CAPTCHA_TOKEN_SCRIPT, safe_type, safe_token)
    data = _coerce_json_object(raw)
    return data if data else {"ok": False}


def _coerce_json_object(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text or text == "null":
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None
