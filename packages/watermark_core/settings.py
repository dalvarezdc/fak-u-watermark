"""Persistent app settings (API keys, base URL, model) in ~/.faku/settings.json."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def settings_path() -> Path:
    root = Path.home() / ".faku"
    root.mkdir(parents=True, exist_ok=True)
    return root / "settings.json"


@dataclass
class AppSettings:
    """User-editable configuration. API keys are stored locally only."""

    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    inpaint_model: str = "dall-e-2"
    inpaint_base_url: str = ""  # empty → reuse base_url
    temperature: float = 0.7
    # Named provider presets (manual keys still win)
    provider: str = "custom"  # openai | deepseek | xai | custom
    extra: dict[str, Any] = field(default_factory=dict)

    def resolved_inpaint_base(self) -> str:
        return (self.inpaint_base_url or self.base_url or "https://api.openai.com/v1").rstrip(
            "/"
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    def masked_dict(self) -> dict[str, Any]:
        """Safe for UI/CLI display — never includes the raw API key."""
        d = self.to_dict()
        key = d.pop("api_key", "") or ""
        if len(key) > 8:
            d["api_key_masked"] = key[:4] + "…" + key[-4:]
            d["has_api_key"] = True
        else:
            d["api_key_masked"] = "(not set)" if not key else "••••"
            d["has_api_key"] = bool(key)
        return d


PROVIDER_PRESETS: dict[str, dict[str, str]] = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "inpaint_model": "dall-e-2",
        "label": "OpenAI",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
        "inpaint_model": "",
        "label": "DeepSeek",
    },
    "xai": {
        "base_url": "https://api.x.ai/v1",
        "model": "grok-2-latest",
        "inpaint_model": "",
        "label": "xAI Grok",
    },
    "custom": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "inpaint_model": "dall-e-2",
        "label": "Custom (OpenAI-compatible)",
    },
}


def load_settings() -> AppSettings:
    path = settings_path()
    data: dict[str, Any] = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            data = {}

    s = AppSettings(
        api_key=str(data.get("api_key") or ""),
        base_url=str(data.get("base_url") or "https://api.openai.com/v1"),
        model=str(data.get("model") or "gpt-4o-mini"),
        inpaint_model=str(data.get("inpaint_model") or "dall-e-2"),
        inpaint_base_url=str(data.get("inpaint_base_url") or ""),
        temperature=float(data.get("temperature") or 0.7),
        provider=str(data.get("provider") or "custom"),
        extra=dict(data.get("extra") or {}),
    )

    # Env vars fill empty fields (never overwrite saved key with empty)
    if not s.api_key:
        s.api_key = (
            os.environ.get("FAKU_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("DEEPSEEK_API_KEY")
            or os.environ.get("XAI_API_KEY")
            or ""
        )
    if s.base_url == "https://api.openai.com/v1":
        env_base = os.environ.get("FAKU_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        if env_base:
            s.base_url = env_base
    if s.model == "gpt-4o-mini":
        env_model = os.environ.get("FAKU_MODEL") or os.environ.get("OPENAI_MODEL")
        if env_model:
            s.model = env_model
    return s


def save_settings(settings: AppSettings) -> Path:
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(settings.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return path


def update_settings(**kwargs: Any) -> AppSettings:
    """Merge kwargs into current settings and save."""
    s = load_settings()
    for k, v in kwargs.items():
        if not hasattr(s, k):
            continue
        if k == "api_key" and (v is None or str(v).strip() == ""):
            # Empty field in UI means "keep existing" when already set —
            # only clear if explicit clear_api_key
            continue
        if k == "api_key" and str(v).strip() in ("***", "••••", "(unchanged)"):
            continue
        setattr(s, k, v if not isinstance(v, str) else v.strip())
    save_settings(s)
    return s


def apply_provider_preset(provider: str, *, keep_key: bool = True) -> AppSettings:
    s = load_settings()
    preset = PROVIDER_PRESETS.get(provider) or PROVIDER_PRESETS["custom"]
    key = s.api_key if keep_key else ""
    s.provider = provider
    s.base_url = preset["base_url"]
    s.model = preset["model"]
    if preset.get("inpaint_model"):
        s.inpaint_model = preset["inpaint_model"]
    s.api_key = key
    save_settings(s)
    return s


def clear_api_key() -> AppSettings:
    s = load_settings()
    s.api_key = ""
    save_settings(s)
    return s
