"""Pluggable LLM provider abstraction layer.

Usage:
  from app.services.llm_providers import get_llm
  llm = get_llm()
  reply = llm.generate(messages=[{"role": "user", "content": "Hello"}])

Providers:
    - EchoProvider: deterministic echo, for tests/local
    - OpenAIProvider: OpenAI Chat Completions

Add new provider by implementing BaseLLMProvider.
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional
import abc

from config import settings

try:
    import openai  # type: ignore
except Exception:  # pragma: no cover
    openai = None  # type: ignore


ChatMessage = Dict[str, str]  # {role: user|assistant|system, content: str}


class BaseLLMProvider(abc.ABC):
    name: str

    @abc.abstractmethod
    def generate(self, messages: List[ChatMessage], **kwargs) -> str:  # returns assistant reply
        ...

    def trim_history(self, messages: List[ChatMessage]) -> List[ChatMessage]:
        limit = settings.LLM_MAX_HISTORY_MESSAGES
        if limit and len(messages) > limit:
            # Keep system + last N
            system_msgs = [m for m in messages if m.get("role") == "system"]
            non_system = [m for m in messages if m.get("role") != "system"]
            return system_msgs + non_system[-limit:]
        return messages


class EchoProvider(BaseLLMProvider):
    name = "echo"

    def generate(self, messages: List[ChatMessage], **kwargs) -> str:
        last_user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        # status convenience (used by caller)
        self._last_status = "ok"
        # expose a model attribute for uniform access
        self.model = self.name
        return last_user
        

class OpenAIProvider(BaseLLMProvider):
    name = "openai"

    def __init__(self):
        if openai is None:
            raise RuntimeError("openai package not installed")
        if not settings.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY missing")
        openai.api_key = settings.OPENAI_API_KEY
        self.model = settings.OPENAI_MODEL_NAME

    def generate(self, messages: List[ChatMessage], **kwargs) -> str:
        formatted: List[Dict[str, Any]] = []
        for m in messages:
            role = m.get("role")
            if role not in {"user", "assistant", "system"}:
                continue
            content = m.get("content", "")
            # extraction_controller 가 vision 사용 시 content 를 list[parts] 로 전달
            if isinstance(content, list):
                formatted.append({"role": role, "content": content})
            else:
                formatted.append({"role": role, "content": str(content)})
        import time
        start = time.time()
        try:
            print(f"[llm:openai] start model={self.model} msg_count={len(formatted)}")
            # Some versions of openai lib (>=1.x) use different client style; keep legacy call.
            resp = openai.chat.completions.create(model=self.model, messages=formatted, timeout=15)  # type: ignore
            dur = time.time() - start
            self._last_status = "ok"
            out = resp.choices[0].message.content  # type: ignore
            print(f"[llm:openai] done model={self.model} latency={dur:.2f}s chars={len(out) if out else 0}")
            return out
        except Exception as e:  # pragma: no cover
            dur = time.time() - start
            print(f"[llm:openai] error model={self.model} latency={dur:.2f}s type={type(e).__name__} msg={str(e)[:180]}")
            self._last_status = "fallback"
            return f"{str(e)[:400]}"


_singleton: Optional[BaseLLMProvider] = None


class AutoProvider(BaseLLMProvider):
    """Dynamic chained provider.

    Order comes from settings.LLM_AUTO_PRIORITY (comma separated).
    Unknown / disabled providers are skipped. Echo is always ensured as final fallback.
    Records last_model_name + last_provider for meta storage.
    """
    name = "auto"

    def __init__(self):
        self._instances: Dict[str, BaseLLMProvider] = {}
        self.last_model_name: Optional[str] = None
        self.last_provider: Optional[str] = None
        # Parse priority list once
        raw = (settings.LLM_AUTO_PRIORITY or "").split(",")
        cleaned = [p.strip().lower() for p in raw if p.strip() and p.strip().lower() != "gemini"]
        # Guarantee echo fallback at end
        if "echo" not in cleaned:
            cleaned.append("echo")
        self.priority = cleaned

    def _get_or_init(self, name: str) -> Optional[BaseLLMProvider]:
        if name in self._instances:
            return self._instances[name]
        try:
            if name == "openai":
                inst = OpenAIProvider()
            elif name == "echo":
                inst = EchoProvider()
            else:
                return None
            self._instances[name] = inst
            return inst
        except Exception as e:  # pragma: no cover
            print(f"[llm:auto] init failed for {name}: {e}")
            return None

    def generate(self, messages: List[ChatMessage], **kwargs) -> str:
        for provider_name in self.priority:
            prov = self._get_or_init(provider_name)
            if prov is None:
                continue
            try:
                reply = prov.generate(messages, **kwargs)
                status = getattr(prov, "_last_status", "ok")
                if status == "ok" and reply:
                    self.last_provider = getattr(prov, "name", provider_name)
                    self.last_model_name = getattr(prov, "model", provider_name)
                    return reply
            except Exception as e:  # pragma: no cover
                print(f"[llm:auto] generate error from {provider_name}: {e}")
                continue
        # Absolute fallback (shouldn't reach unless echo errored w/out reply)
        echo = EchoProvider()
        self.last_provider = echo.name
        self.last_model_name = echo.name
        return echo.generate(messages, **kwargs)


def get_llm() -> BaseLLMProvider:
    global _singleton
    if _singleton:
        return _singleton
    provider = settings.LLM_PROVIDER.lower()
    if provider == "auto":
        _singleton = AutoProvider()
        return _singleton
    if provider == "openai":
        try:
            _singleton = OpenAIProvider()
            return _singleton
        except Exception as e:
            print("[llm] OpenAI init failed, falling back to echo:", e)
    _singleton = EchoProvider()
    return _singleton
