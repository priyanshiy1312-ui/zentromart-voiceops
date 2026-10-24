"""Tiny provider-agnostic LLM wrapper (Anthropic, or any OpenAI-compatible endpoint).

Robust to SDK / model differences: if a model rejects `temperature` (or `max_tokens`),
the call is retried automatically without it. Auth / config problems fail fast with a clear message.
"""
import os
import time

from . import config


class LLM:
    def __init__(self, provider=None, model=None):
        self.provider = (provider or config.LLM_PROVIDER).lower()
        self.model = model or config.LLM_MODEL
        self._client = None
        self._no_temperature = False
        self._max_completion = False
        self._truncated = False

    def _get_client(self):
        if self._client is not None:
            return self._client
        if self.provider == "anthropic":
            if not os.getenv("ANTHROPIC_API_KEY"):
                raise RuntimeError("LLM_PROVIDER is 'anthropic' but ANTHROPIC_API_KEY is empty. "
                                   "If you have an OpenAI/Gemini/Groq key, set LLM_PROVIDER=openai in .env "
                                   "(plus OPENAI_API_KEY, LLM_MODEL and, for Gemini/Groq, OPENAI_BASE_URL).")
            import anthropic

            self._client = anthropic.Anthropic()
        else:
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("LLM_PROVIDER is 'openai' but OPENAI_API_KEY is empty in .env.")
            from openai import OpenAI

            self._client = OpenAI(base_url=config.OPENAI_BASE_URL or None, api_key=os.getenv("OPENAI_API_KEY"))
        return self._client

    def _call(self, system, messages, max_tokens, temperature):
        client = self._get_client()
        if self.provider == "anthropic":
            kw = dict(model=self.model, system=system, messages=messages, max_tokens=max_tokens)
            if not self._no_temperature:
                kw["temperature"] = temperature
            resp = client.messages.create(**kw)
            self._truncated = getattr(resp, "stop_reason", "") == "max_tokens"
            return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        kw = dict(model=self.model, messages=[{"role": "system", "content": system}] + messages)
        kw["max_completion_tokens" if self._max_completion else "max_tokens"] = max_tokens
        if not self._no_temperature:
            kw["temperature"] = temperature
        resp = client.chat.completions.create(**kw)
        self._truncated = resp.choices[0].finish_reason == "length"
        return resp.choices[0].message.content or ""

    def chat(self, system, messages, max_tokens=700, temperature=0.0):
        """messages: [{'role': 'user'|'assistant', 'content': str}, ...] starting with a user turn."""
        last_err = None
        max_tokens = max(max_tokens, 1500)  # "thinking" models spend part of the budget on reasoning
        for attempt in range(6):
            try:
                text = self._call(system, messages, max_tokens, temperature)
                if self._truncated and max_tokens < 8000:  # output was cut off: retry with a bigger budget
                    max_tokens *= 2
                    continue
                return text
            except RuntimeError:
                raise  # configuration problem: do not retry
            except Exception as exc:
                msg = str(exc).lower()
                last_err = exc
                if "temperature" in msg and not self._no_temperature:
                    self._no_temperature = True  # model/SDK does not accept it: retry without
                    continue
                if "max_completion_tokens" in msg and not self._max_completion:
                    self._max_completion = True
                    continue
                if any(w in msg for w in ("api key", "api_key", "authentication", "unauthorized", "401", "403",
                                          "permission", "model_not_found", "not found", "does not exist")):
                    raise RuntimeError(f"LLM configuration error: {exc}") from exc
                time.sleep(2 * (attempt + 1))  # rate limit / network
        raise RuntimeError(f"LLM call failed after retries: {last_err}")
