"""Model clients. OpenAIChat talks to any OpenAI-compatible server (vLLM).
MockLLM is for tests only and never produces reportable results."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable


class ContextOverflow(Exception):
    pass


@dataclass
class Reply:
    content: str = ""
    tool_calls: list[dict] = field(default_factory=list)  # {"id","name","arguments"(str)}
    prompt_tokens: int = 0
    completion_tokens: int = 0


class OpenAIChat:
    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY",
                 temperature: float = 0.7, max_tokens: int = 1024,
                 extra_body: dict | None = None, timeout: float = 300):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens
        self.extra_body = extra_body or {}

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> Reply:
        import openai
        kwargs = dict(model=self.model, messages=messages, temperature=self.temperature,
                      max_tokens=self.max_tokens, extra_body=self.extra_body)
        if tools:
            kwargs.update(tools=tools, tool_choice="auto")
        for attempt in range(3):
            try:
                r = self.client.chat.completions.create(**kwargs)
                break
            except openai.BadRequestError as e:
                msg = str(e).lower()
                if "context" in msg and ("length" in msg or "window" in msg or "too long" in msg):
                    raise ContextOverflow(str(e)) from e
                raise
            except (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError):
                if attempt == 2:
                    raise
                time.sleep(5 * (attempt + 1))
        m = r.choices[0].message
        calls = [{"id": c.id, "name": c.function.name, "arguments": c.function.arguments or "{}"}
                 for c in (m.tool_calls or [])]
        u = r.usage
        return Reply(content=m.content or "", tool_calls=calls,
                     prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
                     completion_tokens=getattr(u, "completion_tokens", 0) or 0)


class MockLLM:
    """Scripted policy for unit tests: policy(messages, tools) -> Reply."""

    def __init__(self, policy: Callable[[list[dict], list[dict] | None], Reply]):
        self.policy = policy

    def chat(self, messages, tools=None) -> Reply:
        return self.policy(messages, tools)
