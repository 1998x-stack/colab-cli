"""Model client for vLLM's OpenAI-compatible API."""

import time
import json
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI


@dataclass
class ModelOutput:
    message: str
    thought: str
    action: str
    tool_calls: list[dict] | None = None
    thinking_blocks: list[dict] | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class ModelStats:
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_queries: int = 0
    total_time: float = 0.0


class VLLMClient:
    def __init__(self, base_url: str = "http://localhost:8000/v1", model: str = "Qwen/Qwen2.5-7B-Instruct-AWQ"):
        self.client = OpenAI(base_url=base_url, api_key="not-needed")
        self.model = model
        self.stats = ModelStats()

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> ModelOutput:
        """Send chat completion request to vLLM."""
        t0 = time.perf_counter()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**kwargs)
        elapsed = time.perf_counter() - t0

        choice = response.choices[0]
        msg = choice.message

        # Track usage via vLLM's usage field if available
        prompt_tokens = 0
        completion_tokens = 0
        if response.usage:
            prompt_tokens = response.usage.prompt_tokens or 0
            completion_tokens = response.usage.completion_tokens or 0

        self.stats.total_prompt_tokens += prompt_tokens
        self.stats.total_completion_tokens += completion_tokens
        self.stats.total_queries += 1
        self.stats.total_time += elapsed

        # Extract tool calls if present
        tool_calls = None
        if msg.tool_calls:
            tool_calls = [
                {"id": tc.id, "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]

        # Extract content
        content = msg.content or ""

        return ModelOutput(
            message=content,
            thought=content,  # thought and action separated during parsing
            action="",
            tool_calls=tool_calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


def check_vllm_health(base_url: str = "http://localhost:8000/v1") -> bool:
    """Check if vLLM server is alive and responding."""
    try:
        client = OpenAI(base_url=base_url, api_key="not-needed")
        models = client.models.list()
        return len(list(models)) > 0
    except Exception:
        return False
