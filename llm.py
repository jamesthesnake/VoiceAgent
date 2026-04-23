from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Sequence

from groq import AsyncGroq


@dataclass(slots=True)
class GroqConfig:
    api_key: str
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.2
    max_completion_tokens: int = 512


class GroqChatClient:
    def __init__(self, config: GroqConfig) -> None:
        self._config = config
        self._client = AsyncGroq(api_key=config.api_key)

    async def stream_response(
        self,
        messages: Sequence[dict[str, str]],
    ) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(
            model=self._config.model,
            messages=list(messages),
            temperature=self._config.temperature,
            max_completion_tokens=self._config.max_completion_tokens,
            stream=True,
        )
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield delta

