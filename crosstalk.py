"""Crosstalk: speculative LLM turn-taking with barge-in for a voice agent.

Architecture
============
Two independent concerns, neither mutates transcriber state mid-turn:

1. **Buffered speculation** — during the user's turn, fire speculative LLM
   calls on each transcript update.  The LLM streams tokens to a *buffer*
   (never to TTS).  When ``speech_final`` arrives and a buffered response
   is ready, we skip the cold LLM call and pipe the buffer straight to TTS.

2. **Barge-in** — during TTS playback, poll the local mic VAD for a
   speech *onset* (silence→speech transition) that occurred after playback
   started.  On detection, abort the speaker immediately.  Uses calibrated
   noise floor so ambient noise doesn't false-trigger.  Assumes headphones
   (no echo cancellation).
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass

from llm import GroqChatClient
from log import LatencyLogger, SessionClock, SessionLogger
from transcriber import DeepgramTranscriber
from tts import ElevenLabsSpeaker


WAIT_TOKEN = "<wait/>"

CROSSTALK_INSTRUCTIONS = (
    "You are receiving the user's speech transcribed in real time, so the "
    "last user message may be incomplete. If the user appears to still be "
    "mid-thought (dangling conjunction, trailing-off phrase, no clear "
    "question or request yet, obvious cut-off mid-sentence), reply with "
    "EXACTLY the single token {wait} and nothing else. If the user's turn "
    "looks complete enough to answer, respond normally — concise and "
    "spoken-style, no preamble, no meta commentary, and never mention "
    "{wait}."
).format(wait=WAIT_TOKEN)


@dataclass(slots=True)
class CrosstalkConfig:
    base_system_prompt: str
    # Minimum transcript length (chars) before we bother speculating.
    min_speculation_chars: int = 2
    # Barge-in: enable onset-based detection during playback.
    bargein_enabled: bool = True
    # Polling interval for barge-in watcher.
    bargein_poll_s: float = 0.03
    # Grace period before barge-in detection starts (let first audio frame play).
    bargein_grace_s: float = 0.5


@dataclass(slots=True)
class CrosstalkResult:
    user_text: str
    assistant_text: str
    interrupted: bool = False


class CrosstalkRunner:
    """Runs one user turn with optional speculation and barge-in."""

    def __init__(
        self,
        transcriber: DeepgramTranscriber,
        llm: GroqChatClient,
        speaker: ElevenLabsSpeaker,
        session_clock: SessionClock,
        logger: SessionLogger,
        latency_logger: LatencyLogger,
        config: CrosstalkConfig,
    ) -> None:
        self._transcriber = transcriber
        self._llm = llm
        self._speaker = speaker
        self._session_clock = session_clock
        self._logger = logger
        self._latency_logger = latency_logger
        self._config = config

    # ------------------------------------------------------------------
    # Main turn loop
    # ------------------------------------------------------------------

    async def run_turn(
        self, conversation: list[dict[str, str]]
    ) -> CrosstalkResult:
        self._transcriber.unmute()
        final_user_text = ""
        spec_task: asyncio.Task[str] | None = None
        last_spec_text: str | None = None

        try:
            async for update in self._transcriber.iter_utterance(
                self._session_clock.now,
                self._logger,
                self._latency_logger,
            ):
                text = update.transcript.strip()
                if text:
                    final_user_text = text

                # Fire / restart speculation (pure LLM buffer, never TTS).
                if (
                    text
                    and len(text) >= self._config.min_speculation_chars
                    and text != last_spec_text
                ):
                    spec_task = self._restart_buffer_spec(
                        spec_task, conversation, text
                    )
                    last_spec_text = text

                if update.speech_final or update.utterance_end:
                    break
        finally:
            self._transcriber.mute()

        # ----------------------------------------------------------
        # Resolve response: use buffered speculation or fall back
        # ----------------------------------------------------------
        buffered_text = ""
        if spec_task is not None:
            if spec_task.done():
                try:
                    buffered_text = spec_task.result()
                except Exception:
                    buffered_text = ""
            else:
                # Speculation still running — give it a short window.
                try:
                    buffered_text = await asyncio.wait_for(spec_task, timeout=0.3)
                except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                    buffered_text = ""
                    spec_task.cancel()

        assistant_text = ""
        interrupted = False

        if buffered_text:
            # Speculation produced a reply — pipe it to TTS.
            assistant_text, interrupted = await self._speak_text(buffered_text)
        elif final_user_text:
            # No usable speculation — cold LLM + TTS.
            assistant_text, interrupted = await self._final_reply(
                conversation, final_user_text
            )

        return CrosstalkResult(
            user_text=final_user_text,
            assistant_text=assistant_text,
            interrupted=interrupted,
        )

    # ------------------------------------------------------------------
    # Buffered speculation (no mute, no TTS — pure LLM text)
    # ------------------------------------------------------------------

    def _restart_buffer_spec(
        self,
        current: asyncio.Task[str] | None,
        conversation: list[dict[str, str]],
        user_text: str,
    ) -> asyncio.Task[str]:
        """Cancel the previous speculation and start a new one.

        Returns the new task.  Does NOT await the old one — fire and forget
        the cancellation so we never block iter_utterance.
        """
        if current is not None and not current.done():
            current.cancel()
        return asyncio.create_task(
            self._buffer_speculate(conversation, user_text)
        )

    async def _buffer_speculate(
        self,
        conversation: list[dict[str, str]],
        user_text: str,
    ) -> str:
        """Run a speculative LLM call and buffer the result as plain text.

        Returns the full reply text, or "" if the LLM said <wait/>.
        Never touches the transcriber or speaker.
        """
        messages = self._build_messages(conversation, user_text)
        token_stream = self._llm.stream_response(messages)
        tokens: list[str] = []
        try:
            async for token in token_stream:
                tokens.append(token)
                accumulated = "".join(tokens).lstrip()
                if self._looks_like_wait(accumulated):
                    await _drain(token_stream)
                    return ""
        except asyncio.CancelledError:
            await _drain_and_close(token_stream)
            raise
        return "".join(tokens).strip()

    # ------------------------------------------------------------------
    # Speaking (TTS with optional barge-in)
    # ------------------------------------------------------------------

    async def _speak_text(self, text: str) -> tuple[str, bool]:
        """Send pre-generated text to TTS with barge-in support."""
        stream = _iter_once(text)
        return await self._speak_with_bargein(stream)

    async def _final_reply(
        self,
        conversation: list[dict[str, str]],
        user_text: str,
    ) -> tuple[str, bool]:
        """Cold path: LLM stream → TTS with barge-in."""
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._config.base_system_prompt},
            *(conversation[1:] if conversation and conversation[0].get("role") == "system" else conversation),
            {"role": "user", "content": user_text},
        ]
        stream = self._llm.stream_response(messages)
        try:
            return await self._speak_with_bargein(stream)
        except Exception as exc:
            print(f"[crosstalk] final reply error: {exc!r}")
            return "", False

    async def _speak_with_bargein(
        self,
        token_source: AsyncIterator[str],
    ) -> tuple[str, bool]:
        """Play TTS.  If barge-in is enabled and a speech onset is detected,
        abort playback and return (partial_text, True)."""
        if not self._config.bargein_enabled:
            text = await self._speaker.speak_stream(
                token_source, self._session_clock, self._logger
            )
            return text, False

        # Clear any stale onset and record the playback start time.
        self._transcriber.clear_onset()
        playback_start = self._session_clock.now()

        capture = _TextCapture(token_source)

        speak_task = asyncio.create_task(
            self._speaker.speak_stream(
                capture, self._session_clock, self._logger
            )
        )
        watcher_task = asyncio.create_task(
            self._bargein_watcher(playback_start)
        )

        done, _pending = await asyncio.wait(
            [speak_task, watcher_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        if watcher_task in done:
            speak_task.cancel()
            try:
                await speak_task
            except (asyncio.CancelledError, Exception):
                pass
            partial = capture.text.strip()
            print("[crosstalk] barge-in: user interrupted, stopping playback")
            return partial, True

        # Normal finish.
        watcher_task.cancel()
        try:
            await watcher_task
        except (asyncio.CancelledError, Exception):
            pass
        return speak_task.result(), False

    async def _bargein_watcher(self, playback_start: float) -> None:
        """Wait for a speech onset that occurred after playback_start."""
        await asyncio.sleep(self._config.bargein_grace_s)
        while True:
            if self._transcriber.speech_onset_since(playback_start):
                return  # Onset detected — trigger barge-in.
            await asyncio.sleep(self._config.bargein_poll_s)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        conversation: list[dict[str, str]],
        user_text: str,
    ) -> list[dict[str, str]]:
        base = conversation[0] if conversation and conversation[0].get("role") == "system" else None
        rest = conversation[1:] if base is not None else conversation
        combined_system = (
            f"{self._config.base_system_prompt}\n\n{CROSSTALK_INSTRUCTIONS}"
            if base is None
            else f"{base['content']}\n\n{CROSSTALK_INSTRUCTIONS}"
        )
        return [
            {"role": "system", "content": combined_system},
            *rest,
            {"role": "user", "content": user_text},
        ]

    @staticmethod
    def _looks_like_wait(accumulated: str) -> bool:
        normalized = accumulated.lower().strip()
        return (
            normalized.startswith("<wait")
            or normalized.startswith("</wait")
            or normalized == "wait"
        )


class _TextCapture:
    """Async iterator wrapper that records all tokens passing through."""

    def __init__(self, source: AsyncIterator[str]) -> None:
        self._source = source
        self.text: str = ""

    def __aiter__(self) -> _TextCapture:
        return self

    async def __anext__(self) -> str:
        token = await self._source.__anext__()
        self.text += token
        return token


async def _iter_once(text: str) -> AsyncIterator[str]:
    yield text


async def _drain(stream: AsyncIterator[str]) -> None:
    try:
        async for _ in stream:
            pass
    except Exception:
        pass


async def _drain_and_close(stream: AsyncIterator[str]) -> None:
    aclose = getattr(stream, "aclose", None)
    if aclose is not None:
        try:
            await aclose()
            return
        except Exception:
            pass
    await _drain(stream)
