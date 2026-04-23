"""Crosstalk: speculative LLM turn-taking for a voice agent.

Inspired by https://github.com/tarzain/crosstalk. The idea: instead of
waiting for a long VAD silence after the user stops speaking, we let the
LLM itself decide whether the user is done. On every interim transcript
update from Deepgram we kick off a speculative LLM call. The LLM either
emits `<wait/>` (user is mid-thought, stay quiet) or an actual assistant
reply (user is done, start speaking now). When a new interim update
arrives we cancel the in-flight speculation and start a fresh one,
getting an effect similar to the `setTranscript`-driven regeneration in
the original crosstalk React app.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass

from llm import GroqChatClient
from log import LatencyLogger, SessionClock, SessionLogger
from transcriber import DeepgramTranscriber, TranscriptUpdate
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
    # Minimum tokens to accumulate before deciding wait vs. speak. Keeping
    # this small lets us start TTS fast; too small risks misreading a
    # reply that coincidentally begins with "<".
    decision_peek_chars: int = 12
    # Minimum transcript length (chars) before we bother speculating. Very
    # short interims like "hi" are cheap but noisy; skipping them avoids
    # thrashing the LLM when the user has barely started.
    min_speculation_chars: int = 2
    # How long the local mic must be silent (seconds) before we commit a
    # speculation to TTS.  This gates on real audio energy, not on
    # Deepgram transcript timing, so it's immune to ASR latency.
    commit_silence_s: float = 0.4
    # Polling interval while waiting for the silence gate.
    silence_poll_s: float = 0.03
    # Barge-in: if mic silence drops below this during agent playback,
    # the user is talking and we should stop TTS. Assumes headphones
    # (no echo cancellation).  Set to 0 to disable barge-in.
    bargein_threshold_s: float = 0.0


@dataclass(slots=True)
class CrosstalkResult:
    user_text: str
    assistant_text: str
    interrupted: bool = False


class CrosstalkRunner:
    """Runs one user turn using speculative LLM-driven turn prediction."""

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

    async def run_turn(
        self, conversation: list[dict[str, str]]
    ) -> CrosstalkResult:
        self._transcriber.unmute()
        final_user_text = ""

        try:
            async for update in self._transcriber.iter_utterance(
                self._session_clock.now,
                self._logger,
                self._latency_logger,
            ):
                text = update.transcript.strip()
                if text:
                    final_user_text = text
                if update.speech_final or update.utterance_end:
                    break
        finally:
            self._transcriber.mute()

        assistant_text = ""
        interrupted = False
        if final_user_text:
            assistant_text, interrupted = await self._final_reply(
                conversation, final_user_text
            )

        return CrosstalkResult(
            user_text=final_user_text,
            assistant_text=assistant_text,
            interrupted=interrupted,
        )

    async def _final_reply(
        self,
        conversation: list[dict[str, str]],
        user_text: str,
    ) -> tuple[str, bool]:
        """Returns (assistant_text, interrupted)."""
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._config.base_system_prompt},
            *(conversation[1:] if conversation and conversation[0].get("role") == "system" else conversation),
            {"role": "user", "content": user_text},
        ]
        stream = self._llm.stream_response(messages)
        self._transcriber.mute()
        try:
            text, interrupted = await self._speak_with_bargein(stream)
            return text, interrupted
        except Exception as exc:
            print(f"[crosstalk] final reply error: {exc!r}")
            return "", False

    async def _restart_speculation(
        self,
        current: asyncio.Task[str] | None,
        conversation: list[dict[str, str]],
        user_text: str,
    ) -> asyncio.Task[str]:
        if current is not None and not current.done():
            current.cancel()
            try:
                await current
            except (asyncio.CancelledError, Exception):
                pass
        return asyncio.create_task(
            self._speculate_and_speak(conversation, user_text)
        )

    async def _speculate_and_speak(
        self,
        conversation: list[dict[str, str]],
        user_text: str,
    ) -> str:
        messages = self._build_messages(conversation, user_text)
        token_stream = self._llm.stream_response(messages)

        buffered: list[str] = []
        decided_reply = False
        try:
            async for token in token_stream:
                buffered.append(token)
                accumulated = "".join(buffered).lstrip()
                if not accumulated:
                    continue
                if self._looks_like_wait(accumulated):
                    # User is still mid-thought. Drain silently and stay
                    # quiet so we don't interrupt them.
                    await _drain(token_stream)
                    return ""
                if len(accumulated) >= self._config.decision_peek_chars or (
                    not accumulated.startswith("<") and len(accumulated) >= 2
                ):
                    decided_reply = True
                    break
        except asyncio.CancelledError:
            await _drain_and_close(token_stream)
            raise

        if not decided_reply:
            # Stream ended before we saw enough to decide. If it was
            # plainly a wait we'd have returned above; treat everything
            # we did get as the reply.
            full = "".join(buffered).strip()
            if not full or self._looks_like_wait(full):
                return ""
            return await self._commit_speak(_iter_once(full), token_stream)

        # We've committed to speaking. Replay buffered prefix then
        # continue pulling from the live stream into TTS.
        combined = _prepend_stream(buffered, token_stream)
        return await self._commit_speak(combined, token_stream)

    async def _commit_speak(
        self,
        token_source: AsyncIterator[str],
        upstream: AsyncIterator[str],
    ) -> str:
        # Wait for real local mic silence before committing. If the user
        # starts talking again while we poll, this task will be cancelled
        # by _restart_speculation on the next transcript update.
        while True:
            silence = self._transcriber.silence_seconds()
            if silence >= self._config.commit_silence_s:
                break
            await asyncio.sleep(self._config.silence_poll_s)
        # Mute the mic while TTS plays so Deepgram doesn't transcribe our
        # own audio and cancel the speculation in a feedback loop.
        self._transcriber.mute()
        try:
            text, _interrupted = await self._speak_with_bargein(token_source)
            return text
        except asyncio.CancelledError:
            await _drain_and_close(upstream)
            raise

    async def _speak_with_bargein(
        self,
        token_source: AsyncIterator[str],
    ) -> tuple[str, bool]:
        """Run speak_stream with a concurrent barge-in watcher.

        Returns (assistant_text, interrupted).  When interrupted is True
        the speaker was aborted mid-playback because the user started
        talking (detected via local mic VAD).  Assumes headphones — no
        echo cancellation.
        """
        threshold = self._config.bargein_threshold_s
        if threshold <= 0:
            # Barge-in disabled.
            text = await self._speaker.speak_stream(
                token_source, self._session_clock, self._logger
            )
            return text, False

        # Wrap the token source so we can capture partial text on barge-in.
        capture = _TextCapture(token_source)

        speak_task = asyncio.create_task(
            self._speaker.speak_stream(
                capture, self._session_clock, self._logger
            )
        )
        watcher_task = asyncio.create_task(
            self._bargein_watcher(threshold)
        )

        done, pending = await asyncio.wait(
            [speak_task, watcher_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        if watcher_task in done:
            # User started talking — cancel TTS.
            speak_task.cancel()
            try:
                await speak_task
            except (asyncio.CancelledError, Exception):
                pass
            partial = capture.text.strip()
            print("[crosstalk] barge-in detected, stopping playback")
            return partial, True

        # Playback finished normally.
        watcher_task.cancel()
        try:
            await watcher_task
        except (asyncio.CancelledError, Exception):
            pass
        text = speak_task.result()
        return text, False

    async def _bargein_watcher(self, threshold: float) -> None:
        """Poll local mic VAD until speech is detected."""
        # Give playback a brief grace period so the first audio frame
        # doesn't false-trigger on ambient noise pickup.
        await asyncio.sleep(0.25)
        while True:
            silence = self._transcriber.silence_seconds()
            if silence < threshold:
                return  # User is speaking.
            await asyncio.sleep(self._config.silence_poll_s)

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
        # Accept minor variations the LLM might emit despite instructions.
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


async def _prepend_stream(
    prefix: list[str], remaining: AsyncIterator[str]
) -> AsyncIterator[str]:
    for token in prefix:
        yield token
    async for token in remaining:
        yield token


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
