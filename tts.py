from __future__ import annotations

import asyncio
import base64
import json
import re
from dataclasses import dataclass
from typing import Any, AsyncIterator
from urllib.parse import urlencode

import sounddevice as sd
import websockets

from log import CharEvent, SessionClock, SessionLogger


def _coerce_device(value: int | str | None) -> int | str | None:
    return value if value not in ("", None) else None


@dataclass(slots=True)
class ElevenLabsConfig:
    api_key: str
    voice_id: str
    model_id: str = "eleven_flash_v2_5"
    output_format: str = "pcm_22050"
    stability: float = 0.45
    similarity_boost: float = 0.8
    speed: float = 1.0
    receive_idle_timeout_s: float = 1.0
    # Safety: no audio ever received this long after we finished sending text.
    post_send_no_audio_timeout_s: float = 5.0
    # Safety: no message of any kind for this long after the last one. With
    # auto_mode=true, ElevenLabs emits isFinal per synthesis context, so we
    # can't use isFinal to detect end-of-stream — we wait for the websocket
    # to close, with this idle gap as a fallback.
    post_send_idle_timeout_s: float = 2.0
    speaker_device: int | str | None = None


class _StreamingTextChunker:
    def __init__(self, min_chars: int = 20, max_chars: int = 140) -> None:
        self._buffer = ""
        self._min_chars = min_chars
        self._max_chars = max_chars

    def push(self, token: str) -> list[str]:
        self._buffer += token
        return self._drain_ready()

    def flush(self) -> list[str]:
        final = self._buffer.strip()
        self._buffer = ""
        return [final] if final else []

    def _drain_ready(self) -> list[str]:
        ready: list[str] = []
        while True:
            sentence = self._pop_sentence()
            if sentence is not None:
                ready.append(sentence)
                continue
            if len(self._buffer) >= self._max_chars:
                wrapped = self._pop_wrapped()
                if wrapped is not None:
                    ready.append(wrapped)
                    continue
            break
        return ready

    def _pop_sentence(self) -> str | None:
        matches = list(re.finditer(r"[.!?](?:\s|$)", self._buffer))
        if not matches:
            return None
        end = matches[-1].end()
        candidate = self._buffer[:end].strip()
        remaining = self._buffer[end:]
        if len(candidate) < self._min_chars and len(remaining.strip()) < self._min_chars:
            return None
        self._buffer = remaining
        return candidate

    def _pop_wrapped(self) -> str | None:
        split_at = self._buffer.rfind(" ", self._min_chars, self._max_chars)
        if split_at == -1:
            return None
        candidate = self._buffer[:split_at].strip()
        self._buffer = self._buffer[split_at + 1 :]
        return candidate or None


class _PCMPlayer:
    def __init__(self, sample_rate: int, device: int | str | None) -> None:
        self._sample_rate = sample_rate
        self._device = _coerce_device(device)
        self._stream: sd.RawOutputStream | None = None
        self._played_frames = 0
        self._playback_origin: float | None = None
        self._scheduled_end: float | None = None

    def start(self) -> None:
        self._stream = sd.RawOutputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="int16",
            device=self._device,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is None:
            return
        self._stream.stop()
        self._stream.close()
        self._stream = None

    def abort(self) -> None:
        """Like stop(), but drop any buffered frames immediately."""
        if self._stream is None:
            return
        try:
            self._stream.abort()
        except Exception:
            pass
        try:
            self._stream.close()
        except Exception:
            pass
        self._stream = None

    async def play(self, audio_chunk: bytes, clock: SessionClock) -> float:
        if self._stream is None:
            raise RuntimeError("Playback stream has not been started.")
        if self._playback_origin is None:
            latency = float(self._stream.latency or 0.0)
            self._playback_origin = clock.now() + latency
        chunk_start = self._playback_origin + (self._played_frames / self._sample_rate)
        await asyncio.to_thread(self._stream.write, audio_chunk)
        frame_count = len(audio_chunk) // 2
        self._played_frames += frame_count
        self._scheduled_end = chunk_start + (frame_count / self._sample_rate)
        return chunk_start

    def pending_audio_seconds(self, clock: SessionClock) -> float:
        if self._scheduled_end is None:
            return 0.0
        return max(self._scheduled_end - clock.now(), 0.0)


class ElevenLabsSpeaker:
    def __init__(self, config: ElevenLabsConfig) -> None:
        self._config = config
        self._sample_rate = self._parse_sample_rate(config.output_format)

    async def speak_stream(
        self,
        token_stream: AsyncIterator[str],
        clock: SessionClock,
        logger: SessionLogger,
    ) -> str:
        params = urlencode(
            {
                "model_id": self._config.model_id,
                "output_format": self._config.output_format,
                "sync_alignment": "true",
                "auto_mode": "true",
            }
        )
        url = (
            "wss://api.elevenlabs.io/v1/text-to-speech/"
            f"{self._config.voice_id}/stream-input?{params}"
        )

        headers = {"xi-api-key": self._config.api_key}
        player = _PCMPlayer(self._sample_rate, self._config.speaker_device)

        aborted = False
        assistant_text = ""
        async with websockets.connect(
            url,
            extra_headers=headers,
            ping_interval=20,
            ping_timeout=20,
            max_size=None,
        ) as websocket:
            player.start()
            send_done = asyncio.Event()
            receiver = asyncio.create_task(
                self._receive_audio(
                    websocket=websocket,
                    player=player,
                    clock=clock,
                    logger=logger,
                    send_done=send_done,
                )
            )
            try:
                await websocket.send(
                    json.dumps(
                        {
                            "text": " ",
                            "voice_settings": {
                                "stability": self._config.stability,
                                "similarity_boost": self._config.similarity_boost,
                                "speed": self._config.speed,
                            },
                        }
                    )
                )
                assistant_text = await self._send_text(websocket, token_stream)
                send_done.set()
                await receiver
            except asyncio.CancelledError:
                aborted = True
                raise
            finally:
                receiver.cancel()
                await asyncio.gather(receiver, return_exceptions=True)
                if aborted:
                    player.abort()
                else:
                    pending_audio = player.pending_audio_seconds(clock)
                    if pending_audio > 0.0:
                        await asyncio.sleep(pending_audio)
                    player.stop()
        return assistant_text

    async def _send_text(
        self,
        websocket: websockets.WebSocketClientProtocol,
        token_stream: AsyncIterator[str],
    ) -> str:
        chunker = _StreamingTextChunker()
        fragments: list[str] = []

        async for token in token_stream:
            fragments.append(token)
            for chunk in chunker.push(token):
                await self._send_text_chunk(websocket, chunk)

        for chunk in chunker.flush():
            await self._send_text_chunk(websocket, chunk)

        # With auto_mode=true, the empty-text EOS message is the synthesis
        # trigger. Don't also send flush=true — mixing the two can cause
        # ElevenLabs to emit isFinal=true before the last audio frames arrive.
        await websocket.send(json.dumps({"text": ""}))
        return "".join(fragments).strip()

    async def _receive_audio(
        self,
        websocket: websockets.WebSocketClientProtocol,
        player: _PCMPlayer,
        clock: SessionClock,
        logger: SessionLogger,
        send_done: asyncio.Event,
    ) -> None:
        heard_audio = False
        send_done_at: float | None = None
        last_message_at: float | None = None
        loop = asyncio.get_running_loop()
        try:
            while True:
                if send_done.is_set() and send_done_at is None:
                    send_done_at = loop.time()
                try:
                    raw_message = await asyncio.wait_for(
                        websocket.recv(),
                        timeout=self._config.receive_idle_timeout_s,
                    )
                except asyncio.TimeoutError:
                    if self._should_finish(
                        now=loop.time(),
                        send_done_at=send_done_at,
                        heard_audio=heard_audio,
                        last_message_at=last_message_at,
                    ):
                        return
                    continue
                except websockets.ConnectionClosed:
                    return

                last_message_at = loop.time()

                if isinstance(raw_message, bytes):
                    continue

                message = json.loads(raw_message)
                audio_base64 = message.get("audio")
                if audio_base64:
                    heard_audio = True
                    audio_chunk = base64.b64decode(audio_base64)
                    chunk_start = await player.play(audio_chunk, clock)
                    alignment = (
                        message.get("normalizedAlignment")
                        or message.get("normalized_alignment")
                        or message.get("alignment")
                    )
                    if alignment:
                        self._log_alignment(alignment, chunk_start, logger)
                # Note: we deliberately ignore isFinal. With auto_mode=true and
                # chunked text, isFinal arrives per synthesis context (multiple
                # times per response), so it's not a reliable end-of-stream
                # marker. We rely on websocket close + idle-timeout fallback.
        except asyncio.CancelledError:
            return

    def _should_finish(
        self,
        *,
        now: float,
        send_done_at: float | None,
        heard_audio: bool,
        last_message_at: float | None,
    ) -> bool:
        if send_done_at is None:
            return False
        if not heard_audio:
            # Synthesis never started — give up after a longer wait.
            return (now - send_done_at) >= self._config.post_send_no_audio_timeout_s
        # Heard audio. Wait for either the websocket to close (handled in the
        # caller) or for a sustained idle gap after the last received message.
        if last_message_at is None:
            return False
        return (now - last_message_at) >= self._config.post_send_idle_timeout_s

    @staticmethod
    async def _send_text_chunk(
        websocket: websockets.WebSocketClientProtocol,
        chunk: str,
        *,
        flush: bool = False,
    ) -> None:
        payload: dict[str, Any] = {"text": chunk}
        if flush:
            payload["flush"] = True
        await websocket.send(json.dumps(payload))

    @staticmethod
    def _parse_sample_rate(output_format: str) -> int:
        parts = output_format.split("_")
        if len(parts) < 2 or not parts[1].isdigit():
            raise ValueError(
                f"Expected an ElevenLabs PCM output format like 'pcm_22050', got {output_format!r}."
            )
        return int(parts[1])

    @staticmethod
    def _log_alignment(
        alignment: dict[str, Any],
        chunk_start: float,
        logger: SessionLogger,
    ) -> None:
        chars = alignment.get("chars") or []
        starts = (
            alignment.get("charStartTimesMs")
            or alignment.get("char_start_times_ms")
            or []
        )
        durations = (
            alignment.get("charDurationsMs")
            or alignment.get("char_durations_ms")
            or []
        )
        events = [
            CharEvent(
                char=char,
                start=chunk_start + (start_ms / 1000.0),
                end=chunk_start + ((start_ms + duration_ms) / 1000.0),
                source="assistant",
                notes="",
            )
            for char, start_ms, duration_ms in zip(chars, starts, durations)
        ]
        logger.write_events(events)
