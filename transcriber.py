from __future__ import annotations

import asyncio
import json
import math
import struct
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode

import sounddevice as sd
import websockets

from log import (
    CharEvent,
    LatencyLogger,
    SessionLogger,
    TurnMetricEvent,
    TurnMetricsLogger,
    WordLatencyEvent,
    append_note,
)


def _coerce_device(value: int | str | None) -> int | str | None:
    return value if value not in ("", None) else None


@dataclass(slots=True)
class DeepgramConfig:
    api_key: str
    model: str = "nova-3"
    language: str = "en-US"
    sample_rate: int = 16000
    punctuate: bool = False
    smart_format: bool = False
    endpointing_ms: int = 120
    utterance_end_ms: int | None = None
    channels: int = 1
    blocksize: int = 480
    mic_device: int | str | None = None


@dataclass(slots=True)
class TimedWord:
    text: str
    start: float
    end: float


@dataclass(slots=True)
class UtteranceResult:
    transcript: str
    started_at: float
    ended_at: float


@dataclass(slots=True)
class TranscriptUpdate:
    """A single incremental update from Deepgram during a user turn.

    `transcript` is the full cumulative user transcript so far in this turn.
    `is_final` means Deepgram marked this segment as final (but the user may
    still continue speaking). `speech_final` / `utterance_end` mean the turn
    is over and no more updates will follow.
    """

    transcript: str
    is_final: bool
    speech_final: bool
    utterance_end: bool
    at: float


@dataclass(slots=True)
class _SegmentState:
    last_words: list[TimedWord] = field(default_factory=list)
    emitted_until: int = 0
    emitted_text: dict[int, str] = field(default_factory=dict)
    final_words: list[TimedWord] = field(default_factory=list)


class _TurnTiming:
    def __init__(self, fallback_origin: float) -> None:
        self._fallback_origin = fallback_origin
        self._capture_origin: float | None = None
        self._lock = threading.Lock()

    def set_capture_origin(self, origin: float) -> None:
        with self._lock:
            if self._capture_origin is None:
                self._capture_origin = origin

    def word_time(self, offset_seconds: float) -> float:
        with self._lock:
            origin = self._capture_origin
        if origin is None:
            origin = self._fallback_origin
        return origin + offset_seconds


class _LocalVAD:
    """Lightweight energy-based voice-activity detector running on the mic thread.

    Supports auto-calibration: call `calibrate()` at startup to measure the
    ambient noise floor, then speech threshold is set to `noise_floor * multiplier`.
    Also provides `speech_onset_since(t)` which detects a *transition* from
    silence to speech after timestamp `t`, avoiding false triggers from
    sustained ambient noise.
    """

    # How many consecutive speech frames count as an onset.
    # At blocksize=160 / 16kHz each frame is 10ms, so 10 = ~100ms sustained.
    ONSET_FRAMES: int = 10

    def __init__(
        self,
        session_time: Callable[[], float],
        threshold: float = 1500.0,
        multiplier: float = 6.0,
    ) -> None:
        self._session_time = session_time
        self._threshold = threshold
        self._multiplier = multiplier
        self._noise_floor: float | None = None
        self._last_speech_at: float = session_time()
        self._lock = threading.Lock()
        # Calibration state
        self._cal_samples: list[float] = []
        self._calibrating = False
        # Onset detection: track consecutive speech frames and when onset began
        self._consec_speech: int = 0
        self._onset_at: float | None = None

    def start_calibration(self) -> None:
        """Begin collecting RMS samples for noise-floor calibration."""
        with self._lock:
            self._cal_samples = []
            self._calibrating = True

    def finish_calibration(self) -> float:
        """Finish calibration, compute threshold, return noise floor RMS."""
        with self._lock:
            self._calibrating = False
            if not self._cal_samples:
                return self._threshold
            self._noise_floor = sum(self._cal_samples) / len(self._cal_samples)
            self._threshold = self._noise_floor * self._multiplier
            return self._noise_floor

    @staticmethod
    def _rms(pcm_bytes: bytes) -> float:
        n_samples = len(pcm_bytes) // 2
        if n_samples == 0:
            return 0.0
        samples = struct.unpack(f"<{n_samples}h", pcm_bytes)
        return math.sqrt(sum(s * s for s in samples) / n_samples)

    def feed(self, pcm_bytes: bytes) -> None:
        """Called from the mic callback thread with raw int16 PCM."""
        rms = self._rms(pcm_bytes)
        now = self._session_time()
        with self._lock:
            if self._calibrating:
                self._cal_samples.append(rms)
                return
            is_speech = rms >= self._threshold
            if is_speech:
                self._last_speech_at = now
                self._consec_speech += 1
                if self._consec_speech >= self.ONSET_FRAMES and self._onset_at is None:
                    self._onset_at = now
            else:
                self._consec_speech = 0

    def silence_seconds(self) -> float:
        """How many seconds of silence since the last speech frame."""
        now = self._session_time()
        with self._lock:
            return max(now - self._last_speech_at, 0.0)

    def last_speech_at(self) -> float:
        with self._lock:
            return self._last_speech_at

    def speech_onset_since(self, since: float) -> bool:
        """True if a speech onset (transition from silence) occurred after `since`."""
        with self._lock:
            return self._onset_at is not None and self._onset_at > since

    def clear_onset(self) -> None:
        """Reset the onset marker (e.g. after consuming it)."""
        with self._lock:
            self._onset_at = None
            self._consec_speech = 0

    def reset(self) -> None:
        """Mark current time as last speech (e.g. at turn start)."""
        now = self._session_time()
        with self._lock:
            self._last_speech_at = now
            self._onset_at = None
            self._consec_speech = 0


class _MicrophoneCapture:
    def __init__(
        self,
        audio_queue: asyncio.Queue[bytes],
        sample_rate: int,
        channels: int,
        blocksize: int,
        device: int | str | None,
        session_time: Callable[[], float],
        timing: _TurnTiming,
        is_muted: Callable[[], bool] | None = None,
        vad: _LocalVAD | None = None,
    ) -> None:
        self._audio_queue = audio_queue
        self._sample_rate = sample_rate
        self._channels = channels
        self._blocksize = blocksize
        self._device = _coerce_device(device)
        self._session_time = session_time
        self._timing = timing
        self._is_muted = is_muted
        self._vad = vad
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stream: sd.RawInputStream | None = None

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._stream = sd.RawInputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype="int16",
            blocksize=self._blocksize,
            device=self._device,
            latency="low",
            callback=self._on_audio,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is None:
            return
        self._stream.stop()
        self._stream.close()
        self._stream = None

    def _on_audio(self, indata: Any, frames: int, time_info: Any, status: Any) -> None:
        del status
        callback_now = self._session_time()
        current_time = float(getattr(time_info, "currentTime", 0.0) or 0.0)
        input_adc_time = float(getattr(time_info, "inputBufferAdcTime", 0.0) or 0.0)
        if current_time and input_adc_time:
            capture_start = callback_now - max(current_time - input_adc_time, 0.0)
        else:
            capture_start = callback_now - (frames / self._sample_rate)
        self._timing.set_capture_origin(capture_start)
        # Always feed the VAD from the real mic audio so local onset/silence
        # detection still works even when we choose to send silence upstream.
        chunk = bytes(indata)
        if self._vad is not None:
            self._vad.feed(chunk)
        if self._loop is None:
            return
        # Keep Deepgram's stream clock aligned with wall clock by continuing
        # to send real-time audio cadence while muted, but replace the mic
        # frame with silence. If we stop sending entirely, Deepgram word
        # offsets collapse all muted gaps and our absolute timestamps drift.
        if self._is_muted is not None and self._is_muted():
            chunk = b"\x00" * len(chunk)
        try:
            self._loop.call_soon_threadsafe(self._audio_queue.put_nowait, chunk)
        except RuntimeError:
            return


class _TranscriptEmitter:
    def __init__(
        self,
        logger: SessionLogger,
        latency_logger: LatencyLogger,
        session_time: Callable[[], float],
        timing: _TurnTiming,
    ) -> None:
        self._logger = logger
        self._latency_logger = latency_logger
        self._session_time = session_time
        self._timing = timing
        self._last_word: TimedWord | None = None

    def emit_word(self, word: TimedWord, notes: str, revised: bool = False) -> None:
        saved_at = self._session_time()  # Timestamp immediately on arrival.
        events: list[CharEvent] = []
        word_start = self._timing.word_time(word.start)
        word_end = self._timing.word_time(word.end)
        if not revised and self._last_word is not None:
            gap_start = self._timing.word_time(self._last_word.end)
            gap_end = self._timing.word_time(max(self._last_word.end, word.start))
            gap_notes = notes
            if gap_end - gap_start > 0.25:
                gap_notes = append_note(gap_notes, "silence")
            events.append(
                CharEvent(
                    char=" ",
                    start=gap_start,
                    end=gap_end,
                    source="user",
                    notes=gap_notes,
                )
            )

        text = word.text
        duration = max(word_end - word_start, 0.0)
        if not text:
            return

        for index, character in enumerate(text):
            char_start = word_start + (duration * index / len(text))
            char_end = word_start + (duration * (index + 1) / len(text))
            events.append(
                CharEvent(
                    char=character,
                    start=char_start,
                    end=char_end,
                    source="user",
                    notes=notes,
                )
            )

        self._logger.write_events(events)
        self._latency_logger.write_event(
            WordLatencyEvent(
                word=text,
                source="user",
                spoken_start=word_start,
                spoken_end=word_end,
                saved_at=saved_at,
                latency_ms=max((saved_at - word_end) * 1000.0, 0.0),
                revised=revised,
                notes=notes,
            )
        )
        if not revised:
            self._last_word = word


class _TranscriptState:
    # Skip events that arrive within this window after state creation.
    # Prevents stale Deepgram results from the previous turn's audio.
    STALE_GUARD_S: float = 0.5

    def __init__(
        self,
        logger: SessionLogger,
        latency_logger: LatencyLogger,
        session_time: Callable[[], float],
        timing: _TurnTiming,
        updates: asyncio.Queue[TranscriptUpdate | None] | None = None,
        turn_id: int | None = None,
        turn_metrics_logger: TurnMetricsLogger | None = None,
    ) -> None:
        self._segments: dict[float, _SegmentState] = {}
        self._final_segments: dict[float, list[TimedWord]] = {}
        self._latest_interim_transcript = ""
        self._latest_final_transcript = ""
        self._completed = asyncio.Event()
        self._session_time = session_time
        self._created_at = session_time()
        self._updates = updates
        self._turn_id = turn_id
        self._turn_metrics_logger = turn_metrics_logger
        self._logged_first_partial = False
        self._emitter = _TranscriptEmitter(
            logger=logger,
            latency_logger=latency_logger,
            session_time=session_time,
            timing=timing,
        )

    def log_metric(
        self,
        event: str,
        *,
        at: float | None = None,
        source: str = "",
        text: str = "",
        info: dict[str, Any] | None = None,
    ) -> None:
        if self._turn_id is None or self._turn_metrics_logger is None:
            return
        self._turn_metrics_logger.write_event(
            TurnMetricEvent(
                turn_id=self._turn_id,
                event=event,
                at=self._session_time() if at is None else at,
                source=source,
                text=text,
                info=info or {},
            )
        )

    def complete(self, utterance_end: bool = False) -> None:
        if self._completed.is_set():
            return
        self._completed.set()
        if self._updates is not None:
            # Emit a terminal update so iter_utterance consumers see end-of-turn.
            self._updates.put_nowait(
                TranscriptUpdate(
                    transcript=self.transcript,
                    is_final=True,
                    speech_final=True,
                    utterance_end=utterance_end,
                    at=self._session_time(),
                )
            )
            # Sentinel so iter_utterance can shut down cleanly.
            self._updates.put_nowait(None)

    async def wait(self) -> None:
        await self._completed.wait()

    @property
    def transcript(self) -> str:
        ordered_words: list[TimedWord] = []
        for key in sorted(self._segments):
            segment = self._segments[key]
            ordered_words.extend(segment.final_words or segment.last_words)
        if ordered_words:
            return " ".join(word.text for word in ordered_words).strip()
        return (self._latest_final_transcript or self._latest_interim_transcript).strip()

    @property
    def has_transcript(self) -> bool:
        return bool(self.transcript)

    def handle_results(self, message: dict[str, Any]) -> None:
        is_final = bool(message.get("is_final"))
        alternative = message.get("channel", {}).get("alternatives", [{}])[0]
        transcript = (alternative.get("transcript") or "").strip()
        if transcript:
            self._latest_interim_transcript = transcript
            if not self._logged_first_partial:
                self.log_metric(
                    "stt_first_partial",
                    source="stt",
                    text=transcript,
                )
                self._logged_first_partial = True

        words = [
            TimedWord(
                text=(raw_word.get("word") or raw_word.get("punctuated_word") or "").strip(),
                start=float(raw_word["start"]),
                end=float(raw_word["end"]),
            )
            for raw_word in alternative.get("words", [])
            if (raw_word.get("word") or raw_word.get("punctuated_word"))
        ]

        segment_key = round(float(message.get("start", 0.0)), 3)
        segment = self._segments.setdefault(segment_key, _SegmentState())

        shared = min(segment.emitted_until, len(words))
        for index in range(shared):
            previous_text = segment.emitted_text.get(index)
            current_text = words[index].text
            if previous_text and previous_text != current_text:
                self._emitter.emit_word(
                    words[index],
                    notes="interpolated,revised",
                    revised=True,
                )
                segment.emitted_text[index] = current_text

        for index in range(segment.emitted_until, len(words)):
            notes = "interpolated"
            if not is_final:
                notes = append_note(notes, "interim")
            self._emitter.emit_word(words[index], notes=notes)
            segment.emitted_text[index] = words[index].text

        if len(words) > segment.emitted_until:
            segment.emitted_until = len(words)

        if is_final:
            segment.final_words = words
            self._final_segments[segment_key] = words
            if transcript:
                self._latest_final_transcript = transcript

        segment.last_words = words

        if self._updates is not None and not self._completed.is_set():
            self._updates.put_nowait(
                TranscriptUpdate(
                    transcript=self.transcript,
                    is_final=is_final,
                    speech_final=bool(message.get("speech_final")),
                    utterance_end=False,
                    at=self._session_time(),
                )
            )


class DeepgramTranscriber:
    def __init__(self, config: DeepgramConfig) -> None:
        self._config = config
        self._timing: _TurnTiming | None = None
        self._mic: _MicrophoneCapture | None = None
        self._vad: _LocalVAD | None = None
        self._websocket: websockets.WebSocketClientProtocol | None = None
        self._audio_queue: asyncio.Queue[bytes] | None = None
        self._sender_task: asyncio.Task[None] | None = None
        self._receiver_task: asyncio.Task[None] | None = None
        self._current_state: _TranscriptState | None = None
        # Start muted so the mic feeds silence to Deepgram until the first
        # capture_utterance() call unmutes us. Real audio only flows during
        # an active turn.
        self._muted: bool = True

    def silence_seconds(self) -> float:
        """Seconds of mic silence since the last speech energy frame."""
        if self._vad is None:
            return 0.0
        return self._vad.silence_seconds()

    def last_speech_at(self) -> float:
        if self._vad is None:
            return 0.0
        return self._vad.last_speech_at()

    def speech_onset_since(self, since: float) -> bool:
        """True if a speech onset occurred after the given timestamp."""
        if self._vad is None:
            return False
        return self._vad.speech_onset_since(since)

    def clear_onset(self) -> None:
        """Consume the current onset marker."""
        if self._vad is not None:
            self._vad.clear_onset()

    async def calibrate_vad(self, duration: float = 1.0) -> float:
        """Measure ambient noise floor for `duration` seconds. Returns noise floor RMS.

        Must be called after start(). Blocks for `duration` while the mic
        is already running, collecting RMS samples to set the speech threshold.
        """
        if self._vad is None:
            raise RuntimeError("Call start() before calibrate_vad().")
        self._vad.start_calibration()
        await asyncio.sleep(duration)
        noise_floor = self._vad.finish_calibration()
        return noise_floor

    def mute(self) -> None:
        self._muted = True
        # Complete the current turn so iter_utterance exits immediately.
        # Without this, Deepgram never sees silence (we stopped sending)
        # and never fires speech_final, so the turn hangs forever.
        state = self._current_state
        if state is not None:
            state.complete(utterance_end=False)
        # Clear the state so stale Deepgram events that arrive after
        # muting don't accidentally land on the next turn's state.
        self._current_state = None

    def unmute(self) -> None:
        # Drain any stale audio frames that accumulated while muted so
        # Deepgram doesn't see old data at the start of the new turn.
        if self._audio_queue is not None:
            while not self._audio_queue.empty():
                try:
                    self._audio_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
        # Reset the VAD so old silence duration doesn't carry over.
        if self._vad is not None:
            self._vad.reset()
        self._muted = False

    async def start(self, session_time: Callable[[], float]) -> None:
        """Open the Deepgram WebSocket and the mic stream once, reuse across turns."""
        if self._websocket is not None:
            return

        loop = asyncio.get_running_loop()
        self._timing = _TurnTiming(session_time())
        self._audio_queue = asyncio.Queue()
        self._vad = _LocalVAD(
            session_time=session_time,
        )
        self._mic = _MicrophoneCapture(
            audio_queue=self._audio_queue,
            sample_rate=self._config.sample_rate,
            channels=self._config.channels,
            blocksize=self._config.blocksize,
            device=self._config.mic_device,
            session_time=session_time,
            timing=self._timing,
            is_muted=lambda: self._muted,
            vad=self._vad,
        )

        params = urlencode(
            {
                "model": self._config.model,
                "language": self._config.language,
                "encoding": "linear16",
                "sample_rate": self._config.sample_rate,
                "channels": self._config.channels,
                "interim_results": "true",
                "vad_events": "true",
                "endpointing": str(self._config.endpointing_ms),
                "punctuate": str(self._config.punctuate).lower(),
                "smart_format": str(self._config.smart_format).lower(),
            }
        )
        if self._config.utterance_end_ms is not None:
            if self._config.utterance_end_ms < 1000:
                raise ValueError(
                    "Deepgram utterance_end_ms must be at least 1000 ms. "
                    "Leave DEEPGRAM_UTTERANCE_END_MS blank to disable it."
                )
            params = f"{params}&utterance_end_ms={self._config.utterance_end_ms}"

        headers = {"Authorization": f"Token {self._config.api_key}"}
        url = f"wss://api.deepgram.com/v1/listen?{params}"

        self._websocket = await websockets.connect(
            url,
            extra_headers=headers,
            ping_interval=20,
            ping_timeout=20,
            max_size=None,
        )
        self._mic.start(loop)
        self._sender_task = asyncio.create_task(self._send_audio())
        self._receiver_task = asyncio.create_task(self._receive_events())

    async def stop(self) -> None:
        if self._mic is not None:
            self._mic.stop()
            self._mic = None
        if self._sender_task is not None:
            self._sender_task.cancel()
        if self._receiver_task is not None:
            self._receiver_task.cancel()
        if self._websocket is not None:
            try:
                await self._websocket.send(json.dumps({"type": "CloseStream"}))
            except Exception:
                pass
            try:
                await self._websocket.close()
            except Exception:
                pass
            self._websocket = None
        pending = [t for t in (self._sender_task, self._receiver_task) if t is not None]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self._sender_task = None
        self._receiver_task = None

    async def capture_utterance(
        self,
        session_time: Callable[[], float],
        logger: SessionLogger,
        latency_logger: LatencyLogger,
        turn_id: int | None = None,
        turn_metrics_logger: TurnMetricsLogger | None = None,
    ) -> UtteranceResult:
        await self.start(session_time)
        assert self._timing is not None
        turn_offset = session_time()
        state = _TranscriptState(
            logger=logger,
            latency_logger=latency_logger,
            session_time=session_time,
            timing=self._timing,
            turn_id=turn_id,
            turn_metrics_logger=turn_metrics_logger,
        )
        self._current_state = state
        try:
            await state.wait()
        finally:
            self._current_state = None
        return UtteranceResult(
            transcript=state.transcript,
            started_at=turn_offset,
            ended_at=session_time(),
        )

    async def iter_utterance(
        self,
        session_time: Callable[[], float],
        logger: SessionLogger,
        latency_logger: LatencyLogger,
        turn_id: int | None = None,
        turn_metrics_logger: TurnMetricsLogger | None = None,
    ):
        """Yield TranscriptUpdate events for a single user turn.

        The final update has speech_final=True. After that the generator
        terminates; callers should break their loop on the final update.
        """
        await self.start(session_time)
        assert self._timing is not None
        updates: asyncio.Queue[TranscriptUpdate | None] = asyncio.Queue()
        state = _TranscriptState(
            logger=logger,
            latency_logger=latency_logger,
            session_time=session_time,
            timing=self._timing,
            updates=updates,
            turn_id=turn_id,
            turn_metrics_logger=turn_metrics_logger,
        )
        self._current_state = state
        try:
            while True:
                try:
                    update = await asyncio.wait_for(updates.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    # No speech detected for 15s — end turn to avoid hanging.
                    return
                if update is None:
                    return
                yield update
                if update.speech_final or update.utterance_end:
                    return
        finally:
            self._current_state = None
            # Reset turn timing so the next turn gets a fresh capture origin.
            # This also prevents stale Deepgram results (still in flight from
            # the previous audio) from being timestamped against the old origin
            # if they arrive after the next iter_utterance installs a new state.
            self._timing = _TurnTiming(session_time())

    async def _send_audio(self) -> None:
        assert self._websocket is not None and self._audio_queue is not None
        websocket = self._websocket
        audio_queue = self._audio_queue
        while True:
            try:
                chunk = await asyncio.wait_for(audio_queue.get(), timeout=3.0)
            except asyncio.TimeoutError:
                try:
                    await websocket.send(json.dumps({"type": "KeepAlive"}))
                except Exception:
                    return
                continue
            except asyncio.CancelledError:
                return
            # Batch: drain all queued chunks into one websocket send.
            # Prevents backlog when the event loop falls behind.
            parts = [chunk]
            while not audio_queue.empty():
                try:
                    parts.append(audio_queue.get_nowait())
                except Exception:
                    break
            payload = b"".join(parts) if len(parts) > 1 else chunk
            try:
                await websocket.send(payload)
            except Exception:
                return

    async def _receive_events(self) -> None:
        assert self._websocket is not None
        websocket = self._websocket
        state: _TranscriptState | None = None
        try:
            async for raw_message in websocket:
                if isinstance(raw_message, bytes):
                    continue
                message = json.loads(raw_message)
                event_type = message.get("type")
                if event_type == "Results":
                    state = self._current_state
                    if state is None:
                        # Between turns (mute period); drop any stray results.
                        continue
                    # Always process interim results so speculation fires
                    # immediately.  Only block *completion* events during the
                    # stale-guard window to prevent ghost turn endings.
                    state.handle_results(message)
                    if message.get("speech_final") and state.has_transcript:
                        age = state._session_time() - state._created_at
                        if age < state.STALE_GUARD_S:
                            continue
                        state.log_metric(
                            "user_last_speech_local",
                            at=self.last_speech_at(),
                            source="user",
                            text=state.transcript,
                        )
                        state.log_metric(
                            "stt_speech_final",
                            source="stt",
                            text=state.transcript,
                        )
                        state.complete()
                elif event_type == "UtteranceEnd":
                    state = self._current_state
                    if state is None:
                        continue
                    age = state._session_time() - state._created_at
                    if age < state.STALE_GUARD_S:
                        continue
                    if state.has_transcript:
                        state.log_metric(
                            "user_last_speech_local",
                            at=self.last_speech_at(),
                            source="user",
                            text=state.transcript,
                            info={"via": "utterance_end"},
                        )
                        state.log_metric(
                            "stt_utterance_end",
                            source="stt",
                            text=state.transcript,
                        )
                        state.complete(utterance_end=True)
        except (websockets.ConnectionClosed, asyncio.CancelledError):
            return
        finally:
            stuck = self._current_state
            if stuck is not None:
                stuck.complete()
