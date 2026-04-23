from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from crosstalk import CrosstalkConfig, CrosstalkRunner
from llm import GroqChatClient, GroqConfig
from log import (
    LatencyLogger,
    SessionClock,
    SessionLogger,
    TurnMetricEvent,
    TurnMetricsLogger,
)
from transcriber import DeepgramConfig, DeepgramTranscriber
from tts import ElevenLabsConfig, ElevenLabsSpeaker, SpeakHooks


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _optional_device(name: str) -> int | str | None:
    value = os.getenv(name, "").strip()
    if not value:
        return None
    return int(value) if value.isdigit() else value


def _optional_int(name: str) -> int | None:
    value = os.getenv(name, "").strip()
    if not value:
        return None
    return int(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"Invalid boolean value for {name}: {value!r}")


def _log_path() -> Path:
    configured = os.getenv("LOG_PATH", "").strip()
    if configured:
        return Path(configured)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("logs") / f"session-{timestamp}.jsonl"


def _latency_log_path() -> Path:
    configured = os.getenv("LATENCY_LOG_PATH", "").strip()
    if configured:
        return Path(configured)
    return Path("logs") / "latency.jsonl"


def _turn_metrics_log_path() -> Path:
    configured = os.getenv("TURN_METRICS_PATH", "").strip()
    if configured:
        return Path(configured)
    return Path("logs") / "turn_metrics.jsonl"


async def _trace_first_token(
    stream: AsyncIterator[str],
    turn_id: int,
    event: str,
    clock: SessionClock,
    logger: TurnMetricsLogger,
) -> AsyncIterator[str]:
    saw_first_token = False
    async for token in stream:
        if not saw_first_token:
            logger.write_event(
                TurnMetricEvent(
                    turn_id=turn_id,
                    event=event,
                    at=clock.now(),
                    source="llm",
                )
            )
            saw_first_token = True
        yield token


async def _run() -> None:
    load_dotenv()

    session_clock = SessionClock()
    logger = SessionLogger(_log_path())
    latency_logger = LatencyLogger(_latency_log_path())
    turn_metrics_logger = TurnMetricsLogger(_turn_metrics_log_path())

    deepgram = DeepgramTranscriber(
        DeepgramConfig(
            api_key=_required_env("DEEPGRAM_API_KEY"),
            model=os.getenv("DEEPGRAM_MODEL", "nova-3"),
            language=os.getenv("DEEPGRAM_LANGUAGE", "en-US"),
            sample_rate=int(os.getenv("DEEPGRAM_SAMPLE_RATE", "16000")),
            punctuate=_env_bool("DEEPGRAM_PUNCTUATE", False),
            smart_format=_env_bool("DEEPGRAM_SMART_FORMAT", False),
            endpointing_ms=int(os.getenv("DEEPGRAM_ENDPOINTING_MS", "120")),
            utterance_end_ms=_optional_int("DEEPGRAM_UTTERANCE_END_MS"),
            blocksize=int(os.getenv("DEEPGRAM_BLOCKSIZE", "480")),
            mic_device=_optional_device("MIC_DEVICE"),
        )
    )
    llm = GroqChatClient(
        GroqConfig(
            api_key=_required_env("GROQ_API_KEY"),
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
            temperature=float(os.getenv("GROQ_TEMPERATURE", "0.2")),
            max_completion_tokens=int(
                os.getenv("GROQ_MAX_COMPLETION_TOKENS", "512")
            ),
        )
    )
    speaker = ElevenLabsSpeaker(
        ElevenLabsConfig(
            api_key=_required_env("ELEVENLABS_API_KEY"),
            voice_id=_required_env("ELEVENLABS_VOICE_ID"),
            model_id=os.getenv("ELEVENLABS_MODEL_ID", "eleven_flash_v2_5"),
            output_format=os.getenv("ELEVENLABS_OUTPUT_FORMAT", "pcm_22050"),
            stability=float(os.getenv("ELEVENLABS_STABILITY", "0.45")),
            similarity_boost=float(
                os.getenv("ELEVENLABS_SIMILARITY_BOOST", "0.8")
            ),
            speed=float(os.getenv("ELEVENLABS_SPEED", "1.0")),
            speaker_device=_optional_device("SPEAKER_DEVICE"),
        )
    )

    system_prompt = os.getenv(
        "SYSTEM_PROMPT",
        "You are a concise, helpful voice assistant.",
    )
    conversation: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt}
    ]

    crosstalk_enabled = _env_bool("CROSSTALK_ENABLED", False)
    crosstalk: CrosstalkRunner | None = None
    if crosstalk_enabled:
        crosstalk = CrosstalkRunner(
            transcriber=deepgram,
            llm=llm,
            speaker=speaker,
            session_clock=session_clock,
            logger=logger,
            latency_logger=latency_logger,
            turn_metrics_logger=turn_metrics_logger,
            config=CrosstalkConfig(
                base_system_prompt=system_prompt,
                bargein_enabled=_env_bool("BARGEIN_ENABLED", False),
            ),
        )

    print(f"Logging char timings to {logger.path}")
    print(f"Logging word latencies to {latency_logger.path}")
    print(f"Logging turn metrics to {turn_metrics_logger.path}")

    # Start the mic + Deepgram connection.
    await deepgram.start(session_clock.now)

    # Only calibrate VAD when crosstalk is enabled (barge-in needs it).
    if crosstalk_enabled:
        print("Calibrating mic noise floor (stay quiet for 1 second)...")
        noise_floor = await deepgram.calibrate_vad(duration=1.0)
        print(f"Noise floor: {noise_floor:.0f} RMS — speech threshold set to {noise_floor * 6:.0f} RMS")

    mode = "crosstalk" if crosstalk_enabled else "turn-based"
    print(f"Listening ({mode}). Say 'exit' or press Ctrl+C to stop.")
    if not crosstalk_enabled:
        print("Tip: Set CROSSTALK_ENABLED=true in .env for speculative turn-taking and barge-in.")

    first_turn = True
    turn_id = 0

    try:
        while True:
            turn_id += 1
            if first_turn:
                first_turn = False
            else:
                print("Listening for next turn...")

            if crosstalk is not None:
                result = await crosstalk.run_turn(conversation, turn_id=turn_id)
                transcript = result.user_text.strip()
                if not transcript:
                    continue
                print(f"user> {transcript}")
                if transcript.lower() in {"exit", "quit", "stop"}:
                    break
                assistant_text = result.assistant_text.strip()
                conversation.append({"role": "user", "content": transcript})
                if result.interrupted:
                    # Agent was interrupted mid-reply. Show what was said
                    # before the interruption but don't add it to
                    # conversation history so the next turn starts cleanly.
                    if assistant_text:
                        print(f"assistant> {assistant_text}...")
                    print("(interrupted)")
                elif assistant_text:
                    print(f"assistant> {assistant_text}")
                    conversation.append(
                        {"role": "assistant", "content": assistant_text}
                    )
                continue

            turn_metrics_logger.write_event(
                TurnMetricEvent(
                    turn_id=turn_id,
                    event="turn_started",
                    at=session_clock.now(),
                    source="system",
                )
            )
            deepgram.unmute()
            turn_metrics_logger.write_event(
                TurnMetricEvent(
                    turn_id=turn_id,
                    event="mic_unmuted",
                    at=session_clock.now(),
                    source="system",
                )
            )
            utterance = await deepgram.capture_utterance(
                session_clock.now,
                logger,
                latency_logger,
                turn_id=turn_id,
                turn_metrics_logger=turn_metrics_logger,
            )
            deepgram.mute()
            transcript = utterance.transcript.strip()
            if not transcript:
                continue
            print(f"user> {transcript}")

            if transcript.lower() in {"exit", "quit", "stop"}:
                break

            conversation.append({"role": "user", "content": transcript})
            turn_metrics_logger.write_event(
                TurnMetricEvent(
                    turn_id=turn_id,
                    event="llm_reply_started",
                    at=session_clock.now(),
                    source="llm",
                    text=transcript,
                )
            )
            llm_stream = _trace_first_token(
                llm.stream_response(conversation),
                turn_id=turn_id,
                event="llm_reply_first_token",
                clock=session_clock,
                logger=turn_metrics_logger,
            )
            assistant_text = await speaker.speak_stream(
                llm_stream,
                session_clock,
                logger,
                hooks=SpeakHooks(
                    on_first_text_sent=lambda at, chunk: turn_metrics_logger.write_event(
                        TurnMetricEvent(
                            turn_id=turn_id,
                            event="tts_first_text_sent",
                            at=at,
                            source="tts",
                            text=chunk[:80],
                        )
                    ),
                    on_first_audio_received=lambda at: turn_metrics_logger.write_event(
                        TurnMetricEvent(
                            turn_id=turn_id,
                            event="tts_first_audio_received",
                            at=at,
                            source="tts",
                        )
                    ),
                    on_first_audio_played=lambda at: turn_metrics_logger.write_event(
                        TurnMetricEvent(
                            turn_id=turn_id,
                            event="tts_first_audio_played",
                            at=at,
                            source="tts",
                        )
                    ),
                ),
            )
            if assistant_text:
                print(f"assistant> {assistant_text}")
                conversation.append({"role": "assistant", "content": assistant_text})
    finally:
        await deepgram.stop()
        logger.close()
        latency_logger.close()
        turn_metrics_logger.close()


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
