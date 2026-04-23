from __future__ import annotations

import asyncio
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from crosstalk import CrosstalkConfig, CrosstalkRunner
from llm import GroqChatClient, GroqConfig
from log import LatencyLogger, SessionClock, SessionLogger
from transcriber import DeepgramConfig, DeepgramTranscriber
from tts import ElevenLabsConfig, ElevenLabsSpeaker


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


async def _run() -> None:
    load_dotenv()

    session_clock = SessionClock()
    logger = SessionLogger(_log_path())
    latency_logger = LatencyLogger(_latency_log_path())

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
            blocksize=int(os.getenv("DEEPGRAM_BLOCKSIZE", "160")),
            mic_device=_optional_device("MIC_DEVICE"),
        )
    )
    llm = GroqChatClient(
        GroqConfig(
            api_key=_required_env("GROQ_API_KEY"),
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
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

    crosstalk_enabled = _env_bool("CROSSTALK_ENABLED", True)
    crosstalk: CrosstalkRunner | None = None
    if crosstalk_enabled:
        crosstalk = CrosstalkRunner(
            transcriber=deepgram,
            llm=llm,
            speaker=speaker,
            session_clock=session_clock,
            logger=logger,
            latency_logger=latency_logger,
            config=CrosstalkConfig(base_system_prompt=system_prompt),
        )

    print(f"Logging char timings to {logger.path}")
    print(f"Logging word latencies to {latency_logger.path}")
    mode = "crosstalk" if crosstalk_enabled else "turn-based"
    print(f"Listening ({mode}). Say 'exit' or press Ctrl+C to stop.")

    first_turn = True

    try:
        while True:
            if first_turn:
                first_turn = False
            else:
                print("Listening for next turn...")

            if crosstalk is not None:
                result = await crosstalk.run_turn(conversation)
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

            deepgram.unmute()
            utterance = await deepgram.capture_utterance(
                session_clock.now,
                logger,
                latency_logger,
            )
            deepgram.mute()
            transcript = utterance.transcript.strip()
            if not transcript:
                continue
            print(f"user> {transcript}")

            if transcript.lower() in {"exit", "quit", "stop"}:
                break

            conversation.append({"role": "user", "content": transcript})
            assistant_text = await speaker.speak_stream(
                llm.stream_response(conversation),
                session_clock,
                logger,
            )
            if assistant_text:
                print(f"assistant> {assistant_text}")
                conversation.append({"role": "assistant", "content": assistant_text})
    finally:
        await deepgram.stop()
        logger.close()
        latency_logger.close()


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
