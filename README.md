# Voice Agent

Part 1 scaffold for a local Python voice agent that:

- streams microphone audio to Deepgram `nova-3`
- streams the transcript to Groq for response generation
- streams Groq output into ElevenLabs WebSocket TTS
- plays the assistant audio locally
- writes both sides into one shared char-timestamp JSONL log
- writes per-word input latency measurements to `latency.jsonl`

## Files

- `main.py`: top-level turn loop
- `transcriber.py`: microphone capture, Deepgram WebSocket client, user-side char timing
- `llm.py`: Groq streaming chat client
- `tts.py`: ElevenLabs WebSocket TTS, PCM playback, assistant-side char timing
- `log.py`: session clock and thread-safe JSONL logger
- `crosstalk.py`: LLM-driven turn-taking (see "Crosstalk" below)

## JSONL schema

Each line is one character event:

```json
{"char":"h","start":1.234,"end":1.267,"source":"user","notes":"interpolated"}
```

`start` and `end` are seconds from the beginning of the local session.

## Setup

1. Create a virtualenv and install dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` and fill in:

- `DEEPGRAM_API_KEY`
- `GROQ_API_KEY`
- `ELEVENLABS_API_KEY`
- `ELEVENLABS_VOICE_ID`

3. Run the agent.

```bash
python3 main.py
```

4. Summarize the input save latency after a run.

```bash
python3 scripts/summarize_latency.py logs/latency.jsonl
```

## Notes

- Deepgram is connected directly to `wss://api.deepgram.com/v1/listen` with `interim_results=true`, `vad_events=true`, and `endpointing`.
- The current turn boundary is driven by `speech_final=true` from Deepgram endpointing. `DEEPGRAM_UTTERANCE_END_MS` is optional and should only be set when you explicitly want server-side UtteranceEnd behavior.
- User-side character timings are derived from Deepgram word timings, so they are tagged `interpolated`.
- User-side latency is measured per emitted word as `saved_at - spoken_end`, where `spoken_end` is Deepgram's `word.end` anchored to the local session clock with the microphone callback ADC timestamp.
- User words are emitted as soon as they first appear in Deepgram interim results. Later interim or final updates that change a previously emitted word are appended with `notes="interpolated,revised"`.
- `DEEPGRAM_SMART_FORMAT` defaults to `false` to keep the realtime path focused on raw transcript timing rather than formatting.
- If you enable `DEEPGRAM_UTTERANCE_END_MS`, Deepgram currently requires a minimum of `1000` ms.
- Assistant-side character timings use ElevenLabs alignment data from the streaming WebSocket response.
- Assistant playback timing is anchored to the local output stream clock plus stream latency. That is the right local reference for playback, but it is still an estimate rather than a hardware loopback measurement.

## Crosstalk

Crosstalk (see [tarzain/crosstalk](https://github.com/tarzain/crosstalk))
removes the awkward pause after the user stops speaking without cutting
them off mid-thought. Instead of waiting for Deepgram's VAD to decide
the turn is over, we let the LLM decide.

**How it works in this repo (`crosstalk.py`):**

1. `DeepgramTranscriber.iter_utterance()` yields a `TranscriptUpdate`
   for every interim/final Deepgram result during a turn, not just at
   `speech_final`.
2. On each update where the cumulative transcript has changed,
   `CrosstalkRunner` cancels any in-flight speculation and starts a new
   Groq completion with the transcript-so-far.
3. The system prompt appends a turn-taking instruction: *"if the user
   appears mid-thought, reply with exactly `<wait/>` and nothing else;
   otherwise respond normally."*
4. We peek at the first tokens of the stream. If they look like
   `<wait/>`, we silently drain the stream and stay quiet. Otherwise we
   replay the buffered tokens into ElevenLabs TTS and keep piping the
   rest of the Groq stream straight through. TTS playback can start
   before Deepgram has declared the turn over.
5. If the user keeps talking, the new interim update cancels the
   in-flight task — Groq stream is closed, `speak_stream` raises
   `CancelledError` and `_PCMPlayer.abort()` drops buffered audio
   immediately (not just `stop()`, which would drain the tail).
6. When `speech_final` / `UtteranceEnd` arrives, we `await` whatever
   speculation is still running and commit its output to the
   conversation history.

This differs from the original crosstalk React app in two ways:

- The original uses Deepgram diarization to detect user speech during AI
  playback and cancels synthesis on any user word. We instead keep the
  mic muted while the assistant speaks (no echo-cancellation dependency)
  and restrict "interruption" to re-speculating across interim updates
  within a user turn. The `<wait/>` / speculative-speak mechanic is the
  same.
- We use Groq chat-completions with a `<wait/>` sentinel instead of the
  original's `Speaker0:` / `Speaker1:` completion-style prompt with stop
  sequences, because chat-only APIs don't expose stop-on-prefix control
  equivalently.

Toggle with `CROSSTALK_ENABLED=true|false` (default `true`). Setting it
to `false` falls back to the plain turn-based loop.

## Known limits

- This is turn-based, not full duplex. The microphone is reopened for each user turn and paused while the assistant speaks.
- The current logger is append-only. Revisions are represented as additional `revised` events, not in-place edits.
- Live validation still depends on working Groq and ElevenLabs keys plus a valid ElevenLabs voice ID.
