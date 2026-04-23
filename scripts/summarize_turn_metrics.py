from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        raise ValueError("Cannot compute a percentile of an empty list.")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * pct
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _load_events(path: Path) -> dict[int, dict[str, list[float]]]:
    grouped: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            turn_id = int(payload["turn_id"])
            grouped[turn_id][payload["event"]].append(float(payload["at"]))
    return grouped


def _first(events: dict[str, list[float]], *names: str) -> float | None:
    values: list[float] = []
    for name in names:
        values.extend(events.get(name, []))
    if not values:
        return None
    return min(values)


def _collect_spans(grouped: dict[int, dict[str, list[float]]]) -> dict[str, list[float]]:
    spans: dict[str, list[float]] = defaultdict(list)
    for events in grouped.values():
        user_last_speech = _first(events, "user_last_speech_local")
        stt_done = _first(events, "stt_speech_final", "stt_utterance_end")
        llm_first_token = _first(
            events,
            "llm_speculation_first_token",
            "llm_reply_first_token",
        )
        tts_first_text = _first(events, "tts_first_text_sent")
        tts_first_audio_received = _first(events, "tts_first_audio_received")
        tts_first_audio_played = _first(events, "tts_first_audio_played")
        if user_last_speech is not None and stt_done is not None:
            spans["last_speech_to_stt_done_ms"].append(
                max((stt_done - user_last_speech) * 1000.0, 0.0)
            )
        if user_last_speech is not None and llm_first_token is not None:
            spans["last_speech_to_llm_first_token_ms"].append(
                max((llm_first_token - user_last_speech) * 1000.0, 0.0)
            )
        if user_last_speech is not None and tts_first_audio_played is not None:
            spans["last_speech_to_first_audio_ms"].append(
                max((tts_first_audio_played - user_last_speech) * 1000.0, 0.0)
            )
        if llm_first_token is not None and tts_first_audio_played is not None:
            spans["llm_first_token_to_first_audio_ms"].append(
                max((tts_first_audio_played - llm_first_token) * 1000.0, 0.0)
            )
        if tts_first_text is not None and tts_first_audio_received is not None:
            spans["tts_text_to_audio_received_ms"].append(
                max((tts_first_audio_received - tts_first_text) * 1000.0, 0.0)
            )
        if tts_first_text is not None and tts_first_audio_played is not None:
            spans["tts_text_to_audio_played_ms"].append(
                max((tts_first_audio_played - tts_first_text) * 1000.0, 0.0)
            )
        if events.get("tts_bargein"):
            spans["bargein_count"].append(float(len(events["tts_bargein"])))
    return spans


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize turn-level latency spans from turn_metrics.jsonl."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="logs/turn_metrics.jsonl",
        help="Path to the turn metrics JSONL file.",
    )
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"Turn metrics log not found: {path}")

    grouped = _load_events(path)
    if not grouped:
        raise SystemExit(f"No turn metrics found in {path}.")

    spans = _collect_spans(grouped)
    print(f"path={path}")
    print(f"turns={len(grouped)}")
    for name in (
        "last_speech_to_stt_done_ms",
        "last_speech_to_llm_first_token_ms",
        "last_speech_to_first_audio_ms",
        "llm_first_token_to_first_audio_ms",
        "tts_text_to_audio_received_ms",
        "tts_text_to_audio_played_ms",
    ):
        values = spans.get(name, [])
        if not values:
            continue
        print(f"{name}_count={len(values)}")
        print(f"{name}_p50={_percentile(values, 0.50):.1f}")
        print(f"{name}_p95={_percentile(values, 0.95):.1f}")
        print(f"{name}_max={max(values):.1f}")
    bargeins = int(sum(spans.get("bargein_count", [])))
    print(f"bargeins={bargeins}")


if __name__ == "__main__":
    main()
