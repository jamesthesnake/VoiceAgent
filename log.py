from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable


@dataclass(slots=True)
class CharEvent:
    char: str
    start: float
    end: float
    source: str
    notes: str = ""


@dataclass(slots=True)
class WordLatencyEvent:
    word: str
    source: str
    spoken_start: float
    spoken_end: float
    saved_at: float
    latency_ms: float
    revised: bool = False
    notes: str = ""
    overhead_us: float = 0.0  # Local emit_word overhead in microseconds.


@dataclass(slots=True)
class TurnMetricEvent:
    turn_id: int
    event: str
    at: float
    source: str = ""
    text: str = ""
    info: dict[str, Any] = field(default_factory=dict)


class SessionClock:
    def __init__(self) -> None:
        self._origin = time.perf_counter()

    def now(self) -> float:
        return time.perf_counter() - self._origin


def append_note(notes: str, extra: str) -> str:
    if not notes:
        return extra
    parts = [part.strip() for part in notes.split(",") if part.strip()]
    if extra in parts:
        return notes
    return f"{notes},{extra}"


# ---------------------------------------------------------------------------
# Fast serialisers — direct attribute access, no asdict() overhead.
# ---------------------------------------------------------------------------

def _ser_char(e: CharEvent) -> str:
    return json.dumps(
        {"char": e.char, "start": e.start, "end": e.end,
         "source": e.source, "notes": e.notes},
        ensure_ascii=False, separators=(",", ":"),
    )


def _ser_latency(e: WordLatencyEvent) -> str:
    return json.dumps(
        {"word": e.word, "source": e.source,
         "spoken_start": e.spoken_start, "spoken_end": e.spoken_end,
         "saved_at": e.saved_at, "latency_ms": e.latency_ms,
         "revised": e.revised, "notes": e.notes,
         "overhead_us": e.overhead_us},
        ensure_ascii=False, separators=(",", ":"),
    )


def _ser_metric(e: TurnMetricEvent) -> str:
    return json.dumps(
        {"turn_id": e.turn_id, "event": e.event, "at": e.at,
         "source": e.source, "text": e.text, "info": e.info},
        ensure_ascii=False, separators=(",", ":"),
    )


_SENTINEL = object()


class _BackgroundFileWriter:
    """Drains a queue of raw dataclass objects to a JSONL file on a daemon
    thread.  Serialisation happens on the background thread so the caller's
    hot path (event loop) only pays the cost of a ``put_nowait``.
    """

    def __init__(self, path: Path, serialise: Callable[[Any], str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._serialise = serialise
        self._q: queue.Queue[Any] = queue.Queue()
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()

    def enqueue(self, obj: Any) -> None:
        self._q.put_nowait(obj)

    def enqueue_many(self, objs: Iterable[Any]) -> None:
        for obj in objs:
            self._q.put_nowait(obj)

    def close(self) -> None:
        self._q.put(_SENTINEL)
        self._thread.join(timeout=5.0)

    def _drain(self) -> None:
        ser = self._serialise
        with self._path.open("a", encoding="utf-8") as handle:
            while True:
                item = self._q.get()
                if item is _SENTINEL:
                    handle.flush()
                    return
                handle.write(ser(item))
                handle.write("\n")
                # Batch: drain anything else already queued before flushing.
                drained = 0
                while not self._q.empty() and drained < 200:
                    item = self._q.get_nowait()
                    if item is _SENTINEL:
                        handle.flush()
                        return
                    handle.write(ser(item))
                    handle.write("\n")
                    drained += 1
                handle.flush()


class SessionLogger:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._writer = _BackgroundFileWriter(self._path, _ser_char)

    @property
    def path(self) -> Path:
        return self._path

    def write_event(self, event: CharEvent) -> None:
        self._writer.enqueue(event)

    def write_events(self, events: Iterable[CharEvent]) -> None:
        self._writer.enqueue_many(events)

    def close(self) -> None:
        self._writer.close()


class LatencyLogger:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._writer = _BackgroundFileWriter(self._path, _ser_latency)

    @property
    def path(self) -> Path:
        return self._path

    def write_event(self, event: WordLatencyEvent) -> None:
        self._writer.enqueue(event)

    def close(self) -> None:
        self._writer.close()


class TurnMetricsLogger:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._writer = _BackgroundFileWriter(self._path, _ser_metric)

    @property
    def path(self) -> Path:
        return self._path

    def write_event(self, event: TurnMetricEvent) -> None:
        self._writer.enqueue(event)

    def close(self) -> None:
        self._writer.close()
