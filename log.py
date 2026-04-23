from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


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


_SENTINEL = object()


class _BackgroundFileWriter:
    """Drains a queue of pre-serialised JSON lines to a file on a daemon thread.

    All serialisation and timestamping happen on the caller's thread *before*
    enqueue, so the hot path never touches the filesystem.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._q: queue.Queue[str | object] = queue.Queue()
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()

    def enqueue(self, line: str) -> None:
        self._q.put_nowait(line)

    def enqueue_many(self, lines: list[str]) -> None:
        for line in lines:
            self._q.put_nowait(line)

    def close(self) -> None:
        self._q.put(_SENTINEL)
        self._thread.join(timeout=5.0)

    def _drain(self) -> None:
        with self._path.open("a", encoding="utf-8") as handle:
            while True:
                item = self._q.get()
                if item is _SENTINEL:
                    handle.flush()
                    return
                handle.write(item)  # type: ignore[arg-type]
                handle.write("\n")
                # Batch: drain anything else already queued before flushing.
                drained = 0
                while not self._q.empty() and drained < 200:
                    item = self._q.get_nowait()
                    if item is _SENTINEL:
                        handle.flush()
                        return
                    handle.write(item)  # type: ignore[arg-type]
                    handle.write("\n")
                    drained += 1
                handle.flush()


class SessionLogger:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._writer = _BackgroundFileWriter(self._path)

    @property
    def path(self) -> Path:
        return self._path

    def write_event(self, event: CharEvent) -> None:
        self.write_events([event])

    def write_events(self, events: Iterable[CharEvent]) -> None:
        lines = [
            json.dumps(asdict(event), ensure_ascii=False, separators=(",", ":"))
            for event in events
        ]
        if lines:
            self._writer.enqueue_many(lines)

    def close(self) -> None:
        self._writer.close()


class LatencyLogger:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._writer = _BackgroundFileWriter(self._path)

    @property
    def path(self) -> Path:
        return self._path

    def write_event(self, event: WordLatencyEvent) -> None:
        line = json.dumps(asdict(event), ensure_ascii=False, separators=(",", ":"))
        self._writer.enqueue(line)

    def close(self) -> None:
        self._writer.close()


class TurnMetricsLogger:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._writer = _BackgroundFileWriter(self._path)

    @property
    def path(self) -> Path:
        return self._path

    def write_event(self, event: TurnMetricEvent) -> None:
        line = json.dumps(asdict(event), ensure_ascii=False, separators=(",", ":"))
        self._writer.enqueue(line)

    def close(self) -> None:
        self._writer.close()
