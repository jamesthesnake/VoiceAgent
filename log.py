from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


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


class SessionLogger:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._handle = self._path.open("a", encoding="utf-8", buffering=1)

    @property
    def path(self) -> Path:
        return self._path

    def write_event(self, event: CharEvent) -> None:
        self.write_events([event])

    def write_events(self, events: Iterable[CharEvent]) -> None:
        payload = [
            json.dumps(asdict(event), ensure_ascii=False, separators=(",", ":"))
            for event in events
        ]
        if not payload:
            return
        with self._lock:
            for line in payload:
                self._handle.write(line)
                self._handle.write("\n")
            self._handle.flush()

    def close(self) -> None:
        with self._lock:
            self._handle.close()


class LatencyLogger:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._handle = self._path.open("a", encoding="utf-8", buffering=1)

    @property
    def path(self) -> Path:
        return self._path

    def write_event(self, event: WordLatencyEvent) -> None:
        line = json.dumps(asdict(event), ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            self._handle.write(line)
            self._handle.write("\n")
            self._handle.flush()

    def close(self) -> None:
        with self._lock:
            self._handle.close()
