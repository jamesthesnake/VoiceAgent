from __future__ import annotations

import argparse
import json
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


def _load_latencies(path: Path, source: str, include_revised: bool) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if payload.get("source") != source:
                continue
            if not include_revised and payload.get("revised"):
                continue
            rows.append(payload)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize per-word save latency from latency.jsonl."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="logs/latency.jsonl",
        help="Path to the latency JSONL file.",
    )
    parser.add_argument(
        "--source",
        default="user",
        help="Source to summarize. Default: user",
    )
    parser.add_argument(
        "--include-revised",
        action="store_true",
        help="Include revised events in the summary.",
    )
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"Latency log not found: {path}")

    rows = _load_latencies(path, source=args.source, include_revised=args.include_revised)
    if not rows:
        raise SystemExit(
            f"No matching latency records found in {path} for source={args.source!r}."
        )

    values = [float(row["latency_ms"]) for row in rows]
    under_300 = sum(1 for value in values if value < 300.0)

    print(f"path={path}")
    print(f"source={args.source}")
    print(f"include_revised={args.include_revised}")
    print(f"count={len(values)}")
    print(f"under_300ms={under_300}/{len(values)} ({under_300 / len(values) * 100:.1f}%)")
    print(f"p50_ms={_percentile(values, 0.50):.1f}")
    print(f"p95_ms={_percentile(values, 0.95):.1f}")
    print(f"max_ms={max(values):.1f}")


if __name__ == "__main__":
    main()
