#!/usr/bin/env python3
"""Runtime profiler for a local machine.

Run through start.py so dependencies and the checkpoint are prepared first:

    python start.py --profile

Prints plain terminal text only; no report files are written.
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "server"
sys.path.insert(0, str(SERVER))

from laya_runtime import RUNTIME, choice, noul, score  # noqa: E402


def percentile(xs: List[float], q: float) -> float:
    if not xs:
        return 0.0
    data = sorted(xs)
    idx = min(len(data) - 1, max(0, round((len(data) - 1) * q)))
    return data[idx]


def run_case(name: str, fn: Callable[[], Dict[str, Any]], repeats: int = 5) -> Dict[str, Any]:
    times: List[float] = []
    questions = tokens = 0
    # one case-local warmup keeps first-call graph/setup noise out of the row
    fn()
    for _ in range(repeats):
        out = fn()
        times.append(float(out.get("latency_ms", 0.0)))
        questions = int(out.get("questions", questions))
        tokens = int(out.get("input_tokens", tokens))
    return {
        "name": name,
        "repeats": repeats,
        "p50": percentile(times, 0.50),
        "p90": percentile(times, 0.90),
        "mean": statistics.mean(times) if times else 0.0,
        "questions": questions,
        "tokens": tokens,
    }


def main() -> int:
    print("=" * 72)
    print("Laya local runtime profile")
    print("=" * 72)
    print(f"LAYA_DEVICE={os.environ.get('LAYA_DEVICE', 'auto')}")

    t0 = time.perf_counter()
    RUNTIME.ensure_loaded()
    info = RUNTIME.info()
    print(f"loaded in {info['load_seconds']}s · device {info['device']} · threads {info['threads']}")
    if info.get("cuda"):
        c = info["cuda"]
        print(f"cuda: {c['name']} · VRAM {c['total_vram_mb']:.0f} MB · reserved {c['reserved_mb']:.0f} MB")
    print()

    one = {"ok": noul("Is this request about a refund?")}
    mixed = {
        "department": choice("Which department should handle this request?", {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, system errors",
            "sales": "pricing, quotes, new contracts",
            "other": "everything else",
        }),
        "urgent": score("How urgent is this request?", ["can wait", "soon", "critical"]),
        "refund": noul("Does the customer ask for a refund?"),
    }
    eight = {f"q{i}": noul("Is this contact actively attacking the player?") for i in range(8)}

    def one_call() -> Dict[str, Any]:
        return RUNTIME.predict("Please refund the duplicate charge on invoice 4411.", one)

    def mixed_call() -> Dict[str, Any]:
        return RUNTIME.predict(
            "I was charged twice for invoice 4411. Refund it today or we cancel.", mixed)

    def eight_call() -> Dict[str, Any]:
        return RUNTIME.predict(
            "A rusher is very close and attacking the player right now.", eight)

    def many_call() -> Dict[str, Any]:
        states = [
            {"contact": "A rusher is very close and attacking the player right now."},
            {"contact": "A wooden crate sits very close. It is scenery and cannot attack."},
            {"contact": "A sniper is far away and has the player in its sights."},
            {"contact": "A wounded scout is very close but fleeing and no longer fighting."},
        ]
        q = {"engaging": choice("Is this contact attacking or about to attack?", {
            "no": "idle, fleeing, harmless or not an enemy",
            "yes": "attacking now or has the player in its sights",
        })}
        _, ms, rows = RUNTIME.predict_many([(s, q) for s in states])
        return {"latency_ms": ms, "questions": rows, "input_tokens": 0}

    rows = [
        run_case("1 noul / short text", one_call, 8),
        run_case("3 mixed questions", mixed_call, 6),
        run_case("8 noul questions", eight_call, 5),
        run_case("4 states x 1 choice", many_call, 5),
    ]

    print(f"{'case':28} {'q':>3} {'tokens':>7} {'p50 ms':>9} {'p90 ms':>9} {'mean ms':>9}")
    print("-" * 72)
    for r in rows:
        print(f"{r['name'][:28]:28} {r['questions']:>3} {r['tokens']:>7} "
              f"{r['p50']:>9.1f} {r['p90']:>9.1f} {r['mean']:>9.1f}")
    print("-" * 72)
    print(f"wall time {time.perf_counter() - t0:.1f}s · output tokens always 0")
    print("If these rows say device=cuda, no Ollama or second terminal is involved.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
