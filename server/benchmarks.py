"""Small, reproducible benchmark suite for the playground UI.

This is intentionally a live smoke benchmark, not a paper-grade evaluation. It answers:
which panels demonstrate Laya clearly, how fast are the decision loops, and which demos
belong in showcase vs failure/legacy buckets.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

import games as G
import shooter as S


def _avg(xs: List[float]) -> Optional[float]:
    return (sum(xs) / len(xs)) if xs else None


def _pct(x: Optional[float]) -> Optional[float]:
    return round(x * 100.0, 1) if x is not None else None


def _row(
    panel: str,
    tier: str,
    score: str,
    avg_ms: Optional[float],
    questions: int,
    note: str,
    accuracy: Optional[float] = None,
    value: Optional[float] = None,
    baseline: str = "—",
) -> Dict[str, Any]:
    return {
        "panel": panel,
        "tier": tier,
        "score": score,
        "baseline": baseline,
        "accuracy_pct": _pct(accuracy),
        "value_pct": _pct(value),
        "avg_ms": round(avg_ms, 1) if avg_ms is not None else None,
        "questions": questions,
        "note": note,
    }


def _snake(seed: int, steps: int) -> Dict[str, Any]:
    game = G.SnakeGame(width=10, height=6, seed=seed, initial_length=4)
    latencies: List[float] = []
    interventions = 0
    done = 0
    for _ in range(steps):
        if not game.alive or game.won:
            break
        d = G.snake_decide(game, guarded=True)
        game.step(d["executed"])
        latencies.append(d["raw"]["latency_ms"])
        interventions += int(d["intervened"])
        done += 1
    return _row(
        "Snake",
        "keep · speed loop",
        f"alive {done}/{steps}",
        _avg(latencies),
        done * 3,
        "Snake stays: it shows repeated low-latency decisions in a visible game loop.",
        accuracy=1.0 if game.alive else 0.0,
        baseline="random move dies quickly",
    ) | {"extra": {"interventions": interventions, "length": len(game.body)}}


def _aim_cascade(seed: int, rounds: int) -> Dict[str, Any]:
    game = G.AimCascade(seed=seed, size=6, decoys=3)
    hits = 0
    stage_hits = 0
    latencies: List[float] = []
    questions = 0
    for _ in range(rounds):
        game.new_round()
        d = game.detect(cascade=True)
        hits += int(d["hit"])
        stage_hits += int(bool(d.get("stage", {}).get("stage1_correct")))
        latencies.append(d["latency_ms"])
        questions += d["questions"]
    return _row(
        "Aim Cascade",
        "showcase",
        f"{hits}/{rounds} hits",
        _avg(latencies),
        questions,
        "Best explanation demo: one yes/no per region beats one huge choice question.",
        accuracy=hits / max(1, rounds),
        baseline="single 36-choice: 0%",
    ) | {"extra": {"stage1_accuracy_pct": _pct(stage_hits / max(1, rounds))}}


def _shooter(seed: int, steps: int, difficulty: str, budget_ms: Optional[float]) -> Dict[str, Any]:
    arena = S.Arena(seed=seed, difficulty=difficulty, budget_ms=budget_ms)
    latencies: List[float] = []
    questions = 0
    out: Optional[Dict[str, Any]] = None
    for _ in range(steps):
        out = arena.step()
        latencies.append(out["latency_ms"])
        questions += int(out.get("questions", 0))
        if not out["alive"]:
            break
    stats = out["stats"] if out else {}
    return _row(
        "3D Shooter",
        "showcase",
        f"score {stats.get('score', 0)}",
        _avg(latencies),
        questions,
        "Strongest game proof: Laya scores contacts, code aims, causal rollout grades value.",
        accuracy=stats.get("priority_accuracy"),
        value=stats.get("value_captured"),
        baseline="random value: 32.9%",
    ) | {"extra": {"hp": out.get("hp") if out else None,
                   "missed_by_deadline": stats.get("missed_by_deadline"),
                   "budget_ms": budget_ms}}


def _triage(seed: int, rounds: int) -> Dict[str, Any]:
    rush = G.TriageRush(seed=seed)
    hits = 0
    latencies: List[float] = []
    for _ in range(rounds):
        text, truth, lang = rush.next_ticket()
        d = rush.triage(text, truth, lang)
        hits += int(d["correct"])
        latencies.append(d["raw"]["latency_ms"])
    return _row(
        "Triage Rush",
        "product proof",
        f"{hits}/{rounds} routed",
        _avg(latencies),
        rounds * 4,
        "Shows Laya's real job: multilingual routing/classification, not game physics.",
        accuracy=hits / max(1, rounds),
        baseline="4-way random: 25%",
    )


def _guard(seed: int, rounds: int) -> Dict[str, Any]:
    arena = G.GuardrailArena(seed=seed)
    hits = 0
    latencies: List[float] = []
    for _ in range(rounds):
        text, is_attack = arena.next_prompt()
        d = arena.judge(text, is_attack, phrasing="choice", threshold=0.5)
        hits += int(d["correct"])
        latencies.append(d["raw"]["latency_ms"])
    return _row(
        "Guardrail Arena",
        "product proof",
        f"{hits}/{rounds} correct",
        _avg(latencies),
        rounds * 2,
        "Real use-case panel; phrasing sensitivity stays visible instead of hidden.",
        accuracy=hits / max(1, rounds),
        baseline="binary random: 50%",
    )


def _calibration() -> Dict[str, Any]:
    lab = G.CalibrationLab()
    out = lab.run(temperature=1.0, phrasing="choice")
    return _row(
        "Calibration Lab",
        "trust layer",
        f"ECE {out['ece']:.3f}",
        _avg([r["latency_ms"] for r in out["rows"]]),
        out["n"],
        "Keeps the playground honest: confidence must be measured, not trusted blindly.",
        accuracy=out["accuracy"],
        baseline="trust without ECE: unknown",
    ) | {"extra": {"brier": out["brier"], "ece": out["ece"]}}


def run_benchmark(
    suite: str = "quick",
    seed: int = 7,
    shooter_difficulty: str = "normal",
    shooter_budget_ms: Optional[float] = None,
) -> Dict[str, Any]:
    """Run the selected benchmark suite synchronously and return UI-ready rows."""
    suite = "full" if str(suite).lower() == "full" else "quick"
    shooter_difficulty = shooter_difficulty if shooter_difficulty in {"easy", "normal", "hard"} else "normal"
    if shooter_budget_ms is not None:
        shooter_budget_ms = max(0.0, min(float(shooter_budget_ms), 5000.0)) or None
    quick = suite != "full"
    plan: List[tuple[str, Callable[[], Dict[str, Any]]]] = [
        ("snake", lambda: _snake(seed, 6 if quick else 18)),
        ("aim_cascade", lambda: _aim_cascade(seed, 3 if quick else 10)),
        ("shooter", lambda: _shooter(seed, 4 if quick else 12, shooter_difficulty, shooter_budget_ms)),
        ("triage", lambda: _triage(seed, 6 if quick else 14)),
        ("guard", lambda: _guard(seed, 6 if quick else 16)),
    ]
    if not quick:
        plan.append(("calibration", _calibration))

    t0 = time.perf_counter()
    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    for name, fn in plan:
        try:
            rows.append(fn())
        except Exception as exc:  # returned to the UI; benchmark should degrade gracefully
            errors.append({"panel": name, "error": f"{type(exc).__name__}: {exc}"})

    return {
        "suite": "quick" if quick else "full",
        "seed": seed,
        "wall_ms": round((time.perf_counter() - t0) * 1000.0, 1),
        "rows": rows,
        "errors": errors,
        "ceo_decision": {
            "headline": "Snake bleibt. Showcase-Fokus: Shooter, Aim Cascade, Triage, Guardrail, Calibration.",
            "keep": ["Snake", "3D Shooter", "Aim Cascade", "Triage Rush", "Guardrail Arena", "Calibration Lab"],
            "secondary": ["Basic Aim", "Maze", "Minesweeper", "Trading Floor"],
            "reason": "Hauptdemos müssen echte Entscheidungen, klare Baselines und ehrliche Messwerte zeigen; schwache Panels bleiben als Failure/Legacy-Lab statt als Beweis.",
        },
    }
