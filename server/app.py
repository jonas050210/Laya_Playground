"""Laya Model Playground -- API server.

Serves the SPA and a JSON API over one resident Laya checkpoint
(convaiinnovations/laya, multilingual subfolder, 322M, mmBERT-base).

Every /api/game/* response carries the exact state string, the exact typed questions,
the raw model output and the latency, so the UI can show what was actually asked.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

import games as G  # noqa: E402
import workflows as W  # noqa: E402
import market as M  # noqa: E402
import shooter as S  # noqa: E402
from laya_runtime import RUNTIME  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(os.path.dirname(HERE), "web")

app = FastAPI(title="Laya Model Playground", version="1.0.0")

SESSIONS: Dict[str, Any] = {}
_LOCK = threading.Lock()


# --------------------------------------------------------------------------------------
# startup: load the model in a worker thread so the UI is reachable immediately
# --------------------------------------------------------------------------------------

def _background_load() -> None:
    try:
        RUNTIME.ensure_loaded()
        print(f"[laya] ready in {RUNTIME.status.load_seconds:.1f}s "
              f"peak RSS {RUNTIME.status.peak_rss_mb:.0f} MB", flush=True)
    except Exception as exc:  # pragma: no cover
        print(f"[laya] load failed: {type(exc).__name__}: {exc}", flush=True)


@app.on_event("startup")
def _startup() -> None:
    threading.Thread(target=_background_load, daemon=True).start()


def _require_ready() -> None:
    if RUNTIME.status.state == "failed":
        raise HTTPException(503, f"model failed to load: {RUNTIME.status.detail}")
    if RUNTIME.status.state != "ready":
        raise HTTPException(503, "model is still loading")


# --------------------------------------------------------------------------------------
# meta
# --------------------------------------------------------------------------------------

@app.get("/api/status")
def status() -> Dict[str, Any]:
    return RUNTIME.info()


class TempBody(BaseModel):
    choice: float = 1.0
    score: float = 1.0
    noul: float = 1.0


@app.post("/api/temperature")
def set_temperature(body: TempBody) -> Dict[str, Any]:
    RUNTIME.temperature = {"choice": max(0.05, body.choice),
                           "score": max(0.05, body.score),
                           "noul": max(0.05, body.noul)}
    return {"temperature": RUNTIME.temperature}


# --------------------------------------------------------------------------------------
# free-form playground
# --------------------------------------------------------------------------------------

class PredictBody(BaseModel):
    state: Any
    questions: Dict[str, Dict[str, Any]]
    show_prompt: bool = False


@app.post("/api/predict")
def predict(body: PredictBody) -> Dict[str, Any]:
    _require_ready()
    if not body.questions:
        raise HTTPException(400, "at least one question is required")
    try:
        result = RUNTIME.predict(body.state, body.questions)
    except Exception as exc:
        raise HTTPException(400, f"{type(exc).__name__}: {exc}")
    if body.show_prompt:
        first = next(iter(body.questions.values()))
        result["prompt"] = RUNTIME.rendered_prompt(body.state, first)
    return result


# --------------------------------------------------------------------------------------
# 1. snake
# --------------------------------------------------------------------------------------

class SnakeNew(BaseModel):
    width: int = 12
    height: int = 8
    seed: int = 7
    guarded: bool = True


@app.post("/api/game/snake/new")
def snake_new(body: SnakeNew) -> Dict[str, Any]:
    _require_ready()
    with _LOCK:
        game = G.SnakeGame(width=body.width, height=body.height, seed=body.seed)
        SESSIONS["snake"] = {"game": game, "guarded": body.guarded,
                             "interventions": 0, "latencies": [], "deaths": 0}
    return {"snapshot": game.snapshot()}


@app.post("/api/game/snake/step")
def snake_step() -> Dict[str, Any]:
    _require_ready()
    sess = SESSIONS.get("snake")
    if not sess:
        raise HTTPException(400, "start a game first")
    game: G.SnakeGame = sess["game"]
    if not game.alive or game.won:
        return {"snapshot": game.snapshot(), "finished": True}
    decision = G.snake_decide(game, guarded=sess["guarded"])
    ate = game.step(decision["executed"])
    if decision["intervened"]:
        sess["interventions"] += 1
    sess["latencies"].append(decision["raw"]["latency_ms"])
    if not game.alive:
        sess["deaths"] += 1
    lat = sess["latencies"]
    return {
        "snapshot": game.snapshot(), "decision": decision, "ate": ate,
        "session": {"interventions": sess["interventions"], "deaths": sess["deaths"],
                    "steps": len(lat), "avg_ms": round(sum(lat) / len(lat), 1),
                    "min_ms": round(min(lat), 1), "max_ms": round(max(lat), 1),
                    "decisions_per_sec": round(1000.0 / (sum(lat) / len(lat)), 2)},
    }


# --------------------------------------------------------------------------------------
# 2. aim trainer
# --------------------------------------------------------------------------------------

class AimNew(BaseModel):
    seed: int = 0
    decoys: int = 2


@app.post("/api/game/aim/new")
def aim_new(body: AimNew) -> Dict[str, Any]:
    _require_ready()
    with _LOCK:
        SESSIONS["aim"] = G.AimTrainer(seed=body.seed, decoys=body.decoys)
    return {"ok": True, "sectors": G.SECTORS}


@app.post("/api/game/aim/round")
def aim_round() -> Dict[str, Any]:
    _require_ready()
    trainer: Optional[G.AimTrainer] = SESSIONS.get("aim")
    if not trainer:
        raise HTTPException(400, "start a session first")
    trainer.new_round()
    return trainer.detect()


# --------------------------------------------------------------------------------------
# 3. maze
# --------------------------------------------------------------------------------------

class MazeNew(BaseModel):
    width: int = 11
    height: int = 9
    seed: int = 1


@app.post("/api/game/maze/new")
def maze_new(body: MazeNew) -> Dict[str, Any]:
    _require_ready()
    with _LOCK:
        maze = G.MazeGame(width=body.width, height=body.height, seed=body.seed)
        SESSIONS["maze"] = {"maze": maze, "latencies": [], "interventions": 0}
    return {"snapshot": maze.snapshot()}


@app.post("/api/game/maze/step")
def maze_step() -> Dict[str, Any]:
    _require_ready()
    sess = SESSIONS.get("maze")
    if not sess:
        raise HTTPException(400, "start a maze first")
    maze: G.MazeGame = sess["maze"]
    if maze.finished:
        return {"snapshot": maze.snapshot(), "finished": True}
    decision = maze.decide()
    if decision["executed"]:
        maze.apply(decision["executed"])
    if decision["intervened"]:
        sess["interventions"] += 1
    sess["latencies"].append(decision["raw"]["latency_ms"])
    lat = sess["latencies"]
    return {"snapshot": maze.snapshot(), "decision": decision,
            "session": {"steps": len(lat), "interventions": sess["interventions"],
                        "avg_ms": round(sum(lat) / len(lat), 1)}}


# --------------------------------------------------------------------------------------
# 4. minesweeper
# --------------------------------------------------------------------------------------

class MineNew(BaseModel):
    width: int = 8
    height: int = 8
    mines: int = 10
    seed: int = 4


@app.post("/api/game/mines/new")
def mines_new(body: MineNew) -> Dict[str, Any]:
    _require_ready()
    with _LOCK:
        game = G.MinesweeperGame(width=body.width, height=body.height,
                                 mines=body.mines, seed=body.seed)
        SESSIONS["mines"] = {"game": game, "calls": 0, "correct": 0}
    return {"snapshot": game.snapshot()}


class MineStep(BaseModel):
    phrasing: str = "criteria"
    auto_open: bool = True


@app.post("/api/game/mines/step")
def mines_step(body: MineStep) -> Dict[str, Any]:
    _require_ready()
    sess = SESSIONS.get("mines")
    if not sess:
        raise HTTPException(400, "start a board first")
    game: G.MinesweeperGame = sess["game"]
    if game.lost:
        return {"snapshot": game.snapshot(), "finished": True}
    frontier = game.frontier()
    if not frontier:
        return {"snapshot": game.snapshot(), "finished": True, "reason": "no frontier left"}
    best = None
    for cell in frontier:
        d = game.decide(cell, phrasing=body.phrasing)
        if best is None or d["p_mine"] < best["p_mine"]:
            best = d
        if d["p_mine"] < 0.05:
            break
    sess["calls"] += 1
    sess["correct"] += int(best["correct"])
    if body.auto_open:
        game.open(tuple(best["cell"]))
    return {"snapshot": game.snapshot(), "decision": best,
            "session": {"calls": sess["calls"], "correct": sess["correct"],
                        "accuracy": sess["correct"] / max(1, sess["calls"])}}


# --------------------------------------------------------------------------------------
# 5. guardrail arena
# --------------------------------------------------------------------------------------

@app.post("/api/game/guard/new")
def guard_new() -> Dict[str, Any]:
    _require_ready()
    with _LOCK:
        SESSIONS["guard"] = G.GuardrailArena()
    return {"ok": True}


class GuardRound(BaseModel):
    phrasing: str = "choice"
    threshold: float = 0.5
    text: Optional[str] = None
    is_attack: Optional[bool] = None


@app.post("/api/game/guard/round")
def guard_round(body: GuardRound) -> Dict[str, Any]:
    _require_ready()
    arena: Optional[G.GuardrailArena] = SESSIONS.get("guard")
    if not arena:
        raise HTTPException(400, "start a session first")
    if body.text:
        text, is_attack = body.text, bool(body.is_attack)
    else:
        text, is_attack = arena.next_prompt()
    return arena.judge(text, is_attack, phrasing=body.phrasing, threshold=body.threshold)


# --------------------------------------------------------------------------------------
# 6. triage rush
# --------------------------------------------------------------------------------------

@app.post("/api/game/triage/new")
def triage_new() -> Dict[str, Any]:
    _require_ready()
    with _LOCK:
        SESSIONS["triage"] = G.TriageRush()
    return {"ok": True, "departments": G.DEPARTMENTS}


class TriageRound(BaseModel):
    text: Optional[str] = None
    truth: Optional[str] = None


@app.post("/api/game/triage/round")
def triage_round(body: TriageRound) -> Dict[str, Any]:
    _require_ready()
    rush: Optional[G.TriageRush] = SESSIONS.get("triage")
    if not rush:
        raise HTTPException(400, "start a session first")
    if body.text and body.truth:
        text, truth, lang = body.text, body.truth, "custom"
    else:
        text, truth, lang = rush.next_ticket()
    return rush.triage(text, truth, lang)


# --------------------------------------------------------------------------------------
# 7. calibration lab
# --------------------------------------------------------------------------------------

class CalibBody(BaseModel):
    temperature: float = 1.0
    phrasing: str = "choice"


@app.post("/api/game/calibration/run")
def calibration_run(body: CalibBody) -> Dict[str, Any]:
    _require_ready()
    lab = SESSIONS.setdefault("calib", G.CalibrationLab())
    return lab.run(temperature=max(0.05, body.temperature), phrasing=body.phrasing)


# --------------------------------------------------------------------------------------
# official demo workflows (ported from the Hugging Face Space)
# --------------------------------------------------------------------------------------

def _wf(fn, *args):
    _require_ready()
    try:
        return fn(*args)
    except Exception as exc:
        raise HTTPException(400, f"{type(exc).__name__}: {exc}")


@app.get("/api/workflow/examples")
def workflow_examples() -> Dict[str, Any]:
    return {
        "triage": W.TRIAGE_EXAMPLES,
        "email": W.EMAIL_EXAMPLES,
        "guard": W.GUARD_EXAMPLES,
        "moderation": W.MOD_EXAMPLES,
        "router": W.ROUTER_EXAMPLES,
        "routing": W.ROUTING_EXAMPLES,
        "rag": {"query": W.RAG_DEFAULT_QUERY, "passages": W.RAG_DEFAULT_PASSAGES},
    }


class TriageBody(BaseModel):
    message: str
    account_tier: str = "free"
    threshold: float = 0.7


@app.post("/api/workflow/triage")
def wf_triage(body: TriageBody) -> Dict[str, Any]:
    return _wf(W.triage, body.message, body.account_tier, body.threshold)


class EmailBody(BaseModel):
    sender: str = ""
    subject: str = ""
    body: str


@app.post("/api/workflow/email")
def wf_email(body: EmailBody) -> Dict[str, Any]:
    return _wf(W.email_triage, body.sender, body.subject, body.body)


class GuardBody(BaseModel):
    prompt: str
    threshold: float = 0.7


@app.post("/api/workflow/guard")
def wf_guard(body: GuardBody) -> Dict[str, Any]:
    return _wf(W.guardrail, body.prompt, body.threshold)


class RagBody(BaseModel):
    query: str
    passages: str
    threshold: float = 0.5
    compare_sequential: bool = False


@app.post("/api/workflow/rag")
def wf_rag(body: RagBody) -> Dict[str, Any]:
    return _wf(W.rag_filter, body.query, body.passages, body.threshold,
               body.compare_sequential)


class ModBody(BaseModel):
    post: str
    threshold: float = 0.7


@app.post("/api/workflow/moderation")
def wf_moderation(body: ModBody) -> Dict[str, Any]:
    return _wf(W.moderate, body.post, body.threshold)


class RouterBody(BaseModel):
    request: str
    small_model: str = "gpt-4o-mini"
    large_model: str = "gpt-4o"


@app.post("/api/workflow/router")
def wf_router(body: RouterBody) -> Dict[str, Any]:
    return _wf(W.route_model, body.request, body.small_model, body.large_model)


class RoutingBody(BaseModel):
    text: str


@app.post("/api/workflow/routing")
def wf_routing(body: RoutingBody) -> Dict[str, Any]:
    return _wf(W.route_language, body.text)


# --------------------------------------------------------------------------------------
# trading floor
# --------------------------------------------------------------------------------------

class MarketBody(BaseModel):
    seed: int = 7
    reset: bool = False
    ticks: int = 1


@app.post("/api/market/tick")
def market_tick(body: MarketBody) -> Dict[str, Any]:
    _require_ready()
    try:
        s = M.get_session(body.seed, body.reset)
        out = None
        for _ in range(max(1, min(body.ticks, 10))):
            out = s.tick()
        return out
    except Exception as exc:
        raise HTTPException(400, f"{type(exc).__name__}: {exc}")


@app.get("/api/market/measured")
def market_measured() -> Dict[str, Any]:
    return M.MEASURED


@app.get("/api/market/state")
def market_state(seed: int = 7) -> Dict[str, Any]:
    s = M.get_session(seed)
    return {"tick": s.ticks, "regime": s.mkt.regime,
            "assets": [{"sym": a["sym"], "name": a["name"], "kind": a["kind"],
                        "price": round(s.mkt.price(a["sym"]), 2)} for a in M.ASSETS],
            "prices": {a["sym"]: [round(x, 3) for x in s.mkt.history[a["sym"]][-120:]]
                       for a in M.ASSETS},
            "portfolio": s.book.stats(s.mkt),
            "questions": M.ACTION_Q}


# --------------------------------------------------------------------------------------
# aim trainer — cascade
# --------------------------------------------------------------------------------------

AIM_CASCADE: Dict[str, Any] = {}


class AimCascadeBody(BaseModel):
    seed: int = 0
    size: int = 6
    decoys: int = 3
    cascade: bool = True
    reset: bool = False


@app.post("/api/game/aimcascade/round")
def aim_cascade_round(body: AimCascadeBody) -> Dict[str, Any]:
    _require_ready()
    try:
        key = f"{body.seed}:{body.size}:{body.decoys}"
        if body.reset or key not in AIM_CASCADE:
            AIM_CASCADE.clear()
            AIM_CASCADE[key] = G.AimCascade(seed=body.seed, size=body.size,
                                            decoys=body.decoys)
        game = AIM_CASCADE[key]
        game.new_round()
        return game.detect(cascade=body.cascade)
    except Exception as exc:
        raise HTTPException(400, f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------------------
# mini 3D shooter
# --------------------------------------------------------------------------------------

class ShooterBody(BaseModel):
    seed: int = 1
    difficulty: str = "normal"
    reset: bool = False
    steps: int = 1
    budget_ms: Optional[float] = None   # None = no deadline


@app.post("/api/shooter/step")
def shooter_step(body: ShooterBody) -> Dict[str, Any]:
    _require_ready()
    try:
        a = S.get_arena(body.seed, body.difficulty, body.reset, body.budget_ms)
        out = None
        for _ in range(max(1, min(body.steps, 10))):
            out = a.step()
            if not out["alive"]:
                break
        return out
    except Exception as exc:
        raise HTTPException(400, f"{type(exc).__name__}: {exc}")


@app.get("/api/shooter/measured")
def shooter_measured() -> Dict[str, Any]:
    return {"measured": S.MEASURED, "questions": S.THREAT_Q}


# --------------------------------------------------------------------------------------
# static SPA
# --------------------------------------------------------------------------------------

@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(WEB, "index.html"))


@app.exception_handler(404)
def spa_fallback(request, exc):  # pragma: no cover
    if request.url.path.startswith("/api/"):
        return JSONResponse({"detail": "not found"}, status_code=404)
    return FileResponse(os.path.join(WEB, "index.html"))


if os.path.isdir(WEB):
    app.mount("/static", StaticFiles(directory=WEB), name="static")
