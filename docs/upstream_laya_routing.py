"""Multilingual routing for the demo Space, using the `laya` package's Router.

Deliberately self-contained and defensive: every entry point degrades to a readable message
instead of raising, so a failure here cannot take the rest of the Space down. The existing
tabs do not import this module.
"""
import os
import time

# Both checkpoints live in the one bundled repo (convaiinnovations/laya): English at the root,
# multilingual in a subfolder. Router only downloads the subfolder it is asked for. The env vars
# stay supported so the Space can be pointed at local paths for testing.
_EN = os.environ.get("LAYA_EN_REPO")
_ML = os.environ.get("LAYA_ML_REPO")
MODELS = {"english": _EN, "multilingual": _ML} if (_EN and _ML) else None

_ROUTER = None
_LOAD_ERROR = None


def available():
    """True when the installed `laya` package is new enough to expose Router."""
    try:
        import laya
        return hasattr(laya, "Router")
    except Exception:
        return False


def get_router():
    """Build the Router once. Returns None (and records why) rather than raising."""
    global _ROUTER, _LOAD_ERROR
    if _ROUTER is not None or _LOAD_ERROR is not None:
        return _ROUTER
    try:
        from laya import Router
        # Two resident checkpoints: a demo alternates languages, and at max_loaded=1 every
        # switch would pay a full model load. MODELS is None in normal operation, which uses
        # the bundled repo.
        # Preload every checkpoint at boot. A cold load costs seconds; detection costs
        # microseconds. Without this the first request in each language pays a full model
        # build, which is what made routing feel slow here.
        # No preload here: building checkpoints on the import path can exceed the Space's
        # start-up window and get the app killed and restarted. max_loaded=2 keeps both
        # routed checkpoints resident once warm_async() has built them in the background.
        kw = {"device": "cpu", "max_loaded": 2}
        if MODELS:
            kw["models"] = MODELS
        _ROUTER = Router(**kw)
    except Exception as e:
        _LOAD_ERROR = "%s: %s" % (type(e).__name__, e)
    return _ROUTER


def load_error():
    return _LOAD_ERROR


_WARM = {"state": "cold", "detail": ""}

# Only the two checkpoints this tab serves.
WARM_MODELS = ["english", "multilingual"]


def warm_status():
    return dict(_WARM)


def attach_existing(agent):
    """Give the router the checkpoint this Space already built.

    app.py calls D.get_agent() at start-up, which builds convaiinnovations/laya -- the exact
    checkpoint the router would otherwise load again as "english". Sharing it saves a duplicate
    421M parameters and the multi-second wait the user would see on their first click.
    """
    router = get_router()
    if router is None or agent is None:
        return False
    try:
        if hasattr(router, "attach"):          # laya >= 0.3.3
            router.attach("english", agent)
        else:                                  # 0.3.2: same effect, private path
            router._agents["english"] = agent
            router._touch("english")
            router.max_loaded = max(router.max_loaded, len(router._agents))
        print("routing: reusing the already-built english checkpoint", flush=True)
        return True
    except Exception as e:
        print("routing: could not attach existing agent: %s" % e, flush=True)
        return False


def warm_now(names=None):
    """Build the routed checkpoints in the calling thread.

    Deliberately synchronous and thread-free: ZeroGPU forks the process for each GPU call, and
    a background thread holding torch state across that fork is a crash waiting to happen. The
    first request pays this once; `max_loaded=2` keeps both resident afterwards.
    """
    import time
    router = get_router()
    if router is None:
        _WARM.update(state="failed", detail=load_error() or "router unavailable")
        return _WARM["state"]
    t, done = time.perf_counter(), []
    for name in (names or WARM_MODELS):
        try:
            if name in router.loaded:      # already attached or built
                done.append(name + " (shared)")
                continue
            router.load(name)
            done.append(name)
        except Exception as e:
            print("routing: %s failed to warm: %s" % (name, e), flush=True)
    _WARM.update(state="warm" if done else "failed",
                 detail="%s in %.0fs" % (", ".join(done) or "nothing", time.perf_counter() - t))
    print("routing warm: %s" % _WARM["detail"], flush=True)
    return _WARM["state"]


QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which team should handle the message in `body`?",
        "criteria": {"billing": "invoices, payments, refunds",
                     "technical": "bugs, outages, integrations",
                     "sales": "pricing, demos, new purchases",
                     "security": "phishing, fraud, account compromise",
                     "other": "none of the above"},
    },
    "is_urgent": {"type": "noul", "instructions": "Does `body` communicate time pressure or a deadline?"},
    "refund_requested": {"type": "noul", "instructions": "Does the sender ask for money back?"},
    "frustration": {"type": "score", "instructions": "How frustrated does the sender sound?",
                    "criteria": ["calm and neutral", "concerned but civil", "clearly annoyed",
                                 "very angry or using strong language"]},
}

EXAMPLES = [
    ["I was charged twice for invoice 4411 and nobody has answered for three days. Refund the duplicate today."],
    ["मुझसे इनवॉइस 4411 के लिए दो बार शुल्क लिया गया और तीन दिनों से कोई जवाब नहीं मिला। कृपया आज ही धनवापसी करें।"],
    ["請求書4411で二重に請求されました。三日間返信がありません。本日中に返金してください。"],
    ["청구서 4411에 대해 두 번 청구되었습니다. 사흘 동안 답변이 없습니다. 오늘 환불해 주세요."],
    ["تم خصم مبلغ الفاتورة 4411 مرتين ولم يرد أحد منذ ثلاثة أيام. يرجى رد المبلغ اليوم."],
    ["Мне дважды списали деньги по счёту 4411, и уже три дня нет ответа. Верните деньги сегодня."],
    ["Der Kunde wurde zweimal fuer Rechnung 4411 belastet und hat seit drei Tagen keine Antwort erhalten."],
    ["发票4411被重复扣款，三天没有人回复。请今天退款。"],
]


def analyse_and_answer(text):
    """Route, then answer. Returns (rows, routing_markdown, raw_json_string)."""
    import json

    text = (text or "").strip()
    if not text:
        return [], "_Enter a message._", "{}"

    router = get_router()
    if router is None:
        return [], ("**Routing unavailable** — `%s`\n\nThe Space needs `laya>=0.3.0`."
                    % (load_error() or "unknown error")), "{}"

    state = {"body": text}
    try:
        decision = router.route(state, QUESTIONS)
    except Exception as e:
        return [], "**Routing failed** — %s: %s" % (type(e).__name__, e), "{}"

    det = decision.get("detection") or {}
    head = ("**Routed to `%s`**  ·  %s\n\n"
            "script `%s`" % (decision["model"], decision["reason"], det.get("script", "?")))
    if det.get("language"):
        head += "  ·  language guess `%s`" % det["language"]
    head += "  ·  repo `%s`" % decision["repo"]

    try:
        if _WARM["state"] == "cold":
            warm_now()
        t = time.perf_counter()
        result = router.predict(state, QUESTIONS)
        ms = (time.perf_counter() - t) * 1000
    except Exception as e:
        return [], head + "\n\n**Inference failed** — %s: %s" % (type(e).__name__, e), "{}"

    a = result["answers"]
    rows = [
        ["department", a["department"]["choice"], "%.2f" % a["department"]["confidence"]],
        ["is_urgent", "%.3f" % a["is_urgent"]["noul"],
         "%.2f" % max(a["is_urgent"]["noul"], 1 - a["is_urgent"]["noul"])],
        ["refund_requested", "%.3f" % a["refund_requested"]["noul"],
         "%.2f" % max(a["refund_requested"]["noul"], 1 - a["refund_requested"]["noul"])],
        ["frustration", "%.2f / 3" % a["frustration"]["score"],
         "%.2f" % a["frustration"]["confidence"]],
    ]
    head += "  ·  **%.0f ms**" % ms
    return rows, head, json.dumps(result, indent=2, ensure_ascii=False)
