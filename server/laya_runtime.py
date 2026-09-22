"""Memory-frugal Laya runtime for the playground.

Why this exists instead of a plain ``laya.load()``:

``laya.load()`` builds the model in fp32 on the heap, then loads a *second* full copy of
the fp16 checkpoint before ``load_state_dict`` copies it in. On the 1.9 GB sandbox that
peaks at ~2.6 GB and the kernel OOM-kills the process (verified: exit 137).

This loader instead:
  1. builds the module graph on the ``meta`` device (no allocation at all),
  2. streams tensors one at a time out of the safetensors file, casting as it goes,
  3. assigns them in place (``assign=True``), so only one copy is ever resident,
  4. re-materialises the rotary ``inv_freq`` buffers that meta-init leaves empty.

Step 4 is not optional and is easy to get wrong. ModernBERT registers rope tables as
*non-persistent* buffers, so they are absent from the checkpoint and stay on ``meta``
after assignment. Zero-filling them silently produces a model that returns an exactly
uniform distribution for every question. We recompute them from the config and assert
they are bit-identical to a reference module (see tools/verify_runtime.py).

Peak RSS with this path: ~1.30 GB, load time ~5 s.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from laya.common import (  # noqa: E402
    QTYPES,
    build_model,
    build_sequence,
    collate_items,
    confidence_from_probs,
    render_options,
)

REPO = "convaiinnovations/laya"
SUBFOLDER = "multilingual"
EMB_KEY = "encoder.embeddings.tok_embeddings.weight"

_TORCH_THREADS = int(os.environ.get("LAYA_THREADS", "2"))
torch.set_num_threads(_TORCH_THREADS)
torch.set_grad_enabled(False)


# --------------------------------------------------------------------------------------
# question helpers
# --------------------------------------------------------------------------------------

def choice(instructions: str, criteria: Dict[str, Optional[str]]) -> Dict[str, Any]:
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def noul(instructions: str, criteria: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    q: Dict[str, Any] = {"type": "noul", "instructions": instructions}
    if criteria:
        q["criteria"] = criteria
    return q


def score(instructions: str, levels: Sequence[str]) -> Dict[str, Any]:
    return {"type": "score", "instructions": instructions, "criteria": list(levels)}


def _internal(qdef: Dict[str, Any]) -> Dict[str, Any]:
    """Mirror of laya.agent.Agent._to_internal."""
    t = qdef["type"]
    crit = qdef.get("criteria")
    if t == "choice" and isinstance(crit, list):
        crit = {c: None for c in crit}
    ins = qdef["instructions"]
    if not isinstance(ins, str):
        ins = json.dumps(ins)
    return {"t": t, "ins": ins, "crit": crit}


def entropy_bits(p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0)
    return float(-(p * np.log2(p)).sum())


# --------------------------------------------------------------------------------------
# loader
# --------------------------------------------------------------------------------------

def _materialize_rope(model: torch.nn.Module) -> List[Dict[str, Any]]:
    """Recompute non-persistent rope buffers left on the meta device.

    Raises on any *other* meta buffer rather than zero-filling it, because a silently
    zeroed buffer degrades the model to a uniform predictor instead of crashing.
    """
    ec = model.encoder.config
    head_dim = ec.hidden_size // ec.num_attention_heads
    rope_params = getattr(ec, "rope_parameters", None) or {}
    fixed: List[Dict[str, Any]] = []

    for mod_name, mod in model.named_modules():
        for buf_name, buf in list(mod.named_buffers(recurse=False)):
            if buf is None or not buf.is_meta:
                continue
            if not buf_name.endswith("inv_freq"):
                raise RuntimeError(
                    f"unexpected meta buffer {mod_name}.{buf_name}; refusing to zero-fill"
                )
            layer_type = "sliding_attention" if buf_name.startswith("sliding") else "full_attention"
            entry = rope_params.get(layer_type) or rope_params.get("full_attention") or {}
            theta = float(entry.get("rope_theta", getattr(ec, "rope_theta", 160000.0)))
            inv = 1.0 / (
                theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim)
            )
            mod.register_buffer(buf_name, inv, persistent=False)
            fixed.append({"buffer": f"{mod_name}.{buf_name}", "theta": theta})
    return fixed


def resolve_snapshot(local_only: bool = False) -> str:
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        REPO,
        allow_patterns=[f"{SUBFOLDER}/*"],
        local_files_only=local_only,
    )
    return os.path.join(path, SUBFOLDER)


@dataclass
class Prediction:
    """One typed answer, with every number the playground can honestly display."""

    qid: str
    qtype: str
    labels: List[str]
    probabilities: List[float]
    top_index: int
    top_label: str
    top_prob: float
    confidence: float
    margin: float
    entropy: float
    act_probability: float
    expected_score: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        d = {
            "qid": self.qid,
            "type": self.qtype,
            "labels": self.labels,
            "probabilities": [round(p, 6) for p in self.probabilities],
            "top_index": self.top_index,
            "top": self.top_label,
            "top_prob": round(self.top_prob, 6),
            "confidence": round(self.confidence, 6),
            "margin": round(self.margin, 6),
            "entropy_bits": round(self.entropy, 6),
            "act_probability": round(self.act_probability, 6),
        }
        if self.qtype == "noul":
            d["noul"] = round(self.probabilities[1], 6)
        if self.expected_score is not None:
            d["score"] = round(self.expected_score, 6)
        return d


@dataclass
class RuntimeStatus:
    state: str = "cold"          # cold | loading | ready | failed
    detail: str = ""
    load_seconds: float = 0.0
    peak_rss_mb: float = 0.0
    rope_fixed: List[Dict[str, Any]] = field(default_factory=list)


class LayaRuntime:
    """Thread-safe wrapper around one resident Laya checkpoint."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._load_lock = threading.Lock()
        self.model = None
        self.tok = None
        self.cfg: Dict[str, Any] = {}
        self.status = RuntimeStatus()
        # user-tunable temperature per question type (calibration lab)
        self.temperature = {"choice": 1.0, "score": 1.0, "noul": 1.0}
        self.total_calls = 0
        self.total_questions = 0
        self.total_input_tokens = 0
        self.total_infer_ms = 0.0

    # -- lifecycle ---------------------------------------------------------------------

    def ensure_loaded(self) -> None:
        if self.status.state == "ready":
            return
        with self._load_lock:
            if self.status.state == "ready":
                return
            self.status = RuntimeStatus(state="loading", detail="resolving checkpoint")
            t0 = time.perf_counter()
            try:
                snap = resolve_snapshot()
                self.status.detail = "building graph on meta device"

                from safetensors import safe_open
                from transformers import AutoTokenizer

                cfg = json.load(open(os.path.join(snap, "rl_agent_config.json")))
                tok = AutoTokenizer.from_pretrained(os.path.join(snap, "tokenizer"))

                with torch.device("meta"):
                    model = build_model(cfg, encoder_dir=os.path.join(snap, "encoder"))

                self.status.detail = "streaming weights"
                state_dict = {}
                with safe_open(os.path.join(snap, "model.safetensors"), framework="pt") as f:
                    for key in f.keys():
                        tensor = f.get_tensor(key)
                        # The 256k-row vocab table is 61% of the file. Keeping it fp16 and
                        # up-casting the looked-up rows saves ~390 MB with no effect on the
                        # matmuls, which all run in fp32.
                        state_dict[key] = (
                            tensor.to(torch.float16) if key == EMB_KEY else tensor.to(torch.float32)
                        )
                        del tensor

                model.load_state_dict(state_dict, strict=True, assign=True)
                del state_dict
                model.eval()

                self.status.rope_fixed = _materialize_rope(model)
                try:
                    model.encoder.config.reference_compile = False
                except Exception:
                    pass
                model.encoder.embeddings.tok_embeddings.register_forward_hook(
                    lambda m, i, o: o.to(torch.float32)
                )
                for p in model.parameters():
                    p.requires_grad_(False)

                leftover = [n for n, p in model.named_parameters() if p.is_meta]
                leftover += [n for n, b in model.named_buffers() if b is not None and b.is_meta]
                if leftover:
                    raise RuntimeError(f"tensors still on meta device: {leftover[:5]}")

                self.model, self.tok, self.cfg = model, tok, cfg
                self.status.state = "ready"
                self.status.detail = "resident"
                self.status.load_seconds = time.perf_counter() - t0
                self.status.peak_rss_mb = peak_rss_mb()

                self.predict("warm up", {"w": noul("Is this a warm up call?")})
            except Exception as exc:  # pragma: no cover
                self.status.state = "failed"
                self.status.detail = f"{type(exc).__name__}: {exc}"
                raise

    # -- inference ---------------------------------------------------------------------

    def predict(self, state: Any, questions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Answer every question in one forward pass. Returns rich per-question stats."""
        if self.status.state != "ready":
            self.ensure_loaded()

        qids = list(questions.keys())
        max_len = self.cfg.get("max_len", 1024)
        head_max_len = self.cfg.get("head_max_len", 256)

        items, internals = [], []
        for qid in qids:
            q = _internal(questions[qid])
            internals.append(q)
            seq, markers = build_sequence(self.tok, state, q, max_len, head_max_len)
            if len(markers) != len(render_options(q)):
                raise ValueError(
                    f"question {qid!r}: options do not fit in head_max_len={head_max_len}"
                )
            items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]]})

        batch = collate_items([items], self.tok.pad_token_id)

        with self._lock:
            t0 = time.perf_counter()
            with torch.inference_mode():
                logits, act = self.model(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["marker_pos"],
                    batch["marker_mask"],
                    batch["qtype"],
                )
            infer_ms = (time.perf_counter() - t0) * 1000.0

        logits_np = logits.float().numpy()
        act_np = torch.softmax(act.float(), -1).numpy()
        n_tokens = int(batch["attention_mask"].sum())

        answers: Dict[str, Dict[str, Any]] = {}
        for row, qid in enumerate(qids):
            q = internals[row]
            k = len(items[row]["markers"])
            temp = max(1e-3, float(self.temperature.get(q["t"], 1.0)))
            z = logits_np[row, :k] / temp
            p = np.exp(z - z.max())
            p = p / p.sum()

            if q["t"] == "choice":
                labels = list(q["crit"].keys())
            elif q["t"] == "score":
                labels = [f"level {i}" for i in range(k)]
            else:
                labels = ["false", "true"]

            order = np.argsort(p)[::-1]
            margin = float(p[order[0]] - p[order[1]]) if k > 1 else 1.0
            pred = Prediction(
                qid=qid,
                qtype=q["t"],
                labels=labels,
                probabilities=[float(x) for x in p],
                top_index=int(order[0]),
                top_label=labels[int(order[0])],
                top_prob=float(p[order[0]]),
                confidence=float(confidence_from_probs(p, k)),
                margin=margin,
                entropy=entropy_bits(p),
                act_probability=float(act_np[row, 0]),
                expected_score=(float((np.arange(k) * p).sum()) if q["t"] == "score" else None),
            )
            answers[qid] = pred.as_dict()
            if q["t"] == "score":
                answers[qid]["legend"] = {str(i): c for i, c in enumerate(q["crit"])}

        self.total_calls += 1
        self.total_questions += len(qids)
        self.total_input_tokens += n_tokens
        self.total_infer_ms += infer_ms

        return {
            "answers": answers,
            "latency_ms": round(infer_ms, 3),
            "input_tokens": n_tokens,
            "output_tokens": 0,
            "questions": len(qids),
            "batch_shape": list(batch["input_ids"].shape),
        }

    def predict_many(
        self,
        pairs: Sequence[tuple],
        token_budget: int = 3072,
    ) -> tuple:
        """Answer several (state, questions) pairs, batching rows across states.

        Upstream's RAG tab does this: each passage keeps its *own* state, because the model
        reads a passage far better than a ``passages[i]`` reference, while batching keeps it
        to a single pass. Ported here with one change -- on 2 vCPU a 36-row batch of
        200-token rows blows the RAM budget, so rows are packed into sub-batches under
        ``token_budget`` and the wall time of all sub-batches is reported as one number.
        """
        if self.status.state != "ready":
            self.ensure_loaded()

        max_len = self.cfg.get("max_len", 1024)
        head_max_len = self.cfg.get("head_max_len", 256)

        rows, index = [], []
        for pair_i, (state, questions) in enumerate(pairs):
            for qid, qdef in questions.items():
                q = _internal(qdef)
                seq, markers = build_sequence(self.tok, state, q, max_len, head_max_len)
                if len(markers) != len(render_options(q)):
                    continue
                rows.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]]})
                index.append((pair_i, qid, q, len(markers)))

        out: List[Dict[str, Any]] = [{} for _ in pairs]
        if not rows:
            return out, 0.0, 0

        chunks, current, current_tokens = [], [], 0
        for row in rows:
            n = len(row["ids"])
            if current and current_tokens + n > token_budget:
                chunks.append(current)
                current, current_tokens = [], 0
            current.append(row)
            current_tokens += n
        if current:
            chunks.append(current)

        total_ms = 0.0
        total_tokens = 0
        logits_all = []
        for chunk in chunks:
            batch = collate_items([chunk], self.tok.pad_token_id)
            with self._lock:
                t0 = time.perf_counter()
                with torch.inference_mode():
                    logits, _ = self.model(
                        batch["input_ids"], batch["attention_mask"],
                        batch["marker_pos"], batch["marker_mask"], batch["qtype"])
                total_ms += (time.perf_counter() - t0) * 1000.0
            total_tokens += int(batch["attention_mask"].sum())
            logits_all.append(logits.float().numpy())

        flat = np.concatenate([np.pad(l, ((0, 0), (0, max(0, max(x.shape[1] for x in logits_all) - l.shape[1]))),
                                      constant_values=-1e4) for l in logits_all], axis=0)

        for row_i, (pair_i, qid, q, k) in enumerate(index):
            temp = max(1e-3, float(self.temperature.get(q["t"], 1.0)))
            z = flat[row_i, :k] / temp
            p = np.exp(z - z.max())
            p = p / p.sum()
            if q["t"] == "choice":
                labels = list(q["crit"].keys())
            elif q["t"] == "score":
                labels = [f"level {i}" for i in range(k)]
            else:
                labels = ["false", "true"]
            entry: Dict[str, Any] = {
                "type": q["t"], "labels": labels,
                "probabilities": [round(float(x), 6) for x in p],
                "top": labels[int(p.argmax())], "top_prob": round(float(p.max()), 6),
                "confidence": round(float(confidence_from_probs(p, k)), 6),
                "entropy_bits": round(entropy_bits(p), 6),
            }
            if q["t"] == "noul":
                entry["noul"] = round(float(p[1]), 6)
            if q["t"] == "score":
                entry["score"] = round(float((np.arange(k) * p).sum()), 6)
            out[pair_i][qid] = entry

        self.total_calls += 1
        self.total_questions += len(rows)
        self.total_input_tokens += total_tokens
        self.total_infer_ms += total_ms
        return out, round(total_ms, 2), len(rows)

    # -- introspection -----------------------------------------------------------------

    def rendered_prompt(self, state: Any, question: Dict[str, Any]) -> Dict[str, Any]:
        """Exactly what the encoder sees, for the transparency panel."""
        q = _internal(question)
        max_len = self.cfg.get("max_len", 1024)
        head_max_len = self.cfg.get("head_max_len", 256)
        seq, markers = build_sequence(self.tok, state, q, max_len, head_max_len)
        return {
            "text": self.tok.decode(seq),
            "tokens": len(seq),
            "markers": markers,
            "options": render_options(q),
        }

    def info(self) -> Dict[str, Any]:
        avg = self.total_infer_ms / self.total_calls if self.total_calls else 0.0
        return {
            "state": self.status.state,
            "detail": self.status.detail,
            "repo": f"{REPO}/{SUBFOLDER}",
            "backbone": self.cfg.get("encoder", "jhu-clsp/mmBERT-base"),
            "params_m": 321.9,
            "context": self.cfg.get("max_len", 1024),
            "head_max_len": self.cfg.get("head_max_len", 256),
            "precision": "fp32 compute · fp16 vocab table",
            "threads": _TORCH_THREADS,
            "load_seconds": round(self.status.load_seconds, 2),
            "peak_rss_mb": round(peak_rss_mb(), 1),
            "rope_buffers_restored": len(self.status.rope_fixed),
            "temperature": self.temperature,
            "totals": {
                "calls": self.total_calls,
                "questions": self.total_questions,
                "input_tokens": self.total_input_tokens,
                "output_tokens": 0,
                "avg_call_ms": round(avg, 2),
            },
        }


def peak_rss_mb() -> float:
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


RUNTIME = LayaRuntime()
