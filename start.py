#!/usr/bin/env python3
"""Laya Model Playground — one-command launcher.

    python3 start.py

Checks Python, installs anything missing (CPU-only torch by default), pre-downloads the
checkpoint with a visible progress line, then serves the playground and opens a browser.

Useful flags:
    --port 7860        port to serve on
    --host 0.0.0.0     bind address
    --no-install       never touch pip; fail if a dependency is missing
    --no-browser       do not open a browser
    --gpu              install the CUDA build of torch instead of the CPU one
    --verify           run the verification suite and exit
    --check            report environment status and exit
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERVER = HERE / "server"
MIN_PY = (3, 9)

# import name -> pip requirement
REQUIREMENTS = {
    "torch": "torch",
    "transformers": "transformers>=4.45",
    "safetensors": "safetensors>=0.4",
    "huggingface_hub": "huggingface_hub>=0.20",
    "numpy": "numpy>=1.20",
    "laya": "laya>=0.3.2",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
}
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"

C = {"d": "\033[2m", "b": "\033[1m", "g": "\033[32m", "y": "\033[33m",
     "r": "\033[31m", "c": "\033[36m", "x": "\033[0m"}
if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
    C = {k: "" for k in C}


def say(msg: str = "", colour: str = "") -> None:
    print(f"{C.get(colour, '')}{msg}{C['x'] if colour else ''}", flush=True)


def banner() -> None:
    say()
    say("  ╭────────────────────────────────────────────╮", "c")
    say("  │   " + C["b"] + "Laya Model Playground" + C["x"] + C["c"] + "                    │", "c")
    say("  │   " + C["d"] + "322M decision model · 18 live panels" + C["x"] + C["c"] + "       │", "c")
    say("  ╰────────────────────────────────────────────╯", "c")
    say()


def missing_packages() -> list:
    out = []
    for module, requirement in REQUIREMENTS.items():
        if importlib.util.find_spec(module) is None:
            out.append((module, requirement))
    return out


def install(missing: list, gpu: bool) -> bool:
    say(f"  Installing {len(missing)} missing package(s). First run only.", "y")
    say()
    torch_missing = any(m == "torch" for m, _ in missing)
    others = [req for mod, req in missing if mod != "torch"]

    if torch_missing:
        cmd = [sys.executable, "-m", "pip", "install", "--no-cache-dir", "torch"]
        if not gpu:
            cmd += ["--index-url", TORCH_CPU_INDEX]
        say(f"    torch ({'CUDA' if gpu else 'CPU-only, ~200 MB'}) …", "d")
        if subprocess.call(cmd, stdout=subprocess.DEVNULL) != 0:
            say("  ✗ torch failed to install", "r")
            return False

    if others:
        say(f"    {', '.join(o.split('>')[0].split('=')[0] for o in others)} …", "d")
        if subprocess.call([sys.executable, "-m", "pip", "install", "--no-cache-dir", *others],
                           stdout=subprocess.DEVNULL) != 0:
            say("  ✗ dependencies failed to install", "r")
            return False

    say("  ✓ dependencies ready", "g")
    return True


def fetch_checkpoint() -> bool:
    """Download the 644 MB multilingual checkpoint before the server needs it."""
    from huggingface_hub import snapshot_download

    os.environ.setdefault("USE_TF", "0")
    try:
        path = snapshot_download("convaiinnovations/laya",
                                 allow_patterns=["multilingual/*"], local_files_only=True)
        say(f"  ✓ checkpoint cached  {C['d']}{path}{C['x']}", "g")
        return True
    except Exception:
        pass

    say("  Downloading checkpoint convaiinnovations/laya · multilingual (644 MB) …", "y")
    t0 = time.perf_counter()
    try:
        snapshot_download("convaiinnovations/laya", allow_patterns=["multilingual/*"])
    except Exception as exc:
        say(f"  ✗ download failed: {type(exc).__name__}: {exc}", "r")
        say("    Check your connection, or set HF_TOKEN for higher rate limits.", "d")
        return False
    say(f"  ✓ checkpoint downloaded in {time.perf_counter() - t0:.0f}s", "g")
    return True


def report_environment() -> None:
    total = free = None
    try:
        import re
        info = Path("/proc/meminfo").read_text()
        total = int(re.search(r"MemTotal:\s+(\d+)", info).group(1)) / 1024 / 1024
        free = int(re.search(r"MemAvailable:\s+(\d+)", info).group(1)) / 1024 / 1024
    except Exception:
        pass

    say(f"  python    {sys.version.split()[0]}  ({sys.executable})", "d")
    if total:
        warn = free is not None and free < 1.6
        say(f"  memory    {free:.1f} GB available of {total:.1f} GB"
            + ("   ← tight; close other apps" if warn else ""), "y" if warn else "d")
    say(f"  cpus      {os.cpu_count()}", "d")

    missing = missing_packages()
    if missing:
        say(f"  packages  {len(missing)} missing: "
            f"{', '.join(m for m, _ in missing)}", "y")
    else:
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                vram = torch.cuda.get_device_properties(0).total_memory / 1e9
                say(f"  packages  all present (torch {torch.__version__} · CUDA)", "g")
                say(f"  gpu       {name} · {vram:.0f} GB VRAM", "g")
                say("            the 322M checkpoint needs ~1.3 GB — it fits easily; "
                    "expect ~30 ms per batched pass instead of ~800 ms", "d")
            else:
                say(f"  packages  all present (torch {torch.__version__} · CPU)", "g")
                try:
                    import importlib.util
                    if importlib.util.find_spec("torch") and sys.platform == "win32":
                        say("            no CUDA build detected. If you have an NVIDIA GPU, "
                            "reinstall with:  python start.py --gpu", "y")
                except Exception:
                    pass
        except Exception:
            say("  packages  all present", "g")

    free_gb = shutil.disk_usage(HERE).free / 1e9
    say(f"  disk      {free_gb:.1f} GB free", "d")


def main() -> int:
    ap = argparse.ArgumentParser(description="Start the Laya Model Playground.")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--no-install", action="store_true", help="never invoke pip")
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser")
    ap.add_argument("--gpu", action="store_true", help="install the CUDA torch build")
    ap.add_argument("--verify", action="store_true", help="run the verification suite and exit")
    ap.add_argument("--check", action="store_true", help="report environment and exit")
    args = ap.parse_args()

    banner()

    if sys.version_info < MIN_PY:
        say(f"  ✗ Python {MIN_PY[0]}.{MIN_PY[1]}+ required, found "
            f"{sys.version_info.major}.{sys.version_info.minor}", "r")
        return 1

    if not SERVER.is_dir():
        say(f"  ✗ server/ not found next to start.py (looked in {HERE})", "r")
        return 1

    report_environment()
    say()

    if args.check:
        return 0

    missing = missing_packages()
    if missing:
        if args.no_install:
            say(f"  ✗ missing: {', '.join(r for _, r in missing)}", "r")
            say("    Install them, or drop --no-install.", "d")
            return 1
        if not install(missing, args.gpu):
            return 1
        say()

    if not fetch_checkpoint():
        return 1
    say()

    sys.path.insert(0, str(SERVER))
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if args.verify:
        say("  Running verification suite …", "c")
        say()
        return subprocess.call([sys.executable, str(HERE / "tools" / "verify_runtime.py")])

    import uvicorn

    url = f"http://localhost:{args.port}"
    say(f"  {C['b']}▸ {url}{C['x']}", "g")
    say("  The UI opens immediately; the model finishes loading in the background (~5 s).", "d")
    say("  Press Ctrl+C to stop.", "d")
    say()

    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    os.chdir(SERVER)
    try:
        uvicorn.run("app:app", host=args.host, port=args.port, log_level="warning")
    except KeyboardInterrupt:
        pass
    say()
    say("  Stopped.", "d")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
