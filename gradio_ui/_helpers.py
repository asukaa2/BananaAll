"""Shared helpers for the BananaAll Gradio UI.

Includes:
  - format_event:   pretty one-line summary of a backend JSON event
  - parse_event:    parse the raw event dict that the worker emitted
  - pretty_json:    pretty-print any value as JSON
  - environment_status: probe Python/GPU/HF-login for the status footer
  - dataset presets, system prompt presets, model presets, benchmark catalog
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import gradio as gr

ROOT = pathlib.Path(os.getcwd()).resolve()
BACKEND = ROOT / "backend"

if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


# ---------------------------------------------------------------------------
# Event parsing & formatting
# ---------------------------------------------------------------------------

def parse_event(raw_line: str):
    """Try to decode a JSON line from the worker; fall back to a log event."""
    line = raw_line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return {"type": "log", "message": line}


def format_event(event: dict) -> str:
    """Convert a backend JSON event into a short display string."""
    kind = event.get("type", "log")
    if kind == "status":
        return f"⏳ {event.get('message', '')}"
    if kind == "log":
        return f"   {event.get('message', '')}"
    if kind == "metric":
        parts = [f"{k}={_fmt(v)}" for k, v in event.items() if k not in ("type",)]
        return "📊 " + "  ".join(parts)
    if kind == "progress":
        step = event.get("step", 0)
        total = event.get("totalSteps", "?")
        acc = event.get("accumulationStep", 0)
        acc_total = event.get("accumulationTotal", "?")
        lr = event.get("learningRate")
        extra = f"  lr={lr:.2e}" if isinstance(lr, (int, float)) else ""
        tokens = ""
        if "tokensSeen" in event:
            tokens = (
                f"  tokens={event['tokensSeen']:,}/"
                f"{event.get('totalTokens', '?'):,}"  # type: ignore[arg-type]
            )
        elapsed = event.get("elapsedSeconds")
        etime = f"  {elapsed:.0f}s" if isinstance(elapsed, (int, float)) else ""
        return (
            f"🔄 step {step}/{total}  micro {acc}/{acc_total}"
            f"{extra}{tokens}{etime}"
        )
    if kind == "architecture":
        return (
            f"🏗️  target={event.get('targetM')}M  "
            f"actual={event.get('parameters'):,} params  "
            f"layers={event.get('layers')}  "
            f"executions={event.get('executions')}"
        )
    if kind == "result":
        payload = json.dumps(event.get("result"), default=str)
        return f"✅ result[{event.get('task')}]: {payload[:400]}"
    if kind == "complete":
        return f"✅ COMPLETE — {event.get('message', '')}"
    if kind == "error":
        return f"❌ ERROR: {event.get('message', '')}"
    return json.dumps(event, default=str)


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.4f}" if abs(value) < 1 else f"{value:.2f}"
    return str(value)


def pretty_json(value, indent: int = 2, limit: int | None = None) -> str:
    """Pretty-print a value as JSON. Returns '' on failure."""
    try:
        text = json.dumps(value, indent=indent, default=str, ensure_ascii=False)
    except Exception:
        return ""
    if limit and len(text) > limit:
        text = text[:limit] + "\n…(truncated)"
    return text


# ---------------------------------------------------------------------------
# Environment status probe (for the footer)
# ---------------------------------------------------------------------------

def _import_version(name: str) -> str:
    try:
        import importlib.metadata as m
        return m.version(name)
    except Exception:
        return "—"


def environment_status() -> dict:
    """Probe Python, GPU, ML libs, and HF login status."""
    info = {
        "python": sys.version.split()[0],
        "gradio": _import_version("gradio"),
        "torch": "—",
        "transformers": "—",
        "datasets": "—",
        "cuda": "unavailable",
        "device_count": 0,
        "hf_user": None,
        "hf_message": "Not signed in",
        "backend_ready": False,
    }
    try:
        import torch  # noqa: F401
        info["torch"] = _import_version("torch")
        if torch.cuda.is_available():
            info["cuda"] = torch.cuda.get_device_name(0)
            info["device_count"] = torch.cuda.device_count()
    except Exception:
        pass
    info["transformers"] = _import_version("transformers")
    info["datasets"] = _import_version("datasets")
    info["backend_ready"] = all(
        v not in ("—",) for v in (info["torch"], info["transformers"], info["datasets"])
    )
    # HF login probe — call backend/auth_status.py's status() in-process.
    try:
        import importlib
        auth = importlib.import_module("auth_status")
        payload = auth.status()
        if payload.get("loggedIn"):
            info["hf_user"] = payload.get("username")
            info["hf_message"] = f"Signed in as {payload['username']}"
        else:
            info["hf_message"] = payload.get("message", "Not signed in")
    except Exception as exc:
        info["hf_message"] = f"auth probe failed: {exc.__class__.__name__}"
    return info


def render_status_chips(status: dict) -> str:
    """Render the status footer as markdown chips."""
    chips = [
        f"🐍 Python `{status['python']}`",
        f"🤗 Gradio `{status['gradio']}`",
        f"🔥 torch `{status['torch']}`",
        f"🤖 transformers `{status['transformers']}`",
        f"📚 datasets `{status['datasets']}`",
    ]
    if status["device_count"]:
        chips.append(f"⚡ CUDA `{status['cuda']}` ×{status['device_count']}")
    else:
        chips.append("🐢 CUDA `unavailable`")
    chips.append(f"🔑 HF `{status['hf_message']}`")
    return "  ".join(chips)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

DATASET_PRESETS = [
    {"label": "BananaMind Base Bench 1.1",
     "id": "BananaMind/BananaMind-Base-Bench-1.1", "split": "train"},
    {"label": "BananaMind Safety Bench 1.1",
     "id": "BananaMind/BananaMind-Safety-Bench-1.1", "split": "train"},
    {"label": "Arithmark 3.0",
     "id": "AxiomicLabs/Arithmark-3.0", "split": "train"},
    {"label": "Tiny Theory of Mind",
     "id": "AxiomicLabs/Tiny_Theory_of_Mind", "split": "train"},
    {"label": "HellaSwag",
     "id": "Rowan/hellaswag", "split": "train"},
    {"label": "PIQA",
     "id": "ybisk/piqa", "split": "train"},
]

MODEL_PRESETS = [
    "BananaMind/BananaMind-2-25M",
    "BananaMind/BananaMind-2-50M",
    "BananaMind/BananaMind-2-140M",
    "HuggingFaceTB/SmolLM-135M-Instruct",
    "HuggingFaceTB/SmolLM-360M-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
]

SYSTEM_PROMPT_PRESETS = [
    ("BananaMind default",
     "You are BananaMind, a helpful, harmless, and honest assistant."),
    ("Concise",
     "Be concise. Answer in 1-3 sentences unless asked for more."),
    ("Code reviewer",
     "You are a senior code reviewer. Review the code for bugs, "
     "style, and security. Provide concrete, prioritized feedback."),
    ("Plain chatbot",
     "You are a friendly assistant. Keep replies short and natural."),
]

BENCHMARK_CATALOG = {
    "piqa":           {"label": "PIQA",            "kind": "Standard (lm-eval-harness)", "desc": "Physical commonsense reasoning"},
    "lambada":        {"label": "LAMBADA",        "kind": "Standard (lm-eval-harness)", "desc": "Sentence completion accuracy"},
    "arc_easy":       {"label": "ARC-Easy",       "kind": "Standard (lm-eval-harness)", "desc": "Grade-school science MC, easy set"},
    "arc_challenge":  {"label": "ARC-Challenge",  "kind": "Standard (lm-eval-harness)", "desc": "Grade-school science MC, hard set"},
    "hellaswag":      {"label": "HellaSwag",      "kind": "Standard (lm-eval-harness)", "desc": "Commonsense NLI"},
    "arithmark":     {"label": "ArithMark 3.0",  "kind": "Continuation",             "desc": "Multi-grade arithmetic"},
    "tiny_tom":      {"label": "Tiny ToM",        "kind": "Continuation",             "desc": "Theory of mind reasoning"},
    "base_bench":     {"label": "Base Bench 1.1", "kind": "Official script",          "desc": "BananaMind base benchmark"},
    "safety_bench":   {"label": "Safety Bench 1.1","kind": "Official script",         "desc": "BananaMind safety benchmark"},
}


# ---------------------------------------------------------------------------
# CSS — small refinements on top of the dark theme
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
#banana-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 8px 4px 4px 4px;
}
#banana-header img.brand {
    width: 48px;
    height: 48px;
    border-radius: 12px;
    background: #1A1818;
    border: 1px solid #2A2626;
    padding: 4px;
}
#banana-header h1 {
    margin: 0;
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--body-text-color);
}
#banana-header .subtitle {
    margin: 2px 0 0 0;
    font-size: 0.85rem;
    color: var(--body-text-color-subdued);
}
#banana-status-bar {
    padding: 6px 12px;
    border-radius: 10px;
    background: #1A1818;
    border: 1px solid #2A2626;
    font-size: 0.78rem;
    color: var(--body-text-color-subdued);
    margin-bottom: 12px;
    display: flex;
    flex-wrap: wrap;
    gap: 6px 18px;
}
.tab-nav {
    border-bottom: 1px solid var(--border-color-primary) !important;
}
.tab-nav button {
    font-weight: 500 !important;
    padding: 8px 14px !important;
}
.tab-nav button.selected {
    border-bottom: 2px solid var(--color-accent) !important;
    color: var(--body-text-color) !important;
}
.code, .gr-code {
    font-size: 0.82rem !important;
}
.preset-row button {
    font-size: 0.78rem !important;
    padding: 4px 10px !important;
    height: 32px !important;
}
"""


# ---------------------------------------------------------------------------
# Reusable UI builders
# ---------------------------------------------------------------------------

def preset_row(items, *, container_id: str | None = None):
    """Render a horizontal scrollable row of small preset buttons."""
    buttons = []
    with gr.Row(elem_classes=["preset-row"]):
        for item in items:
            label = item["label"] if isinstance(item, dict) else item[0]
            buttons.append(gr.Button(label, size="sm", variant="secondary"))
    return buttons


def json_viewer(label: str, *, language: str = "json", interactive: bool = False):
    """A consistent JSON viewer across tabs."""
    return gr.Code(label=label, language=language, interactive=interactive,
                   elem_classes=["json-viewer"])
