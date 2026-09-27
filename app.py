"""BananaAll Gradio WebUI - main entry point."""
""" Made by asukaa2 for colab"""
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import threading
from queue import Queue, Empty
from argparse import ArgumentParser

import gradio as gr
from gradio_ui.theme.dark import Dark

theme = Dark()


ROOT = pathlib.Path(__file__).parent.resolve()
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from gradio_ui.tabs import (
    build_train_tab,
    build_inference_tab,
    build_evaluate_tab,
    build_dataset_tab,
    build_architecture_tab,
)


def run_worker(script_name, config, cancel_flag=None):
    """Run backend/<script_name> with a temp JSON config; yield parsed events."""
    script = BACKEND / script_name
    if not script.is_file():
        yield {"type": "error", "message": f"Missing backend worker: {script}"}
        return

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as fh:
        json.dump(config, fh, default=str)
        config_path = fh.name

    proc = subprocess.Popen(
        [sys.executable, str(script), config_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        cwd=str(BACKEND),
    )

    def _read_stderr():
        for line in proc.stderr:
            if line.strip():
                print(f"[stderr:{script_name}] {line.rstrip()}", file=sys.stderr, flush=True)

    stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
    stderr_thread.start()

    try:
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                yield {"type": "log", "message": line}
    finally:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        os.unlink(config_path)


def format_event(event):
    """Convert a backend JSON event into a short display string."""
    kind = event.get("type", "log")
    if kind == "status":
        return f"⏳ {event.get('message', '')}"
    if kind == "log":
        return f"   {event.get('message', '')}"
    if kind == "metric":
        parts = [f"{k}={v}" for k, v in event.items() if k not in ("type",)]
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
            tokens = f"  tokens={event['tokensSeen']:,}/{event.get('totalTokens', '?'):,}"
        elapsed = event.get("elapsedSeconds")
        etime = f"  {elapsed:.0f}s" if isinstance(elapsed, (int, float)) else ""
        return f"🔄 step {step}/{total}  micro {acc}/{acc_total}{lr and extra}{tokens}{etime}"
    if kind == "architecture":
        return (f"🏗️  target={event.get('targetM')}M  actual={event.get('parameters'):,} params  "
                f"layers={event.get('layers')}  executions={event.get('executions')}")
    if kind == "result":
        return f"✅ result[{event.get('task')}]: {json.dumps(event.get('result'), default=str)[:400]}"
    if kind == "complete":
        return f"✅ COMPLETE — {event.get('message', '')}"
    if kind == "error":
        return f"❌ ERROR: {event.get('message', '')}"
    return json.dumps(event, default=str)


def stream_logs(config, script_name):
    """Yield accumulating log text for a backend run."""
    lines = []
    for event in run_worker(script_name, config):
        lines.append(format_event(event))
        yield "\n".join(lines[-400:])

if __name__ == '__main__':
    parser = ArgumentParser(description='BananaAll Studio.', add_help=True)
    parser.add_argument("--share", action="store_true", dest="share_enabled", default=False, help="Enable sharing")
        
    args = parser.parse_args()
    
    def build_app():
        with gr.Blocks(title="BananaAll Studio") as app:
            gr.Markdown("# 🍌 BananaAll Studio\n"
                        "Train, fine-tune, evaluate, and chat with BananaAll / BananaMind models.")
            with gr.Tabs():
                build_architecture_tab()
                build_dataset_tab()
                build_train_tab(stream_logs)
                build_inference_tab(stream_logs)
            build_evaluate_tab(stream_logs)
            
            return app
    
    if __name__ == "__main__":
        build_app().launch(
            share=args.share_enabled,
            server_name="0.0.0.0",
            server_port=7860,
            theme=theme
        )
