"""Dataset inspection tab — wraps backend/inspect_dataset.py."""
import json
import pathlib
import subprocess
import sys

import gradio as gr

ROOT = pathlib.Path(os.getcwd()).resolve()
BACKEND = ROOT / "backend"

if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from common import auto_mapping  # noqa: E402  (used if direct import available)


def _preview(dataset_id, config_name, split):
    if not dataset_id.strip():
        return "❌ Enter a dataset ID or local path.", ""
    source = {"id": dataset_id.strip(), "config": config_name.strip() or None, "split": split.strip() or "train"}
    try:
        proc = subprocess.run(
            [sys.executable, str(BACKEND / "inspect_dataset.py"), json.dumps(source)],
            capture_output=True, text=True, timeout=120, cwd=str(BACKEND),
        )
    except subprocess.TimeoutExpired:
        return "❌ Preview timed out after 120s.", ""
    if proc.returncode != 0:
        return f"❌ {proc.stderr.strip() or proc.stdout.strip()}", ""
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return f"❌ Unexpected output:\n{proc.stdout}", ""
    cols = payload.get("columns", [])
    samples = payload.get("samples", [])
    mapping = payload.get("mapping", {})
    lines = [
        f"**Columns:** {', '.join(cols)}",
        f"**Auto mapping:** `{json.dumps(mapping)}`",
        "",
        "**First rows:**",
        "```json",
        json.dumps(samples, indent=2, default=str)[:6000],
        "```",
    ]
    return "\n".join(lines), json.dumps(mapping, indent=2)


def build_dataset_tab():
    with gr.Tab("📊 Dataset"):
        gr.Markdown("Preview a dataset and inspect its auto-detected field mapping.")
        with gr.Row():
            dataset_id = gr.Textbox(label="Dataset ID or local path", placeholder="BananaMind/BananaMind-Base-Bench-1.1")
            config_name = gr.Textbox(label="Config (optional)", placeholder="default")
            split = gr.Textbox(label="Split", value="train")
        preview_btn = gr.Button("Preview", variant="primary")
        preview_md = gr.Markdown()
        mapping_json = gr.Code(label="Auto mapping", language="json")

        preview_btn.click(_preview, inputs=[dataset_id, config_name, split],
                          outputs=[preview_md, mapping_json])
