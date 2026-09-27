"""Architecture sizing tab — uses backend/architecture.py directly."""
import pathlib
import sys

import gradio as gr

ROOT = pathlib.Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from architecture import ANCHORS, architecture_for_size, estimated_parameters  # noqa: E402


def _compute(parameters_m):
    try:
        spec = architecture_for_size(parameters_m)
    except Exception as exc:
        return f"❌ {exc}", ""
    params = estimated_parameters(spec)
    lines = [
        f"### Spec for {parameters_m}M target",
        f"**Estimated parameters:** {params:,}  ({params/1e6:.2f}M)",
        "",
        "```json",
        str(spec),
        "```",
    ]
    return "\n".join(lines), str(spec)


def build_architecture_tab():
    with gr.Tab("🏗️ Architecture"):
        gr.Markdown(
            "Interpolate a continuous BananaAll size (3M–200M). "
            "Anchors are shown on the slider; exact anchors snap to the published shape."
        )
        with gr.Row():
            slider = gr.Slider(
                minimum=3, maximum=200, value=25, step=0.5, label="Target parameters (M)"
            )
            compute_btn = gr.Button("Compute spec", variant="primary")
        output_md = gr.Markdown()
        output_json = gr.Code(label="Config JSON", language="json")

        anchors_md = "**Anchors:** " + ", ".join(f"{k}M" for k in ANCHORS)
        gr.Markdown(anchors_md)

        compute_btn.click(_compute, inputs=[slider], outputs=[output_md, output_json])
        slider.release(_compute, inputs=[slider], outputs=[output_md, output_json])
