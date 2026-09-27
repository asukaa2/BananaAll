"""Architecture sizing tab — uses backend/architecture.py directly.

Improved version:
  - shows the full anchor table as a side-by-side reference,
  - displays the spec as a clean key/value table instead of a JSON blob,
  - includes a copyable JSON output for use with `config.json`.
"""
import pathlib
import os
import sys

import gradio as gr

from .._helpers import (
    BACKEND,
    MODEL_PRESETS,
    json_viewer,
    pretty_json,
)

if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from architecture import ANCHORS, architecture_for_size, estimated_parameters  # noqa: E402


def _spec_rows(spec: dict) -> list[list]:
    return [
        ["vocab_size",          spec["vocab_size"]],
        ["hidden_size",         spec["hidden_size"]],
        ["num_hidden_layers",   spec["num_hidden_layers"]],
        ["num_attention_heads", spec["num_attention_heads"]],
        ["num_key_value_heads", spec["num_key_value_heads"]],
        ["head_dim",            spec["head_dim"]],
        ["intermediate_size",   spec["intermediate_size"]],
        ["max_position_embeddings", spec["max_position_embeddings"]],
        ["rope_theta",          spec["rope_theta"]],
    ]


def _anchor_rows():
    rows = []
    for size, spec in ANCHORS.items():
        params = estimated_parameters(spec)
        rows.append([
            f"{size}M",
            f"{params:,}",
            spec["hidden_size"],
            spec["num_hidden_layers"],
            spec["num_attention_heads"],
            spec["num_key_value_heads"],
            spec["head_dim"],
            spec["vocab_size"],
            spec["intermediate_size"],
        ])
    return rows


def _compute(parameters_m):
    try:
        spec = architecture_for_size(parameters_m)
    except Exception as exc:
        return (
            f"❌ {exc}",
            [],
            "",
        )
    params = estimated_parameters(spec)
    md = (
        f"### Spec for **{parameters_m}M** target\n\n"
        f"- **Estimated parameters:** `{params:,}` ({params / 1e6:.2f}M)\n"
        f"- **Layers × heads × head_dim:** "
        f"{spec['num_hidden_layers']} × {spec['num_attention_heads']} × "
        f"{spec['head_dim']}\n"
        f"- **Hidden size:** {spec['hidden_size']}\n"
        f"- **Intermediate size:** {spec['intermediate_size']}\n"
        f"- **Vocab size:** {spec['vocab_size']}\n"
        f"- **Context length:** {spec['max_position_embeddings']}\n"
        f"- **RoPE theta:** {spec['rope_theta']}\n"
    )
    return md, _spec_rows(spec), pretty_json(spec)


def build_architecture_tab():
    with gr.Tab("🏗️ Architecture"):
        gr.Markdown(
            "### Continuous BananaAll sizing (3M–200M)\n"
            "Pick a target size. The published shapes (3M, 10M, 25M, 50M, "
            "140M, 200M) are **exact anchors** — anything in between is "
            "interpolated to land on a sensible head/layer configuration while "
            "hitting the parameter target."
        )

        with gr.Row():
            slider = gr.Slider(
                minimum=3, maximum=200, value=25, step=0.5,
                label="Target parameters (M)", info="Drag to interpolate",
            )
            compute_btn = gr.Button("Compute spec", variant="primary")

        with gr.Row(equal_height=True):
            with gr.Column(scale=3):
                output_md = gr.Markdown()
            with gr.Column(scale=2):
                spec_table = gr.Dataframe(
                    headers=["Field", "Value"],
                    datatype=["str", "str"],
                    row_count=(9, "dynamic"),
                    column_count=(2, "fixed"),
                    interactive=False,
                    wrap=True,
                    elem_classes=["spec-table"],
                    label="Generated configuration",
                )

        output_json = json_viewer("Config JSON (copyable)")
        gr.Markdown("**Reference — published anchor shapes:**")
        anchor_table = gr.Dataframe(
            headers=["Target", "Params", "hidden", "layers", "heads",
                     "kv heads", "head_dim", "vocab", "intermediate"],
            datatype=["str"] * 9,
            value=_anchor_rows(),
            interactive=False,
            wrap=True,
            elem_classes=["anchor-table"],
        )

        compute_btn.click(_compute, inputs=[slider],
                          outputs=[output_md, spec_table, output_json])
        slider.release(_compute, inputs=[slider],
                       outputs=[output_md, spec_table, output_json])
