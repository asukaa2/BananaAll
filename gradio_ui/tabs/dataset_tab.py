"""Dataset inspection tab — calls backend/inspect_dataset.py.preview() directly."""
import json
import os
import pathlib
import sys

import gradio as gr

from .._helpers import (
    BACKEND,
    DATASET_PRESETS,
    pretty_json,
)

if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from inspect_dataset import preview as _preview_dataset  # noqa: E402


def _format_preview(payload: dict):
    """Build the (markdown, dataframe_update, mapping_json) outputs."""
    cols = payload.get("columns", [])
    samples = payload.get("samples", [])
    mapping = payload.get("mapping", {})

    md_lines = [
        f"**Columns:** {', '.join(f'`{c}`' for c in cols) or '—'}",
        f"**Sample count:** {len(samples)}",
        f"**Auto mapping:** `{json.dumps(mapping)}`",
    ]
    table_rows = [
        [_stringify(row.get(c, "")) for c in cols]
        for row in samples
    ]
    df_update = gr.update(
        value=table_rows,
        headers=cols or ["(no rows)"],
        datatype=["str"] * max(len(cols), 1),
        column_count=(max(len(cols), 1), "dynamic"),
        row_count=(len(table_rows), "dynamic"),
    )
    return "\n".join(md_lines), df_update, pretty_json(mapping)


def _stringify(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)[:200]
    return str(value)


def _do_preview(dataset_id, config_name, split):
    if not dataset_id.strip():
        return "❌ Enter a dataset ID or local path.", gr.update(), ""
    source = {
        "id": dataset_id.strip(),
        "config": config_name.strip() or None,
        "split": split.strip() or "train",
    }
    try:
        payload = _preview_dataset(source)
    except Exception as exc:
        return (f"❌ {exc.__class__.__name__}: {exc}", gr.update(), "")
    return _format_preview(payload)


def _apply_preset(preset_label: str):
    """Look up the preset by label and return (id, split)."""
    for item in DATASET_PRESETS:
        if item["label"] == preset_label:
            return item["id"], item.get("split", "train")
    return "", "train"


def _label_to_id(label: str) -> str:
    for item in DATASET_PRESETS:
        if item["label"] == label:
            return item["id"]
    return ""


def _label_to_split(label: str) -> str:
    for item in DATASET_PRESETS:
        if item["label"] == label:
            return item.get("split", "train")
    return "train"


def build_dataset_tab():
    with gr.Tab("📊 Dataset"):
        gr.Markdown(
            "### Preview a dataset before training\n"
            "Pick a curated preset or paste a Hugging Face dataset ID / "
            "local path. The auto-detected field mapping drives the "
            "training parser — BananaAll will use it to pull `messages`, "
            "`system`, `user`, `assistant`, or raw `text` from each row."
        )

        # Quick presets row
        preset_btns_row = gr.Row()
        with preset_btns_row:
            preset_buttons = []
            for item in DATASET_PRESETS:
                preset_buttons.append(
                    gr.Button(item["label"], size="sm", variant="secondary",
                              elem_classes=["preset-row"])
                )

        with gr.Row():
            dataset_id = gr.Textbox(
                label="Dataset ID or local path",
                placeholder="BananaMind/BananaMind-Base-Bench-1.1",
                scale=4,
            )
            config_name = gr.Textbox(
                label="Config (optional)", placeholder="default", scale=1,
            )
            split = gr.Textbox(label="Split", value="train", scale=1)
        preview_btn = gr.Button("🔍 Preview", variant="primary")

        preview_md = gr.Markdown()
        samples_table = gr.Dataframe(
            headers=["(no data)"],
            datatype=["str"],
            interactive=False,
            wrap=True,
            label="First rows",
            column_count=(1, "dynamic"),
            row_count=(0, "dynamic"),
        )
        mapping_json = gr.Code(label="Auto mapping", language="json",
                               interactive=False)

        preview_btn.click(_do_preview,
                          inputs=[dataset_id, config_name, split],
                          outputs=[preview_md, samples_table, mapping_json])

        # Wire each preset button to fill in dataset_id + split
        for btn, item in zip(preset_buttons, DATASET_PRESETS):
            btn.click(
                lambda _label=item["label"]: (
                    _label_to_id(_label), _label_to_split(_label),
                ),
                outputs=[dataset_id, split],
            )
