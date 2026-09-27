"""Training tab — builds a config and streams backend/train.py events in-process.

Improvements (rev 2):
  - Fixes `ValueError: truth value of a DataFrame is ambiguous` — Gradio
    passes a pandas DataFrame, not a list of lists; we explicitly coerce
    via `.values.tolist()` before iterating.
  - Adds `revision` and `mapping` columns to the dataset DataFrame, so
    users can pull a dataset from a specific HF revision/branch and
    override the auto-detected field mapping when a custom dataset has
    non-standard column names.
  - Adds "Add row" / "Clear" buttons and a row of dataset-preset chips.
"""
import json

import gradio as gr

from .._helpers import DATASET_PRESETS
from .._worker import stream_events


PRETRAINING_DEFAULT = {
    "mode": "pretraining",
    "parametersM": 25,
    "architecture": "bananamind2",
    "trustRemoteCode": False,
    "outputPath": "./output/pretrain",
    "sequenceLength": 1024,
    "batchSize": 2,
    "gradientAccumulation": 8,
    "learningRate": 2e-4,
    "warmupRatio": 0.03,
    "weightDecay": 0.01,
    "maxGradNorm": 1.0,
    "scheduler": "cosine",
    "loggingSteps": 10,
    "saveSteps": 100,
    "seed": 1337,
    "precision": "auto",
    "tokenizerSamples": 2000,
    "datasets": [],
}

# Order MUST match the datatype list below.
DATASET_HEADERS = [
    "id", "split", "config", "revision",
    "weight", "tokenLimit", "mapping", "enabled",
]
DATASET_DATATYPES = [
    "str", "str", "str", "str",
    "number", "number", "str", "bool",
]
DATASET_PLACEHOLDER_ROW = [
    "BananaMind/BananaMind-Base-Bench-1.1", "train", "", "", 1.0, 5000000, "", True
]


def _coerce_rows(value):
    """Coerce a gr.Dataframe value (pandas DataFrame / numpy / list) to a
    plain list of lists, without triggering ambiguous DataFrame truthiness.

    This is the fix for `ValueError: The truth value of a DataFrame is
    ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().`
    """
    if value is None:
        return []
    # pandas DataFrame
    if hasattr(value, "values") and hasattr(value.values, "tolist"):
        return value.values.tolist()
    # numpy array / 2D ndarray
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    # Already a list of lists (or list of pandas Series)
    if isinstance(value, (list, tuple)):
        out = []
        for row in value:
            if row is None:
                continue
            if hasattr(row, "tolist"):
                out.append(row.tolist())
            elif isinstance(row, (list, tuple)):
                out.append(list(row))
            else:
                out.append([row])
        return out
    return []


def _parse_mapping(raw):
    """Parse a JSON object string into a mapping dict, or return None."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict):
        return None
    return obj


def _dataframe_to_datasets(value):
    """Convert the editable DataFrame into backend dataset dicts."""
    rows = _coerce_rows(value)
    out = []
    for row in rows:
        if not any(str(v).strip() for v in row if v is not None):
            continue
        # Pad to expected length so we don't IndexError on partial rows.
        cells = list(row) + [None] * (len(DATASET_HEADERS) - len(row))
        id_val, split, config, revision, weight, token_limit, mapping, enabled = cells[:8]
        if id_val is None or not str(id_val).strip():
            continue
        try:
            weight_f = float(weight) if weight not in (None, "") else 1.0
        except (TypeError, ValueError):
            weight_f = 1.0
        try:
            token_limit_i = (int(token_limit)
                              if token_limit not in (None, "") else None)
        except (TypeError, ValueError):
            token_limit_i = None
        try:
            enabled_b = bool(enabled)
        except Exception:
            enabled_b = True

        ds = {
            "id": str(id_val).strip(),
            "split": str(split or "").strip() or "train",
            "config": str(config or "").strip() or None,
            "weight": weight_f,
            "enabled": enabled_b,
        }
        if token_limit_i is not None:
            ds["tokenLimit"] = token_limit_i
        if str(revision or "").strip():
            ds["revision"] = str(revision).strip()
        mapping_dict = _parse_mapping(mapping)
        if mapping_dict:
            ds["mapping"] = mapping_dict
        out.append(ds)
    return out


def _build_config(mode, parameters_m, architecture, trust_remote, output_path,
                  sequence_length, batch_size, grad_accum, lr, warmup, wd,
                  grad_norm, scheduler, logging_steps, save_steps, seed, precision,
                  tokenizer_samples, dataset_rows, base_model, max_steps,
                  lora_rank, lora_alpha, lora_dropout,
                  custom_config_path, custom_modeling_path):
    cfg = dict(PRETRAINING_DEFAULT)
    cfg.update({
        "mode": mode,
        "parametersM": parameters_m,
        "architecture": architecture,
        "trustRemoteCode": trust_remote,
        "outputPath": output_path,
        "sequenceLength": int(sequence_length),
        "batchSize": int(batch_size),
        "gradientAccumulation": int(grad_accum),
        "learningRate": float(lr),
        "warmupRatio": float(warmup),
        "weightDecay": float(wd),
        "maxGradNorm": float(grad_norm),
        "scheduler": scheduler,
        "loggingSteps": int(logging_steps),
        "saveSteps": int(save_steps),
        "seed": int(seed),
        "precision": precision,
        "tokenizerSamples": int(tokenizer_samples),
    })
    cfg["datasets"] = _dataframe_to_datasets(dataset_rows)
    if mode in ("finetune", "lora", "full"):
        cfg["model"] = base_model.strip()
    if mode == "finetune":
        cfg["maxSteps"] = int(max_steps)
    if mode == "lora":
        cfg["loraRank"] = int(lora_rank)
        cfg["loraAlpha"] = int(lora_alpha)
        cfg["loraDropout"] = float(lora_dropout)
        cfg["maxSteps"] = int(max_steps)
    if mode == "full":
        cfg["maxSteps"] = int(max_steps)
    if architecture == "custom":
        cfg["customConfigPath"] = custom_config_path.strip()
        cfg["customModelingPath"] = custom_modeling_path.strip()
    return cfg


def _on_mode_change(mode):
    return (
        gr.update(visible=mode == "pretraining"),
        gr.update(visible=mode in ("finetune", "lora", "full")),
        gr.update(visible=mode in ("finetune", "lora", "full")),
        gr.update(visible=mode == "lora"),
    )


def _on_arch_change(arch):
    return gr.update(visible=arch == "custom")


def _add_row(current):
    rows = _coerce_rows(current)
    rows.append(list(DATASET_PLACEHOLDER_ROW))
    return gr.update(value=rows)


def _clear_rows():
    return gr.update(value=[])


def _append_preset_dataset(current, preset_label):
    """Append a new row for the given preset label."""
    rows = _coerce_rows(current)
    preset = next((p for p in DATASET_PRESETS if p["label"] == preset_label), None)
    if preset is None:
        return gr.update()
    rows.append([
        preset["id"],
        preset.get("split", "train"),
        "", "", 1.0, 1_000_000, "", True,
    ])
    return gr.update(value=rows)


def build_train_tab():
    with gr.Tab("🚀 Train"):
        mode = gr.Radio(
            choices=["pretraining", "finetune", "lora", "full"],
            value="pretraining", label="Training mode",
            info="pretraining = from scratch · finetune = full weights · lora = adapters",
        )

        with gr.Group(visible=True) as pretrain_group:
            with gr.Row():
                parameters_m = gr.Slider(3, 200, value=25, step=0.5,
                                         label="Parameters (M)")
                architecture = gr.Dropdown(
                    ["bananamind2", "lft", "bananamind2lft", "custom"],
                    value="bananamind2", label="Architecture",
                )
            with gr.Row() as custom_row:
                custom_config_path = gr.Textbox(
                    label="Custom config.json path", visible=False, scale=1)
                custom_modeling_path = gr.Textbox(
                    label="Custom modeling .py path", visible=False, scale=1)
            gr.Markdown(
                "### Dataset mix\n"
                "One row per source. Type any Hugging Face dataset ID, "
                "local path, or HF URL. `tokenLimit` = total tokens to "
                "consume from that source; `weight` = mix weight for "
                "selecting the next active source; `revision` = optional "
                "HF branch/commit; `mapping` = optional JSON override "
                "(e.g. `{\"messages\":\"conversations\"}`) for "
                "non-standard column names."
            )

            # Quick-add preset chips for common HF datasets
            with gr.Row(elem_classes=["preset-row"]):
                preset_buttons = [
                    gr.Button(item["label"], size="sm", variant="secondary")
                    for item in DATASET_PRESETS
                ]

            datasets_df = gr.Dataframe(
                headers=DATASET_HEADERS,
                datatype=DATASET_DATATYPES,
                value=[list(DATASET_PLACEHOLDER_ROW)],
                row_count=(1, "dynamic"),
                column_count=(len(DATASET_HEADERS), "fixed"),
                interactive=True,
                wrap=True,
                label="Datasets",
            )
            with gr.Row():
                add_btn = gr.Button("➕ Add row", variant="secondary", size="sm")
                clear_btn = gr.Button("🗑️ Clear rows", variant="secondary", size="sm")

            tokenizer_samples = gr.Number(value=2000,
                                          label="Tokenizer training samples",
                                          precision=0)

        with gr.Group(visible=False) as model_group:
            base_model = gr.Textbox(
                label="Base model ID / local folder / HF URL",
                placeholder="BananaMind/BananaMind-2-25M",
            )

        with gr.Group(visible=False) as steps_group:
            max_steps = gr.Number(value=500, label="Max optimizer steps",
                                  precision=0)

        with gr.Group(visible=False) as lora_group:
            with gr.Row():
                lora_rank = gr.Number(value=16, label="LoRA rank", precision=0)
                lora_alpha = gr.Number(value=32, label="LoRA alpha", precision=0)
                lora_dropout = gr.Slider(0, 0.5, value=0.05,
                                          label="LoRA dropout")

        with gr.Accordion("Hyperparameters", open=False):
            with gr.Row():
                sequence_length = gr.Number(value=1024, label="Sequence length", precision=0)
                batch_size = gr.Number(value=2, label="Batch size", precision=0)
                grad_accum = gr.Number(value=8, label="Grad accumulation", precision=0)
            with gr.Row():
                lr = gr.Number(value=2e-4, label="Learning rate")
                warmup = gr.Number(value=0.03, label="Warmup ratio")
                wd = gr.Number(value=0.01, label="Weight decay")
                grad_norm = gr.Number(value=1.0, label="Max grad norm")
            with gr.Row():
                scheduler = gr.Dropdown(["cosine", "linear", "constant"],
                                          value="cosine", label="Scheduler")
                precision = gr.Dropdown(["auto", "bf16", "fp16", "fp32"],
                                          value="auto", label="Precision")
                logging_steps = gr.Number(value=10, label="Log every N steps", precision=0)
                save_steps = gr.Number(value=100, label="Save every N steps", precision=0)
            with gr.Row():
                seed = gr.Number(value=1337, label="Seed", precision=0)
                trust_remote = gr.Checkbox(value=False, label="Trust remote code")
                output_path = gr.Textbox(value="./output/pretrain",
                                          label="Output folder", scale=2)

        with gr.Row():
            start_btn = gr.Button("▶️ Start training", variant="primary")
            stop_btn = gr.Button("⏹️ Stop", variant="stop")

        status_pill = gr.Markdown("", elem_classes=["status-pill"])
        log_box = gr.Textbox(label="Live log", lines=24, max_lines=24,
                              autoscroll=True, interactive=False)

        mode.change(_on_mode_change, inputs=[mode],
                    outputs=[pretrain_group, model_group, steps_group, lora_group])
        architecture.change(_on_arch_change, inputs=[architecture],
                            outputs=[custom_row])

        # DataFrame row management
        add_btn.click(_add_row, inputs=[datasets_df], outputs=[datasets_df])
        clear_btn.click(_clear_rows, outputs=[datasets_df])
        for btn, item in zip(preset_buttons, DATASET_PRESETS):
            btn.click(
                _append_preset_dataset,
                inputs=[datasets_df, gr.State(item["label"])],
                outputs=[datasets_df],
            )

        def _run(*args):
            cfg = _build_config(*args)
            lines: list[str] = []
            last_status = ""
            for event in stream_events(cfg, "train.py"):
                text = _format_event_for_log(event)
                if text:
                    lines.append(text)
                if event.get("type") == "status":
                    last_status = event.get("message", "")
                status_md = f"**Status:** {last_status}" if last_status else ""
                yield status_md, "\n".join(lines[-400:])
            yield (f"**Final:** {last_status}" if last_status else "**Done.**",
                  "\n".join(lines[-400:]))

        inputs = [
            mode, parameters_m, architecture, trust_remote, output_path,
            sequence_length, batch_size, grad_accum, lr, warmup, wd, grad_norm,
            scheduler, logging_steps, save_steps, seed, precision,
            tokenizer_samples, datasets_df, base_model, max_steps,
            lora_rank, lora_alpha, lora_dropout,
            custom_config_path, custom_modeling_path,
        ]
        start_btn.click(_run, inputs=inputs, outputs=[status_pill, log_box])


def _format_event_for_log(event: dict) -> str:
    """Compact one-line summary for a worker event."""
    kind = event.get("type", "log")
    if kind == "status":
        return f"⏳ {event.get('message', '')}"
    if kind == "log":
        return f"   {event.get('message', '')}"
    if kind == "metric":
        parts = [f"{k}={_fmt(v)}" for k, v in event.items()
                 if k not in ("type",)]
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
            tokens = (f"  tokens={event['tokensSeen']:,}/"
                      f"{event.get('totalTokens', '?'):,}")
        elapsed = event.get("elapsedSeconds")
        etime = f"  {elapsed:.0f}s" if isinstance(elapsed, (int, float)) else ""
        return (f"🔄 step {step}/{total}  micro {acc}/{acc_total}"
                f"{extra}{tokens}{etime}")
    if kind == "architecture":
        return (f"🏗️ target={event.get('targetM')}M  "
                f"actual={event.get('parameters'):,} params  "
                f"layers={event.get('layers')}  "
                f"executions={event.get('executions')}")
    if kind == "result":
        payload = str(event.get("result"))[:400]
        return f"✅ result[{event.get('task')}]: {payload}"
    if kind == "complete":
        return f"✅ COMPLETE — {event.get('message', '')}"
    if kind == "error":
        return f"❌ ERROR: {event.get('message', '')}"
    return str(event)


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.4f}" if abs(value) < 1 else f"{value:.2f}"
    return str(value)
