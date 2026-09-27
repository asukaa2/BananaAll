"""Training tab — builds a config and streams backend/train.py events."""
import json

import gradio as gr


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


def _parse_datasets(text):
    """Each non-empty line: id | split | config | weight | tokenLimit"""
    out = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split("|")]
        while len(parts) < 5:
            parts.append("")
        ds = {
            "id": parts[0],
            "split": parts[1] or "train",
            "config": parts[2] or None,
            "weight": float(parts[3]) if parts[3] else 1.0,
            "enabled": True,
        }
        if parts[4]:
            ds["tokenLimit"] = int(parts[4])
        out.append(ds)
    return out


def _build_config(mode, parameters_m, architecture, trust_remote, output_path,
                  sequence_length, batch_size, grad_accum, lr, warmup, wd,
                  grad_norm, scheduler, logging_steps, save_steps, seed, precision,
                  tokenizer_samples, datasets_text, base_model, max_steps,
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
    cfg["datasets"] = _parse_datasets(datasets_text)
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


def build_train_tab(stream_logs):
    with gr.Tab("🚀 Train"):
        mode = gr.Radio(
            choices=["pretraining", "finetune", "lora", "full"],
            value="pretraining", label="Training mode",
        )

        with gr.Group(visible=True) as pretrain_group:
            with gr.Row():
                parameters_m = gr.Slider(3, 200, value=25, step=0.5, label="Parameters (M)")
                architecture = gr.Dropdown(
                    ["bananamind2", "lft", "bananamind2lft", "custom"],
                    value="bananamind2", label="Architecture",
                )
            with gr.Row() as custom_row:
                custom_config_path = gr.Textbox(label="Custom config.json path", visible=False)
                custom_modeling_path = gr.Textbox(label="Custom modeling .py path", visible=False)
            gr.Markdown("**Datasets** — one per line: `id | split | config | weight | tokenLimit`")
            datasets_text = gr.Textbox(
                lines=4, label="Dataset mix",
                placeholder="BananaMind/BananaMind-Base-Bench-1.1 | train | | 1 | 5000000",
            )
            tokenizer_samples = gr.Number(value=2000, label="Tokenizer training samples", precision=0)

        with gr.Group(visible=False) as model_group:
            base_model = gr.Textbox(label="Base model ID / local folder / HF URL")

        with gr.Group(visible=False) as steps_group:
            max_steps = gr.Number(value=500, label="Max optimizer steps", precision=0)

        with gr.Group(visible=False) as lora_group:
            with gr.Row():
                lora_rank = gr.Number(value=16, label="LoRA rank", precision=0)
                lora_alpha = gr.Number(value=32, label="LoRA alpha", precision=0)
                lora_dropout = gr.Slider(0, 0.5, value=0.05, label="LoRA dropout")

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
                scheduler = gr.Dropdown(["cosine", "linear", "constant"], value="cosine", label="Scheduler")
                precision = gr.Dropdown(["auto", "bf16", "fp16", "fp32"], value="auto", label="Precision")
                logging_steps = gr.Number(value=10, label="Log every N steps", precision=0)
                save_steps = gr.Number(value=100, label="Save every N steps", precision=0)
            with gr.Row():
                seed = gr.Number(value=1337, label="Seed", precision=0)
                trust_remote = gr.Checkbox(value=False, label="Trust remote code")
                output_path = gr.Textbox(value="./output/pretrain", label="Output folder")

        start_btn = gr.Button("▶️ Start training", variant="primary")
        stop_btn = gr.Button("⏹️ Stop", variant="stop")
        log_box = gr.Textbox(label="Live log", lines=24, max_lines=24, autoscroll=True)

        mode.change(_on_mode_change, inputs=[mode],
                    outputs=[pretrain_group, model_group, steps_group, lora_group])
        architecture.change(_on_arch_change, inputs=[architecture], outputs=[custom_row])

        def _run(*args, **kwargs):
            mode_val = args[0]
            cfg = _build_config(*args)
            yield from stream_logs(cfg, "train.py")

        inputs = [
            mode, parameters_m, architecture, trust_remote, output_path,
            sequence_length, batch_size, grad_accum, lr, warmup, wd, grad_norm,
            scheduler, logging_steps, save_steps, seed, precision,
            tokenizer_samples, datasets_text, base_model, max_steps,
            lora_rank, lora_alpha, lora_dropout,
            custom_config_path, custom_modeling_path,
        ]
        start_btn.click(_run, inputs=inputs, outputs=[log_box])
