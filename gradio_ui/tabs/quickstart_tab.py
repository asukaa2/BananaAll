"""Quick Start tab — one-click presets that jump to other tabs."""
import gradio as gr

from .._helpers import (
    BENCHMARK_CATALOG,
    DATASET_PRESETS,
    MODEL_PRESETS,
    SYSTEM_PROMPT_PRESETS,
    environment_status,
    pretty_json,
)


PRESETS = [
    {
        "id": "pretrain-25m",
        "title": "🍞 Pretrain a 25M model",
        "summary": (
            "BananaMind-2 architecture, 25M parameters, byte-level BPE "
            "tokenizer trained on a 5M-token slice of BananaMind Base Bench. "
            "Cosine schedule, 2e-4 LR, bf16 auto, ~25 optimizer steps."
        ),
        "tab": "🚀 Train",
    },
    {
        "id": "pretrain-140m",
        "title": "🍌 Pretrain a 140M model",
        "summary": (
            "Same family, scaled to the 140M published anchor. Needs a "
            "real GPU and several hours. 25M tokens of mixed data → ~50 "
            "optimizer steps at batch 4 × grad-accum 8."
        ),
        "tab": "🚀 Train",
    },
    {
        "id": "lora-finetune",
        "title": "🔧 LoRA fine-tune a base model",
        "summary": (
            "Load a base model (e.g. SmolLM-135M), attach LoRA adapters "
            "(rank 16, alpha 32), train 500 steps on an instruct dataset. "
            "Adapters save to ./output/lora."
        ),
        "tab": "🚀 Train",
    },
    {
        "id": "chat-smol",
        "title": "💬 Chat with SmolLM-135M-Instruct",
        "summary": (
            "Instruct-mode inference. System prompt defaults to a friendly "
            "assistant. 256 max new tokens, 0.7 temperature, 0.95 top-p."
        ),
        "tab": "💬 Inference",
    },
    {
        "id": "eval-piqa",
        "title": "🧪 Evaluate on PIQA + HellaSwag",
        "summary": (
            "Standard lm-evaluation-harness tasks. Run both at once for "
            "a quick quality signal. Limit 0 = full benchmark."
        ),
        "tab": "🧪 Evaluate",
    },
    {
        "id": "inspect-base",
        "title": "📊 Inspect BananaMind Base Bench 1.1",
        "summary": (
            "Preview the first 3 rows and inspect the auto-detected field "
            "mapping before training on this dataset."
        ),
        "tab": "📊 Dataset",
    },
]


def _presets_as_rows():
    return [[p["title"], p["summary"], p["tab"]] for p in PRESETS]


def build_quickstart_tab():
    status = environment_status()

    with gr.Tab("✨ Quick Start"):
        gr.Markdown(
            "### Welcome to BananaAll Studio\n"
            "BananaAll is a local-first toolkit for training, fine-tuning, "
            "evaluating, and chatting with small language models (3M–200M "
            "parameters). This tab collects curated starting points — pick "
            "one and the relevant tab will guide you through the rest."
        )

        # Environment readiness banner
        if status["backend_ready"]:
            readiness = (
                "🟢 **Backend ready** — torch, transformers, and datasets "
                "are installed. You can train and evaluate now."
            )
        else:
            missing = []
            if status["torch"] == "—":
                missing.append("torch")
            if status["transformers"] == "—":
                missing.append("transformers")
            if status["datasets"] == "—":
                missing.append("datasets")
            readiness = (
                "🟡 **Backend not ready** — the following packages are "
                f"missing: {', '.join(missing)}. Install them with "
                f"`pip install -r requirements.txt`. You can still browse "
                f"the UI and preview datasets, but training/inference will "
                f"fail until they're installed."
            )
        gr.Markdown(readiness)

        gr.Markdown("### Curated presets")
        gr.Dataframe(
            headers=["Preset", "What it does", "Goes to"],
            datatype=["str", "str", "str"],
            value=_presets_as_rows(),
            interactive=False,
            wrap=True,
            row_count=(len(PRESETS), "fixed"),
            column_count=(3, "fixed"),
        )

        gr.Markdown(
            "### Available models\n"
            + " · ".join(f"`{m}`" for m in MODEL_PRESETS)
        )
        gr.Markdown(
            "### Available datasets\n"
            + " · ".join(f"`{d['id']}`" for d in DATASET_PRESETS)
        )
        gr.Markdown(
            "### Available benchmarks\n"
            + " · ".join(
                f"`{info['label']}` ({info['kind']})"
                for info in BENCHMARK_CATALOG.values()
            )
        )
        gr.Markdown(
            "### System-prompt presets\n"
            + " · ".join(f"`{label}`" for label, _ in SYSTEM_PROMPT_PRESETS)
        )

        gr.Markdown(
            "---\n"
            "<span style='font-size:0.82rem;color:var(--body-text-color-subdued);'>"
            "Tip: open the relevant tab from the row above, then come back "
            "here to try another preset. The status bar at the top of the "
            "page always shows your environment, GPU, and Hugging Face "
            "login state.</span>"
        )
