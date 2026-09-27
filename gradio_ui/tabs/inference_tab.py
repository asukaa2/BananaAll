"""Inference tab — chat-style generation via backend/infer.py.

Improvements:
  - No more file-reading hack: the worker emits a `complete` event with
    `text` inline, so we just read the event stream.
  - System-prompt presets and model presets for one-click setup.
  - Generates-side log box shows the raw event trail (token throughput,
    generation metadata) without polluting the chat window.
  - Send button is disabled while a generation is in flight (handled by
    Gradio generator + chat state).
"""
import gradio as gr

from .._helpers import MODEL_PRESETS, SYSTEM_PROMPT_PRESETS
from .._worker import stream_events


def _build_config(model, trust_remote, mode, system_prompt, messages, prompt,
                  max_new_tokens, temperature, top_p, top_k,
                  repetition_penalty, output_path):
    return {
        "model": model.strip(),
        "trustRemoteCode": bool(trust_remote),
        "inferenceMode": mode,
        "systemPrompt": system_prompt or "",
        "messages": messages or [],
        "prompt": prompt or "",
        "maxNewTokens": int(max_new_tokens),
        "temperature": float(temperature),
        "topP": float(top_p),
        "topK": int(top_k),
        "repetitionPenalty": float(repetition_penalty),
        "outputPath": output_path,
    }


def _format_event_for_log(event: dict) -> str:
    kind = event.get("type", "log")
    if kind == "status":
        return f"⏳ {event.get('message', '')}"
    if kind == "log":
        return f"   {event.get('message', '')}"
    if kind == "complete":
        msg = event.get("message", "")
        path = event.get("outputPath", "")
        return f"✅ {msg}  → {path}" if path else f"✅ {msg}"
    if kind == "error":
        return f"❌ ERROR: {event.get('message', '')}"
    return str(event)[:200]


def build_inference_tab():
    with gr.Tab("💬 Inference"):
        with gr.Row():
            model = gr.Textbox(
                label="Model ID / local folder / HF URL",
                placeholder="BananaMind/BananaMind-2-25M",
                scale=3,
            )
            trust_remote = gr.Checkbox(label="Trust remote code", value=False)
            mode = gr.Dropdown(["instruct", "base"], value="instruct",
                                label="Mode")

        # Quick model presets
        with gr.Row(elem_classes=["preset-row"]):
            model_preset_btns = [
                gr.Button(label, size="sm", variant="secondary")
                for label in MODEL_PRESETS
            ]

        # System prompt with presets
        system_prompt = gr.Textbox(
            label="System prompt (instruct mode)",
            lines=2,
            placeholder="You are BananaMind, a helpful assistant.",
        )
        with gr.Row(elem_classes=["preset-row"]):
            system_preset_btns = [
                gr.Button(label, size="sm", variant="secondary")
                for label, _ in SYSTEM_PROMPT_PRESETS
            ]

        chatbot = gr.Chatbot(label="Conversation", height=380)

        with gr.Row():
            user_msg = gr.Textbox(
                label="Your message", scale=4,
                placeholder="Type your message and press Enter or Send.",
            )
            send_btn = gr.Button("Send", variant="primary", scale=1)
            clear_btn = gr.Button("Clear", variant="secondary", scale=1)

        with gr.Accordion("Sampling", open=False):
            with gr.Row():
                max_new_tokens = gr.Slider(16, 4096, value=256, step=16,
                                            label="Max new tokens")
                temperature = gr.Slider(0.0, 2.0, value=0.7, step=0.05,
                                         label="Temperature")
            with gr.Row():
                top_p = gr.Slider(0.0, 1.0, value=0.95, step=0.01,
                                   label="Top-p")
                top_k = gr.Slider(0, 200, value=50, step=1, label="Top-k")
                rep_pen = gr.Slider(0.5, 2.0, value=1.0, step=0.01,
                                     label="Repetition penalty")

        output_path = gr.Textbox(value="./output/infer", label="Output folder")
        log_box = gr.Textbox(label="Log", lines=8, max_lines=8,
                              interactive=False)

        def _send(history, text, *args):
            if not text.strip():
                yield history, "", ""
                return
            new_history = list(history) + [{"role": "user", "content": text}]
            cfg = _build_config(*args)
            cfg["messages"] = new_history
            cfg["prompt"] = text
            yield new_history, "", ""  # show user turn immediately

            logs: list[str] = []
            reply = ""
            for event in stream_events(cfg, "infer.py"):
                line = _format_event_for_log(event)
                if line:
                    logs.append(line)
                if event.get("type") == "complete":
                    reply = event.get("text", "") or ""
                yield new_history, "", "\n".join(logs[-200:])

            if not reply:
                # Fall back to the file if the worker wrote one without
                # emitting a complete event (defensive).
                import json, pathlib
                gen_path = pathlib.Path(cfg["outputPath"]) / "generation.json"
                if gen_path.is_file():
                    try:
                        reply = json.loads(
                            gen_path.read_text(encoding="utf-8")
                        ).get("response", "")
                    except Exception:
                        pass
            if not reply:
                reply = "(no response from worker)"
            new_history.append({"role": "assistant", "content": reply})
            yield new_history, "", "\n".join(logs[-200:])

        inputs = [
            model, trust_remote, mode, system_prompt, chatbot, user_msg,
            max_new_tokens, temperature, top_p, top_k, rep_pen, output_path,
        ]
        send_btn.click(_send, inputs=inputs, outputs=[chatbot, user_msg, log_box])
        user_msg.submit(_send, inputs=inputs, outputs=[chatbot, user_msg, log_box])
        clear_btn.click(lambda: ([], "", ""), outputs=[chatbot, user_msg, log_box])

        # Wire model presets
        for btn, model_id in zip(model_preset_btns, MODEL_PRESETS):
            btn.click(lambda _m=model_id: _m, outputs=[model])

        # Wire system-prompt presets
        for btn, (_, prompt_text) in zip(system_preset_btns, SYSTEM_PROMPT_PRESETS):
            btn.click(lambda _p=prompt_text: _p, outputs=[system_prompt])
