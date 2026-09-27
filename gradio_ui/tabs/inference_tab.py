"""Inference tab — chat-style generation via backend/infer.py."""
import json

import gradio as gr


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


def build_inference_tab(stream_logs):
    with gr.Tab("💬 Inference"):
        with gr.Row():
            model = gr.Textbox(label="Model ID / local folder / HF URL", scale=3)
            trust_remote = gr.Checkbox(label="Trust remote code", value=False)
            mode = gr.Dropdown(["instruct", "base"], value="instruct", label="Mode")

        system_prompt = gr.Textbox(label="System prompt (instruct mode)", lines=2)
        chatbot = gr.Chatbot(label="Conversation", height=380, type="messages")

        with gr.Row():
            user_msg = gr.Textbox(label="Your message", scale=4)
            send_btn = gr.Button("Send", variant="primary", scale=1)

        with gr.Accordion("Sampling", open=False):
            with gr.Row():
                max_new_tokens = gr.Slider(16, 4096, value=256, step=16, label="Max new tokens")
                temperature = gr.Slider(0.0, 2.0, value=0.7, step=0.05, label="Temperature")
            with gr.Row():
                top_p = gr.Slider(0.0, 1.0, value=0.95, step=0.01, label="Top-p")
                top_k = gr.Slider(0, 200, value=50, step=1, label="Top-k")
                rep_pen = gr.Slider(0.5, 2.0, value=1.0, step=0.01, label="Repetition penalty")

        output_path = gr.Textbox(value="./output/infer", label="Output folder")
        log_box = gr.Textbox(label="Log", lines=8, max_lines=8)

        def _send(history, text, *args):
            if not text.strip():
                return history, "", ""
            new_history = history + [
                {"role": "user", "content": text},
            ]
            cfg = _build_config(*args)
            cfg["messages"] = new_history
            cfg["prompt"] = text
            logs = []
            reply = ""
            for event in stream_logs(cfg, "infer.py"):
                logs = event.splitlines()
                if event.rstrip().endswith("generation complete") or "generation.json" in event:
                    pass
            # The backend writes the actual text into generation.json.
            try:
                with open(f"{cfg['outputPath']}/generation.json", encoding="utf-8") as fh:
                    reply = json.load(fh).get("response", "")
            except Exception as exc:
                reply = f"(could not read response: {exc})"
            new_history.append({"role": "assistant", "content": reply})
            return new_history, "", "\n".join(logs[-40:])

        inputs = [
            model, trust_remote, mode, system_prompt, chatbot, user_msg,
            max_new_tokens, temperature, top_p, top_k, rep_pen, output_path,
        ]
        send_btn.click(_send, inputs=inputs, outputs=[chatbot, user_msg, log_box])
        user_msg.submit(_send, inputs=inputs, outputs=[chatbot, user_msg, log_box])
