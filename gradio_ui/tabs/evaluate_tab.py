"""Evaluation tab — wraps backend/evaluate.py."""
import gradio as gr


TASK_CHOICES = [
    "piqa", "lambada", "arc_easy", "arc_challenge", "hellaswag",
    "arithmark", "tiny_tom", "base_bench", "safety_bench",
]


def _build_config(model, trust_remote, tasks, limit, batch_size, few_shot, output_path):
    return {
        "model": model.strip(),
        "trustRemoteCode": bool(trust_remote),
        "tasks": list(tasks or []),
        "limit": int(limit or 0),
        "batchSize": int(batch_size),
        "fewShot": int(few_shot),
        "outputPath": output_path,
    }


def build_evaluate_tab(stream_logs):
    with gr.Tab("🧪 Evaluate"):
        gr.Markdown("Run standard lm-eval-harness tasks and official BananaMind benchmarks.")
        with gr.Row():
            model = gr.Textbox(label="Model ID / local folder / HF URL", scale=3)
            trust_remote = gr.Checkbox(label="Trust remote code", value=False)
        tasks = gr.CheckboxGroup(choices=TASK_CHOICES, value=["piqa"], label="Benchmarks")
        with gr.Row():
            limit = gr.Number(value=0, label="Limit examples (0 = all)", precision=0)
            batch_size = gr.Number(value=1, label="Batch size", precision=0)
            few_shot = gr.Number(value=0, label="Few-shot", precision=0)
        output_path = gr.Textbox(value="./output/eval", label="Output folder")
        run_btn = gr.Button("▶️ Run evaluation", variant="primary")
        log_box = gr.Textbox(label="Live log", lines=26, max_lines=26, autoscroll=True)
        results_box = gr.Code(label="Results JSON", language="json")

        def _run(*args):
            cfg = _build_config(*args)
            if not cfg["tasks"]:
                yield "❌ Select at least one benchmark.", ""
                return
            lines = []
            final_results = ""
            for event in stream_logs(cfg, "evaluate.py"):
                lines = event.splitlines()
                if '"type": "complete"' in event or event.rstrip().endswith("Evaluation complete"):
                    pass
                if '"type": "result"' in event:
                    try:
                        import json
                        obj = json.loads(event.splitlines()[-1])
                        final_results = json.dumps(obj.get("result", {}), indent=2, default=str)
                    except Exception:
                        pass
            # Fall back to reading the written file.
            if not final_results:
                try:
                    import json
                    with open(f"{cfg['outputPath']}/evaluation-results.json", encoding="utf-8") as fh:
                        final_results = json.dumps(json.load(fh), indent=2)[:8000]
                except Exception:
                    final_results = ""
            yield "\n".join(lines[-200:]), final_results

        run_btn.click(_run,
                      inputs=[model, trust_remote, tasks, limit, batch_size, few_shot, output_path],
                      outputs=[log_box, results_box])
