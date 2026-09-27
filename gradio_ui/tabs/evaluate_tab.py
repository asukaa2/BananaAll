"""Evaluation tab — calls backend/evaluate.py in-process.

Improvements over the original:
  - Fixed the event-parsing bug: the original tried to json-parse the
    formatted log string, which never worked. Now we iterate parsed
    event dicts directly and read the `complete.results` payload.
  - Per-task results table (one row per benchmark) instead of a raw
    JSON blob. The raw JSON is still available in the accordion.
  - Shows the benchmark catalog with kind + description.
"""
import json

import gradio as gr

from .._helpers import BENCHMARK_CATALOG, pretty_json
from .._worker import stream_events


TASK_CHOICES = list(BENCHMARK_CATALOG.keys())


def _build_config(model, trust_remote, tasks, limit, batch_size,
                  few_shot, output_path):
    return {
        "model": model.strip(),
        "trustRemoteCode": bool(trust_remote),
        "tasks": list(tasks or []),
        "limit": int(limit or 0),
        "batchSize": int(batch_size),
        "fewShot": int(few_shot),
        "outputPath": output_path,
    }


def _format_event_for_log(event: dict) -> str:
    kind = event.get("type", "log")
    if kind == "status":
        return f"⏳ {event.get('message', '')}"
    if kind == "log":
        return f"   {event.get('message', '')}"
    if kind == "metric":
        parts = [f"{k}={_fmt(v)}" for k, v in event.items()
                 if k not in ("type",)]
        return "📊 " + "  ".join(parts)
    if kind == "result":
        payload = json.dumps(event.get("result"), default=str)
        return f"✅ result[{event.get('task')}]: {payload[:400]}"
    if kind == "complete":
        return f"✅ COMPLETE — {event.get('message', '')}"
    if kind == "error":
        return f"❌ ERROR: {event.get('message', '')}"
    return json.dumps(event, default=str)


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.4f}" if abs(value) < 1 else f"{value:.2f}"
    return str(value)


def _results_to_rows(results: dict) -> list[list]:
    """Flatten the {task: result} dict into table rows."""
    rows = []
    for task, payload in (results or {}).items():
        if not isinstance(payload, dict):
            rows.append([task, str(payload)[:80], "—", "—"])
            continue
        meta = BENCHMARK_CATALOG.get(task, {})
        label = meta.get("label", task)
        if "accuracy" in payload:
            acc = payload["accuracy"]
            if isinstance(acc, dict):
                # Some lm-eval results are nested {acc, stderr}
                score = acc.get("acc,none") or acc.get("acc") or acc.get("accuracy")
            else:
                score = acc
            rows.append([label, f"{float(score):.4f}" if score is not None else "—",
                         payload.get("correct", "—"), payload.get("total", "—")])
        elif "overall_elo" in payload:
            rows.append([label, f"{float(payload['overall_elo']):.1f}",
                         payload.get("correct", "—"), payload.get("total", "—")])
        elif "overall" in payload:
            rows.append([label, f"{float(payload['overall']):.4f}",
                         payload.get("correct", "—"), payload.get("total", "—")])
        else:
            rows.append([label, str(payload)[:80], "—", "—"])
    return rows


def build_evaluate_tab():
    with gr.Tab("🧪 Evaluate"):
        gr.Markdown(
            "### Benchmarks\n"
            "Standard tasks run through `lm-evaluation-harness`; "
            "BananaMind Base/Safety run the dataset's own `benchmark.py`; "
            "ArithMark and Tiny ToM use continuation likelihood scoring."
        )

        with gr.Row():
            model = gr.Textbox(
                label="Model ID / local folder / HF URL",
                placeholder="BananaMind/BananaMind-2-25M",
                scale=3,
            )
            trust_remote = gr.Checkbox(label="Trust remote code", value=False)

        # Build the catalog as a reference table
        catalog_rows = [
            [key, info["label"], info["kind"], info["desc"]]
            for key, info in BENCHMARK_CATALOG.items()
        ]
        gr.Dataframe(
            headers=["Key", "Name", "Kind", "Description"],
            datatype=["str"] * 4,
            value=catalog_rows,
            interactive=False,
            wrap=True,
            row_count=(len(catalog_rows), "fixed"),
            column_count=(4, "fixed"),
        )

        tasks = gr.CheckboxGroup(
            choices=TASK_CHOICES,
            value=["piqa"],
            label="Benchmarks",
            info="Pick one or more.",
        )

        with gr.Row():
            limit = gr.Number(value=0, label="Limit examples (0 = all)",
                              precision=0)
            batch_size = gr.Number(value=1, label="Batch size", precision=0)
            few_shot = gr.Number(value=0, label="Few-shot", precision=0)
        output_path = gr.Textbox(value="./output/eval", label="Output folder")

        run_btn = gr.Button("▶️ Run evaluation", variant="primary")

        status_pill = gr.Markdown("")
        log_box = gr.Textbox(label="Live log", lines=20, max_lines=20,
                             autoscroll=True, interactive=False)

        gr.Markdown("### Per-task results")
        results_table = gr.Dataframe(
            headers=["Benchmark", "Score", "Correct", "Total"],
            datatype=["str", "str", "str", "str"],
            interactive=False,
            wrap=True,
            row_count=(0, "dynamic"),
            column_count=(4, "fixed"),
        )
        results_json = gr.Code(label="Raw results JSON", language="json",
                               interactive=False)

        def _run(*args):
            cfg = _build_config(*args)
            if not cfg["tasks"]:
                yield "❌ Select at least one benchmark.", "", [], ""
                return
            lines: list[str] = []
            final_results: dict = {}
            last_status = ""
            for event in stream_events(cfg, "evaluate.py"):
                line = _format_event_for_log(event)
                if line:
                    lines.append(line)
                if event.get("type") == "status":
                    last_status = event.get("message", "")
                if event.get("type") == "result":
                    task = event.get("task")
                    if task:
                        final_results[task] = event.get("result", {})
                if event.get("type") == "complete":
                    payload_results = event.get("results")
                    if isinstance(payload_results, dict):
                        final_results.update(payload_results)
                yield (f"**Status:** {last_status}" if last_status else "",
                       "\n".join(lines[-200:]),
                       _results_to_rows(final_results),
                       pretty_json(final_results) or "")
            yield (f"**Final:** {last_status}" if last_status else "**Done.**",
                   "\n".join(lines[-200:]),
                   _results_to_rows(final_results),
                   pretty_json(final_results) or "")

        run_btn.click(_run,
                      inputs=[model, trust_remote, tasks, limit, batch_size,
                              few_shot, output_path],
                      outputs=[status_pill, log_box, results_table, results_json])
