"""BananaAll Gradio WebUI — improved entry point."""
import argparse
import pathlib
import sys

import gradio as gr

from gradio_ui._helpers import (
    CUSTOM_CSS,
    environment_status,
    render_status_chips,
)
from gradio_ui.theme.dark import Dark
from gradio_ui.tabs import (
    build_architecture_tab,
    build_dataset_tab,
    build_train_tab,
    build_inference_tab,
    build_evaluate_tab,
    build_quickstart_tab,
)

LOGO_PATH = pathlib.Path(__file__).resolve().parent / "gradio_ui" / "assets" / "bananaall-logo.webp"

theme = Dark()


def build_app():
    status = environment_status()

    with gr.Blocks(
        title="BananaAll Studio",
        analytics_enabled=False,
    ) as app:
        # ----- Branded header -------------------------------------------
        with gr.Row(elem_id="banana-header"):
            gr.Image(
                value=str(LOGO_PATH) if LOGO_PATH.is_file() else None,
                elem_classes=["brand"],
                show_label=False,
                container=False,
                width=48,
                height=48,
                interactive=False,
            )
            with gr.Column(scale=8):
                gr.Markdown(
                    "<h1 style='margin:0;font-size:1.6rem;font-weight:700;"
                    "letter-spacing:-0.02em;'>BananaAll Studio</h1>"
                    "<div class='subtitle' style='margin:2px 0 0 0;font-size:0.85rem;"
                    "color:var(--body-text-color-subdued);'>"
                    "Train, fine-tune, evaluate, and chat with small language "
                    "models — locally.</div>"
                )
            with gr.Column(scale=2, min_width=180):
                gr.Markdown(
                    f"""<div style="text-align:right;font-size:0.78rem;
                    color:var(--body-text-color-subdued);">
                    v0.2 · gradio {status['gradio']}<br/>
                    {status['hf_message']}
                    </div>"""
                )

        # ----- Status bar -----------------------------------------------
        gr.Markdown(render_status_chips(status), elem_id="banana-status-bar")

        # ----- Tabs -----------------------------------------------------
        with gr.Tabs():
            build_quickstart_tab()
            build_architecture_tab()
            build_dataset_tab()
            build_train_tab()
            build_inference_tab()
            build_evaluate_tab()

        # ----- Footer ---------------------------------------------------
        gr.Markdown(
            "---\n"
            "<span style='font-size:0.75rem;color:var(--body-text-color-subdued)'>"
            "BananaAll · local-first SLM super app · "
            "<a href='https://github.com/asukaa2/BananaAll' target='_blank'>"
            "github.com/asukaa2/BananaAll</a></span>"
        )

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BananaAll Studio", add_help=True)
    parser.add_argument("--share", action="store_true", dest="share_enabled",
                        default=False, help="Enable a public share link.")
    parser.add_argument("--port", type=int, default=7860,
                        help="Port to listen on (default: 7860)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable Gradio debug mode.")
    args = parser.parse_args()

    app = build_app()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share_enabled,
        debug=args.debug,
        show_error=True,
        theme=theme,
        css=CUSTOM_CSS,
    )
