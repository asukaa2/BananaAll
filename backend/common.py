"""Shared dataset and event helpers for BananaAll workers."""
import json
import os
import pathlib
import sys
import threading
from contextlib import contextmanager
from queue import Queue, Empty

# Module-level pointer to the active capturing emitter (if any).
# When None (default), `emit()` prints to stdout, preserving the CLI behavior.
# BananaAll runs at most one worker at a time, so a single slot is enough.
_emitter_obj: "CapturingEmitter | None" = None
_emitter_lock = threading.Lock()


def _current_emitter():
    return _emitter_obj


def emit(kind, **fields):
    """Emit a worker event.

    When a capturing emitter is active (in-process run), push to its queue.
    Otherwise (CLI run), print a JSON line to stdout — that keeps the
    original subprocess contract intact for users who still want to launch
    workers via the command line.
    """
    global _emitter_obj
    emitter = _emitter_obj
    event = {"type": kind, **fields}
    if emitter is not None:
        emitter.add(event)
        return
    print(json.dumps(event, default=str), flush=True)


def fail(exc):
    """Emit an error event and exit the CLI worker.

    In-process runs let the exception propagate to `run_in_process`, which
    captures it and emits a clean error event (no double-report).
    """
    import traceback
    emit("error", message=str(exc), detail=traceback.format_exc())
    if _current_emitter() is None:
        sys.exit(1)
    raise exc


class CapturingEmitter:
    """Thread-safe queue that captures `emit()` calls for in-process runs."""

    _SENTINEL = object()

    def __init__(self):
        self.queue: "Queue[dict]" = Queue()
        self._closed = False

    def add(self, event: dict) -> None:
        if self._closed:
            return
        self.queue.put(event)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.queue.put(self._SENTINEL)


@contextmanager
def capture_events():
    """Activate a capturing emitter for the current thread.

    Any `emit()` call (from any thread) will be pushed to the emitter's
    queue while this context is active. Restores the previous emitter on
    exit so nested/sequential runs are safe.
    """
    global _emitter_obj
    emitter = CapturingEmitter()
    with _emitter_lock:
        previous = _emitter_obj
        _emitter_obj = emitter
    try:
        yield emitter
    finally:
        with _emitter_lock:
            _emitter_obj = previous
        emitter.close()


def run_in_process(module_name: str, config: dict):
    """Run `backend.<module_name>.main(config)` in a worker thread, yielding
    captured events on the calling thread.

    This replaces the subprocess spawn — the UI gets parsed event dicts in
    real time without spawning a separate Python process.
    """
    import importlib
    module = importlib.import_module(module_name)
    if not hasattr(module, "main"):
        yield {"type": "error",
               "message": f"Module {module_name} has no main(config)"}
        return

    captured_exc: list[BaseException] = []

    def worker():
        try:
            module.main(config)
        except BaseException as exc:  # noqa: BLE001 — surface to UI
            captured_exc.append(exc)

    with capture_events() as emitter:
        thread = threading.Thread(target=worker, daemon=True,
                                  name=f"bananaall-{module_name}")
        thread.start()
        # Poll the queue, periodically checking the worker thread's liveness.
        # This lets the consumer (Gradio generator) cancel between events.
        while True:
            try:
                event = emitter.queue.get(timeout=0.3)
                yield event
            except Empty:
                if not thread.is_alive() and emitter.queue.empty():
                    break
        thread.join(timeout=2.0)

    if captured_exc:
        exc = captured_exc[0]
        if not isinstance(exc, (SystemExit, KeyboardInterrupt)):
            yield {"type": "error",
                   "message": f"{exc.__class__.__name__}: {exc}"}


def normalize_repo(value):
    value = str(value or "").strip()
    prefix = "https://huggingface.co/datasets/"
    if value.startswith(prefix):
        return value[len(prefix):].strip("/")
    return value


def load_source(source, streaming=True):
    from datasets import load_dataset
    repo = normalize_repo(source.get("id"))
    if not repo:
        raise ValueError("Dataset source is empty")
    split = source.get("split") or "train"
    config = source.get("config") or None
    path = pathlib.Path(os.path.expanduser(repo))
    if path.exists():
        extension = path.suffix.lower()
        format_name = {".json": "json", ".jsonl": "json", ".csv": "csv", ".parquet": "parquet"}.get(extension)
        if not format_name:
            raise ValueError(f"Unsupported local dataset format: {extension}")
        return load_dataset(format_name, data_files={split: str(path)}, split=split, streaming=streaming)
    kwargs = {"split": split, "streaming": streaming}
    if config:
        kwargs["name"] = config
    return load_dataset(repo, **kwargs)


def auto_mapping(sample):
    keys = set(sample)
    pick = lambda names: next((key for key in names if key in keys), "")
    messages = pick(["messages", "conversations", "conversation", "chat"])
    system = pick(["system", "system_prompt"])
    user = pick(["user", "prompt", "question", "instruction", "input", "problem"])
    assistant = pick(["assistant", "completion", "response", "answer", "output", "generated_solution", "solution"])
    text = pick(["text", "content", "document", "body", "raw_content", "article"])
    if messages:
        return {"messages": messages, "text": "", "system": "", "user": "", "assistant": ""}
    if not text and not (user and assistant):
        text = next((key for key, value in sample.items() if isinstance(value, str) and len(value) > 30), "")
    return {"messages": "", "text": text, "system": system, "user": user, "assistant": assistant}


def stringify(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(stringify(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def extract_example(row, mapping, mode):
    mapping = mapping or auto_mapping(row)
    if mode == "pretraining":
        text_field = mapping.get("text")
        if text_field and row.get(text_field) is not None:
            return {"text": stringify(row[text_field])}
        parts = [stringify(row.get(mapping.get(name))) for name in ("system", "user", "assistant") if mapping.get(name)]
        return {"text": "\n\n".join(part for part in parts if part)}
    messages_field = mapping.get("messages")
    if messages_field and isinstance(row.get(messages_field), list):
        messages = []
        for item in row[messages_field]:
            if isinstance(item, dict):
                role = item.get("role") or item.get("from") or item.get("speaker") or "user"
                role = {"human": "user", "gpt": "assistant", "bot": "assistant"}.get(role, role)
                content = item.get("content") or item.get("value") or item.get("text") or ""
                if role in ("system", "user", "assistant"):
                    messages.append({"role": role, "content": stringify(content)})
        return {"messages": messages}
    messages = []
    for role in ("system", "user", "assistant"):
        field = mapping.get(role)
        if field and row.get(field) is not None:
            content = stringify(row[field])
            if content:
                messages.append({"role": role, "content": content})
    if not messages and mapping.get("text"):
        return {"text": stringify(row.get(mapping["text"]))}
    return {"messages": messages}
