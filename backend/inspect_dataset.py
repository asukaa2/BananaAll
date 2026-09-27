"""Dataset inspection — preview first rows and detect field mapping.

Used by the Dataset tab. The Gradio UI calls `preview(source)` directly;
the CLI form (`python inspect_dataset.py '<json>'`) is still supported.
"""
import json
import pathlib
import sys
import urllib.parse
import urllib.request

from common import auto_mapping, load_source, normalize_repo


def hub_preview(source):
    """Fetch the first rows from the Hugging Face datasets-server."""
    from huggingface_hub import get_token
    query = urllib.parse.urlencode({
        "dataset": normalize_repo(source.get("id")),
        "config": source.get("config") or "default",
        "split": source.get("split") or "train",
    })
    headers = {"User-Agent": "BananaAll/0.1"}
    token = get_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        f"https://datasets-server.huggingface.co/first-rows?{query}",
        headers=headers,
    )
    with urllib.request.urlopen(request, timeout=18) as response:
        payload = json.load(response)
    samples = [item["row"] for item in payload.get("rows", [])[:3]]
    if not samples:
        raise ValueError("No preview rows returned by Hugging Face")
    return samples


def preview(source: dict) -> dict:
    """Return {columns, samples, mapping} for the given dataset source."""
    samples: list[dict] = []
    if not pathlib.Path(normalize_repo(source.get("id"))).expanduser().exists():
        try:
            samples = hub_preview(source)
        except Exception:
            pass
    if not samples:
        dataset = load_source(source, streaming=True)
        for row in dataset.take(3):
            samples.append(json.loads(json.dumps(row, default=str)))
    if not samples:
        raise ValueError("This split has no rows")
    return {
        "columns": list(samples[0]),
        "samples": samples,
        "mapping": auto_mapping(samples[0]),
    }


def main():
    source = json.loads(sys.argv[1])
    print(json.dumps(preview(source), default=str))


if __name__ == "__main__":
    main()
