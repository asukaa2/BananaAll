"""Probe the local Hugging Face login state.

Used by the Gradio UI's status bar to show "Signed in as <user>" without
asking the user to paste a token in the app.
"""
import json
import sys


def status() -> dict:
    """Return a dict: {loggedIn, username?, message?}."""
    try:
        from huggingface_hub import HfApi, get_token
        token = get_token()
        if not token:
            return {"loggedIn": False, "message": "No local Hugging Face login"}
        user = HfApi().whoami(token=token)
        return {
            "loggedIn": True,
            "username": user.get("name") or user.get("fullname") or "Signed in",
        }
    except Exception as exc:
        return {"loggedIn": False, "message": str(exc).splitlines()[0][:160]}


def main():
    print(json.dumps(status()))


if __name__ == "__main__":
    main()
