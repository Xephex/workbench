"""Plain terminal output. Every user-facing message goes through say()."""

import sys


def say(msg: str = "") -> None:
    print(msg, flush=True)


def ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        answer = input(f"{prompt}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    return answer or default


def gb(n_bytes: int) -> str:
    return f"{n_bytes / 1e9:.1f} GB"
