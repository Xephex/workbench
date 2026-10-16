"""Install the workbench chat demo into a local n8n.

Reads the workbench address and key the same way the server does, and the n8n
API key from N8N_API_KEY. Nothing secret is printed.

    set N8N_API_KEY=...          (Windows)
    py -3.14 examples/n8n/install_demo.py

N8N_BASE_URL defaults to http://127.0.0.1:5678. WORKBENCH_BASE_URL defaults to
http://127.0.0.1:8080/v1. A loopback workbench with no key of its own accepts
the placeholder n8n requires.
"""

from __future__ import annotations

import argparse, json, os, pathlib, urllib.error, urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
WORKFLOW_PATH = pathlib.Path(__file__).with_name("workbench-chat.json")
CREDENTIAL_NAME = "Workbench local"
def workbench_key(root: pathlib.Path) -> str | None:
    key = os.environ.get("WORKBENCH_API_KEY")
    if key:
        return key.strip() or None
    try:
        text = (root / ".workbench" / "api-key.txt").read_text().strip()
    except FileNotFoundError:
        return None
    return text or None


def workbench_base() -> str:
    return os.environ.get("WORKBENCH_BASE_URL", "http://127.0.0.1:8080/v1").rstrip("/")


def n8n_base() -> str:
    return os.environ.get("N8N_BASE_URL", "http://127.0.0.1:5678").rstrip("/")


def load_workflow(base: str, model: str) -> dict:
    workflow = json.loads(WORKFLOW_PATH.read_text())
    for node in workflow["nodes"]:
        if node["name"] != "OpenAI Chat Model":
            continue
        node["parameters"]["model"]["value"] = model
        node["parameters"]["model"]["cachedResultName"] = model
        node["parameters"]["options"]["baseURL"] = base
        node["parameters"]["responsesApiEnabled"] = False
    return workflow


def scrub(text: str, secrets: list[str]) -> str:
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[redacted]")
    return text


def request(base: str, key: str, method: str, path: str, body: dict | None, secrets: list[str]) -> dict:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        base + "/api/v1" + path, data=data, method=method,
        headers={"X-N8N-API-KEY": key, "Accept": "application/json", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            raw = response.read().decode()
    except urllib.error.HTTPError as exc:
        detail = scrub(exc.read().decode(errors="replace"), secrets)
        raise SystemExit(f"n8n {method} {path} returned {exc.code}: {detail}") from None
    return json.loads(raw) if raw else {}


def credential_id(listed: dict, name: str) -> str | None:
    rows = listed.get("data", listed if isinstance(listed, list) else [])
    for row in rows:
        if row.get("name") == name:
            return row.get("id")
    return None


def install(root: pathlib.Path, model: str, dry_run: bool) -> None:
    n8n_key = os.environ.get("N8N_API_KEY", "").strip()
    if not n8n_key and not dry_run:
        raise SystemExit("Set N8N_API_KEY to the local n8n API key, then run this again.")
    base = workbench_base()
    wb_key = workbench_key(root) or "local"
    secrets = [n8n_key, wb_key if wb_key != "local" else ""]
    workflow = load_workflow(base, model)
    print(f"Workbench URL: {base}")
    print(f"Workbench key: {'found' if wb_key != 'local' else 'none (n8n will store a placeholder)'}")
    print(f"Model: {model}")
    print(f"n8n: {n8n_base()}")
    if dry_run:
        print("Dry run: no credential or workflow was created.")
        return
    n8n = n8n_base()
    existing = credential_id(request(n8n, n8n_key, "GET", "/credentials", None, secrets), CREDENTIAL_NAME)
    if existing:
        cred_id = existing
        print(f"Using the existing {CREDENTIAL_NAME} credential. Its saved key was not changed.")
    else:
        created = request(n8n, n8n_key, "POST", "/credentials", {
            "name": CREDENTIAL_NAME, "type": "openAiApi",
            "data": {"apiKey": wb_key, "url": base, "header": False}}, secrets)
        cred_id = created["id"]
        print(f"Created the {CREDENTIAL_NAME} credential.")
    for node in workflow["nodes"]:
        if node["name"] == "OpenAI Chat Model":
            node["credentials"] = {"openAiApi": {"id": cred_id, "name": CREDENTIAL_NAME}}
    made = request(n8n, n8n_key, "POST", "/workflows", {
        "name": workflow["name"], "nodes": workflow["nodes"],
        "connections": workflow["connections"], "settings": workflow["settings"]}, secrets)
    print(f"Installed workflow {made.get('id', '')} ({workflow['name']}). Open it in n8n and click Execute.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Install the workbench n8n chat demo.")
    parser.add_argument("--model", default="qwen3.5-4b")
    parser.add_argument("--root", type=pathlib.Path, default=ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    install(args.root, args.model, args.dry_run)


if __name__ == "__main__":
    main()
