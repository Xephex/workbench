"""Maintainer tool: chunk blob IDs and sizes for a catalog entry, from GitHub's tree API.

    python tools/catalog_chunks.py --repo Xephex/playground_stuff --ref <commit> \
        --prefix llm-kit/models/Qwen3.5-4B-GGUF --name Qwen3.5-4B-Q4_K_M.gguf

Reads metadata only; no file content is downloaded. Users never run this.
"""

from __future__ import annotations

import argparse, json, re, urllib.request


def chunks_from_tree(tree: list[dict], prefix: str, name: str) -> list[dict]:
    pat = re.compile(re.escape(f"{prefix}/{name}.part") + r"(\d{3})$")
    found = {}
    for t in tree:
        m = pat.match(t["path"])
        if m and t["type"] == "blob":
            found[int(m[1])] = {"blob": t["sha"], "bytes": t["size"]}
    if not found or sorted(found) != list(range(len(found))):
        raise ValueError(f"chunks of {prefix}/{name} are missing or have gaps: {sorted(found)}")
    return [found[i] for i in range(len(found))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    url = f"https://api.github.com/repos/{a.repo}/git/trees/{a.ref}?recursive=1"
    with urllib.request.urlopen(url, timeout=60) as r:
        data = json.load(r)
    if data.get("truncated"):
        raise SystemExit("tree listing truncated by GitHub; list a subtree instead")
    chunks = chunks_from_tree(data["tree"], a.prefix, a.name)
    total = sum(c["bytes"] for c in chunks)
    print(f"# {a.name}: {len(chunks)} chunks, {total} bytes")
    print("chunks = [ " + ", ".join(f'{{ blob = "{c["blob"]}", bytes = {c["bytes"]} }}' for c in chunks) + " ]")


if __name__ == "__main__":
    main()
