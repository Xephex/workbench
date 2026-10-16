"""Maintainer tool: write catalog.toml in its generated layout, and add or replace one model.

dump() reproduces the committed catalog byte for byte, so an upsert changes
only the model it touches. Users never run this.
"""

from __future__ import annotations

import json, pathlib, tomllib

MODEL_KEYS = ("id", "kinds", "engine", "ram_gb", "vram_gb", "recommended_for", "license", "source")
FILE_KEYS = ("name", "role", "sha256", "bytes", "repo")


def _value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, list):
        return "[" + ", ".join(_value(x) for x in v) + "]"
    if isinstance(v, dict):
        return '{ ' + ', '.join(f'{k} = {_value(x)}' for k, x in v.items()) + ' }'
    raise TypeError(f"cannot write {type(v).__name__} to the catalog")


def _chunks(chunks: list[dict]) -> str:
    return "[" + ", ".join(f"{{ blob = {_value(c['blob'])}, bytes = {_value(c['bytes'])} }}" for c in chunks) + "]"


def dump(data: dict, header: str) -> str:
    out = [header]
    for m in data.get("model", []):
        out.append("\n[[model]]\n")
        out += [f"{k} = {_value(m[k])}\n" for k in MODEL_KEYS]
        if 'settings' in m:
            out.append(f"settings = {_value(m['settings'])}\n")
        for f in m["files"]:
            out.append("\n[[model.files]]\n")
            out += [f"{k} = {_value(f[k])}\n" for k in FILE_KEYS]
            out.append(f"chunks = {_chunks(f['chunks'])}\n")
    return "".join(out)


def upsert(path: pathlib.Path, entry: dict) -> None:
    """Replace the model with entry's id, else append it. Refuses if the file is not in the generated layout."""
    text = path.read_text(encoding="utf-8")
    header = text.splitlines(keepends=True)[0]
    data = tomllib.loads(text)
    if dump(data, header) != text:
        raise ValueError(f"{path} has hand edits that rewriting would lose; "
                         "fix the layout (or move notes into the first line) first")
    models = data.setdefault("model", [])
    for i, m in enumerate(models):
        if m["id"] == entry["id"]:
            models[i] = entry
            break
    else:
        models.append(entry)
    new = dump(data, header)
    if new != text:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(new, encoding="utf-8", newline="\n")
        tmp.replace(path)
