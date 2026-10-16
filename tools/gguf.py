"""Maintainer tool: read a GGUF header (metadata and tensor names), never the weights.

The publisher uses it to refuse reranker files without the classification head
(spec B3, B5): such files load but return meaningless scores.
"""

from __future__ import annotations

import struct
from typing import BinaryIO

RANK_POOLING = 4  # llama.cpp LLAMA_POOLING_TYPE_RANK
SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
STRING, ARRAY = 8, 9
MAX_STRING = 1 << 26  # a sane cap so a corrupt length cannot make us read gigabytes


def _read(f: BinaryIO, n: int) -> bytes:
    b = f.read(n)
    if len(b) != n:
        raise ValueError("GGUF header is truncated")
    return b


def _unpack(f: BinaryIO, fmt: str):
    return struct.unpack(fmt, _read(f, struct.calcsize(fmt)))[0]


def _string(f: BinaryIO) -> str:
    n = _unpack(f, "<Q")
    if n > MAX_STRING:
        raise ValueError(f"GGUF string length {n} is not plausible")
    return _read(f, n).decode("utf-8", errors="replace")


def _value(f: BinaryIO, vtype: int):
    if vtype in SCALARS:
        return _unpack(f, SCALARS[vtype])
    if vtype == STRING:
        return _string(f)
    if vtype == ARRAY:
        etype, count = _unpack(f, "<I"), _unpack(f, "<Q")
        if etype in SCALARS:  # read numeric arrays in one go (token types can be 150k long)
            fmt = SCALARS[etype]
            size = struct.calcsize(fmt)
            return list(struct.unpack(f"<{count}{fmt[1]}", _read(f, size * count)))
        return [_value(f, etype) for _ in range(count)]
    raise ValueError(f"unknown GGUF value type {vtype}")


def read_header(f: BinaryIO) -> tuple[dict, list[str]]:
    """Return (metadata key -> value, tensor names in file order)."""
    if f.read(4) != b"GGUF":
        raise ValueError("not a GGUF file")
    version = _unpack(f, "<I")
    if version not in (2, 3):
        raise ValueError(f"GGUF version {version} is not supported")
    n_tensors, n_kv = _unpack(f, "<Q"), _unpack(f, "<Q")
    kv = {}
    for _ in range(n_kv):
        key = _string(f)
        kv[key] = _value(f, _unpack(f, "<I"))
    tensors = []
    for _ in range(n_tensors):
        tensors.append(_string(f))
        n_dims = _unpack(f, "<I")
        _read(f, 8 * n_dims + 4 + 8)  # dims, type, offset
    return kv, tensors


def rerank_problem(kv: dict, tensors: list[str]) -> str | None:
    """Why this file cannot rerank, or None when it has rank pooling and a classification head."""
    arch = kv.get("general.architecture", "")
    pooling = kv.get(f"{arch}.pooling_type")
    problems = []
    if pooling != RANK_POOLING:
        problems.append(f"{arch}.pooling_type is {pooling}, rank pooling needs {RANK_POOLING}")
    if "cls.output.weight" not in tensors:
        problems.append("no cls.output.weight tensor (classification head)")
    return "; ".join(problems) or None
