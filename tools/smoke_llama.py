"""CI smoke test for a built llama-cpp-python wheel: load a GGUF embedder and embed two strings.

    python tools/smoke_llama.py MODEL.gguf 0.4.2+vulkan

Put the folder the wheel was installed into on PYTHONPATH. Fails (exit 1) if the
installed version does not end with LABEL, if the package holds any .exe, or if the
embeddings are not 2 finite, unit-length vectors of 1024 values.
"""

from __future__ import annotations

import math, pathlib, sys

DIM = 1024


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def main(argv: list[str]) -> None:
    if len(argv) != 3:
        fail("usage: smoke_llama.py MODEL.gguf LABEL")
    model, label = pathlib.Path(argv[1]), argv[2]
    if not model.is_file():
        fail(f"model file not found: {model}")

    import llama_cpp
    from llama_cpp.llama_embedding import LlamaEmbedding

    pkg = pathlib.Path(llama_cpp.__file__).parent
    print(f"llama_cpp {llama_cpp.__version__} from {pkg}", flush=True)
    if not llama_cpp.__version__.endswith(label):
        fail(f"llama_cpp.__version__ is {llama_cpp.__version__!r}, expected it to end with {label!r}")

    lib = pkg / "lib"
    print(f"files in {lib}:", flush=True)
    for p in sorted(lib.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(lib).as_posix()}  {p.stat().st_size}", flush=True)
    exes = sorted(p.relative_to(pkg).as_posix() for p in pkg.rglob("*") if p.suffix.lower() == ".exe")
    if exes:
        fail(f".exe files inside the llama_cpp package: {', '.join(exes)}")

    e = LlamaEmbedding(model_path=str(model), n_gpu_layers=0, verbose=False)
    # After the model load: with dynamic backends, nothing is loaded before it.
    print("system info:", llama_cpp.llama_print_system_info().decode(), flush=True)
    from llama_cpp import _ggml
    regs = [_ggml.ggml_backend_reg_name(_ggml.ggml_backend_reg_get(i)).decode()
            for i in range(_ggml.ggml_backend_reg_count())]
    print("backends:", ", ".join(regs), flush=True)
    v = e.embed(["hello world", "bonjour"])
    if len(v) != 2:
        fail(f"expected 2 vectors, got {len(v)}")
    for i, vec in enumerate(v):
        vals = [float(x) for x in vec]
        if len(vals) != DIM:
            fail(f"vector {i} has {len(vals)} values, expected {DIM}")
        if not all(math.isfinite(x) for x in vals):
            fail(f"vector {i} has non-finite values")
        norm = math.sqrt(sum(x * x for x in vals))
        if abs(norm - 1.0) > 1e-3:
            fail(f"vector {i} has norm {norm:.6f}, expected 1.0 within 1e-3")
    print(f"ok: {len(v)} x {DIM}", flush=True)


if __name__ == "__main__":
    main(sys.argv)
