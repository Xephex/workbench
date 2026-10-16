# workbench

An offline LLM workbench for laptops. If you have Python 3.14 and git, it detects
your machine, fetches only the models and packages that machine can run (from
GitHub, nothing else), and then works with no network: chat, documents, speech,
agents, evals, tracing and fine-tuning.

Status: Stage 1 in progress. `setup` detects the machine, installs its
platform's Python packages offline from this repository, and downloads the
models you pick. The server and app come next. Design:
[docs/specs/2026-10-15-workbench-design.md](docs/specs/2026-10-15-workbench-design.md).

```
git clone --filter=blob:none --sparse https://github.com/Xephex/workbench
cd workbench
python workbench.py setup          (Windows: py -3.14 workbench.py setup)
```

Setup options: `--no-trace-viewer` skips the trace viewer (about 170 MB),
`--packages-only` installs packages without choosing models, and `--pack ID`
forces a platform pack (for example `win-x64-vulkan` on an NVIDIA machine).
Packages: Windows x64 (NVIDIA CUDA or Vulkan/CPU) and Linux x64 (NVIDIA CUDA or
Vulkan/CPU) and arm64 (CPU). macOS is not packaged yet.

Model files live in [Xephex/workbench-models](https://github.com/Xephex/workbench-models)
(published with `tools/publish_model.py`) and, for the files llm-kit already uploaded,
[Xephex/playground_stuff](https://github.com/Xephex/playground_stuff). `catalog.toml` lists all
14 Stage 1 models and names the repo of every file.

License: Apache-2.0. Each model keeps its own license, recorded with its files.
