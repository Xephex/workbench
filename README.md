# workbench

An offline LLM workbench for laptops. If you have Python 3.14 and git, it detects
your machine, fetches only the models and packages that machine can run (from
GitHub, nothing else), and then works with no network.

Status: beta. Setup, model download, and the local server are ready to try.
Chat, ask, embed, rank, and status work from the terminal. The app, documents,
speech, agents, evals, and fine-tuning are not in this beta. Design:
[docs/specs/2026-10-15-workbench-design.md](docs/specs/2026-10-15-workbench-design.md).
Server details: [docs/server.md](docs/server.md).

```
git clone --filter=blob:none --sparse https://github.com/Xephex/workbench
cd workbench
python workbench.py setup          (Windows: py -3.14 workbench.py setup)
python workbench.py serve          (Windows: py -3.14 workbench.py serve)
```

On Windows, an NVIDIA GPU from Turing onward with driver 570 or newer gets
`win-x64-cuda12`, which includes Blackwell. Setup says when it has to use
Vulkan instead. The server listens on `127.0.0.1:8080` and loads a model on
the first request. Leave it on that port for this beta: `models` only
coordinates with a server there. Stop the server with Ctrl+C before running
setup again.

```
python workbench.py chat --model qwen3.5-4b
python workbench.py ask "Explain a binary search"
python workbench.py status
```

`chat` and `ask` reuse the server when it is already running. On Windows,
prefix those commands with `py -3.14` as well.

`examples/n8n/workbench-chat.json` is a plain n8n chat, with Responses API off
and no tools. To install it with your local n8n API key and this server's
address, set `N8N_API_KEY` and run `py -3.14 examples/n8n/install_demo.py`.
The key is not written into the file.

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
