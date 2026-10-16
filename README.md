# workbench

An offline LLM workbench for laptops. If you have Python 3.14 and git, it detects
your machine, fetches only the models and packages that machine can run (from
GitHub, nothing else), and then works with no network: chat, documents, speech,
agents, evals, tracing and fine-tuning.

Status: design. The architecture and the Stage 1 design are in
[docs/specs/2026-10-15-workbench-design.md](docs/specs/2026-10-15-workbench-design.md).
Nothing is installable yet.

Model files live in [Xephex/workbench-models](https://github.com/Xephex/workbench-models)
(published with `tools/publish_model.py`) and, for the files llm-kit already uploaded,
[Xephex/playground_stuff](https://github.com/Xephex/playground_stuff). `catalog.toml` lists all
14 Stage 1 models and names the repo of every file.

License: Apache-2.0. Each model keeps its own license, recorded with its files.
