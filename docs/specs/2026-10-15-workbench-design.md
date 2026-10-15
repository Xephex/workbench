# Workbench: design

Status: draft for review, 2026-10-15
Repos: `Xephex/workbench` (code), `Xephex/workbench-models` (model files)
License: Apache-2.0 (code); each model keeps its own license, recorded in the catalog

This document has two parts. Part A is the overall architecture that every stage
builds on. Part B is the detailed design for Stage 1. Stages 2-7 get their own
specs when they start.

---

## Part A: architecture

### A1. What it is

An offline LLM workbench that anyone with **Python 3.14 and git** can run on a
laptop. It detects the machine, offers only what that machine can run, fetches
only those pieces from GitHub, and from then on works with no network.

It covers: chat (text, images, audio), document Q&A, OCR, Jev-style typed
decisions, embeddings and reranking, extraction, guard and translation models,
speech (meeting transcription with speaker labels, live voice conversation,
read-aloud, clean-up), image generation, agents (Pi plus seven agent
frameworks), batch jobs, model comparison, evals, tracing, a prompt library,
speed tuning, fine-tuning (text and speech), model conversion, and a resource
monitor.

### A2. Requirements

What the operator asked for:

- **R1. Python and git only.** The user has Python 3.14 and git. Nothing else
  is installed by hand: no CUDA toolkit, no Docker, no Node.js (Pi is the one
  exception: it needs Node.js 20.6+ and is hidden without it).
- **R2. Offline after git.** The only network activity is git talking to
  GitHub during setup, add-model and update. No pip index, no Hugging Face, no
  CDN. Everything else runs with the network unplugged.
- **R3. GitHub only.** All code, Python packages and model weights are hosted on
  GitHub. Files over 95 MB are split into chunks and joined on the machine
  (the llm-kit approach). Nothing is fetched from Hugging Face or any other host.
- **R4. Any machine.** Windows, Linux and macOS; NVIDIA, AMD, Intel, Apple or no
  GPU. Hardware is detected, and only runnable models and features are shown.
- **R5. Lean.** A machine downloads only the packages for its own platform and
  only the models the user picked, never the full catalog.
- **R6. No executables on locked-down Windows.** Everything runs as Python plus
  libraries loaded by Python. Setup removes the `.exe` launchers pip creates
  for workbench packages. The workbench never launches an `.exe` itself
  (detection uses ctypes and the registry, not `nvidia-smi.exe`).
- **R7. Laptop class.** Sized for CPU-only laptops up to 16 GB laptop GPUs and
  32-48 GB Apple Silicon Macs.
- **R8. Every kind of model** someone would want on such a laptop (catalog in
  A6), with open weights whose license allows re-hosting.
- **R9. Hand-over-and-run.** Copy-paste setup, no "go download X". Clear
  messages when something is missing.
- **R10. Apache-2.0**, public on GitHub.

Assumptions (correct these if wrong):

- One user per install; the local server binds to 127.0.0.1 unless sharing is
  turned on with a key (as llm-kit option 7 does today).
- English UI only.
- llm-kit in `Xephex/playground_stuff` stays as-is for the operator's laptop
  until the workbench covers everything it does; then its README points here.

### A3. Repositories

```
Xephex/workbench                 code + Python packages (small clone)
  workbench.py                   entry point: launcher menu and every command
  workbench/                     the package
    hw.py                        hardware detection (stdlib + ctypes only)
    catalog.py                   loads catalog.toml, filters by hardware
    fetch.py                     git fetch of chunks, join, verify
    install.py                   offline pip install of a platform pack
    engines/                     llama.cpp, onnxruntime, speech, torch adapters
    server/                      FastAPI app, model manager, endpoints, jobs
    tools/                       one module per command (chat, docs, ...)
    app/                         Gradio UI, one module per tab
    trace.py                     OpenTelemetry setup
  catalog.toml                   every model and pack: needs, size, chunks
  packs/<platform>/              wheel files for one platform, chunked if >95 MB
  examples/                      agent examples, eval sets, prompts, n8n flows
  tests/                         unit tests (no models needed)
  tools/                         maintainer tools: build packs, publish models
  .github/workflows/             CI on Windows, Linux, macOS

Xephex/workbench-models          model weights only
  <model-id>/<file>.partNNN      95 MB chunks
  <model-id>/<file>.sha256       sha256 of the joined file
  <model-id>/LICENSE             the model's license text
```

The code repo is cloned with `--filter=blob:none --sparse`, so a clone
downloads only the code, not other platforms' packs. The models repo is never
cloned or checked out (see A5).

If either repo grows past GitHub's comfortable size, it can be split (for
example `workbench-models-2`), because the catalog names the repo for every
chunk. Users would not notice.

### A4. How the pieces fit

```
                 +--------------------------------------------+
  browser  --->  | Gradio app (tabs)                           |
                 +----------------------+---------------------+
  scripts, n8n,                         | HTTP (same API)
  agents, Pi  -------------------------->+
                 +----------------------v---------------------+
                 | Local server (FastAPI, 127.0.0.1:8080)       |
                 |  OpenAI-compatible + /v1/systemone + extras |
                 |  model manager (load on demand, evict LRU)  |
                 |  job queue (long tasks)                     |
                 |  tracing (OpenTelemetry -> local store)     |
                 +----------------------+---------------------+
                                        |
           +--------------+-------------+-------------+--------------+
           | llama.cpp    | onnxruntime | speech      | torch        |
           | (chat, VLM,  | (OCR, VAD,  | (whisper,   | (fine-tune,  |
           |  embed, rank,|  diarize,   |  TTS)       |  train only) |
           |  decide L)   |  decide S)  |             |              |
           +--------------+-------------+-------------+--------------+

  workbench.py setup / models / update:  hw.py -> catalog.py -> fetch.py + install.py
```

- **Everything goes through the server.** The app, the commands, the agents,
  n8n and outside scripts all call the same HTTP API. That gives one place for
  model loading, memory limits and tracing. Commands start the server in the
  background when it is not running (as `kit.py pi` does today) and stop it
  afterwards unless it was already running.
- **Commands are thin.** `workbench.py transcribe meeting.mp3` is a small client
  of `/v1/audio/transcriptions` plus formatting. So every feature works the same
  from the app, a command, or n8n, and each can be tested over HTTP.
- **Engines are adapters.** Each engine is a module exposing `load(entry,
  settings) -> handle`, `unload(handle)` and the calls it supports. The model
  manager does not know about llama.cpp or onnxruntime directly.

### A5. Setup, fetching and updates

**First install (user's view):**

```
git clone --filter=blob:none --sparse https://github.com/Xephex/workbench
cd workbench
python workbench.py setup
```

On Windows use `py -3.14 workbench.py setup`. If the default `python` is
not 3.14, the launcher finds 3.14 and re-runs itself with it (as `kit.py` does).

**What setup does:**

1. **Detect** the machine (`hw.py`, A7) and print the profile.
2. **Pick the platform pack** (A8), for example `win-x64-cuda12`.
3. **Fetch the pack** with `git sparse-checkout add packs/<platform>/<layer>`
   for each layer this machine needs (B4). Git downloads only those folders.
4. **Install offline:** join chunked wheels, check sha256, then `pip install
   --no-index --find-links <layer folder> -r <layer>/requirements.lock` per layer.
   Then remove the `.exe` launchers pip created for those packages (llm-kit's
   RECORD-based removal).
5. **Choose models.** Show the recommended set for this machine with sizes and
   the disk space available; the user accepts it or adds and removes models.
6. **Fetch models** (below).
7. **Self-check:** start the server, send one request per installed model kind,
   and print a summary.

**Model fetch (verified 2026-10-15 against github.com):**

The catalog stores, for each model file, the models repo name, the git blob ID
of every chunk, and the sha256 of the joined file. To fetch a model:

1. Create a throwaway bare repo under `.workbench/fetch/` with
   `remote.origin.promisor=true` and `partialclonefilter=blob:none`.
2. For each chunk, run `git fetch --no-tags --filter=blob:none origin <blob-id>`,
   which downloads exactly that one object, then stream `git cat-file blob
   <blob-id>` onto the end of the output file.
3. Once a chunk has been appended, drop it (repack or delete the throwaway
   repo), so peak extra disk use is about one chunk (95 MB), not the whole model.
4. Check the sha256 of the finished file against the catalog. On mismatch, keep
   nothing and report the model as failed.
5. Re-running resumes: chunks already appended are skipped. Progress goes in
   `<file>.fetching` (bytes written, chunks done).

Tested: fetching one 41 MB chunk by blob ID from `Xephex/playground_stuff` took
1.4 s, used 39 MB on disk, and returned bytes matching the committed chunk's
sha256. Pitfall found: `git ls-tree -l` on a blobless repo downloads every blob
to report sizes, so the fetcher never asks git for sizes; they come from the
catalog.

This needs no sparse-checkout bookkeeping in the models repo, never stores a
model twice, and ties each download to exact, checkable bytes. `git` is the
only tool that talks to the network.

**Adding or removing a model later:** `workbench.py models` (or the Models
tab) runs the same fetch. Removing deletes the joined file.

**Update:** `workbench.py update` runs `git pull --no-rebase` on the code repo,
reinstalls the pack if its lock changed, and re-fetches any installed model
whose catalog entry changed (new sha256). It never deletes user data (chats,
traces, datasets, fine-tunes in `~/.workbench` or the configured data folder).

**Where data lives:** models in `<workbench>/models/` (or a folder the user
picks, for example a second drive); user data in `<workbench>/data/`. Neither
is tracked by git (`.gitignore`), so `git pull` never touches them.

### A6. Model catalog

Every model is one entry in `catalog.toml`:

```toml
[[model]]
id = "qwen3.5-9b"
kind = ["chat", "vision", "tools"]   # what it can do
engine = "llama.cpp"
files = [
  { name = "Qwen3.5-9B-Q4_K_M.gguf", sha256 = "...", bytes = 5680522464,
    repo = "Xephex/workbench-models", chunks = ["<blob-id>", "..."] },
  { name = "mmproj-F16.gguf", role = "mmproj", sha256 = "...", bytes = 918166080,
    repo = "Xephex/workbench-models", chunks = ["..."] },
]
needs = { ram_gb = 12, vram_gb = 7, backends = ["cuda", "vulkan", "metal", "rocm", "cpu"] }
recommended_for = ["gpu-8"]           # tiers where it is in the default set
settings = { n_ctx = 12288 }          # engine defaults (tuning can override)
license = "Apache-2.0"
source = "unsloth/Qwen3.5-9B-GGUF"    # where the maintainer got it (not used by users)
```

Categories (the Stage-1 subset is in B3; the rest come in later stages):

| Category | Purpose | Stage |
|---|---|---|
| General chat | everyday assistant | 1 |
| Reasoning | harder problems, planning | 1 |
| Coding | code, the Pi agent | 1 |
| Tool calling | agents, n8n | 1 |
| Vision | images, screenshots | 1 |
| Embeddings | search, document Q&A | 1 |
| Rerankers | better search ranking | 1 |
| Speed-up heads (MTP) | faster generation (speculative decoding) | after Stage 1 selftest |
| Speech-to-text | transcription, accents | 2 |
| Diarization | who spoke when | 2 |
| Voice activity | live voice turn-taking | 2 |
| Text-to-speech | read-aloud, voice replies | 2 |
| Jev-style decisions | typed Choice / Score / yes-no | 3 |
| OCR | scans, tables, handwriting | 3 |
| Extraction | names, dates, fields | 3 |
| Guard | screen prompts and output | 3 |
| Translation | offline translation | 3 |
| Image generation | text to image | 7 |

A model is only added once someone has: confirmed that its license allows
redistribution (license text copied next to the chunks), run it with the
workbench's engine on at least CPU, and recorded its measured RAM/VRAM need.

### A7. Hardware detection

`hw.py` uses only the standard library and ctypes, so it runs before any
package is installed and never launches an executable.

Profile fields: `os`, `arch`, `cpu_name`, `cpu_cores`, `ram_gb`, `disk_free_gb`
(at the models folder), `gpus: [{vendor, name, vram_gb, driver}]`, `backend`
(cuda / rocm / vulkan / metal / cpu), `tier`.

Tiers (R7):

| Tier | Rule | Typical machine |
|---|---|---|
| `cpu` | no usable GPU backend | any laptop, 8 GB+ RAM |
| `gpu-8` | GPU with 6-11 GB VRAM, or Apple with 16-24 GB memory | RTX PRO 2000 / 4060 laptop, 16 GB Mac |
| `gpu-16` | GPU with 12 GB+ VRAM, or Apple with 32 GB+ memory | RTX 4080/5080 laptop, 32-48 GB Mac |

Apple counts about 70% of unified memory as usable by the GPU. System RAM also
gates the MoE models (they keep experts in RAM).

Detection methods per platform: see B2.

Models and features above the machine's tier are **shown greyed out with the
reason** ("needs 12 GB of video memory; this machine has 8"). Features the
machine cannot do at all (fine-tuning without NVIDIA or Apple) are listed once
at the bottom of the app as "not available on this machine" with the reason.

### A8. Platform packs

A pack is the full set of wheels for one platform, with a lock file and
checksums, built and verified by CI (llm-kit's `build_bundle.py`, generalized).
Pack names: `<os>-<arch>-<backend>`.

The pack set, and which ones need wheels built by us, is in B4.

Packs are layered so a machine installs only what it uses:

- `core`: server, app, engines' Python glue, tracing (pure or per-platform)
- `llama-<backend>`: the llama.cpp wheel for that backend (+ CUDA runtime wheels)
- `speech`, `onnx`: Stage 2 and 3 additions
- `torch-<backend>`: only on machines that can fine-tune (Stage 6)
- `agents-<framework>`: one per agent framework (Stage 5)

Every pack is verified in CI by installing it with `--no-index` on its target
platform (or with `uv --python-platform` for platforms with no runner), the
check that caught the missing colorama in llm-kit.

### A9. Local server

FastAPI on `127.0.0.1:8080`. Sharing to the network (`--share`) binds to all
addresses and requires an API key (generated and stored as in llm-kit option 7).

- **Model manager.** Loads models on first use. Keeps a memory budget from the
  hardware profile (VRAM and RAM separately). When a new model does not fit, it
  unloads the least recently used one. Small models (embedding, reranker,
  decisions, OCR, VAD) usually stay resident next to one chat model. A request
  waits while its model loads; a second request for a busy chat model queues.
- **Endpoints:** OpenAI-compatible: `/v1/models`, `/v1/chat/completions`
  (streaming, tools, images, audio), `/v1/embeddings`; Stage 2: `/v1/audio/transcriptions`,
  `/v1/audio/speech`; Stage 7: `/v1/images/generations`. Jev-compatible:
  `/v1/systemone` (Stage 3). Extras: `/v1/rerank` (Stage 1), `/v1/ocr`,
  `/v1/extract`, `/v1/diarize` (Stages 2-3). Workbench: `/wb/status`,
  `/wb/jobs`, `/wb/models` (install state).
- **Jobs.** Long tasks (meeting transcription, batch, fine-tuning, conversion,
  evals, model fetch) run in a background worker with progress, cancel, and
  results kept on disk. Closing the browser does not stop them.
- **Tool-call parsing.** llm-kit's `toolcalls.py` (Qwen/Granite XML, Hermes,
  Harmony, Gemma) moves over unchanged with its tests.

### A10. App

Gradio, mounted into the same FastAPI process, offline (no CDN fonts or
scripts; see B6). Tabs: Chat, Documents, Speech, Decide, Images, Agents, Batch,
Tune, Evaluate, Traces, Monitor, Models. Each tab appears in the stage that
delivers it. A tab whose features cannot run on this machine is hidden and
listed at the bottom with the reason.

The terminal launcher (`python workbench.py` with no arguments) offers: open
the app, run a command, setup, models, update, status.

### A11. Tracing and evals

- Every server request creates an OpenTelemetry span: model, inputs and outputs
  (switchable off per tab for privacy), timing, tokens, memory, errors. Agent
  steps and tool calls nest under the request that started them.
- Spans go to a local store with a browser viewer that works offline
  (tool chosen in B6).
- Optional second destination: a Langfuse server over OTLP/HTTP, set in
  settings with its address and keys. Off by default.
- Evals (Stage 4) use the same store: datasets, runs, scores, comparisons.

### A12. Commands

Every feature is a command: `python workbench.py <command> [args]`.

Stage 1: `setup`, `models`, `update`, `status`, `app`, `serve`, `chat`, `ask`,
`image`, `docs`, `embed`, `rank`, `tune`, `pi`, `monitor`, `selftest`.
Later: `transcribe`, `speak`, `voice`, `decide`, `ocr`, `extract`, `batch`,
`compare`, `eval`, `traces`, `dataset`, `finetune`, `convert`, `agent`, `imagine`.

Every command takes `--model ID` (otherwise it uses the recommended model for
the machine) and `--json` for machine-readable output.

### A13. Testing (all stages)

- **Unit tests, no models:** hardware detection against recorded profiles for
  each machine kind (fake registry, sysfs and sysctl data), catalog filtering
  per tier, chunk fetch/join/resume against a local test repo, tool-call
  parsing, API schemas.
- **CI on GitHub Actions: Windows, Linux, macOS (arm64).** Each run does the
  real user flow: sparse clone of the pushed commit, `setup` with the CPU pack
  and the smallest models, one request to every endpoint, the Windows `.exe`
  check, and the network blocked after the git steps (so a missing wheel fails
  CI instead of being downloaded).
- **Pack checks:** every pack is installed offline for its target platform
  before commit.
- **GPU builds:** GitHub runners have no GPUs. `workbench.py selftest` loads
  every installed model, measures speed and memory, and writes a report. The
  operator's laptop (Windows, RTX PRO 2000 Blackwell, 8 GB) is the reference
  NVIDIA machine. Catalog entries record which platforms have passed a selftest;
  the others are marked "untested on this hardware" in the app.

### A14. Build stages

| Stage | Delivers |
|---|---|
| 1. Foundation | repos, detection, catalog, fetch/join/install, packs, server with model manager (chat, vision, embeddings, rerank), llm-kit's features (chat, documents, Pi, speed tuning, n8n sharing), basic tracing, app shell (Chat, Documents, Models, Monitor), CI on three OSes |
| 2. Speech | STT models incl. accented-English checks, diarization, VAD, TTS, meeting pipeline with clean-up and notes, live voice, vocabulary |
| 3. Decide and documents | Jev-style models + `/v1/systemone`, OCR (two tiers), extraction, guard, translation, batch |
| 4. Evaluate and trace | trace viewer, eval sets and runs, model comparison, prompt library, Langfuse export |
| 5. Agent frameworks | LangGraph, Google ADK, OpenAI Agents SDK, Pydantic AI, smolagents, CrewAI, LlamaIndex packs, each with an example |
| 6. Fine-tuning | dataset builder, text LoRA, Whisper fine-tuning, conversion to GGUF, before/after evals |
| 7. Image generation | stable-diffusion.cpp models (GPU tiers) |


### A15. Research findings that shape later stages

From `~/Work/notes/workbench/later-stages-research/models.md` (2026-10-15).
Recorded here so the architecture does not block them; each stage's own spec
confirms them.

- **Decisions (Stage 3):** llama.cpp gained a native `/v1/systemone` endpoint
  and decision-head GGUFs (PR #29818, 2026-10-02), with Laya (449 MB), Julia-1
  (168 MB), Kev-4B (3.0 GB) and lev as ready GGUFs, all Apache-2.0. Whether the
  pinned llama-cpp-python fork exposes the decision head through Python must be
  checked; if not, the decision models run through ONNX Runtime (Laya, Von and
  Verdict have ONNX exports), which keeps everything in Python.
- **Accented English (Stage 2):** published numbers show Whisper large-v3 at 19%
  word errors on Indian-accented clips against 7.8% for Qwen3-ASR-1.7B. Stage 2
  measures Parakeet-TDT-0.6B and Qwen3-ASR-0.6B (via sherpa-onnx, Python 3.14
  wheels) against Whisper on Svarah (Indian English, CC BY 4.0, can be
  bundled). No redistributable Middle-Eastern-accent set exists (L2-ARCTIC is
  non-commercial), so Stage 2 includes a way to build a small private eval set
  from the user's own corrected recordings.
- **Speaker labels and voice activity (Stage 2):** sherpa-onnx with
  pyannote-segmentation-3.0 and WeSpeaker ResNet34 plus Silero VAD: about 34 MB,
  no Hugging Face login, no PyTorch.
- **Python 3.14 gaps found:** `kokoro-onnx` and `misaki` (text-to-speech helpers)
  and PaddleOCR 3.x have no 3.14 support; `stable-diffusion-cpp-python` has no
  wheels at all. Plans: Kokoro through sherpa-onnx instead; OCR through RapidOCR
  (ONNX) and llama.cpp vision models; image generation needs our own
  stable-diffusion.cpp Python build (same approach as B4) because the
  ready-made `sd-cli` program is an executable.
- **Licenses that rule models out:** NLLB (translation), DiariZen, FLUX.1-dev,
  FLUX.2-klein-9B, SDXL-Turbo, Llama Guard, ShieldGemma, Nanonets-OCR, some
  Piper voices, and the large OpenJev GGUF. Picks per category are in the
  research notes.

---

## Part B: Stage 1 detailed design

### B1. Scope

Stage 1 ships a workbench that does everything llm-kit does today, on any
machine, with the new install model:

- `setup`, `models`, `update`, `status`, `selftest`
- server with model manager: chat (streaming, tools, images), embeddings,
  rerank; tracing to a local store
- commands: `chat`, `ask`, `image`, `docs`, `embed`, `rank`, `tune`, `pi`,
  `serve` (incl. `--share` for n8n), `monitor`, `app`
- app tabs: Chat, Documents, Models, Monitor
- packs for the platforms in B4; CI on Windows, Linux, macOS
- the Stage 1 models in B3 published to `workbench-models`
- llm-kit's n8n workflows, Pi integration and tool-call fixes carried over

Out of Stage 1: everything in stages 2-7 (A14).

### B2. Hardware detection (`hw.py`)

Source: `~/Work/notes/workbench/stage1-research/engines.md` section 4. Standard
library and ctypes only; no `subprocess`, WMI or PowerShell on Windows. Every
probe is optional and returns "unknown" instead of failing. All foreign
functions get explicit `argtypes`/`restype`; Win32 structs use fixed-width
integers.

| Item | Windows | Linux (incl. WSL) | macOS |
|---|---|---|---|
| NVIDIA GPU, VRAM, driver, CUDA version, compute capability | NVML: `nvml.dll` (System32, else NVSMI folder): `nvmlInit_v2`, `nvmlDeviceGetCount_v2`, `...GetHandleByIndex_v2`, `...GetName`, `...GetMemoryInfo`, `nvmlSystemGetDriverVersion`, `nvmlSystemGetCudaDriverVersion`, `...GetCudaComputeCapability`. Fallback: `nvcuda.dll` `cuInit`, `cuDeviceTotalMem_v2`, attributes 75/76 | same with `libnvidia-ml.so.1` / `libcuda.so.1` (WSL: `/usr/lib/wsl/lib`) | n/a |
| AMD / Intel GPU and VRAM | DXGI `CreateDXGIFactory1` → `EnumAdapters1` → `GetDesc1` (vendor id, name, dedicated VRAM; skip the software adapter). Cross-check: registry display class `{4d36e968-...}\NNNN` `DriverDesc`, `MatchingDeviceId`, `HardwareInformation.qwMemorySize` | `/sys/class/drm/cardN/device/{vendor,device,driver}`; AMD `mem_info_vram_total`; Intel Arc VRAM unknown (no sysfs) | n/a |
| Apple GPU | n/a | n/a | `sysctlbyname`: `hw.memsize`, `machdep.cpu.brand_string`, `hw.optional.arm64`, `sysctl.proc_translated` (refuse x86_64 Python under Rosetta) |
| Vulkan loader present | load `vulkan-1.dll` | load `libvulkan.so.1` | n/a |
| RAM | `GlobalMemoryStatusEx` | `/proc/meminfo` (`MemTotal`, `MemAvailable`) | `hw.memsize` |
| Free disk | `shutil.disk_usage(models folder)` | same | same |

GPU memory budget used for tiers: NVIDIA/AMD dedicated VRAM; integrated GPUs
count as no GPU for tiering (their models run on CPU or Vulkan with shared
memory, offered as "may be slow"); Apple: 0.66 × unified memory before load.
After a model loads, the real backend and memory are read back through ggml
(`ggml_backend_dev_count/_name/_memory`), because a backend that fails to load
falls back to CPU **silently**. `status` and `selftest` report the device that
actually ran, and say so plainly when a GPU was expected but not used (llm-kit's
`check_gpu.py` diagnostics carry over, including the CUDA driver query).

Pack choice (B4) from the profile:

| Profile | Pack |
|---|---|
| Windows, NVIDIA Turing or newer, driver ≥ 570 | `win-x64-cuda12` |
| Windows, other GPU, older NVIDIA, or none | `win-x64-vulkan` (CPU + Vulkan) |
| Linux/WSL, NVIDIA Turing or newer, driver ≥ 580 | `linux-x64-cuda13` |
| Linux/WSL, anything else | `linux-x64-vulkan` (CPU + Vulkan) |
| Linux arm64 | `linux-arm64-cpu` |
| macOS Apple Silicon | `macos-arm64-metal` |
| Intel Mac, Windows on ARM | not supported in Stage 1 (no Python 3.14 wheels for several dependencies); setup says so and stops |

An NVIDIA driver below the minimum gets the Vulkan pack plus a message naming
the driver version needed for CUDA.

**WSL.** WSL is detected (`/proc/sys/kernel/osrelease` contains `microsoft`,
or `WSL_DISTRO_NAME` is set) and treated as Linux, with these differences:

- The NVIDIA GPU does not appear in `/sys/class/drm` (WSL routes GPU access
  through `/dev/dxg`); detection uses NVML from `/usr/lib/wsl/lib`, which works.
- WSL gives Linux 50% of the host's RAM by default. Tiers use the RAM Linux
  sees. When a chosen MoE model (gpt-oss-20b, Qwen3.6-35B-A3B) would fit the
  host but not the WSL limit, setup explains how to raise `memory=` in
  `%UserProfile%\.wslconfig`.
- The app's address `http://127.0.0.1:8080` opens from the Windows browser
  (WSL forwards localhost by default).
- A workbench folder under `/mnt/c/...` works but loads models slowly (Windows
  files are reached through a network file system); setup warns and suggests
  keeping the workbench in the Linux home folder.
- Stage 6 (fine-tuning) will prefer WSL on Windows machines with NVIDIA,
  because the Linux training stack is better supported.

### B3. Stage 1 models

Source: `~/Work/notes/workbench/stage1-research/models.md` (sizes and sha256
read from the source repos at pinned revisions on 2026-10-15; the 11 files
llm-kit already ships match byte for byte). All are Apache-2.0; none of the
source repos ships a LICENSE file, so each model folder in `workbench-models`
gets the Apache-2.0 text and a `SOURCE` file (source repo, revision, file,
sha256).

| id | Kinds | File (Q4_K_M unless noted) | GB | Vision file | Tiers |
|---|---|---|---:|---|---|
| qwen3.5-0.8b | chat | Qwen3.5-0.8B | 0.53 | 0.20 GB | optional (all) |
| qwen3.5-2b | chat | Qwen3.5-2B | 1.28 | 0.67 GB | cpu default |
| qwen3.5-4b | chat, vision, tools | Qwen3.5-4B | 2.74 | 0.67 GB | cpu, gpu-8 default |
| qwen3.5-9b | chat, vision, tools | Qwen3.5-9B | 5.68 | 0.92 GB | gpu-8, gpu-16 default |
| gemma4-e4b | chat, vision, audio | gemma-4-E4B QAT q4_0 | 5.15 | 0.99 GB | optional (gpu-8+) |
| granite-4.2-8b | chat, tools | granite-4.2-8b | 5.35 | none | optional (gpu-8+) |
| gpt-oss-20b | reasoning, tools | gpt-oss-20b MXFP4 | 12.11 | none | gpu-16 default; gpu-8 with MoE offload |
| qwen3.6-35b-a3b | chat, coding, vision, tools | Qwen3.6-35B-A3B UD-Q4_K_M | 22.13 | 0.90 GB | optional (gpu-16 with 32 GB RAM, 48 GB Mac) |
| qwen3-embed-0.6b | embeddings | Qwen3-Embedding-0.6B Q8_0 | 0.64 | none | cpu, gpu-8 default |
| qwen3-embed-4b | embeddings | Qwen3-Embedding-4B | 2.50 | none | gpu-16 default |
| qwen3-embed-8b | embeddings | Qwen3-Embedding-8B | 4.68 | none | optional |
| qwen3-rerank-0.6b | rerank | Qwen3-Reranker-0.6B Q8_0 (ggml-org) | 0.64 | none | cpu, gpu-8 default |
| qwen3-rerank-4b | rerank | Qwen3-Reranker-4B (Voodisss) | 2.50 | none | gpu-16 default |
| qwen3-rerank-8b | rerank | Qwen3-Reranker-8B (Voodisss) | 4.68 | none | optional |

Recommended sets (weights only): **cpu** 5.3 GB; **gpu-8** 9.7 GB (plus 0.67 GB
for vision on the 4B); **gpu-16** 22.8 GB, 44.9 GB with Qwen3.6-35B-A3B.
The whole Stage 1 catalog is about 75 GB in about 790 chunks.

Decisions from the research:

- **No separate coding model.** There is no small Qwen3.x coder, and the small
  Qwen2.5-Coder sizes are under a non-commercial license. Coding uses
  Qwen3.5-9B / Qwen3.6-35B-A3B (Pi's default).
- **Rerankers must have the classification head.** Some community reranker
  GGUFs lack `cls.output.weight` and return meaningless scores. The chosen
  files were checked; the publish tool (B5) rejects a reranker without it.
- **Draft models: not in Stage 1.** The llama-cpp-python fork does not implement
  plain draft-model decoding. Its working speed-up is MTP (the model's own
  next-token head), which needs the `-MTP` builds of the Qwen files (2-3% larger).
  Stage 1 ships the plain files; switching to MTP files is a measured follow-up
  once `selftest` shows the speed-up and that vision still works with them.
- **Thinking mode.** Qwen3.5-4B/9B think by default. On the `cpu` tier chat
  turns thinking off by default (a toggle in the app and `--think` on commands).

### B4. Platform packs and engines

Source: `engines.md` sections 1-3.

**llama.cpp engine: llama-cpp-python 0.4.2 from JamePeng's fork**, pinned to
release `...-20261003` (vendored llama.cpp `cb7934c5`). Needed because it is
the only build with reranking (`LlamaEmbedding.rank()`), Qwen3.5/3.6 support,
MTP speculation and the 0.4.x API llm-kit already uses. Upstream 0.3.36 has no
`rank()` and PyPI has no binary wheels at all.

| Pack | llama-cpp-python wheel | Extra runtime wheels | Built by |
|---|---|---|---|
| `win-x64-cuda12` | JamePeng 0.4.2 cu128 cp314 win_amd64 (156 MB): GPUs sm_75-sm_120 incl. Blackwell | `nvidia-cuda-runtime-cu12`, `nvidia-cublas-cu12` (568 MB), `msvc-runtime` | JamePeng (llm-kit's proven set) |
| `win-x64-vulkan` | our build: CPU variants + Vulkan, dynamic backends | `msvc-runtime` | our CI, ~10 min |
| `linux-x64-cuda13` | JamePeng 0.4.2 cu131 cp314 linux_x86_64 (183 MB; no NCCL dependency, unlike cu128) | `nvidia-cuda-runtime` 13, `nvidia-cublas` 13 (440 MB) | JamePeng |
| `linux-x64-vulkan` | our build: CPU variants + Vulkan | none | our CI, ~10 min |
| `linux-arm64-cpu` | our build: CPU (single variant) | none | our CI, ~6 min |
| `macos-arm64-metal` | JamePeng 0.4.2 Metal cp314 arm64 (14 MB) | none | JamePeng |

- **Our builds** use the fork's own recipe (`GGML_BACKEND_DL=ON`,
  `GGML_CPU_ALL_VARIANTS=ON`, `GGML_NATIVE=OFF`, plus `GGML_VULKAN=ON`), so a
  machine without a Vulkan driver still loads the CPU backend. The Linux build
  runs on a `manylinux_2_28` image so it works on older distributions. Plain
  CPU builds must not assume AVX2 (upstream's do).
- **Windows DLL loading:** ggml loads backends with plain `LoadLibraryW`, so
  the CUDA DLL folders go on `PATH` (llm-kit's working fix). The engine also
  preloads `cudart64_12.dll`, `cublasLt64_12.dll`, `cublas64_12.dll` with
  their full paths before the first model load, which works no matter how the
  search path is set.
- **MSVC runtime:** `ggml-cuda.dll` needs `msvcp140.dll`, which python.org's
  installer does not ship. Windows packs include the `msvc-runtime` wheel and
  preload its `msvcp140.dll`.
- **Linux CUDA libraries** are preloaded from the pip `nvidia/*/lib` folders
  (`RTLD_GLOBAL`), since `libggml-cuda.so` only searches its own folder.
- **Pack layers.** Each platform folder has `core/` (server, app, numpy,
  pydantic, Gradio and the rest, about 80 MB), `llama/` (table above) and
  `tracing/` (Phoenix, about 170 MB, skippable). Later stages add `speech/`,
  `onnx/`, `agents-*/`, `torch/`. Every layer has `requirements.lock` and
  `SHA256SUMS`; files over 95 MB are chunked.
- **ONNX Runtime** (Stage 2-3, recorded now): `onnxruntime` 1.31 (CPU; CoreML
  on macOS), `onnxruntime-directml` 1.24.4 for non-NVIDIA Windows GPUs,
  `onnxruntime-gpu` 1.31 (CUDA 13, driver ≥ 580) for NVIDIA. Only one can be
  installed per machine.
- **Out of Stage 1:** ROCm (AMD on Linux uses Vulkan), Windows HIP, NVIDIA
  Pascal or older (Vulkan), Intel Mac, Windows ARM.

`tools/build_packs.py` (llm-kit's `build_bundle.py`, generalized) resolves each
layer with `uv pip compile --python-platform <target> --python-version 3.14`,
downloads the pinned wheels, chunks big files, writes checksums, and verifies
an offline install for the target platform. A GitHub Actions workflow builds
the three llama.cpp wheels we compile ourselves and uploads them as artifacts;
the maintainer commits them into the pack.

### B5. Publishing models (`tools/publish_model.py`)

Maintainer-side only (users never run it). For one catalog entry:

1. Download the file from its source at the pinned revision (B3's research
   notes record every revision and sha256) and check the sha256.
2. For rerankers, read the GGUF header and refuse files without
   `cls.output.weight` and rank pooling.
3. Split into 95 MiB chunks (llm-kit's `split_tool.py` format), add `LICENSE`
   and `SOURCE`, commit to `workbench-models` (one commit per model, pushed in
   batches under GitHub's 2 GB per push).
4. Read each chunk's blob ID from the pushed commit and write the catalog entry
   (`files`, `chunks`, `bytes`, `sha256`).
5. Fetch the model back from GitHub with the user-side fetcher into a temp
   folder and compare the sha256 (catches anything wrong before users see it).

**Re-using llm-kit's uploads.** The 11 files llm-kit already put in
`Xephex/playground_stuff` (about 57 GB) are in the same chunk format. A catalog
entry may name that repo, and fetching by blob ID from it was tested. Stage 1
points those entries at `playground_stuff` and uploads only the new files
(about 18 GB: Qwen3.5-0.8B/2B, the 4B/8B embedding models, the rerankers),
which saves roughly 5 hours of upload at the measured 3 MB/s. They can move to
`workbench-models` later by changing only the catalog.

### B6. Tracing and app stack

Source: `~/Work/notes/workbench/stage1-research/tracing-app.md` (resolved for
win-x64, linux-x64 and mac-arm64 with Python 3.14; run end to end on Linux).

**Trace viewer: Arize Phoenix 20.22.0.** Its full dependency set resolves to
Python 3.14 wheels on all three platforms. It has datasets, experiments and
evals built in (Stage 4 builds on them), and ready-made OpenInference
instrumentation for OpenAI-style clients and LangChain/LangGraph (Stage 5). It
runs as `python -m phoenix.server.main serve`. Spans sent over OTLP/HTTP were
ingested. With `PHOENIX_ALLOW_EXTERNAL_RESOURCES=false` and
`PHOENIX_TELEMETRY_ENABLED=false`, 14 UI pages made zero outside requests; by
default it contacts Google Fonts, scarf.sh, PyPI and GitHub, so the workbench
always sets both.

Phoenix's caveats, and how the design handles them:

- **Size:** about 160-190 MB of wheels per platform (pyarrow alone 51 MB). It
  goes in its own `tracing` pack layer, installed by default but skippable
  (`setup --no-trace-viewer`). Recording spans needs only the small
  OpenTelemetry SDK, which is always installed; spans are kept in a local file
  until a viewer is available.
- **`.exe` files on Windows:** `pydantic-monty-runtime` ships `monty.exe`, a
  sandbox worker Phoenix can start, and `pywin32` ships two `.exe` tools.
  Phoenix ran on Linux with the monty packages removed. On Windows the pack
  leaves out `pydantic-monty*`, and the CI Windows job checks that Phoenix
  starts and ingests without them and without launching any `.exe`. If it
  cannot, Windows falls back to MLflow's server-only set (about 63 MB, only
  `pywin32` carries `.exe` files and it is not run).
- **License:** Phoenix's server is Elastic License 2.0 (source-available; the
  restriction is on offering it as a hosted service). Bundling it is allowed
  with its license text and notices, which the pack keeps. The workbench code
  stays Apache-2.0. `ldap3` in its dependencies is LGPL-3.
- Phoenix and MLflow cannot share one environment (conflicting `graphql-core`),
  so only one is ever installed.
- No Windows on ARM (no pyarrow wheel); the trace viewer is hidden there.

**Langfuse (optional second destination):** OTLP/HTTP (JSON or protobuf, not
gRPC) to `<host>/api/public/otel/v1/traces`, header
`Authorization: Basic base64(<public-key>:<secret-key>)` plus
`x-langfuse-ingestion-version: 4`; self-hosted Langfuse 3.22.0 or newer. The
exporter is added next to Phoenix's (`replace_default_processor=False`;
otherwise Phoenix's own exporter is replaced). Tested only against a capture
server; Langfuse maps OpenInference attributes partially.

**App: Gradio 6.30.0**, mounted into the FastAPI server with
`gr.mount_gradio_app`. Offline settings: `GRADIO_ANALYTICS_ENABLED=False`, no
`share=True`, no server-side rendering, a theme using local/system fonts. Tabs,
microphone input, file upload and mounting all worked with zero outside
requests. Components that pull scripts from unpkg/jsdelivr (video trim,
Plotly/Bokeh maps, 3D viewers) are not used. A CI test loads every tab with
outside network blocked and fails on any outside request.

**Server:** FastAPI, uvicorn and sse-starlette have Python 3.14 wheels on all
three platforms (about 3.5 MB).

### B7. Server and model manager

The Stage 1 server has its own FastAPI routes rather than llama-cpp-python's
bundled server: llm-kit had to patch that server's request and response types
for tool calls, and owning the routes removes those patches.

- **Routes:** `GET /v1/models`; `POST /v1/chat/completions` (streaming via SSE,
  `tools`/`tool_choice`, image parts, `response_format` JSON schema through the
  engine's grammar support); `POST /v1/embeddings`; `POST /v1/rerank`
  (`{model, query, documents, top_n}` → `{results: [{index, relevance_score}]}`,
  the common Jina/Cohere shape); `GET /wb/status`, `GET/POST /wb/models`,
  `GET /wb/jobs`. API key required when bound to anything but loopback.
- **Model manager:** a memory budget from B2 (VRAM minus 0.8 GB headroom, and
  RAM minus 4 GB). Each loaded model records its measured footprint (from
  `tune`, or the catalog estimate until tuned). Loading a model that does not
  fit evicts the least recently used one. Embedding and reranker models load on
  the GPU when they fit next to the chat model, otherwise on the CPU. One lock
  per model instance (llama-cpp-python runs one sequence at a time); requests
  for a busy model queue in arrival order.
- **Tool calls:** llm-kit's `toolcalls.py` parser and its 13 tests move over
  unchanged.
- **Tracing:** an OpenTelemetry span per request, with child spans for model
  load, prompt processing and generation; attributes follow the OpenInference
  conventions so Phoenix shows them as LLM spans.

### B8. Testing and CI for Stage 1

- **Unit tests** (no models): `hw.py` against recorded fixtures (NVML results,
  DXGI descriptors, registry values, sysfs trees, sysctl values, WSL
  osrelease); pack choice per profile; catalog filtering per tier; fetcher
  against a local bare repo (chunk by blob ID, resume, sha256 mismatch);
  toolcalls; route schemas.
- **CI matrix:** `windows-latest`, `ubuntu-24.04`, `ubuntu-24.04-arm`,
  `macos-15` (arm64). Each job: sparse clone of the pushed commit, `setup` with
  the platform's CPU-capable pack and the smallest set (Qwen3.5-0.8B,
  Qwen3-Embedding-0.6B, Qwen3-Reranker-0.6B, about 1.8 GB, cached between runs
  by sha256), then one request to every Stage 1 route, then the app's tabs
  loaded with an outside-network guard (any non-loopback connection fails the
  job). On Windows, also: no `.exe` launchers left for workbench packages, no
  process other than `python.exe` started, Phoenix starts without the monty
  packages.
- **Pack verification:** every layer installed offline for its target before it
  is committed.
- **Real hardware:** `workbench.py selftest` on the operator's laptop (Windows,
  RTX PRO 2000 Blackwell 8 GB) for `win-x64-cuda12`; and, if the operator's WSL
  Ubuntu is on an NVIDIA machine, `linux-x64-cuda13` there too. Platforms
  without a selftest report are marked "untested on this hardware".

### B9. Risks and open items

| Risk | Plan |
|---|---|
| Our Vulkan/CPU llama.cpp builds may not load cleanly side by side | CI runs a model on CPU with the Vulkan build; real Vulkan GPUs tested by selftest when available |
| Intel Arc VRAM is not detectable before load | tier from the Vulkan device memory after the first load; until then treat as unknown and offer `cpu`-tier defaults |
| `workbench-models` grows past what GitHub tolerates (whole catalog ~75 GB in Stage 1, more later) | catalog names the repo per file, so models can be spread over several repos without user impact |
| Phoenix on Windows without `pydantic-monty` untested | CI check; fallback MLflow server-only set |
| Hybrid Qwen3.5/3.6 models have had backend-specific bugs upstream | selftest per backend; catalog marks models per backend once verified |
| JamePeng fork is a single maintainer's project | pinned release, our own builds for three packs already; the recipe is in our CI so we can rebuild CUDA ourselves if needed (about 2-3 h per build) |
| CPU speed of 2B/4B models not measured | selftest on CI runners records it; tier defaults adjusted from those numbers |
