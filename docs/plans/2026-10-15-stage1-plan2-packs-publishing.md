# Stage 1, Plan 2: Packs and publishing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `python workbench.py setup` installs this machine's platform pack offline from the repo, and every Stage 1 model is published to GitHub with a real catalog entry, so a Windows or Linux machine goes from `git clone` to installed packages and verified models.

**Architecture:** Packs live in the code repo as `packs/<pack>/<layer>/` folders: wheels (chunked over 95 MiB), a hashed `requirements.lock` and `SHA256SUMS`. `workbench/install.py` sparse-checks-out only this machine's layers, joins chunks, and runs pip from its own wheel (`python -I pip.whl/pip install --isolated --no-index --require-hashes --target`) into a fresh `.workbench/site.new`. It then deletes every `.exe` and swaps the result into `.workbench/site`. The maintainer tools are `tools/build_packs.py` (uv locks per layer, per-platform downloads, offline verify) and `tools/publish_model.py` (runs on miniRUD: download at the pinned revision, check, split, commit into a blobless clone with git plumbing, push in slices, fetch back). CI builds the three llama.cpp wheels nobody publishes.

**Tech Stack:** Python 3.14 standard library (user side). Maintainer side: uv 0.11, pip 26.2.1 (run from its wheel), git ≥ 2.40, GitHub Actions (manylinux_2_28 containers, windows-2022), JamePeng llama-cpp-python 0.4.2 at commit `528cf248460b49312597db98eb4f4e74a73214b8`.

**Spec:** `docs/specs/2026-10-15-workbench-design.md` (A3, A5 steps 2-4, A8, B3, B4, B5, B8 "pack verification").

**This is plan 2 of 4 for Stage 1:**
1. Install foundation (done, main 26d0abf)
2. Packs and publishing (this plan)
3. Server and engines
4. App, Pi, docs, CI

**Scope cut (operator, 2026-10-15):** macOS is low priority. This plan builds and publishes no `macos-arm64-metal` pack. `packs.choose_pack` keeps returning it, and `setup` then says that pack is not published yet.

## Global Constraints

- User-side code (`workbench/`, `workbench.py`) uses only the standard library. Only `git` and `sys.executable` (running pip from its wheel) are ever started as processes (spec R6).
- The pack install never reads the network: `--no-index`, `--isolated`, `--require-hashes`, `--only-binary :all:`, `--find-links` local folders only (spec R2).
- Pack ids: `win-x64-cuda12`, `win-x64-vulkan`, `linux-x64-cuda13`, `linux-x64-vulkan`, `linux-arm64-cpu`. Layers, in install order: `core`, `llama`, `tracing` (optional; `setup --no-trace-viewer` skips it).
- A wheel stored in more than one layer's lock lives in the first layer folder of `core`, `llama`, `tracing` that locks it. `core` and `llama` are always installed.
- Chunks: `<file>.partNNN` (95 MiB = 99,614,720 bytes) plus a `<file>.sha256` sidecar `"<hex>  <name>\n"` (llm-kit `split_tool.py` format).
- No `.exe` file is left anywhere under `.workbench/site` after an install (spec R6, B8).
- Windows locks leave out `pydantic-monty`, `pydantic-monty-client` and `pydantic-monty-runtime` (uv `--overrides` with `; sys_platform != "win32"`), per spec B6.
- llama-cpp-python wheels: JamePeng `v0.4.2-cu128-win-20261003` (win-x64-cuda12), `v0.4.2-cu131-linux-20261003` (linux-x64-cuda13); our CI builds `0.4.2+vulkan` (win, linux x64) and `0.4.2+cpu` (linux arm64) from commit `528cf248460b49312597db98eb4f4e74a73214b8`.
- CUDA runtime wheels: win `nvidia-cuda-runtime-cu12==12.9.79`, `nvidia-cublas-cu12==12.9.2.10`, `msvc-runtime==14.44.35112`; linux `nvidia-cuda-runtime==13.1.80`, `nvidia-cublas==13.1.1.3`.
- Model publishing never force-pushes. Each push is under 2 GB; each commit holds at most 15 chunks (about 1.5 GB).
- The GitHub token reaches miniRUD only on ssh standard input. It is held in memory and passed to git through a credential-helper environment variable, never through argv, files or logs.
- Commits: author `Xephex <Xephex@users.noreply.github.com>`, message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Deletes name literal targets, or remove a path built from a fixed parent plus a validated name (`.workbench/site.old`, files ending in `.exe` under `.workbench/site.new`).

## Review Focus

1. **Install interrupted part-way** (Ctrl+C or a pip failure mid-install): the previous `.workbench/site` must still work, and re-running `setup` completes. Tested in Task 1 (`test_failed_install_keeps_old_site`).
2. **User's own pip configuration** (`PIP_REQUIRE_VIRTUALENV=1`, `PIP_INDEX_URL`, a `pip.conf`): must not break or redirect the offline install. Tested in Task 1 (`test_user_pip_config_ignored`).
3. **Corrupt or tampered pack file** (bad chunk, edited wheel): setup stops, names the file, and installs nothing. Tested in Task 1 (`test_bad_chunk_named`, `test_hash_mismatch_installs_nothing`).
4. **Windows path too long** for Phoenix's deepest file (113 characters below the site folder) when the clone sits in a deep folder and long paths are off: say so before installing and suggest a shorter folder. Tested in Task 1 (`test_long_path_problem`).
5. **Publishing interrupted or the remote moved** (network drop mid-push, push rejected): a re-run continues from what GitHub already has, never re-uploads pushed chunks, and never force-pushes. Tested in Task 3 (`test_rerun_skips_pushed_slices`, `test_rejected_push_rebases_on_remote`).

---

## File Structure

```
workbench/install.py            pack fetch (sparse checkout), chunk join, sums check, pip install, .exe removal, activate()
workbench/cli.py                setup installs the pack (--no-trace-viewer, --packages-only); status shows packages
packs/pip/pip-26.2.1-py3-none-any.whl (+ .sha256)   the pip that installs every pack
packs/requirements/core.in, tracing.in, llama-<pack>.in, overrides.txt   pack inputs
packs/<pack>/<layer>/           requirements.lock, SHA256SUMS, wheels / chunks   (generated by build_packs)
tools/build_packs.py            lock | fetch | verify | all  --pack ID [--wheels DIR]
tools/gguf.py                   GGUF header reader; rerank_problem()
tools/catalog_write.py          catalog.toml writer (byte-identical round trip) and upsert
tools/publish_model.py          maintainer: publish models to Xephex/workbench-models
tools/publish.toml              the models to publish (source, revision, sizes, sha256, tiers)
tools/smoke_llama.py            CI: load a GGUF with a built wheel and embed two strings
.github/workflows/llama-wheels.yml   build win-x64-vulkan, linux-x64-vulkan, linux-arm64-cpu wheels
.github/workflows/packs.yml     sparse clone + setup --packages-only on Windows, Linux x64, Linux arm64
tests/wheelfixture.py           builds tiny wheels for install tests
tests/test_install.py, test_build_packs.py, test_gguf.py, test_catalog_write.py, test_publish.py
```

Run all tests with `python3.14 -m unittest discover -s tests -v` from the repo root.

---

### Task 1: Offline pack install (`workbench/install.py`) and `setup` wiring

**Files:**
- Create: `workbench/install.py`, `tests/wheelfixture.py`, `tests/test_install.py`, `packs/pip/pip-26.2.1-py3-none-any.whl`, `packs/pip/pip-26.2.1-py3-none-any.whl.sha256`
- Modify: `workbench/cli.py` (setup, status), `tests/test_cli.py`

**Interfaces:**
- Consumes: `packs.choose_pack(p) -> PackChoice` (Plan 1).
- Produces:
  - `install.LAYERS = ("core", "llama", "tracing")`, `install.OPTIONAL = frozenset({"tracing"})`
  - `install.InstallError(RuntimeError)`
  - `install.site_dir(root: Path) -> Path` (`root/.workbench/site`)
  - `install.activate(root: Path) -> bool` (`site.addsitedir` when the site exists)
  - `install.fetch_pack(root: Path, pack: str, layers: list[str]) -> None`
  - `install.join_chunks(layer_dir: Path, out_dir: Path) -> None`
  - `install.check_sums(layer_dir: Path, joined_dir: Path) -> None`
  - `install.remove_exe(site: Path) -> list[Path]`
  - `install.long_path_problem(site: Path, enabled: bool | None) -> str | None`
  - `install.install_pack(root: Path, pack: str, with_tracing: bool = True) -> str` (returns `"installed"` or `"up to date"`)
  - `install.installed_pack(root: Path) -> str | None`

The pip invocation (verified 2026-10-15 with pip 26.2.1 on Python 3.14):

```python
[sys.executable, "-I", str(root / PIP_WHEEL / "pip"), "install", "--isolated", "--no-index",
 "--require-hashes", "--only-binary", ":all:", "--no-deps", "--disable-pip-version-check",
 "--no-warn-script-location", "--target", str(new_site),
 *[a for d in find_links for a in ("--find-links", str(d))],
 *[a for lock in locks for a in ("-r", str(lock))]]
```

Several `-r` files may pin the same package with the same hash (tested: pip installs it once).

- [ ] **Step 1: Write the failing tests**

`tests/wheelfixture.py` builds pure `py3-none-any` wheels: `make_wheel(dir, name, version, script=False, extra_files=None) -> Path`, writing a valid RECORD, plus `lock_line(path) -> str` (`"name==ver \\\n    --hash=sha256:<hex>\n"`).

`tests/test_install.py` builds a fake repo `root/` with `packs/pip/<real pip wheel copied from the repo>` and `packs/linux-x64-vulkan/{core,llama,tracing}/` holding fixture wheels, locks and SHA256SUMS. One core wheel is stored as `.part000`/`.part001` chunks, one llama wheel has a console script (so pip writes a launcher), and one wheel carries a file `tool.exe`. Tests:

```python
def test_installs_all_layers_and_activates(self):
    self.assertEqual(install.install_pack(self.root, PACK), "installed")
    self.assertTrue(install.activate(self.root))
    import wbcore, wbllama, wbtrace  # noqa: F401  (fixture packages)
    self.assertEqual(install.installed_pack(self.root), PACK)

def test_second_run_is_up_to_date(self):
    install.install_pack(self.root, PACK)
    self.assertEqual(install.install_pack(self.root, PACK), "up to date")

def test_no_trace_viewer_skips_layer(self):
    install.install_pack(self.root, PACK, with_tracing=False)
    self.assertFalse((install.site_dir(self.root) / "wbtrace").exists())

def test_no_exe_left(self):
    install.install_pack(self.root, PACK)
    self.assertEqual([p for p in install.site_dir(self.root).rglob("*") if p.suffix.lower() == ".exe"], [])

def test_bad_chunk_named(self):                      # review focus 3
    part = self.layer("core") / "wbbig-1.0-py3-none-any.whl.part001"
    part.write_bytes(b"x" + part.read_bytes()[1:])
    with self.assertRaisesRegex(install.InstallError, "wbbig-1.0-py3-none-any.whl"):
        install.install_pack(self.root, PACK)
    self.assertFalse(install.site_dir(self.root).exists())

def test_hash_mismatch_installs_nothing(self):       # review focus 3
    w = self.layer("llama") / "wbllama-1.0-py3-none-any.whl"
    w.write_bytes(w.read_bytes() + b"\0")
    with self.assertRaises(install.InstallError):
        install.install_pack(self.root, PACK)
    self.assertFalse(install.site_dir(self.root).exists())

def test_failed_install_keeps_old_site(self):        # review focus 1
    install.install_pack(self.root, PACK)
    (self.layer("core") / "requirements.lock").write_text("nosuchpkg==1.0 --hash=sha256:" + "0" * 64 + "\n")
    with self.assertRaises(install.InstallError):
        install.install_pack(self.root, PACK)
    install.activate(self.root)
    import wbcore  # noqa: F401  (old site still in place)

def test_user_pip_config_ignored(self):              # review focus 2
    with mock.patch.dict(os.environ, {"PIP_REQUIRE_VIRTUALENV": "1", "PIP_INDEX_URL": "http://127.0.0.1:9/simple",
                                      "PIP_TARGET": str(self.root / "elsewhere")}):
        self.assertEqual(install.install_pack(self.root, PACK), "installed")
    self.assertFalse((self.root / "elsewhere").exists())

def test_long_path_problem(self):                    # review focus 4
    deep = pathlib.PureWindowsPath("C:/" + "d" * 150 + "/workbench/.workbench/site")
    self.assertIn("shorter folder", install.long_path_problem(deep, enabled=False))
    self.assertIsNone(install.long_path_problem(deep, enabled=True))
    self.assertIsNone(install.long_path_problem(pathlib.PureWindowsPath("C:/wb/.workbench/site"), enabled=False))

def test_missing_pack_folder(self):
    with self.assertRaisesRegex(install.InstallError, "not published"):
        install.install_pack(self.root, "macos-arm64-metal")
```

`fetch_pack` test: a local git repo (as in `test_launcher.test_sparse_clone_fetches_its_own_package`) holding `packs/linux-x64-vulkan/core/x.txt` and `packs/win-x64-cuda12/core/y.txt`. After `--sparse` clone plus `fetch_pack(clone, "linux-x64-vulkan", ["core"])`, `x.txt` exists and `packs/win-x64-cuda12` does not.

`tests/test_cli.py` additions: `setup --yes` on a profile whose pack folder is missing prints "not published yet" and returns 1. `status` prints `Packages: not installed (run setup)`. `setup --packages-only --yes` calls `install.install_pack` (mocked) and never calls `fetch.fetch_model`.

- [ ] **Step 2: Run them to see them fail**

Run: `python3.14 -m unittest tests.test_install tests.test_cli -v`
Expected: FAIL / ERROR, because `workbench.install` does not exist.

- [ ] **Step 3: Implement `workbench/install.py`**

```python
"""Install this machine's platform pack offline (spec A5 steps 3-4, A8, B4).

The pack's wheels are already in the repo (git sparse checkout). Pip runs from
its own wheel into a fresh folder, which replaces .workbench/site only once the
install succeeded, so an interrupted install never breaks a working one.
"""

from __future__ import annotations

import hashlib, json, os, pathlib, shutil, site, subprocess, sys

LAYERS = ("core", "llama", "tracing")
OPTIONAL = frozenset({"tracing"})
PIP_WHEEL = pathlib.Path("packs/pip/pip-26.2.1-py3-none-any.whl")
DEEPEST = 120  # longest path pip writes below the site folder (Phoenix: 113) plus margin
MAX_PATH = 260


class InstallError(RuntimeError):
    pass


def site_dir(root): return root / ".workbench" / "site"
def _marker(root): return root / ".workbench" / "site.json"


def activate(root) -> bool:
    s = site_dir(root)
    if not s.is_dir():
        return False
    site.addsitedir(str(s))
    return True


def installed_pack(root) -> str | None:
    try:
        return json.loads(_marker(root).read_text())["pack"]
    except (OSError, ValueError, KeyError):
        return None


def fetch_pack(root, pack, layers) -> None:
    if not (root / ".git").exists():
        return  # a full copy (zip or CI checkout) already has every folder
    paths = [f"packs/pip"] + [f"packs/{pack}/{layer}" for layer in layers]
    r = subprocess.run(["git", "-C", str(root), "sparse-checkout", "add", *paths],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        msg = r.stderr.decode(errors="replace").strip().splitlines()
        raise InstallError("git could not fetch the packages: " + (msg[-1] if msg else f"exit {r.returncode}"))
```

Then:

- `join_chunks(layer_dir, out_dir)`: for each `*.part000`, join `partNNN` in numeric order (gaps are an `InstallError` naming the file) into `out_dir/<name>`, and compare the sha256 with the `.sha256` sidecar (`InstallError(f"{name}: joined chunks do not match its checksum; run git pull or setup again")`). Already-joined matching files are skipped.
- `check_sums(layer_dir, joined_dir)`: every `SHA256SUMS` entry exists whole in `layer_dir` or in `joined_dir`, and its sha256 matches. Mismatch → `InstallError` naming the file.
- `remove_exe(site)`: `[p for p in site.rglob("*") if p.is_file() and p.suffix.lower() == ".exe"]`, unlinking each and returning the list.
- `long_path_problem(site, enabled)`: `enabled=None` reads `HKLM\SYSTEM\CurrentControlSet\Control\FileSystem\LongPathsEnabled` with winreg on Windows (and returns None off Windows). Returns a message containing "shorter folder" when `len(str(site)) + DEEPEST >= MAX_PATH` and long paths are off.
- `install_pack(root, pack, with_tracing=True)`:
  1. layers = core, llama (+ tracing).
  2. `fetch_pack`.
  3. If any layer folder or the pip wheel is missing → `InstallError(f"Packages for {pack} are not published yet.")`.
  4. Long-path check → `InstallError`.
  5. Digest = sha256 over (pack, layers, each layer's `requirements.lock` and `SHA256SUMS` bytes). If it equals the marker's digest and the site exists → `"up to date"`.
  6. Free disk must be ≥ 3 × the layer folders' bytes, else `InstallError` with both numbers.
  7. Join into `.workbench/wheels/<pack>/<layer>/`, then `check_sums`.
  8. Remove a leftover `.workbench/site.new` (fixed path) and run the pip command into it. Non-zero exit → `InstallError` with pip's last 15 output lines; the old site is untouched.
  9. `remove_exe(new)`.
  10. Swap: old site → `site.old`, `site.new` → `site`, remove `site.old`.
  11. Write the marker `{"pack", "layers", "digest"}` last, then return `"installed"`.

- [ ] **Step 4: Wire into the CLI**

`cli.py`:
- `setup` gets `--no-trace-viewer` and `--packages-only`.
- `cmd_setup` does the following after `_profile_lines`:

```python
    say("")
    say(f"Installing the {choice.pack} packages (offline, from this folder)...")
    try:
        state = install.install_pack(ROOT, choice.pack, with_tracing=not a.no_trace_viewer)
    except install.InstallError as e:
        say(f"  {e}")
        return 1
    say(f"  Packages {state}.")
    if a.packages_only:
        return 0
```

- `cmd_status` adds `Packages: <pack> installed` or `Packages: not installed (run setup)`, using `install.installed_pack(ROOT)`.
- `ROOT` stays the repo root. Tests patch `cli.ROOT`.
- Existing setup tests patch `install.install_pack` to return `"installed"` so they stay model-only.

- [ ] **Step 5: Run the tests**

Run: `python3.14 -m unittest discover -s tests -v`
Expected: all pass (76 earlier tests plus the new ones).

- [ ] **Step 6: Commit**

```bash
git add workbench/install.py workbench/cli.py packs/pip tests/wheelfixture.py tests/test_install.py tests/test_cli.py
git commit -m "feat: offline pack install with pip from its wheel, .exe removal, safe swap"
```

---

### Task 2: Pack builder (`tools/build_packs.py`) and requirement inputs

**Files:**
- Create: `tools/build_packs.py`, `packs/requirements/core.in`, `packs/requirements/tracing.in`, `packs/requirements/llama-win-x64-cuda12.in`, `llama-win-x64-vulkan.in`, `llama-linux-x64-cuda13.in`, `llama-linux-x64-vulkan.in`, `llama-linux-arm64-cpu.in`, `packs/requirements/overrides.txt`, `tests/test_build_packs.py`

**Interfaces:**
- Consumes: the pack layout Task 1 installs (`packs/<pack>/<layer>/{requirements.lock,SHA256SUMS,wheels|chunks}`), `install.join_chunks`, `install.check_sums`.
- Produces:
  - CLI `python tools/build_packs.py {lock,fetch,verify,all} --pack ID [--wheels DIR]`.
  - Pure functions:
    - `read_lock(text) -> dict[str, Pin]`, with `Pin(name, version, hashes)`.
    - `assign_layers(locks: dict[str, dict[str, Pin]]) -> dict[str, str]` (package → first layer locking it).
    - `split_big(dir: Path) -> None`.
    - `write_sums(dir: Path, digests: dict[str, str]) -> None`.
  - `PACKS: dict[str, Target]`, with `Target(uv_platform, pip_platforms, jamepeng_tag | None, local_suffix | None)`.

Targets:

| pack | uv `--python-platform` | pip `--platform` list | llama source |
|---|---|---|---|
| win-x64-cuda12 | `x86_64-pc-windows-msvc` | `win_amd64` | JamePeng `v0.4.2-cu128-win-20261003` |
| win-x64-vulkan | `x86_64-pc-windows-msvc` | `win_amd64` | our `+vulkan` wheel (`--wheels`) |
| linux-x64-cuda13 | `x86_64-manylinux_2_34` | `manylinux_2_34..2_17`, `manylinux2014`, `linux` (x86_64) | JamePeng `v0.4.2-cu131-linux-20261003` |
| linux-x64-vulkan | `x86_64-manylinux_2_28` | `manylinux_2_28..2_17`, `manylinux2014`, `linux` (x86_64) | our `+vulkan` wheel |
| linux-arm64-cpu | `aarch64-manylinux_2_28` | same list (aarch64) | our `+cpu` wheel |

Inputs:
- `core.in`: `fastapi==0.143.2`, `uvicorn==0.54.0`, `sse-starlette==3.5.0`, `gradio==6.30.0`, `opentelemetry-sdk==1.45.1`, `opentelemetry-exporter-otlp-proto-http==1.45.1`, `openinference-semantic-conventions==0.1.41`.
- `tracing.in`: `arize-phoenix==20.22.0`.
- `overrides.txt`: the three `pydantic-monty*` lines with `; sys_platform != "win32"`.
- `llama-<pack>.in`: the wheel pins from Global Constraints.

The steps:
- **lock:**
  1. Full resolve: `uv pip compile core.in llama-<pack>.in tracing.in --python-platform T --python-version 3.14 --overrides overrides.txt --find-links <llama dir> --no-build --no-config --generate-hashes`.
  2. Then per layer: `uv pip compile <layer>.in -c full.lock …` with the same flags, written to `packs/<pack>/<layer>/requirements.lock`.
  3. The llama dir holds JamePeng assets (downloaded via the GitHub release API, with each asset's sha256 digest checked) or the `--wheels` folder.
- **fetch:** download with pip from its wheel:
  - command: `python -I packs/pip/…whl/pip download --isolated --no-deps --only-binary :all: --require-hashes --implementation cp --python-version 3.14 --abi cp314 --abi abi3 --abi none <platform flags> --find-links <llama dir> -r <layer lock> -d <layer dir>`;
  - each package is stored only in its `assign_layers` layer;
  - files over 95 MiB are split (`split_big`);
  - `SHA256SUMS` is written;
  - wheels that are no longer locked are listed for the maintainer, not deleted.
- **verify:**
  1. Join into a temp folder and `check_sums`.
  2. Every locked package has a stored wheel, and every stored wheel is locked.
  3. Run `uv pip install --target <tmp> --python-platform T --python-version 3.14 --no-index --require-hashes --no-deps --find-links <all layer dirs + joined> -r <each lock>`.
  4. On Windows targets, `grep -i monty` over the locks must be empty.

- [ ] **Step 1: Write the failing tests** (no network; uv-dependent tests skip when uv is missing)

```python
def test_read_lock_parses_hashes(self):
    pins = build_packs.read_lock("a==1.0 \\\n    --hash=sha256:" + "a" * 64 + "\n    # via x\nb==2 \\\n    --hash=sha256:" + "b" * 64 + "\n")
    self.assertEqual(pins["a"].version, "1.0")
    self.assertEqual(pins["b"].hashes, ["b" * 64])

def test_assign_layers_first_layer_wins(self):
    locks = {"core": {"numpy": P("numpy")}, "llama": {"numpy": P("numpy"), "llama-cpp-python": P("llama-cpp-python")},
             "tracing": {"numpy": P("numpy"), "arize-phoenix": P("arize-phoenix")}}
    self.assertEqual(build_packs.assign_layers(locks),
                     {"numpy": "core", "llama-cpp-python": "llama", "arize-phoenix": "tracing"})

def test_split_big_and_sums_roundtrip_with_install_join(self):
    # 250-byte file with SPLIT_BYTES patched to 100 -> 3 parts + sidecar; install.join_chunks rebuilds it
    ...

def test_windows_locks_have_no_monty(self):          # skips without uv; uses fixture wheels + overrides.txt
    ...
```

- [ ] **Step 2: Run them to see them fail.** `python3.14 -m unittest tests.test_build_packs -v` → ERROR (no module).
- [ ] **Step 3: Implement** `tools/build_packs.py` as described (stdlib + subprocess to uv/pip; maintainer-only, network allowed).
- [ ] **Step 4: Run the tests.** All pass.
- [ ] **Step 5: Commit** `feat: pack builder: per-layer hashed locks, per-platform wheels, offline verify`.

---

### Task 3: Model publisher (`tools/publish_model.py`, `tools/gguf.py`, `tools/catalog_write.py`)

**Files:**
- Create: `tools/gguf.py`, `tools/catalog_write.py`, `tools/publish_model.py`, `tools/publish.toml`, `tests/test_gguf.py`, `tests/test_catalog_write.py`, `tests/test_publish.py`

**Interfaces:**
- Consumes:
  - `workbench.fetch.fetch_file(f, dest, work)`, used to fetch back.
  - `workbench.catalog.load`.
  - `tests.gitfixture` style local bare repos.
- Produces:
  - `gguf.read_header(f: BinaryIO) -> tuple[dict, list[str]]`
  - `gguf.rerank_problem(kv, tensors) -> str | None`: the reason when the file lacks `<arch>.pooling_type == 4` or a `cls.output.weight` tensor.
  - `catalog_write.dump(data: dict, header: str) -> str` and `catalog_write.upsert(path: Path, entry: dict) -> None` (replaces the model with the same id, else appends).
  - `publish_model.blob_id(path) -> str`: git's `sha1("blob <n>\0" + bytes)`, verified equal to `git hash-object`.
  - `publish_model.split(path, out_dir) -> list[Part]`, with `Part(name, path, bytes, blob)`.
  - `publish_model.slices(paths, max_files=15) -> list[list]`.
  - `publish_model.commit_files(bare, parent: str | None, files: dict[str, Path], message) -> str`.
  - `publish_model.push(bare, commit, url, token: str | None) -> None`.
  - `publish_model.main(argv)`.

Verified git mechanics (2026-10-15, local file:// repos), in a blobless bare clone (`git clone --bare --filter=blob:none --no-tags`) with `GIT_NO_LAZY_FETCH=1` and a private `GIT_INDEX_FILE`:
1. `git read-tree <parent>`.
2. `git hash-object -w --stdin < chunk`.
3. `git update-index --add --cacheinfo 100644,<blob>,<path>`.
4. `git write-tree --missing-ok`: plain `write-tree` tries to lazy-fetch the parent's blobs and fails.
5. `git commit-tree <tree> -p <parent> -m <msg>`.
6. `git push origin <commit>:refs/heads/main`.
7. `git ls-remote origin refs/heads/main` must equal `<commit>`.

The commit identity comes from `GIT_AUTHOR_*` / `GIT_COMMITTER_*` = Xephex. The credential is passed as `-c credential.helper= -c 'credential.helper=!f() { echo username=Xephex; echo "password=$WB_PUSH_TOKEN"; }; f'`, with `WB_PUSH_TOKEN` only in the git child's environment.

Model folder in `workbench-models`: `<id>/<file>.partNNN`, `<id>/<file>.sha256`, `<id>/LICENSE` (the repo's Apache-2.0 text), `<id>/SOURCE` (`source repo@revision`, file names, bytes, sha256).

Re-run logic:
1. Clone, then `git ls-tree -r origin/main -- <id>/` (names and blob ids only; no blob download).
2. Every local part whose path and blob id are already there is skipped.
3. The remaining parts go out in slices of ≤ 15 files, one commit and one push each, 3 tries per push.
4. A rejected push (the remote moved) re-fetches `main`, rebuilds that slice's commit on the new head, and tries again. Never `--force`.

- [ ] **Step 1: Write the failing tests**

```python
# test_gguf.py: builds GGUF v3 bytes in memory (header, kv: general.architecture="qwen3",
# qwen3.pooling_type=uint32 4, tokenizer list; tensors: cls.output.weight, blk.0.attn_q.weight)
def test_reranker_with_head_passes(self): self.assertIsNone(gguf.rerank_problem(*gguf.read_header(buf(pool=4, cls=True))))
def test_reranker_without_head_refused(self): self.assertIn("cls.output.weight", gguf.rerank_problem(*gguf.read_header(buf(pool=4, cls=False))))
def test_wrong_pooling_refused(self): self.assertIn("pooling", gguf.rerank_problem(*gguf.read_header(buf(pool=3, cls=True))))
def test_not_gguf(self): self.assertRaises(ValueError, gguf.read_header, io.BytesIO(b"PK\x03\x04"))

# test_catalog_write.py
def test_roundtrip_is_byte_identical(self):
    text = (ROOT / "catalog.toml").read_text(encoding="utf-8")
    self.assertEqual(catalog_write.dump(tomllib.loads(text), text.splitlines(keepends=True)[0]), text)
def test_upsert_replaces_and_appends(self): ...  # same id replaced in place; new id appended; catalog.load() still works

# test_publish.py  (local bare "GitHub" over file://; publish_model.repo_url patched like fetch.WORKBENCH_REPO_BASE)
def test_blob_id_matches_git(self): ...
def test_publish_new_model_then_fetch_back(self): ...   # 5,000 random bytes, chunk size patched to 2,000 -> 3 parts;
                                                        # entry chunks equal the remote blobs; fetch_file returns the bytes
def test_rerun_skips_pushed_slices(self):               # review focus 5
    ...  # interrupt after the first slice (push raises on 2nd call), re-run: remote log shows 1 + remaining commits,
         # first slice's parts not re-committed
def test_rejected_push_rebases_on_remote(self):         # review focus 5
    ...  # another commit lands on remote main between clone and push -> publish still succeeds, both commits in history
def test_reranker_without_head_not_published(self): ...
def test_sha_mismatch_stops_before_any_commit(self): ...
```

- [ ] **Step 2: Run them to see them fail.** ERROR (no modules).
- [ ] **Step 3: Implement.**
  - `publish_model.py` reads the `tools/publish.toml` entries (`id`, `kinds`, `ram_gb`, `vram_gb`, `recommended_for`, `license`, `source = "repo@rev"`, `files = [{name, role, sha256, bytes}]`).
  - Download: from `https://huggingface.co/<repo>/resolve/<rev>/<name>`, resumable via a `.download` file and a Range header, then size and sha256 are checked. Maintainer-only network.
  - Rerankers get the GGUF check. Then split, commit/push, verify with ls-remote and ls-tree, fetch back with `workbench.fetch.fetch_file` into `<work>/fetchback/` (skippable with `--no-fetch-back`), and upsert the catalog entry.
  - Flags: `--spec`, `--work`, `--repo Xephex/workbench-models`, `--catalog catalog.toml`, `--only ID…`, `--token-stdin` (reads one line from stdin).
- [ ] **Step 4: Run the tests.** All pass.
- [ ] **Step 5: Commit** `feat: model publisher: pinned download, reranker head check, slice pushes, fetch-back`.

`tools/publish.toml` holds the 7 models / 9 files from `notes/workbench/stage1-research/models.md` (revisions and sha256 copied from there), with tiers from spec B3:

| id | kinds | ram/vram GB (estimate until `tune` measures) | recommended_for |
|---|---|---|---|
| qwen3.5-0.8b | chat, vision | 2 / 1.5 | [] |
| qwen3.5-2b | chat, vision | 3 / 2.5 | ["cpu"] |
| qwen3-embed-4b | embeddings | 4 / 3 | ["gpu-16"] |
| qwen3-embed-8b | embeddings | 7 / 5.5 | [] |
| qwen3-rerank-0.6b | rerank | 2 / 0 | ["cpu", "gpu-8"] |
| qwen3-rerank-4b | rerank | 4 / 3 | ["gpu-16"] |
| qwen3-rerank-8b | rerank | 7 / 5.5 | [] |

---

### Task 4: llama.cpp wheels in CI (`.github/workflows/llama-wheels.yml`)

**Files:**
- Create: `.github/workflows/llama-wheels.yml`, `tools/smoke_llama.py`

**Interfaces:**
- Consumes:
  - JamePeng source at `528cf248460b49312597db98eb4f4e74a73214b8` (checkout with `repository: JamePeng/llama-cpp-python`, `submodules: recursive`).
  - The seed catalog's `qwen3-embed-0.6b` entry and `workbench.fetch`, for the smoke test.
- Produces: artifacts `llama-win-x64-vulkan`, `llama-linux-x64-vulkan`, `llama-linux-arm64-cpu`, each holding one `llama_cpp_python-0.4.2+<label>-cp314-cp314-<tag>.whl`. Task 6 passes those wheels to `build_packs --wheels`.

Jobs (workflow_dispatch):

- **linux-x64-vulkan**:
  - `ubuntu-24.04`, container `quay.io/pypa/manylinux_2_28_x86_64`, Python `/opt/python/cp314-cp314/bin/python`.
  - Vulkan SDK: micromamba `shaderc libvulkan-loader spirv-headers` (conda-forge), plus LunarG SDK `1.4.341.0` headers with sha256 `ed66477d587a5587dc3601b1c2cdcc1fab5529c505f53a00171876cecd9b4fbe` (upstream's recipe).
  - `CMAKE_ARGS="-DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DGGML_NATIVE=OFF -DGGML_VULKAN=ON -DGGML_OPENMP=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_OPENSSL=OFF -DCMAKE_PREFIX_PATH=/opt/vulkan -DVulkan_INCLUDE_DIR=… -DVulkan_LIBRARY=/opt/vulkan/lib/libvulkan.so -DVulkan_GLSLC_EXECUTABLE=/opt/vulkan/bin/glslc"`.
  - Version label via `sed -i 's/__version__ = "0.4.2"/__version__ = "0.4.2+vulkan"/' llama_cpp/__init__.py`.
  - `python -m build --wheel`, then `auditwheel repair --exclude libvulkan.so.1` (yields a `manylinux_2_28` tag).
- **linux-arm64-cpu**: `ubuntu-24.04-arm`, container `quay.io/pypa/manylinux_2_28_aarch64`, `CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8-a -DGGML_OPENMP=OFF -DLLAMA_OPENSSL=OFF"` plus the same `LLAMA_BUILD_*=OFF`, label `+cpu`, auditwheel repair.
- **win-x64-vulkan**:
  - `windows-2022`, `ilammy/msvc-dev-cmd@v1`, `choco install vulkan-sdk --version=1.4.341.0`, `actions/setup-python@v6` with `3.14`, `pip install build ninja`.
  - Generator `Ninja Multi-Config` with `-DCMAKE_TOOLCHAIN_FILE=vendor/llama.cpp/cmake/x64-windows-llvm.cmake` (JamePeng's: MSVC skips CPU variants), plus the DL/variants/Vulkan args with `-DGGML_OPENMP=OFF`.
  - Label `+vulkan`.
- **Each job's smoke test:**
  1. Install the wheel into a fresh `--target` dir.
  2. Fetch `qwen3-embed-0.6b` with `workbench.fetch` (cached by `actions/cache` keyed on its sha256).
  3. Run `tools/smoke_llama.py`, which:
     - prints `llama_print_system_info()`;
     - runs `LlamaEmbedding(model_path, n_gpu_layers=0).embed(["hello world", "bonjour"])`;
     - asserts 2 vectors of length 1024, finite values, norm ≈ 1.
  4. On Windows, also assert no `.exe` inside the installed `llama_cpp` package.

- [ ] **Step 1: Write `tools/smoke_llama.py`** and run it locally against the cu-wheel venv in `~/Work/scratch/llm-kit-kittest2` with the 0.6B embedder. Expected: `ok: 2 x 1024`.
- [ ] **Step 2: Write the workflow**, then push and dispatch it (`gh workflow run llama-wheels.yml`).
- [ ] **Step 3: Watch the run** (`gh run watch`). Expected: 3 green jobs, smoke lines `ok: 2 x 1024`, 3 artifacts. On failure, fix and re-run. Each rerun is about 10-15 minutes.
- [ ] **Step 4: Commit** `ci: build Vulkan and arm64 CPU llama.cpp wheels from the pinned JamePeng source`.

---

### Task 5: Publish the Stage 1 models and write the real catalog

**Files:**
- Modify: `catalog.toml` (7 new models, 9 files), `tests/test_catalog_chunks.py` (seed-catalog test → real-catalog test), `README.md` (models line)
- Create: `~/Work/notes/workbench/stage1-plan2/publish-log.md` (outside the repo)

**Interfaces:**
- Consumes: Task 3's tool and `tools/publish.toml`.
- Produces: `catalog.toml` with 14 models, whose new entries point at `Xephex/workbench-models`.

- [ ] **Step 1: Prepare miniRUD.**
  1. `ssh minirud` with `AGENT_SHELL=1 HISTFILE=/dev/null`.
  2. Clone the workbench repo at the pushed commit into `~/Work/scratch/workbench-publish/wb`. Work dir: `~/Work/scratch/workbench-publish/work`. 64 GB is free; the largest model needs about 9.4 GB (download plus parts), and parts are deleted after a verified push.
  3. Python: miniRUD's 3.13 runs the tool (stdlib only, `tomllib` present). `workbench.fetch` needs 3.10 or newer.
- [ ] **Step 2: Publish.**
  Run from the laptop:
  `printf 'protocol=https\nhost=github.com\n\n' | git credential fill | sed -n 's/^password=//p' | ssh minirud 'cd ~/Work/scratch/workbench-publish/wb && AGENT_SHELL=1 HISTFILE=/dev/null python3 tools/publish_model.py --spec tools/publish.toml --work ../work --repo Xephex/workbench-models --catalog catalog.toml --token-stdin'`.
  This runs in the background with a one-shot wakeup at the expected finish (about 18 GB: download at about 38 MB/s ≈ 8 min, plus upload and fetch-back). Expected output: `published <id>` and `fetch-back ok <file>` for all 9 files.
- [ ] **Step 3: Bring the catalog back.** `scp minirud:~/Work/scratch/workbench-publish/wb/catalog.toml catalog.toml`. Check the diff: only additions, 7 new `[[model]]` blocks.
- [ ] **Step 4: Update the catalog test.**

```python
def test_real_catalog(self):
    models = {m.id: m for m in catalog.load(ROOT / "catalog.toml")}
    self.assertEqual(len(models), 14)
    main = lambda tier: sum(f.bytes for m in models.values() if tier in m.recommended_for
                            for f in m.files if f.role == "main")
    self.assertAlmostEqual(main("cpu") / 1e9, 5.3, delta=0.05)    # spec B3 recommended sets
    self.assertAlmostEqual(main("gpu-8") / 1e9, 9.7, delta=0.05)
    self.assertAlmostEqual(main("gpu-16") / 1e9, 22.8, delta=0.05)
    for mid in ("qwen3-rerank-0.6b", "qwen3-rerank-4b", "qwen3-rerank-8b"):
        self.assertEqual(models[mid].files[0].repo, "Xephex/workbench-models")
```

- [ ] **Step 5: Spot-check from the laptop:** `python3.14 workbench.py models --add qwen3-rerank-0.6b` downloads from `workbench-models` and verifies, then `--remove` it.
- [ ] **Step 6: Run the tests** (all pass), then commit `feat: real Stage 1 catalog (14 models; new files on Xephex/workbench-models)`.

---

### Task 6: Build, verify and commit the packs; pack CI

**Files:**
- Create: `packs/<5 packs>/{core,llama,tracing}/…` (generated), `.github/workflows/packs.yml`
- Modify: `README.md` (setup now installs packages)

**Interfaces:**
- Consumes: Task 2's builder, Task 4's artifacts, Task 1's installer.
- Produces: a repo where `setup` works on Windows x64 and Linux x64/arm64.

- [ ] **Step 1: Download the CI wheels.** `gh run download <run> -D ~/Work/scratch/workbench-built/` (three wheels).
- [ ] **Step 2: Build every pack.**
  - Run `uv run --python 3.14 tools/build_packs.py all --pack <id> --wheels ~/Work/scratch/workbench-built` on miniRUD (fast GitHub upload; `uv python` downloads 3.14 there).
  - Expected: `verify OK` for each pack. Windows locks have 0 `monty` lines. Sizes are about 250 MB (core + tracing) per pack plus llama: cuda12 ≈ 730 MB, cuda13 ≈ 625 MB, vulkan/cpu ≈ 30 MB.
- [ ] **Step 3: Commit and push.**
  - One commit per pack (each push under 2 GB), using the token-over-stdin push.
  - Before each push, `git ls-remote` must show the expected parent.
  - The laptop then `git pull`s.
- [ ] **Step 4: Write `.github/workflows/packs.yml`.**
  - Matrix: `windows-latest`, `ubuntu-24.04`, `ubuntu-24.04-arm`; Python 3.14 via `actions/setup-python`.
  - Each job:
    1. `git clone --filter=blob:none --sparse` of the pushed commit, then `git checkout ${{ github.sha }}`.
    2. `python workbench.py setup --yes --packages-only` (prints the chosen pack; runners have no GPU, so they get `*-vulkan` / `linux-arm64-cpu`).
    3. Assert that `.workbench/site` exists, `import fastapi, gradio, llama_cpp, phoenix` works with the site activated, no `.exe` is under `.workbench/site`, and on Windows `pydantic_monty` is not importable.
  - A second step sets `WB_PACK=win-x64-cuda12` / `linux-x64-cuda13` (an override honoured only by `install_pack` callers in CI, through a `--pack` flag on `setup`). It installs the CUDA pack and imports `llama_cpp`: the CPU backend loads without a GPU.
- [ ] **Step 5: Push and watch.** Expected: green on all three runners.
- [ ] **Step 6: Commit** `ci: install every pack offline on Windows, Linux x64 and Linux arm64`.

`setup --pack ID` (Step 4) is added to `cli.py` with a test (`test_setup_pack_override`). It is also how a user forces the Vulkan pack on an NVIDIA machine.

---

## After this plan

Plan 3 (server and engines) consumes `install.activate()`, the installed `llama_cpp`, and the catalog's `recommended_for`.

Known follow-ups, deferred:
- The macOS pack.
- MTP model files.
- Moving the llm-kit models from `playground_stuff` to `workbench-models` (a catalog-only change).
- The RAM/VRAM estimates in Task 3's table, to be replaced by `tune` measurements (Plan 4).
