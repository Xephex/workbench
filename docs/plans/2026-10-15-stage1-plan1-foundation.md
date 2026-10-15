# Stage 1, Plan 1: Install foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `python workbench.py setup` (and `models`, `status`) detects the machine, chooses its tier and platform pack, and fetches and verifies the chosen models from GitHub by chunk blob ID, using only Python 3.14's standard library and git.

**Architecture:** A `workbench/` package of stdlib-only modules: `hw.py` (hardware profile via ctypes/winreg/sysfs/sysctl, never a subprocess on Windows), `catalog.py` (TOML catalog, tier filtering), `fetch.py` (git partial fetch of one blob at a time into a throwaway bare repo, streamed into the target file, sha256-checked, resumable), `packs.py` (platform pack choice), and `cli.py` (commands). Each module is tested without network or models; the fetcher's tests use a local git repo served over `file://`.

**Tech Stack:** Python 3.14 standard library only (`ctypes`, `winreg`, `tomllib`, `hashlib`, `subprocess` for git only, `unittest`), git ≥ 2.40.

**Spec:** `docs/specs/2026-10-15-workbench-design.md` (sections A3, A5, A6, A7, A12, B2, B3, B5).

**This is plan 1 of 4 for Stage 1:**
1. Install foundation: detection, catalog, fetch, CLI (this plan)
2. Packs and publishing: `tools/build_packs.py`, our Vulkan/CPU llama.cpp builds in CI, offline pack install with `.exe` removal, `tools/publish_model.py` run on miniRUD, the real Stage 1 catalog
3. Server and engines: FastAPI routes, model manager, llama.cpp adapter, toolcalls, tracing, commands that call the server
4. App, Pi, docs, CI: Gradio tabs, Pi, n8n, tune, monitor, selftest, `update`, three-OS CI

## Global Constraints

- Python 3.14 exactly (`sys.version_info[:2] == (3, 14)`); the launcher re-executes itself with a 3.14 interpreter when started with another version (spec A5).
- Standard library only in this plan. No third-party imports anywhere under `workbench/` or `tests/`.
- Only git talks to the network. No `urllib` or `http.client` calls in user-side code (spec R2).
- Windows: never start a process other than `git` (spec R6). Detection uses ctypes and winreg only.
- Chunk files are named `<file>.partNNN` (three digits) with a `<file>.sha256` sidecar `"<hex>  <name>\n"` (llm-kit `split_tool.py` format).
- Tiers: `cpu`, `gpu-8`, `gpu-16` (spec A7). Apple usable GPU memory = 0.66 × unified memory (spec B2).
- Pack ids: `win-x64-cuda12`, `win-x64-vulkan`, `linux-x64-cuda13`, `linux-x64-vulkan`, `linux-arm64-cpu`, `macos-arm64-metal` (spec B2/B4).
- `catalog.toml` uses TOML 1.0 only (Python 3.14's `tomllib`): no multi-line inline tables; each file is a `[[model.files]]` table and `chunks` an inline array on one line.
- Data that git must never track: `models/`, `data/`, `.workbench/` (spec A5).
- Commits: author `Xephex <Xephex@users.noreply.github.com>`, message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Never delete via a variable or glob that could be empty; deletes name literal targets or use `shutil.rmtree` on a path built from a fixed parent plus a validated name.

## Review Focus

1. **Interrupted model download** (network drop, Ctrl+C mid-chunk): re-running `models` must resume from the last whole chunk, never append a chunk twice, and never leave a file that looks finished. Tested in Task 5 (`test_resume_after_partial_chunk`).
2. **Not enough disk** for the chosen models: setup must say how much is needed and how much is free before downloading anything. Tested in Task 7 (`test_models_refuses_when_disk_short`).
3. **Machine with no GPU and no NVIDIA libraries at all** (most laptops): detection must return a valid `cpu` profile with no exceptions, on every OS. Tested in Task 2 (`test_profile_without_any_gpu_library`).
4. **NVIDIA driver too old for CUDA** (e.g. 535): must choose the Vulkan pack and say which driver version CUDA needs, not fail. Tested in Task 4 (`test_old_nvidia_driver_gets_vulkan_with_reason`).
5. **WSL with the 50% memory limit**: a model that fits the host but not WSL must be shown greyed with the `.wslconfig` hint, not offered. Tested in Task 3 (`test_wsl_reason_mentions_wslconfig`).

---

## File Structure

```
workbench.py                    launcher: Python 3.14 check/re-exec, then workbench.cli.main()
workbench/__init__.py           version string
workbench/hw.py                 Profile/Gpu dataclasses, detect(), per-OS probes, tier_for()
workbench/catalog.py            Model/ModelFile dataclasses, load(), offered(), recommended()
workbench/packs.py              choose_pack(profile) -> PackChoice
workbench/fetch.py              fetch_file(), fetch_model(), ChunkSource, FetchError
workbench/ui.py                 say(), ask(), gb(): plain terminal output
workbench/cli.py                argparse commands: status, models, setup
catalog.toml                    seed catalog: the 11 llm-kit models (blob IDs from playground_stuff)
tools/catalog_chunks.py         maintainer tool: fill chunk blob IDs/sizes from the GitHub tree API
tests/fixtures/                 recorded probe data (sysfs trees, meminfo, osrelease)
tests/test_hw.py
tests/test_catalog.py
tests/test_packs.py
tests/test_fetch.py
tests/test_cli.py
tests/gitfixture.py             builds a local git repo with chunked files for fetch tests
.gitignore
```

Run all tests with: `python3.14 -m unittest discover -s tests -v` from the repo root.

---

### Task 1: Launcher, package skeleton, terminal helpers

**Files:**
- Create: `workbench.py`, `workbench/__init__.py`, `workbench/ui.py`, `.gitignore`, `tests/__init__.py`, `tests/test_ui.py`

**Interfaces:**
- Produces: `workbench.__version__: str`; `workbench.ui.say(msg: str = "") -> None`; `workbench.ui.ask(prompt: str, default: str = "") -> str`; `workbench.ui.gb(n_bytes: int) -> str` (e.g. `gb(5_300_000_000) == "5.3 GB"`); `workbench.py` calls `workbench.cli.main(argv: list[str]) -> int`.

- [ ] **Step 1: Write the failing test**

`tests/__init__.py`: empty file.

`tests/test_ui.py`:
```python
import io, unittest
from unittest import mock

from workbench import ui


class UiTest(unittest.TestCase):
    def test_gb_formats_one_decimal(self):
        self.assertEqual(ui.gb(5_300_000_000), "5.3 GB")
        self.assertEqual(ui.gb(639_150_592), "0.6 GB")
        self.assertEqual(ui.gb(0), "0.0 GB")

    def test_ask_returns_default_on_empty_input(self):
        with mock.patch("builtins.input", return_value=""):
            self.assertEqual(ui.ask("Pick", "y"), "y")

    def test_ask_returns_default_on_eof(self):
        with mock.patch("builtins.input", side_effect=EOFError):
            self.assertEqual(ui.ask("Pick", "n"), "n")

    def test_say_prints_line(self):
        buf = io.StringIO()
        with mock.patch("sys.stdout", buf):
            ui.say("hello")
        self.assertEqual(buf.getvalue(), "hello\n")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3.14 -m unittest tests.test_ui -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'workbench'`

- [ ] **Step 3: Write minimal implementation**

`workbench/__init__.py`:
```python
"""Offline LLM workbench."""

__version__ = "0.1.0"
```

`workbench/ui.py`:
```python
"""Plain terminal output. Every user-facing message goes through say()."""

import sys


def say(msg: str = "") -> None:
    print(msg, flush=True)


def ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        answer = input(f"{prompt}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    return answer or default


def gb(n_bytes: int) -> str:
    return f"{n_bytes / 1e9:.1f} GB"
```

`workbench.py`:
```python
r"""Offline LLM workbench.

    python workbench.py setup      detect this machine, install, choose and fetch models
    python workbench.py models     add or remove models
    python workbench.py status     what this machine is and what is installed

On Windows use:  py -3.14 workbench.py setup
"""

import os, pathlib, shutil, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parent
NEED = (3, 14)


def find_python_314():
    """Return a command list for an installed Python 3.14, or None."""
    cands = []
    if shutil.which("py"):
        cands.append(["py", "-3.14"])
    for name in ("python3.14", "python3", "python"):
        p = shutil.which(name)
        if p:
            cands.append([p])
    for c in cands:
        try:
            out = subprocess.run([*c, "-c", "import sys;print(sys.version_info[:2]==(3,14))"],
                                 capture_output=True, text=True, timeout=20).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            continue
        if out == "True":
            return c
    return None


def ensure_python():
    if sys.version_info[:2] == NEED:
        return
    other = find_python_314()
    if other and not os.environ.get("WORKBENCH_REEXEC"):
        print(f"You started this with Python {sys.version.split()[0]}; switching to Python 3.14.", flush=True)
        env = dict(os.environ, WORKBENCH_REEXEC="1")
        sys.exit(subprocess.run([*other, str(ROOT / "workbench.py"), *sys.argv[1:]], env=env).returncode)
    print(f"The workbench needs Python 3.14. You are on {sys.version.split()[0]}, and no Python 3.14 was found.")
    print("Install Python 3.14 (python.org, or the Microsoft Store on Windows), then run:")
    print("  py -3.14 workbench.py setup      (Windows)")
    print("  python3.14 workbench.py setup    (Linux, macOS)")
    sys.exit(1)


if __name__ == "__main__":
    ensure_python()
    sys.path.insert(0, str(ROOT))
    from workbench import cli

    sys.exit(cli.main(sys.argv[1:]))
```

`.gitignore`:
```
/models/
/data/
/.workbench/
__pycache__/
*.pyc
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3.14 -m unittest tests.test_ui -v`
Expected: 4 tests, OK

- [ ] **Step 5: Commit**

```bash
git add workbench.py workbench/__init__.py workbench/ui.py .gitignore tests/__init__.py tests/test_ui.py
git -c user.name=Xephex -c user.email=Xephex@users.noreply.github.com commit -m "feat: launcher with Python 3.14 re-exec and terminal helpers

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Hardware detection (`hw.py`)

**Files:**
- Create: `workbench/hw.py`, `tests/test_hw.py`, `tests/fixtures/meminfo.txt`, `tests/fixtures/osrelease-wsl.txt`

**Interfaces:**
- Produces:
  - `@dataclass class Gpu: vendor: str` (`"nvidia" | "amd" | "intel" | "apple" | "other"`), `name: str`, `vram_bytes: int | None`, `driver: str | None`, `cuda_driver: int | None` (e.g. `13010` = CUDA 13.1), `compute_capability: tuple[int, int] | None`, `integrated: bool`, `source: str` (`"nvml" | "cuda" | "dxgi" | "registry" | "sysfs" | "sysctl"`)
  - `@dataclass class Profile: os: str` (`"windows" | "linux" | "macos"`), `arch: str` (`"x86_64" | "arm64"`), `wsl: bool`, `rosetta: bool`, `cpu_name: str`, `cpu_cores: int`, `ram_bytes: int`, `ram_available_bytes: int | None`, `disk_free_bytes: int`, `gpus: list[Gpu]`, `vulkan_loader: bool`; property `gpu_budget_bytes -> int` (largest usable GPU memory: dedicated VRAM of a non-integrated GPU, or 0.66 × RAM on Apple, else 0); property `tier -> str`
  - `detect(models_dir: pathlib.Path, plat: str | None = None) -> Profile` (`plat` overrides OS detection; tests use it)
  - `tier_for(gpu_budget_bytes: int, os: str) -> str`
  - Probe functions, each returning `list[Gpu]` and never raising: `probe_nvml() -> list[Gpu]`, `probe_cuda() -> list[Gpu]`, `probe_dxgi() -> list[Gpu]`, `probe_registry() -> list[Gpu]`, `probe_sysfs(root: pathlib.Path = pathlib.Path("/sys/class/drm")) -> list[Gpu]`, `probe_apple() -> list[Gpu]`
  - Helpers: `read_meminfo(path: pathlib.Path = pathlib.Path("/proc/meminfo")) -> tuple[int, int | None]` (total, available); `is_wsl(osrelease: pathlib.Path = pathlib.Path("/proc/sys/kernel/osrelease")) -> bool`

Tier rule (spec A7): `gpu-16` if budget ≥ 11.5 GB; `gpu-8` if budget ≥ 5.5 GB; else `cpu`. (Thresholds slightly under 12/6 GB because cards report e.g. 11.99 GB or 7.96 GB.)

- [ ] **Step 1: Write the failing tests**

`tests/fixtures/meminfo.txt`:
```
MemTotal:       65574956 kB
MemFree:         5123456 kB
MemAvailable:   38114240 kB
Buffers:          123456 kB
```

`tests/fixtures/osrelease-wsl.txt`:
```
6.6.87.2-microsoft-standard-WSL2
```

`tests/test_hw.py`:
```python
import pathlib, tempfile, unittest
from unittest import mock

from workbench import hw

FIX = pathlib.Path(__file__).parent / "fixtures"
GB = 1_000_000_000


def make_sysfs(root: pathlib.Path, cards: dict) -> None:
    """cards: {"card0": {"vendor": "0x1002", "device": "0x73df", "driver": "amdgpu", "vram": 12*GB}}"""
    for card, info in cards.items():
        dev = root / card / "device"
        dev.mkdir(parents=True)
        (dev / "vendor").write_text(info["vendor"] + "\n")
        (dev / "device").write_text(info["device"] + "\n")
        drv = root / "drivers" / info["driver"]
        drv.mkdir(parents=True, exist_ok=True)
        (dev / "driver").symlink_to(drv)
        if "vram" in info:
            (dev / "mem_info_vram_total").write_text(f"{info['vram']}\n")
    (root / "card0-HDMI-A-1").mkdir()  # connector entry, must be skipped
    (root / "renderD128").mkdir()       # render node, must be skipped


class TierTest(unittest.TestCase):
    def test_tiers(self):
        self.assertEqual(hw.tier_for(0, "windows"), "cpu")
        self.assertEqual(hw.tier_for(4 * GB, "linux"), "cpu")
        self.assertEqual(hw.tier_for(int(7.96 * GB), "windows"), "gpu-8")
        self.assertEqual(hw.tier_for(int(11.99 * GB), "linux"), "gpu-16")
        self.assertEqual(hw.tier_for(16 * GB, "macos"), "gpu-16")


class BudgetTest(unittest.TestCase):
    def profile(self, **kw):
        base = dict(os="linux", arch="x86_64", wsl=False, rosetta=False, cpu_name="x", cpu_cores=8,
                    ram_bytes=32 * GB, ram_available_bytes=None, disk_free_bytes=500 * GB, gpus=[],
                    vulkan_loader=False)
        base.update(kw)
        return hw.Profile(**base)

    def test_integrated_gpu_does_not_count(self):
        igpu = hw.Gpu("intel", "Intel UHD 620", 512 * 1024**2, None, None, None, True, "sysfs")
        self.assertEqual(self.profile(gpus=[igpu]).gpu_budget_bytes, 0)
        self.assertEqual(self.profile(gpus=[igpu]).tier, "cpu")

    def test_dedicated_gpu_counts(self):
        dgpu = hw.Gpu("nvidia", "RTX PRO 2000", 8 * GB, "581.42", 13010, (12, 0), False, "nvml")
        self.assertEqual(self.profile(gpus=[dgpu]).tier, "gpu-8")

    def test_apple_uses_two_thirds_of_unified_memory(self):
        apple = hw.Gpu("apple", "Apple M3 Pro", None, None, None, None, True, "sysctl")
        p = self.profile(os="macos", arch="arm64", ram_bytes=36 * GB, gpus=[apple])
        self.assertEqual(p.gpu_budget_bytes, int(36 * GB * 0.66))
        self.assertEqual(p.tier, "gpu-16")
        p = self.profile(os="macos", arch="arm64", ram_bytes=16 * GB, gpus=[apple])
        self.assertEqual(p.tier, "gpu-8")


class LinuxProbeTest(unittest.TestCase):
    def test_read_meminfo(self):
        total, avail = hw.read_meminfo(FIX / "meminfo.txt")
        self.assertEqual(total, 65574956 * 1024)
        self.assertEqual(avail, 38114240 * 1024)

    def test_is_wsl(self):
        self.assertTrue(hw.is_wsl(FIX / "osrelease-wsl.txt"))
        self.assertFalse(hw.is_wsl(FIX / "meminfo.txt"))
        self.assertFalse(hw.is_wsl(FIX / "does-not-exist"))

    def test_sysfs_amd_with_vram(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            make_sysfs(root, {"card0": {"vendor": "0x1002", "device": "0x73df", "driver": "amdgpu", "vram": 12 * GB}})
            gpus = hw.probe_sysfs(root)
        self.assertEqual(len(gpus), 1)
        self.assertEqual(gpus[0].vendor, "amd")
        self.assertEqual(gpus[0].vram_bytes, 12 * GB)
        self.assertFalse(gpus[0].integrated)

    def test_sysfs_intel_igpu_has_unknown_vram_and_is_integrated(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            make_sysfs(root, {"card1": {"vendor": "0x8086", "device": "0x5917", "driver": "i915"}})
            gpus = hw.probe_sysfs(root)
        self.assertEqual([(g.vendor, g.vram_bytes, g.integrated) for g in gpus], [("intel", None, True)])

    def test_sysfs_skips_nvidia_left_to_nvml(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            make_sysfs(root, {"card0": {"vendor": "0x10de", "device": "0x2c38", "driver": "nvidia"}})
            self.assertEqual(hw.probe_sysfs(root), [])

    def test_sysfs_missing_root(self):
        self.assertEqual(hw.probe_sysfs(pathlib.Path("/nonexistent/drm")), [])


class DetectTest(unittest.TestCase):
    def test_profile_without_any_gpu_library(self):
        """Review focus 3: a plain laptop with no GPU libraries must give a valid cpu profile."""
        with tempfile.TemporaryDirectory() as d, \
             mock.patch.object(hw, "probe_nvml", return_value=[]), \
             mock.patch.object(hw, "probe_cuda", return_value=[]), \
             mock.patch.object(hw, "probe_dxgi", return_value=[]), \
             mock.patch.object(hw, "probe_registry", return_value=[]), \
             mock.patch.object(hw, "probe_sysfs", return_value=[]), \
             mock.patch.object(hw, "probe_apple", return_value=[]):
            p = hw.detect(pathlib.Path(d))
        self.assertEqual(p.gpus, [])
        self.assertEqual(p.tier, "cpu")
        self.assertGreater(p.ram_bytes, 0)
        self.assertGreater(p.disk_free_bytes, 0)
        self.assertIn(p.os, ("windows", "linux", "macos"))

    def test_real_probes_never_raise_on_this_machine(self):
        with tempfile.TemporaryDirectory() as d:
            p = hw.detect(pathlib.Path(d))
        self.assertIsInstance(p.gpus, list)

    def test_windows_nvml_result_suppresses_duplicate_nvidia_from_dxgi(self):
        nv = hw.Gpu("nvidia", "RTX PRO 2000", 8 * GB, "581.42", 13010, (12, 0), False, "nvml")
        dx_nv = hw.Gpu("nvidia", "NVIDIA RTX PRO 2000", 8 * GB, None, None, None, False, "dxgi")
        dx_intel = hw.Gpu("intel", "Intel Arc 130T", None, None, None, None, True, "dxgi")
        with tempfile.TemporaryDirectory() as d, \
             mock.patch.object(hw, "probe_nvml", return_value=[nv]), \
             mock.patch.object(hw, "probe_cuda", return_value=[]), \
             mock.patch.object(hw, "probe_dxgi", return_value=[dx_nv, dx_intel]), \
             mock.patch.object(hw, "probe_registry", return_value=[]), \
             mock.patch.object(hw, "_windows_ram", return_value=(64 * GB, 40 * GB)), \
             mock.patch.object(hw, "_cpu_name", return_value="Intel Core Ultra 7 265H"), \
             mock.patch.object(hw, "_vulkan_loader", return_value=True):
            p = hw.detect(pathlib.Path(d), plat="windows")
        self.assertEqual([(g.vendor, g.source) for g in p.gpus], [("nvidia", "nvml"), ("intel", "dxgi")])
        self.assertEqual(p.tier, "gpu-8")
        self.assertEqual(p.os, "windows")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3.14 -m unittest tests.test_hw -v`
Expected: FAIL with `ImportError: cannot import name 'hw'`

- [ ] **Step 3: Write the implementation**

`workbench/hw.py`:
```python
"""Hardware profile, using only the standard library and ctypes.

Never starts a process. Every probe is optional: a probe that cannot run
returns [] and detection carries on.
"""

from __future__ import annotations

import ctypes, ctypes.util, dataclasses, os, pathlib, platform, shutil, sys

GB = 1_000_000_000
APPLE_GPU_SHARE = 0.66
TIER_16 = int(11.5 * GB)
TIER_8 = int(5.5 * GB)
PCI_VENDORS = {0x10DE: "nvidia", 0x1002: "amd", 0x8086: "intel"}


@dataclasses.dataclass
class Gpu:
    vendor: str
    name: str
    vram_bytes: int | None
    driver: str | None
    cuda_driver: int | None
    compute_capability: tuple[int, int] | None
    integrated: bool
    source: str


@dataclasses.dataclass
class Profile:
    os: str
    arch: str
    wsl: bool
    rosetta: bool
    cpu_name: str
    cpu_cores: int
    ram_bytes: int
    ram_available_bytes: int | None
    disk_free_bytes: int
    gpus: list[Gpu]
    vulkan_loader: bool

    @property
    def gpu_budget_bytes(self) -> int:
        best = 0
        for g in self.gpus:
            if g.vendor == "apple":
                best = max(best, int(self.ram_bytes * APPLE_GPU_SHARE))
            elif not g.integrated and g.vram_bytes:
                best = max(best, g.vram_bytes)
        return best

    @property
    def tier(self) -> str:
        return tier_for(self.gpu_budget_bytes, self.os)


def tier_for(gpu_budget_bytes: int, os: str) -> str:
    if gpu_budget_bytes >= TIER_16:
        return "gpu-16"
    if gpu_budget_bytes >= TIER_8:
        return "gpu-8"
    return "cpu"


# ---------- Linux files ----------

def read_meminfo(path: pathlib.Path = pathlib.Path("/proc/meminfo")) -> tuple[int, int | None]:
    total = avail = None
    for line in path.read_text().splitlines():
        key, _, rest = line.partition(":")
        if key == "MemTotal":
            total = int(rest.split()[0]) * 1024
        elif key == "MemAvailable":
            avail = int(rest.split()[0]) * 1024
    return total or 0, avail


def is_wsl(osrelease: pathlib.Path = pathlib.Path("/proc/sys/kernel/osrelease")) -> bool:
    if os.environ.get("WSL_DISTRO_NAME") and osrelease == pathlib.Path("/proc/sys/kernel/osrelease"):
        return True
    try:
        return "microsoft" in osrelease.read_text().lower()
    except OSError:
        return False


def probe_sysfs(root: pathlib.Path = pathlib.Path("/sys/class/drm")) -> list[Gpu]:
    """AMD and Intel GPUs from /sys/class/drm. NVIDIA is left to NVML."""
    gpus = []
    try:
        cards = sorted(p for p in root.iterdir() if p.name.startswith("card") and p.name[4:].isdigit())
    except OSError:
        return []
    for card in cards:
        dev = card / "device"
        try:
            vendor_id = int((dev / "vendor").read_text().strip(), 16)
        except (OSError, ValueError):
            continue
        vendor = PCI_VENDORS.get(vendor_id, "other")
        if vendor == "nvidia":
            continue
        try:
            driver = os.path.basename(os.readlink(dev / "driver"))
        except OSError:
            driver = None
        vram = None
        try:
            vram = int((dev / "mem_info_vram_total").read_text().strip())
        except (OSError, ValueError):
            pass
        # No VRAM figure (Intel) or a small BIOS carve-out (AMD APU): treat as integrated.
        integrated = vram is None or vram < 2 * GB
        gpus.append(Gpu(vendor, f"{vendor} {card.name}", vram, driver, None, None, integrated, "sysfs"))
    return gpus


# ---------- NVIDIA (Windows and Linux) ----------

class _NvmlMemory(ctypes.Structure):
    _fields_ = [("total", ctypes.c_uint64), ("free", ctypes.c_uint64), ("used", ctypes.c_uint64)]


def _load_nvml():
    if sys.platform == "win32":
        cands = [os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "System32", "nvml.dll"),
                 os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"),
                              "NVIDIA Corporation", "NVSMI", "nvml.dll")]
    else:
        cands = ["libnvidia-ml.so.1", "/usr/lib/wsl/lib/libnvidia-ml.so.1"]
    for c in cands:
        try:
            return ctypes.CDLL(c)
        except OSError:
            continue
    return None


def probe_nvml() -> list[Gpu]:
    lib = _load_nvml()
    if lib is None:
        return []
    try:
        lib.nvmlInit_v2.restype = ctypes.c_int
        if lib.nvmlInit_v2() != 0:
            return []
        try:
            buf = ctypes.create_string_buffer(96)
            lib.nvmlSystemGetDriverVersion.argtypes = [ctypes.c_char_p, ctypes.c_uint]
            driver = buf.value.decode() if lib.nvmlSystemGetDriverVersion(buf, 96) == 0 else None
            cuda = ctypes.c_int(0)
            lib.nvmlSystemGetCudaDriverVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
            cuda_driver = cuda.value if lib.nvmlSystemGetCudaDriverVersion(ctypes.byref(cuda)) == 0 else None
            count = ctypes.c_uint(0)
            lib.nvmlDeviceGetCount_v2.argtypes = [ctypes.POINTER(ctypes.c_uint)]
            if lib.nvmlDeviceGetCount_v2(ctypes.byref(count)) != 0:
                return []
            lib.nvmlDeviceGetHandleByIndex_v2.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
            lib.nvmlDeviceGetName.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint]
            lib.nvmlDeviceGetMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(_NvmlMemory)]
            lib.nvmlDeviceGetCudaComputeCapability.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
            gpus = []
            for i in range(count.value):
                h = ctypes.c_void_p()
                if lib.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(h)) != 0:
                    continue
                name = ctypes.create_string_buffer(96)
                lib.nvmlDeviceGetName(h, name, 96)
                mem = _NvmlMemory()
                vram = mem.total if lib.nvmlDeviceGetMemoryInfo(h, ctypes.byref(mem)) == 0 else None
                ma, mi = ctypes.c_int(0), ctypes.c_int(0)
                cc = (ma.value, mi.value) if lib.nvmlDeviceGetCudaComputeCapability(
                    h, ctypes.byref(ma), ctypes.byref(mi)) == 0 else None
                gpus.append(Gpu("nvidia", name.value.decode(errors="replace"), vram, driver,
                                cuda_driver, cc, False, "nvml"))
            return gpus
        finally:
            lib.nvmlShutdown()
    except (AttributeError, OSError):
        return []


def probe_cuda() -> list[Gpu]:
    """Fallback when NVML is missing but the CUDA driver library exists."""
    names = ["nvcuda.dll"] if sys.platform == "win32" else ["libcuda.so.1", "/usr/lib/wsl/lib/libcuda.so.1"]
    lib = None
    for n in names:
        try:
            lib = ctypes.CDLL(n)
            break
        except OSError:
            continue
    if lib is None:
        return []
    try:
        if lib.cuInit(0) != 0:
            return []
        ver, count = ctypes.c_int(0), ctypes.c_int(0)
        lib.cuDriverGetVersion(ctypes.byref(ver))
        if lib.cuDeviceGetCount(ctypes.byref(count)) != 0:
            return []
        lib.cuDeviceTotalMem_v2.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.c_int]
        gpus = []
        for i in range(count.value):
            dev = ctypes.c_int(0)
            if lib.cuDeviceGet(ctypes.byref(dev), i) != 0:
                continue
            name = ctypes.create_string_buffer(256)
            lib.cuDeviceGetName(name, 256, dev)
            total = ctypes.c_size_t(0)
            vram = total.value if lib.cuDeviceTotalMem_v2(ctypes.byref(total), dev) == 0 else None
            ma, mi = ctypes.c_int(0), ctypes.c_int(0)
            lib.cuDeviceGetAttribute(ctypes.byref(ma), 75, dev)
            lib.cuDeviceGetAttribute(ctypes.byref(mi), 76, dev)
            gpus.append(Gpu("nvidia", name.value.decode(errors="replace"), vram, None, ver.value,
                            (ma.value, mi.value), False, "cuda"))
        return gpus
    except (AttributeError, OSError):
        return []


# ---------- Windows: DXGI and registry ----------

class _Guid(ctypes.Structure):
    _fields_ = [("d1", ctypes.c_uint32), ("d2", ctypes.c_uint16), ("d3", ctypes.c_uint16),
                ("d4", ctypes.c_uint8 * 8)]


class _AdapterDesc1(ctypes.Structure):
    _fields_ = [("Description", ctypes.c_wchar * 128), ("VendorId", ctypes.c_uint32),
                ("DeviceId", ctypes.c_uint32), ("SubSysId", ctypes.c_uint32), ("Revision", ctypes.c_uint32),
                ("DedicatedVideoMemory", ctypes.c_size_t), ("DedicatedSystemMemory", ctypes.c_size_t),
                ("SharedSystemMemory", ctypes.c_size_t), ("AdapterLuid", ctypes.c_uint64),
                ("Flags", ctypes.c_uint32)]


def _vtable_call(obj, slot, restype, *argtypes):
    vtbl = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    return ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(vtbl[slot])


def probe_dxgi() -> list[Gpu]:
    if sys.platform != "win32":
        return []
    try:
        dxgi = ctypes.WinDLL("dxgi.dll")
        iid = _Guid(0x770AAE78, 0xF26F, 0x4DBA, (ctypes.c_uint8 * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87))
        factory = ctypes.c_void_p()
        dxgi.CreateDXGIFactory1.argtypes = [ctypes.POINTER(_Guid), ctypes.POINTER(ctypes.c_void_p)]
        if dxgi.CreateDXGIFactory1(ctypes.byref(iid), ctypes.byref(factory)) != 0:
            return []
        gpus = []
        enum = _vtable_call(factory, 12, ctypes.c_long, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p))
        i = 0
        while True:
            adapter = ctypes.c_void_p()
            if enum(factory, i, ctypes.byref(adapter)) != 0:
                break
            desc = _AdapterDesc1()
            get_desc = _vtable_call(adapter, 10, ctypes.c_long, ctypes.POINTER(_AdapterDesc1))
            if get_desc(adapter, ctypes.byref(desc)) == 0 and not desc.Flags & 2:
                vendor = PCI_VENDORS.get(desc.VendorId, "other")
                vram = desc.DedicatedVideoMemory or None
                gpus.append(Gpu(vendor, desc.Description, vram, None, None, None,
                                vram is None or vram < 2 * GB, "dxgi"))
            _vtable_call(adapter, 2, ctypes.c_ulong)(adapter)
            i += 1
        _vtable_call(factory, 2, ctypes.c_ulong)(factory)
        return gpus
    except (OSError, AttributeError, ValueError):
        return []


def probe_registry() -> list[Gpu]:
    if sys.platform != "win32":
        return []
    try:
        import winreg
    except ImportError:
        return []
    key = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
    gpus = []
    try:
        root = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key)
    except OSError:
        return []
    with root:
        i = 0
        while True:
            try:
                sub = winreg.EnumKey(root, i)
            except OSError:
                break
            i += 1
            if not sub.isdigit():
                continue
            try:
                with winreg.OpenKey(root, sub) as k:
                    def val(name):
                        try:
                            return winreg.QueryValueEx(k, name)[0]
                        except OSError:
                            return None
                    desc = val("DriverDesc") or "unknown"
                    match = (val("MatchingDeviceId") or "").lower()
                    mem = val("HardwareInformation.qwMemorySize")
                    if isinstance(mem, bytes):
                        mem = int.from_bytes(mem, "little")
                    vendor = "other"
                    for vid, name in PCI_VENDORS.items():
                        if f"ven_{vid:04x}" in match:
                            vendor = name
                    vram = mem if isinstance(mem, int) and mem > 0 else None
                    gpus.append(Gpu(vendor, desc, vram, val("DriverVersion"), None, None,
                                    vram is None or vram < 2 * GB, "registry"))
            except OSError:
                continue
    return gpus


def _windows_ram() -> tuple[int, int | None]:
    class MemStatus(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32),
                    ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64), ("ullAvailVirtual", ctypes.c_uint64),
                    ("ullAvailExtendedVirtual", ctypes.c_uint64)]
    m = MemStatus()
    m.dwLength = ctypes.sizeof(MemStatus)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullTotalPhys, m.ullAvailPhys


# ---------- macOS ----------

def _sysctl(name: str, as_int: bool):
    libc = ctypes.CDLL(ctypes.util.find_library("c"))
    libc.sysctlbyname.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
                                  ctypes.c_void_p, ctypes.c_size_t]
    size = ctypes.c_size_t(0)
    if libc.sysctlbyname(name.encode(), None, ctypes.byref(size), None, 0) != 0:
        return None
    buf = ctypes.create_string_buffer(size.value)
    if libc.sysctlbyname(name.encode(), buf, ctypes.byref(size), None, 0) != 0:
        return None
    if as_int:
        return int.from_bytes(buf.raw[: size.value], "little")
    return buf.value.decode(errors="replace")


def probe_apple() -> list[Gpu]:
    if sys.platform != "darwin":
        return []
    try:
        if _sysctl("hw.optional.arm64", True) != 1:
            return []
        name = _sysctl("machdep.cpu.brand_string", False) or _sysctl("hw.model", False) or "Apple Silicon"
        return [Gpu("apple", name, None, None, None, None, True, "sysctl")]
    except OSError:
        return []


# ---------- detect ----------

def _vulkan_loader() -> bool:
    name = "vulkan-1.dll" if sys.platform == "win32" else ("libvulkan.so.1" if sys.platform != "darwin" else None)
    if not name:
        return False
    try:
        ctypes.CDLL(name)
        return True
    except OSError:
        return False


def _cpu_name(plat: str) -> str:
    """CPU model name without starting a process (platform.processor() may run uname or sysctl)."""
    if plat == "windows":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
        except (ImportError, OSError):
            return os.environ.get("PROCESSOR_IDENTIFIER", "unknown CPU")
    if plat == "macos":
        try:
            return _sysctl("machdep.cpu.brand_string", False) or "Apple Silicon"
        except OSError:
            return "Apple Silicon"
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith(("model name", "Model")):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown CPU"


def detect(models_dir: pathlib.Path, plat: str | None = None) -> Profile:
    plat = plat or {"win32": "windows", "darwin": "macos"}.get(sys.platform, "linux")
    machine = platform.machine().lower()
    arch = "arm64" if machine in ("arm64", "aarch64") else "x86_64"
    rosetta = False
    cpu_name = _cpu_name(plat)
    if plat == "windows":
        ram, avail = _windows_ram()
        gpus = probe_nvml() or probe_cuda()
        others = [g for g in (probe_dxgi() or probe_registry()) if not (gpus and g.vendor == "nvidia")]
        gpus = gpus + others
    elif plat == "macos":
        ram = _sysctl("hw.memsize", True) or 0
        avail = None
        rosetta = _sysctl("sysctl.proc_translated", True) == 1
        gpus = probe_apple()
    else:
        ram, avail = read_meminfo()
        gpus = (probe_nvml() or probe_cuda()) + probe_sysfs()
    probe_dir = models_dir if models_dir.exists() else models_dir.parent
    while not probe_dir.exists() and probe_dir != probe_dir.parent:
        probe_dir = probe_dir.parent
    return Profile(
        os=plat, arch=arch, wsl=plat == "linux" and is_wsl(), rosetta=rosetta,
        cpu_name=cpu_name, cpu_cores=os.cpu_count() or 1,
        ram_bytes=ram, ram_available_bytes=avail,
        disk_free_bytes=shutil.disk_usage(probe_dir).free,
        gpus=gpus, vulkan_loader=_vulkan_loader(),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3.14 -m unittest tests.test_hw -v`
Expected: 13 tests, OK

- [ ] **Step 5: Print this machine's profile as a smoke check**

Run: `python3.14 -c "import pathlib,pprint; from workbench import hw; p=hw.detect(pathlib.Path('models')); pprint.pprint(p); print(p.tier)"`
Expected: a `Profile(...)` with `os='linux'`, real `ram_bytes`, an Intel `card` entry from sysfs on this laptop, `tier` `cpu`; no traceback.

- [ ] **Step 6: Commit**

```bash
git add workbench/hw.py tests/test_hw.py tests/fixtures/meminfo.txt tests/fixtures/osrelease-wsl.txt
git -c user.name=Xephex -c user.email=Xephex@users.noreply.github.com commit -m "feat: hardware detection with ctypes, winreg, sysfs and sysctl (no subprocess)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Catalog (`catalog.py`) and tier filtering

**Files:**
- Create: `workbench/catalog.py`, `tests/test_catalog.py`, `tests/fixtures/catalog-small.toml`

**Interfaces:**
- Consumes: `hw.Profile` (Task 2).
- Produces:
  - `@dataclass(frozen=True) class Chunk: blob: str, bytes: int`
  - `@dataclass(frozen=True) class ModelFile: name: str, sha256: str, bytes: int, repo: str, chunks: tuple[Chunk, ...], role: str` (`"main" | "mmproj"`)
  - `@dataclass(frozen=True) class Model: id: str, kinds: tuple[str, ...], engine: str, files: tuple[ModelFile, ...], ram_gb: float, vram_gb: float, recommended_for: tuple[str, ...], license: str, source: str`; property `total_bytes -> int`; property `folder -> str` (equals `id`)
  - `load(path: pathlib.Path) -> list[Model]` (raises `CatalogError` with the model id and field on bad entries; checks that chunk byte counts add up to the file size)
  - `@dataclass class Offer: model: Model, runnable: bool, reason: str | None`
  - `offered(models: list[Model], profile: hw.Profile) -> list[Offer]` (every model, with runnable/reason, in catalog order)
  - `recommended(offers: list[Offer], tier: str) -> list[Model]`
  - `class CatalogError(ValueError)`

Rules for `offered` (spec A7/B2): a model is runnable when `vram_gb == 0` and `ram_bytes ≥ ram_gb`, or the GPU budget ≥ `vram_gb` and RAM ≥ `ram_gb`. On the `cpu` tier, a model with `vram_gb > 0` is still runnable on CPU when RAM ≥ `ram_gb + vram_gb` (GPU models run from RAM, slowly) but its reason says `"runs on CPU only; slow"`. Not runnable reasons: `"needs {x} GB of video memory; this machine has {y}"` or `"needs {x} GB of RAM; this machine has {y}"`. On WSL, when RAM is the limit and the model would fit 2 × RAM (WSL's default 50% split), the reason adds `"; WSL can use more memory if you raise memory= in %UserProfile%\\.wslconfig"`.

- [ ] **Step 1: Write the failing tests**

`tests/fixtures/catalog-small.toml`:
```toml
[[model]]
id = "tiny-chat"
kinds = ["chat"]
engine = "llama.cpp"
ram_gb = 2
vram_gb = 0
recommended_for = ["cpu", "gpu-8"]
license = "Apache-2.0"
source = "example/tiny"

[[model.files]]
name = "tiny.gguf"
role = "main"
sha256 = "aa"
bytes = 150
repo = "Xephex/workbench-models"
chunks = [{ blob = "b1", bytes = 100 }, { blob = "b2", bytes = 50 }]

[[model]]
id = "mid-chat"
kinds = ["chat", "vision"]
engine = "llama.cpp"
ram_gb = 8
vram_gb = 7
recommended_for = ["gpu-8", "gpu-16"]
license = "Apache-2.0"
source = "example/mid"

[[model.files]]
name = "mid.gguf"
role = "main"
sha256 = "bb"
bytes = 10
repo = "Xephex/workbench-models"
chunks = [{ blob = "b3", bytes = 10 }]

[[model.files]]
name = "mmproj.gguf"
role = "mmproj"
sha256 = "cc"
bytes = 5
repo = "Xephex/workbench-models"
chunks = [{ blob = "b4", bytes = 5 }]

[[model]]
id = "big-moe"
kinds = ["reasoning"]
engine = "llama.cpp"
ram_gb = 28
vram_gb = 12
recommended_for = ["gpu-16"]
license = "Apache-2.0"
source = "example/big"

[[model.files]]
name = "big.gguf"
role = "main"
sha256 = "dd"
bytes = 1
repo = "Xephex/workbench-models"
chunks = [{ blob = "b5", bytes = 1 }]
```

`tests/test_catalog.py`:
```python
import pathlib, tempfile, unittest

from workbench import catalog, hw

FIX = pathlib.Path(__file__).parent / "fixtures"
GB = 1_000_000_000


def profile(ram_gb, vram_gb=0, wsl=False, os="linux"):
    gpus = [hw.Gpu("nvidia", "x", int(vram_gb * GB), "581", 13010, (12, 0), False, "nvml")] if vram_gb else []
    return hw.Profile(os=os, arch="x86_64", wsl=wsl, rosetta=False, cpu_name="x", cpu_cores=8,
                      ram_bytes=int(ram_gb * GB), ram_available_bytes=None, disk_free_bytes=900 * GB,
                      gpus=gpus, vulkan_loader=False)


class LoadTest(unittest.TestCase):
    def test_load_small(self):
        models = catalog.load(FIX / "catalog-small.toml")
        self.assertEqual([m.id for m in models], ["tiny-chat", "mid-chat", "big-moe"])
        tiny = models[0]
        self.assertEqual(tiny.total_bytes, 150)
        self.assertEqual(tiny.files[0].chunks, (catalog.Chunk("b1", 100), catalog.Chunk("b2", 50)))
        self.assertEqual(models[1].files[1].role, "mmproj")

    def test_chunk_sizes_must_add_up(self):
        bad = (FIX / "catalog-small.toml").read_text().replace("bytes = 150", "bytes = 151")
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "c.toml"
            p.write_text(bad)
            with self.assertRaisesRegex(catalog.CatalogError, "tiny-chat.*tiny.gguf"):
                catalog.load(p)

    def test_missing_field_names_model(self):
        bad = (FIX / "catalog-small.toml").read_text().replace('engine = "llama.cpp"\nram_gb = 2', "ram_gb = 2", 1)
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "c.toml"
            p.write_text(bad)
            with self.assertRaisesRegex(catalog.CatalogError, "tiny-chat.*engine"):
                catalog.load(p)


class OfferTest(unittest.TestCase):
    def setUp(self):
        self.models = catalog.load(FIX / "catalog-small.toml")

    def offers(self, prof):
        return {o.model.id: o for o in catalog.offered(self.models, prof)}

    def test_cpu_laptop(self):
        o = self.offers(profile(ram_gb=16))
        self.assertTrue(o["tiny-chat"].runnable)
        self.assertIsNone(o["tiny-chat"].reason)
        self.assertTrue(o["mid-chat"].runnable)
        self.assertEqual(o["mid-chat"].reason, "runs on CPU only; slow")
        self.assertFalse(o["big-moe"].runnable)
        self.assertIn("needs 40 GB of RAM; this machine has 16", o["big-moe"].reason)

    def test_gpu8_laptop(self):
        o = self.offers(profile(ram_gb=32, vram_gb=8))
        self.assertTrue(o["mid-chat"].runnable)
        self.assertIsNone(o["mid-chat"].reason)
        self.assertFalse(o["big-moe"].runnable)
        self.assertIn("needs 12 GB of video memory; this machine has 8", o["big-moe"].reason)

    def test_recommended_follows_tier(self):
        prof = profile(ram_gb=32, vram_gb=8)
        rec = catalog.recommended(catalog.offered(self.models, prof), prof.tier)
        self.assertEqual([m.id for m in rec], ["tiny-chat", "mid-chat"])

    def test_recommended_never_includes_unrunnable(self):
        prof = profile(ram_gb=4)
        rec = catalog.recommended(catalog.offered(self.models, prof), "gpu-8")
        self.assertEqual([m.id for m in rec], ["tiny-chat"])

    def test_wsl_reason_mentions_wslconfig(self):
        """Review focus 5: fits the host's RAM but not WSL's half."""
        o = self.offers(profile(ram_gb=16, vram_gb=16, wsl=True))
        self.assertFalse(o["big-moe"].runnable)
        self.assertIn(".wslconfig", o["big-moe"].reason)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3.14 -m unittest tests.test_catalog -v`
Expected: FAIL with `ImportError: cannot import name 'catalog'`

- [ ] **Step 3: Write the implementation**

`workbench/catalog.py`:
```python
"""The model catalog: what can be downloaded, and what this machine can run."""

from __future__ import annotations

import dataclasses, pathlib, tomllib

from workbench import hw

GB = 1_000_000_000
REQUIRED = ("id", "kinds", "engine", "ram_gb", "vram_gb", "recommended_for", "license", "source", "files")


class CatalogError(ValueError):
    pass


@dataclasses.dataclass(frozen=True)
class Chunk:
    blob: str
    bytes: int


@dataclasses.dataclass(frozen=True)
class ModelFile:
    name: str
    sha256: str
    bytes: int
    repo: str
    chunks: tuple[Chunk, ...]
    role: str


@dataclasses.dataclass(frozen=True)
class Model:
    id: str
    kinds: tuple[str, ...]
    engine: str
    files: tuple[ModelFile, ...]
    ram_gb: float
    vram_gb: float
    recommended_for: tuple[str, ...]
    license: str
    source: str

    @property
    def total_bytes(self) -> int:
        return sum(f.bytes for f in self.files)

    @property
    def folder(self) -> str:
        return self.id


@dataclasses.dataclass
class Offer:
    model: Model
    runnable: bool
    reason: str | None


def _file(mid: str, raw: dict) -> ModelFile:
    try:
        chunks = tuple(Chunk(c["blob"], int(c["bytes"])) for c in raw["chunks"])
        f = ModelFile(raw["name"], raw["sha256"], int(raw["bytes"]), raw["repo"], chunks, raw.get("role", "main"))
    except KeyError as e:
        raise CatalogError(f"model {mid}: file entry is missing {e.args[0]}") from None
    if sum(c.bytes for c in chunks) != f.bytes:
        raise CatalogError(f"model {mid}: chunks of {f.name} add up to {sum(c.bytes for c in chunks)} "
                           f"bytes, the file is {f.bytes}")
    return f


def load(path: pathlib.Path) -> list[Model]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    models = []
    for raw in data.get("model", []):
        mid = raw.get("id", "<no id>")
        for key in REQUIRED:
            if key not in raw:
                raise CatalogError(f"model {mid}: missing field {key}")
        models.append(Model(
            id=raw["id"], kinds=tuple(raw["kinds"]), engine=raw["engine"],
            files=tuple(_file(mid, f) for f in raw["files"]),
            ram_gb=float(raw["ram_gb"]), vram_gb=float(raw["vram_gb"]),
            recommended_for=tuple(raw["recommended_for"]), license=raw["license"], source=raw["source"],
        ))
    return models


def _fmt(n: float) -> str:
    return f"{n:.0f}" if abs(n - round(n)) < 0.05 else f"{n:.1f}"


def _offer(m: Model, p: hw.Profile) -> Offer:
    ram = p.ram_bytes / GB
    budget = p.gpu_budget_bytes / GB
    if m.vram_gb == 0:
        if ram >= m.ram_gb:
            return Offer(m, True, None)
        return Offer(m, False, _ram_reason(m.ram_gb, ram, p))
    if budget >= m.vram_gb and ram >= m.ram_gb:
        return Offer(m, True, None)
    if budget == 0 and ram >= m.ram_gb + m.vram_gb:
        return Offer(m, True, "runs on CPU only; slow")
    if budget and budget < m.vram_gb:
        return Offer(m, False, f"needs {_fmt(m.vram_gb)} GB of video memory; this machine has {_fmt(budget)}")
    need = m.ram_gb if budget else m.ram_gb + m.vram_gb
    return Offer(m, False, _ram_reason(need, ram, p))


def _ram_reason(need: float, have: float, p: hw.Profile) -> str:
    reason = f"needs {_fmt(need)} GB of RAM; this machine has {_fmt(have)}"
    if p.wsl and have * 2 >= need:
        reason += "; WSL can use more memory if you raise memory= in %UserProfile%\\.wslconfig"
    return reason


def offered(models: list[Model], profile: hw.Profile) -> list[Offer]:
    return [_offer(m, profile) for m in models]


def recommended(offers: list[Offer], tier: str) -> list[Model]:
    return [o.model for o in offers if o.runnable and tier in o.model.recommended_for]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3.14 -m unittest tests.test_catalog -v`
Expected: 8 tests, OK

- [ ] **Step 5: Commit**

```bash
git add workbench/catalog.py tests/test_catalog.py tests/fixtures/catalog-small.toml
git -c user.name=Xephex -c user.email=Xephex@users.noreply.github.com commit -m "feat: model catalog with per-machine offers and reasons

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Platform pack choice (`packs.py`)

**Files:**
- Create: `workbench/packs.py`, `tests/test_packs.py`

**Interfaces:**
- Consumes: `hw.Profile`, `hw.Gpu` (Task 2).
- Produces: `@dataclass class PackChoice: pack: str | None, note: str | None` and `choose_pack(p: hw.Profile) -> PackChoice`. `pack is None` means unsupported in Stage 1; `note` then says why.

Rules (spec B2 table): NVIDIA usable for CUDA when compute capability ≥ (7, 5) (Turing). Windows CUDA needs CUDA driver version ≥ 12080 (driver ≥ 570); Linux/WSL CUDA 13 needs ≥ 13000 (driver ≥ 580). Rosetta Python → unsupported with the note to install arm64 Python.

- [ ] **Step 1: Write the failing tests**

`tests/test_packs.py`:
```python
import unittest

from workbench import hw, packs

GB = 1_000_000_000


def prof(os="windows", arch="x86_64", gpus=(), rosetta=False, wsl=False):
    return hw.Profile(os=os, arch=arch, wsl=wsl, rosetta=rosetta, cpu_name="x", cpu_cores=8, ram_bytes=32 * GB,
                      ram_available_bytes=None, disk_free_bytes=500 * GB, gpus=list(gpus), vulkan_loader=True)


def nvidia(cuda=13010, cc=(12, 0)):
    return hw.Gpu("nvidia", "RTX PRO 2000", 8 * GB, "581.42", cuda, cc, False, "nvml")


class PackTest(unittest.TestCase):
    def test_windows_blackwell(self):
        self.assertEqual(packs.choose_pack(prof(gpus=[nvidia()])), packs.PackChoice("win-x64-cuda12", None))

    def test_windows_no_gpu(self):
        self.assertEqual(packs.choose_pack(prof()).pack, "win-x64-vulkan")

    def test_windows_amd(self):
        amd = hw.Gpu("amd", "RX 7600M", 8 * GB, None, None, None, False, "dxgi")
        self.assertEqual(packs.choose_pack(prof(gpus=[amd])).pack, "win-x64-vulkan")

    def test_old_nvidia_driver_gets_vulkan_with_reason(self):
        """Review focus 4."""
        c = packs.choose_pack(prof(gpus=[nvidia(cuda=12020)]))
        self.assertEqual(c.pack, "win-x64-vulkan")
        self.assertIn("570", c.note)

    def test_pascal_gets_vulkan(self):
        c = packs.choose_pack(prof(gpus=[nvidia(cc=(6, 1))]))
        self.assertEqual(c.pack, "win-x64-vulkan")
        self.assertIn("older than Turing", c.note)

    def test_linux_and_wsl_cuda13(self):
        self.assertEqual(packs.choose_pack(prof(os="linux", gpus=[nvidia()])).pack, "linux-x64-cuda13")
        self.assertEqual(packs.choose_pack(prof(os="linux", wsl=True, gpus=[nvidia()])).pack, "linux-x64-cuda13")

    def test_linux_driver_570_gets_vulkan_and_580_note(self):
        c = packs.choose_pack(prof(os="linux", gpus=[nvidia(cuda=12080)]))
        self.assertEqual(c.pack, "linux-x64-vulkan")
        self.assertIn("580", c.note)

    def test_linux_arm(self):
        self.assertEqual(packs.choose_pack(prof(os="linux", arch="arm64")).pack, "linux-arm64-cpu")

    def test_mac(self):
        apple = hw.Gpu("apple", "M3", None, None, None, None, True, "sysctl")
        self.assertEqual(packs.choose_pack(prof(os="macos", arch="arm64", gpus=[apple])).pack, "macos-arm64-metal")

    def test_unsupported(self):
        self.assertIsNone(packs.choose_pack(prof(os="macos", arch="x86_64")).pack)
        self.assertIsNone(packs.choose_pack(prof(os="windows", arch="arm64")).pack)
        c = packs.choose_pack(prof(os="macos", arch="x86_64", rosetta=True))
        self.assertIsNone(c.pack)
        self.assertIn("arm64", c.note)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3.14 -m unittest tests.test_packs -v`
Expected: FAIL with `ImportError: cannot import name 'packs'`

- [ ] **Step 3: Write the implementation**

`workbench/packs.py`:
```python
"""Which platform pack this machine installs (spec B2, B4)."""

from __future__ import annotations

import dataclasses

from workbench import hw

TURING = (7, 5)
CUDA_MIN = {"windows": (12080, "570"), "linux": (13000, "580")}


@dataclasses.dataclass
class PackChoice:
    pack: str | None
    note: str | None


def _cuda_ok(p: hw.Profile) -> tuple[bool, str | None]:
    nv = [g for g in p.gpus if g.vendor == "nvidia"]
    if not nv:
        return False, None
    g = nv[0]
    if g.compute_capability and g.compute_capability < TURING:
        return False, f"{g.name} is older than Turing, so it runs through Vulkan instead of CUDA."
    need, driver = CUDA_MIN[p.os]
    if not g.cuda_driver or g.cuda_driver < need:
        return False, (f"NVIDIA driver {g.driver or 'unknown'} is too old for CUDA here; update to driver "
                       f"{driver} or newer for full GPU speed. Using Vulkan meanwhile.")
    return True, None


def choose_pack(p: hw.Profile) -> PackChoice:
    if p.rosetta:
        return PackChoice(None, "This Python runs under Rosetta (x86_64). Install the arm64 Python 3.14 "
                                "from python.org and run setup with it.")
    if p.os == "macos":
        if p.arch == "arm64":
            return PackChoice("macos-arm64-metal", None)
        return PackChoice(None, "Intel Macs are not supported yet (several packages have no Python 3.14 build).")
    if p.os == "windows":
        if p.arch != "x86_64":
            return PackChoice(None, "Windows on ARM is not supported yet.")
        ok, note = _cuda_ok(p)
        return PackChoice("win-x64-cuda12" if ok else "win-x64-vulkan", note)
    if p.arch == "arm64":
        return PackChoice("linux-arm64-cpu", None)
    ok, note = _cuda_ok(p)
    return PackChoice("linux-x64-cuda13" if ok else "linux-x64-vulkan", note)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3.14 -m unittest tests.test_packs -v`
Expected: 10 tests, OK

- [ ] **Step 5: Commit**

```bash
git add workbench/packs.py tests/test_packs.py
git -c user.name=Xephex -c user.email=Xephex@users.noreply.github.com commit -m "feat: choose the platform pack from the hardware profile

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Chunk fetcher (`fetch.py`)

**Files:**
- Create: `workbench/fetch.py`, `tests/gitfixture.py`, `tests/test_fetch.py`

**Interfaces:**
- Consumes: `catalog.ModelFile`, `catalog.Chunk`, `catalog.Model` (Task 3).
- Produces:
  - `class FetchError(RuntimeError)`
  - `repo_url(repo: str) -> str`: `"https://github.com/<repo>.git"`, unless env `WORKBENCH_REPO_BASE` is set (tests set it to a `file://` folder), then `f"{base}/{repo}.git"`
  - `fetch_file(f: ModelFile, dest_dir: pathlib.Path, work_dir: pathlib.Path, progress: Callable[[int, int], None] | None = None) -> pathlib.Path`
  - `fetch_model(m: Model, models_dir: pathlib.Path, work_dir: pathlib.Path, progress: Callable[[str, int, int], None] | None = None) -> pathlib.Path` (returns `models_dir / m.folder`)
  - `is_installed(m: Model, models_dir: pathlib.Path) -> bool` (every file present with the right size and a `.ok` marker)

Behaviour (spec A5):
- Target file `dest_dir/<name>`; while incomplete it is written as `<name>.partial` with a sidecar `<name>.fetching` holding the number of whole chunks appended.
- For each chunk not yet appended: in a bare repo at `work_dir/<sanitised repo>.git` (created on first use with `remote.origin.promisor=true`, `remote.origin.partialclonefilter=blob:none`), run `git fetch --no-tags --filter=blob:none origin <blob>`, then stream `git cat-file blob <blob>` onto the `.partial` file, check the appended length equals `chunk.bytes`, flush and fsync, and update `.fetching`. The bare repo is deleted and recreated every 20 chunks and at the end, so it never holds more than 20 chunks (about 2 GB).
- On resume: truncate `.partial` to the sum of the chunk sizes recorded as done (drops a half-written chunk).
- At the end, sha256 the `.partial`; on match rename to `<name>`, write `<name>.ok`, remove `.fetching`; on mismatch delete `.partial` and `.fetching` and raise `FetchError` naming the file.
- A file already present with a `.ok` marker and the right size is skipped.
- git failures raise `FetchError` with git's last stderr line.

- [ ] **Step 1: Write the git fixture helper**

`tests/gitfixture.py`:
```python
"""Build a local git repo with chunked files, served over file:// like GitHub."""

import hashlib, os, pathlib, subprocess

from workbench import catalog


def git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def make_repo(base: pathlib.Path, repo: str, files: dict[str, bytes], chunk: int) -> list[catalog.ModelFile]:
    """Create base/<repo>.git with each file split into `chunk`-byte .partNNN pieces.

    Returns catalog.ModelFile entries whose blob IDs point at the committed chunks.
    """
    work = base / "work" / repo.replace("/", "_")
    work.mkdir(parents=True)
    git("init", "-q", "-b", "main", cwd=work)
    for name, data in files.items():
        for i in range(0, max(len(data), 1), chunk):
            (work / f"{name}.part{i // chunk:03d}").write_bytes(data[i:i + chunk])
    git("add", ".", cwd=work)
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "chunks", cwd=work)
    bare = base / f"{repo}.git"
    bare.parent.mkdir(parents=True, exist_ok=True)
    git("clone", "-q", "--bare", str(work), str(bare), cwd=base)
    git("config", "uploadpack.allowFilter", "true", cwd=bare)
    git("config", "uploadpack.allowAnySHA1InWant", "true", cwd=bare)
    out = []
    for name, data in files.items():
        chunks = []
        for i in range(0, max(len(data), 1), chunk):
            part = f"{name}.part{i // chunk:03d}"
            blob = subprocess.run(["git", "rev-parse", f"HEAD:{part}"], cwd=work, check=True,
                                  capture_output=True, text=True).stdout.strip()
            chunks.append(catalog.Chunk(blob, len(data[i:i + chunk])))
        out.append(catalog.ModelFile(name, hashlib.sha256(data).hexdigest(), len(data), repo,
                                     tuple(chunks), "main"))
    return out
```

- [ ] **Step 2: Write the failing tests**

`tests/test_fetch.py`:
```python
import os, pathlib, tempfile, unittest
from unittest import mock

from tests import gitfixture
from workbench import catalog, fetch


class FetchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = pathlib.Path(self.tmp.name)
        self.data = os.urandom(10_000)
        (self.f,) = gitfixture.make_repo(self.base / "remote", "Xephex/models", {"m.gguf": self.data}, chunk=3_000)
        self.env = mock.patch.dict(os.environ, {"WORKBENCH_REPO_BASE": (self.base / "remote").as_uri()})
        self.env.start()
        self.dest = self.base / "models" / "m"
        self.work = self.base / "fetchwork"

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def test_repo_url(self):
        self.assertTrue(fetch.repo_url("Xephex/models").endswith("/remote/Xephex/models.git"))
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(fetch.repo_url("Xephex/workbench-models"),
                             "https://github.com/Xephex/workbench-models.git")

    def test_fetch_file_assembles_and_verifies(self):
        seen = []
        out = fetch.fetch_file(self.f, self.dest, self.work, progress=lambda done, total: seen.append(done))
        self.assertEqual(out.read_bytes(), self.data)
        self.assertTrue((self.dest / "m.gguf.ok").exists())
        self.assertFalse((self.dest / "m.gguf.partial").exists())
        self.assertFalse((self.dest / "m.gguf.fetching").exists())
        self.assertEqual(seen[-1], 10_000)

    def test_second_fetch_is_a_no_op(self):
        fetch.fetch_file(self.f, self.dest, self.work)
        with mock.patch.object(fetch, "_git") as g:
            fetch.fetch_file(self.f, self.dest, self.work)
        g.assert_not_called()

    def test_resume_after_partial_chunk(self):
        """Review focus 1: a crash mid-chunk must not duplicate or corrupt data."""
        real = fetch._append_chunk
        calls = {"n": 0}

        def flaky(bare, chunk, out):
            calls["n"] += 1
            if calls["n"] == 3:
                out.write(b"half-written-garbage")  # simulate a crash in the middle of chunk 3
                raise KeyboardInterrupt
            return real(bare, chunk, out)

        with mock.patch.object(fetch, "_append_chunk", side_effect=flaky):
            with self.assertRaises(KeyboardInterrupt):
                fetch.fetch_file(self.f, self.dest, self.work)
        self.assertFalse((self.dest / "m.gguf").exists())
        self.assertEqual((self.dest / "m.gguf.fetching").read_text().strip(), "2")
        out = fetch.fetch_file(self.f, self.dest, self.work)
        self.assertEqual(out.read_bytes(), self.data)

    def test_sha_mismatch_removes_partial_and_raises(self):
        bad = catalog.ModelFile(self.f.name, "0" * 64, self.f.bytes, self.f.repo, self.f.chunks, "main")
        with self.assertRaisesRegex(fetch.FetchError, "m.gguf"):
            fetch.fetch_file(bad, self.dest, self.work)
        self.assertFalse((self.dest / "m.gguf.partial").exists())
        self.assertFalse((self.dest / "m.gguf").exists())

    def test_unknown_blob_raises_fetch_error(self):
        bad = catalog.ModelFile(self.f.name, self.f.sha256, self.f.bytes, self.f.repo,
                                (catalog.Chunk("1" * 40, self.f.bytes),), "main")
        with self.assertRaises(fetch.FetchError):
            fetch.fetch_file(bad, self.dest, self.work)

    def test_wrong_chunk_size_raises(self):
        c0 = self.f.chunks[0]
        bad_chunks = (catalog.Chunk(c0.blob, c0.bytes + 1),) + self.f.chunks[1:]
        bad = catalog.ModelFile(self.f.name, self.f.sha256, self.f.bytes + 1, self.f.repo, bad_chunks, "main")
        with self.assertRaisesRegex(fetch.FetchError, "size"):
            fetch.fetch_file(bad, self.dest, self.work)

    def test_fetch_model_and_is_installed(self):
        m = catalog.Model("m", ("chat",), "llama.cpp", (self.f,), 1, 0, ("cpu",), "Apache-2.0", "x")
        models_dir = self.base / "models"
        self.assertFalse(fetch.is_installed(m, models_dir))
        fetch.fetch_model(m, models_dir, self.work)
        self.assertTrue(fetch.is_installed(m, models_dir))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3.14 -m unittest tests.test_fetch -v`
Expected: FAIL with `ImportError: cannot import name 'fetch'`

- [ ] **Step 4: Write the implementation**

`workbench/fetch.py`:
```python
"""Fetch model files from GitHub one chunk at a time, by git blob ID.

Git is the only thing that touches the network. Each chunk is fetched into a
throwaway bare repo (partial clone, so only that object downloads), streamed
onto the end of <name>.partial, and the repo is recreated regularly so it
never holds more than a few chunks. Re-running resumes from the last whole chunk.
"""

from __future__ import annotations

import hashlib, os, pathlib, re, shutil, subprocess
from typing import BinaryIO, Callable

from workbench import catalog

RECREATE_EVERY = 20


class FetchError(RuntimeError):
    pass


def repo_url(repo: str) -> str:
    base = os.environ.get("WORKBENCH_REPO_BASE")
    if base:
        return f"{base.rstrip('/')}/{repo}.git"
    return f"https://github.com/{repo}.git"


def _git(*args: str, cwd: pathlib.Path) -> subprocess.CompletedProcess:
    r = subprocess.run(["git", *args], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        lines = r.stderr.decode(errors="replace").strip().splitlines()
        raise FetchError(f"git {args[0]} failed: {lines[-1] if lines else 'exit ' + str(r.returncode)}")
    return r


def _bare(work_dir: pathlib.Path, repo: str) -> pathlib.Path:
    bare = work_dir / (re.sub(r"[^A-Za-z0-9._-]", "_", repo) + ".git")
    if not (bare / "HEAD").exists():
        bare.mkdir(parents=True, exist_ok=True)
        _git("init", "-q", "--bare", cwd=bare)
        _git("config", "remote.origin.url", repo_url(repo), cwd=bare)
        _git("config", "remote.origin.promisor", "true", cwd=bare)
        _git("config", "remote.origin.partialclonefilter", "blob:none", cwd=bare)
    return bare


def _reset_bare(bare: pathlib.Path) -> None:
    if bare.name.endswith(".git") and (bare / "HEAD").exists():
        shutil.rmtree(bare)


def _append_chunk(bare: pathlib.Path, chunk: catalog.Chunk, out: BinaryIO) -> None:
    _git("fetch", "-q", "--no-tags", "--filter=blob:none", "origin", chunk.blob, cwd=bare)
    start = out.tell()
    with subprocess.Popen(["git", "cat-file", "blob", chunk.blob], cwd=bare,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE) as p:
        shutil.copyfileobj(p.stdout, out, 1 << 22)
        err = p.stderr.read()
    if p.returncode != 0:
        raise FetchError(f"git cat-file failed for {chunk.blob}: {err.decode(errors='replace').strip()}")
    if out.tell() - start != chunk.bytes:
        raise FetchError(f"chunk {chunk.blob} has size {out.tell() - start}, catalog says {chunk.bytes}")


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def _done_ok(f: catalog.ModelFile, dest_dir: pathlib.Path) -> bool:
    target = dest_dir / f.name
    return (dest_dir / f"{f.name}.ok").exists() and target.exists() and target.stat().st_size == f.bytes


def fetch_file(f: catalog.ModelFile, dest_dir: pathlib.Path, work_dir: pathlib.Path,
               progress: Callable[[int, int], None] | None = None) -> pathlib.Path:
    target = dest_dir / f.name
    if _done_ok(f, dest_dir):
        return target
    dest_dir.mkdir(parents=True, exist_ok=True)
    partial = dest_dir / f"{f.name}.partial"
    state = dest_dir / f"{f.name}.fetching"
    done = int(state.read_text().strip()) if state.exists() else 0
    done_bytes = sum(c.bytes for c in f.chunks[:done])
    bare = _bare(work_dir, f.repo)
    with open(partial, "r+b" if partial.exists() else "w+b") as out:
        out.truncate(done_bytes)  # drop any half-written chunk
        out.seek(done_bytes)
        for i in range(done, len(f.chunks)):
            _append_chunk(bare, f.chunks[i], out)
            out.flush()
            os.fsync(out.fileno())
            state.write_text(f"{i + 1}\n")
            done_bytes += f.chunks[i].bytes
            if progress:
                progress(done_bytes, f.bytes)
            if (i + 1) % RECREATE_EVERY == 0:
                _reset_bare(bare)
                bare = _bare(work_dir, f.repo)
    _reset_bare(bare)
    if _sha256(partial) != f.sha256:
        partial.unlink()
        state.unlink(missing_ok=True)
        raise FetchError(f"{f.name}: sha256 does not match the catalog; the download was discarded. "
                         "Run the same command again to retry.")
    partial.replace(target)
    (dest_dir / f"{f.name}.ok").write_text(f.sha256 + "\n")
    state.unlink(missing_ok=True)
    return target


def fetch_model(m: catalog.Model, models_dir: pathlib.Path, work_dir: pathlib.Path,
                progress: Callable[[str, int, int], None] | None = None) -> pathlib.Path:
    folder = models_dir / m.folder
    for f in m.files:
        cb = (lambda done, total, name=f.name: progress(name, done, total)) if progress else None
        fetch_file(f, folder, work_dir, cb)
    return folder


def is_installed(m: catalog.Model, models_dir: pathlib.Path) -> bool:
    return all(_done_ok(f, models_dir / m.folder) for f in m.files)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3.14 -m unittest tests.test_fetch -v`
Expected: 8 tests, OK

- [ ] **Step 6: Commit**

```bash
git add workbench/fetch.py tests/gitfixture.py tests/test_fetch.py
git -c user.name=Xephex -c user.email=Xephex@users.noreply.github.com commit -m "feat: resumable chunk fetcher using git partial fetch by blob ID

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Seed catalog from llm-kit's uploads + maintainer tool

**Files:**
- Create: `tools/catalog_chunks.py`, `catalog.toml`, `tests/test_catalog_chunks.py`

**Interfaces:**
- Consumes: `catalog.load` (Task 3).
- Produces: `tools/catalog_chunks.py` with `chunks_from_tree(tree: list[dict], prefix: str, name: str) -> list[dict]` (pure function: picks `<prefix>/<name>.partNNN` entries in order, returns `[{"blob": sha, "bytes": size}]`, raises `ValueError` on a gap) and a CLI `python tools/catalog_chunks.py --repo OWNER/REPO --ref SHA --prefix PATH --name FILE` that fetches `https://api.github.com/repos/OWNER/REPO/git/trees/SHA?recursive=1` (maintainer-only; allowed to use the network) and prints the TOML `chunks = [...]` line. `catalog.toml` with the 11 models llm-kit already uploaded to `Xephex/playground_stuff` at commit `56301091d467a6e6759e39e4ff910797eabe3cfa`.

- [ ] **Step 1: Write the failing test**

`tests/test_catalog_chunks.py`:
```python
import pathlib, sys, unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
import catalog_chunks  # noqa: E402

from workbench import catalog  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


class ChunksFromTreeTest(unittest.TestCase):
    TREE = [
        {"path": "m/x.gguf.part001", "type": "blob", "sha": "b", "size": 5},
        {"path": "m/x.gguf.part000", "type": "blob", "sha": "a", "size": 10},
        {"path": "m/x.gguf.sha256", "type": "blob", "sha": "s", "size": 80},
        {"path": "m/y.gguf.part000", "type": "blob", "sha": "c", "size": 1},
    ]

    def test_orders_and_filters(self):
        self.assertEqual(catalog_chunks.chunks_from_tree(self.TREE, "m", "x.gguf"),
                         [{"blob": "a", "bytes": 10}, {"blob": "b", "bytes": 5}])

    def test_gap_raises(self):
        tree = [t for t in self.TREE if not t["path"].endswith("part000")]
        with self.assertRaises(ValueError):
            catalog_chunks.chunks_from_tree(tree, "m", "x.gguf")


class SeedCatalogTest(unittest.TestCase):
    def test_seed_catalog_loads_and_matches_known_sizes(self):
        models = {m.id: m for m in catalog.load(ROOT / "catalog.toml")}
        self.assertEqual(models["qwen3-embed-0.6b"].files[0].bytes, 639150592)
        self.assertEqual(models["qwen3-embed-0.6b"].files[0].sha256,
                         "06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439")
        self.assertEqual(len(models["qwen3-embed-0.6b"].files[0].chunks), 7)
        self.assertEqual(models["qwen3.5-4b"].files[0].bytes, 2740937888)
        self.assertEqual(len(models), 7)  # 7 models; 4 of them also have a vision file
        self.assertEqual(sum(len(m.files) for m in models.values()), 11)
        for m in models.values():
            for f in m.files:
                self.assertEqual(f.repo, "Xephex/playground_stuff")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3.14 -m unittest tests.test_catalog_chunks -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'catalog_chunks'`

- [ ] **Step 3: Write the tool**

`tools/catalog_chunks.py`:
```python
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
```

- [ ] **Step 4: Generate `catalog.toml` for the 11 llm-kit models**

The source values (file sizes, sha256, RAM/VRAM needs) come from spec B3 and `llm-kit/models/MANIFEST.txt`. Run this generator once (it calls the tool's function against one tree listing) and commit its output:

```bash
python3.14 - <<'EOF'
import json, pathlib, sys, urllib.request
sys.path.insert(0, "tools")
from catalog_chunks import chunks_from_tree

REF = "56301091d467a6e6759e39e4ff910797eabe3cfa"
tree = json.load(urllib.request.urlopen(
    f"https://api.github.com/repos/Xephex/playground_stuff/git/trees/{REF}?recursive=1", timeout=60))["tree"]
P = "llm-kit/models"
# id, kinds, folder, [(file, role, sha256, bytes)], ram_gb, vram_gb, recommended_for, source
M = [
 ("qwen3.5-4b", ["chat","vision","tools"], "Qwen3.5-4B-GGUF",
  [("Qwen3.5-4B-Q4_K_M.gguf","main","00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4",2740937888),
   ("mmproj-F16.gguf","mmproj","cd88edcf8d031894960bb0c9c5b9b7e1fea6ebee02b9f7ce925a00d12891f864",672423616)],
  6, 4, ["cpu","gpu-8"], "unsloth/Qwen3.5-4B-GGUF@e87f176479d0855a907a41277aca2f8ee7a09523"),
 ("qwen3.5-9b", ["chat","vision","tools"], "Qwen3.5-9B-GGUF",
  [("Qwen3.5-9B-Q4_K_M.gguf","main","03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8",5680522464),
   ("mmproj-F16.gguf","mmproj","f70dc3509053962b0d0d3ee8a7eacebf5d60aa560cad78254ae8698516ae029f",918166080)],
  8, 7, ["gpu-8","gpu-16"], "unsloth/Qwen3.5-9B-GGUF@3885219b6810b007914f3a7950a8d1b469d598a5"),
 ("gemma4-e4b", ["chat","vision","audio"], "gemma-4-E4B-it-qat-q4_0-gguf",
  [("gemma-4-E4B_q4_0-it.gguf","main","676c35070db6dbe52f93e9c864ee0fba4eddea94b9c875d9cb10daff453fbaee",5154941280),
   ("gemma-4-E4B-it-mmproj.gguf","mmproj","7498a37cb619e55f2fcf87eb931f56e99389ed6d432e4c5c66110694c0d65578",991552256)],
  8, 7, [], "google/gemma-4-E4B-it-qat-q4_0-gguf@4b4a2c1d584be7264f87aac328a1bc739ce81b6c"),
 ("granite-4.2-8b", ["chat","tools"], "granite-4.2-8b-GGUF",
  [("granite-4.2-8b-Q4_K_M.gguf","main","16a9369d0805f80b7377d25d87f937a90c05dc04ad79173a52001e42c9aab311",5347917952)],
  8, 6.5, [], "ibm-granite/granite-4.2-8b-GGUF@93f3f6a8938ee922b784cf4e5b4203cd3428df8f"),
 ("gpt-oss-20b", ["reasoning","tools"], "gpt-oss-20b-GGUF",
  [("gpt-oss-20b-MXFP4.gguf","main","27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901",12109566624)],
  16, 5, ["gpu-16"], "ggml-org/gpt-oss-20b-GGUF@ef9b12f2ff56c69cf32153a02784e7a3c88bf524"),
 ("qwen3.6-35b-a3b", ["chat","coding","vision","tools"], "Qwen3.6-35B-A3B-GGUF",
  [("Qwen3.6-35B-A3B-UD-Q4_K_M.gguf","main","ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61",22134528992),
   ("mmproj-F16.gguf","mmproj","8971ee4f331ff0a4c609374f32984b3d4e6dc086c0aa35f1d637fad1829e887f",899283680)],
  28, 7, [], "unsloth/Qwen3.6-35B-A3B-GGUF@a483e9e6cbd595906af30beda3187c2663a1118c"),
 ("qwen3-embed-0.6b", ["embeddings"], "Qwen3-Embedding-0.6B-GGUF",
  [("Qwen3-Embedding-0.6B-Q8_0.gguf","main","06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439",639150592)],
  2, 0, ["cpu","gpu-8"], "Qwen/Qwen3-Embedding-0.6B-GGUF@370f27d7550e0def9b39c1f16d3fbaa13aa67728"),
]
out = ["# Workbench model catalog. Generated entries; see spec B3 and tools/catalog_chunks.py.", ""]
for mid, kinds, folder, files, ram, vram, rec, src in M:
    out += ["[[model]]", f'id = "{mid}"', f"kinds = {json.dumps(kinds)}", 'engine = "llama.cpp"',
            f"ram_gb = {ram}", f"vram_gb = {vram}", f"recommended_for = {json.dumps(rec)}",
            'license = "Apache-2.0"', f'source = "{src}"', ""]
    for name, role, sha, size in files:
        ch = chunks_from_tree(tree, f"{P}/{folder}", name)
        assert sum(c["bytes"] for c in ch) == size, name
        chs = ", ".join(f'{{ blob = "{c["blob"]}", bytes = {c["bytes"]} }}' for c in ch)
        out += ["[[model.files]]", f'name = "{name}"', f'role = "{role}"', f'sha256 = "{sha}"', f"bytes = {size}",
                'repo = "Xephex/playground_stuff"', f"chunks = [{chs}]", ""]
pathlib.Path("catalog.toml").write_text("\n".join(out))
print("wrote", len(M), "models")
EOF
```

Expected: `wrote 7 models` (the 11 files llm-kit uploaded belong to these 7 models).

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3.14 -m unittest tests.test_catalog_chunks -v`
Expected: 3 tests, OK

- [ ] **Step 6: Real-network check of one small model** (manual, not part of the unit tests)

Run:
```bash
python3.14 -c "
import pathlib, tempfile
from workbench import catalog, fetch
m = {m.id: m for m in catalog.load(pathlib.Path('catalog.toml'))}['qwen3-embed-0.6b']
d = pathlib.Path(tempfile.mkdtemp(dir='/tmp'))
print(fetch.fetch_model(m, d / 'models', d / 'work', progress=lambda n, a, b: print(n, a, b, end='\r')))
print('installed', fetch.is_installed(m, d / 'models'), 'work dir left:', list((d / 'work').iterdir()))
"
```
Expected: about 640 MB downloaded from GitHub, ending with `installed True work dir left: []`.

- [ ] **Step 7: Commit**

```bash
git add tools/catalog_chunks.py catalog.toml tests/test_catalog_chunks.py
git -c user.name=Xephex -c user.email=Xephex@users.noreply.github.com commit -m "feat: seed catalog from llm-kit's uploaded models; chunk listing tool

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Commands: `status`, `models`, `setup`

**Files:**
- Create: `workbench/cli.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `hw.detect`, `hw.Profile` (Task 2); `catalog.load`, `catalog.offered`, `catalog.recommended`, `catalog.Offer` (Task 3); `packs.choose_pack` (Task 4); `fetch.fetch_model`, `fetch.is_installed`, `fetch.FetchError` (Task 5); `ui.say/ask/gb` (Task 1).
- Produces: `cli.main(argv: list[str]) -> int`; constants `cli.ROOT` (repo root), `cli.MODELS = ROOT / "models"`, `cli.WORK = ROOT / ".workbench" / "fetch"`, `cli.CATALOG = ROOT / "catalog.toml"`; `cli.plan_downloads(chosen: list[catalog.Model], models_dir) -> int` (bytes still to download); flags `models --add ID ...`, `models --remove ID ...`, `models --yes` (accept the recommended set without prompting), `setup --yes`.

Behaviour:
- `status`: prints OS/arch (+ "WSL"), CPU, RAM, free disk, each GPU (name, VRAM, driver), tier, chosen pack (+ note), then every catalog model as `[x]` installed / `[ ]` runnable / `[-]` not runnable with its reason.
- `models` (no flags): shows runnable models with sizes and `*` on recommended ones, asks for a comma-separated list (default: recommended), then downloads.
- `models --add ID`/`--remove ID`: no prompt. Unknown ID → exit 2 with the list of valid IDs. Removing deletes `MODELS/<id>/<file>` and its `.ok` for each catalog file of that model (named targets only), then the folder if empty.
- Before any download: if bytes to download × 1.02 > free disk, print `"Not enough disk: need X, free Y. Free up space or remove models, then run again."` and exit 1 without fetching.
- `setup`: prints the profile summary (as status), stops with exit 1 and the pack note if `pack is None`, prints `"Platform pack: <id>"` (installing the pack is Plan 2), then runs the `models` flow.
- Fetch errors print the message and exit 1; already-finished models stay installed.

- [ ] **Step 1: Write the failing tests**

`tests/test_cli.py`:
```python
import io, os, pathlib, tempfile, unittest
from unittest import mock

from tests import gitfixture
from workbench import catalog, cli, hw

GB = 1_000_000_000


def fake_profile(disk_free=500 * GB, ram=32 * GB, gpus=()):
    return hw.Profile(os="linux", arch="x86_64", wsl=False, rosetta=False, cpu_name="Test CPU", cpu_cores=8,
                      ram_bytes=ram, ram_available_bytes=None, disk_free_bytes=disk_free,
                      gpus=list(gpus), vulkan_loader=False)


class CliTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = pathlib.Path(self.tmp.name)
        self.data = os.urandom(5_000)
        (f,) = gitfixture.make_repo(base / "remote", "Xephex/models", {"t.gguf": self.data}, chunk=2_000)
        chunks = ", ".join(f'{{ blob = "{c.blob}", bytes = {c.bytes} }}' for c in f.chunks)
        (base / "catalog.toml").write_text(f'''
[[model]]
id = "tiny"
kinds = ["chat"]
engine = "llama.cpp"
ram_gb = 1
vram_gb = 0
recommended_for = ["cpu"]
license = "Apache-2.0"
source = "x"

[[model.files]]
name = "t.gguf"
role = "main"
sha256 = "{f.sha256}"
bytes = {f.bytes}
repo = "Xephex/models"
chunks = [{chunks}]

[[model]]
id = "huge"
kinds = ["chat"]
engine = "llama.cpp"
ram_gb = 999
vram_gb = 0
recommended_for = ["cpu"]
license = "Apache-2.0"
source = "x"

[[model.files]]
name = "h.gguf"
role = "main"
sha256 = "00"
bytes = 1
repo = "Xephex/models"
chunks = [{{ blob = "ab", bytes = 1 }}]
''')
        self.patches = [
            mock.patch.dict(os.environ, {"WORKBENCH_REPO_BASE": (base / "remote").as_uri()}),
            mock.patch.object(cli, "CATALOG", base / "catalog.toml"),
            mock.patch.object(cli, "MODELS", base / "models"),
            mock.patch.object(cli, "WORK", base / "work"),
            mock.patch.object(hw, "detect", return_value=fake_profile()),
        ]
        for p in self.patches:
            p.start()
        self.base = base

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def run_cli(self, *argv, stdin=""):
        out = io.StringIO()
        with mock.patch("sys.stdout", out), mock.patch("builtins.input", side_effect=list(stdin.splitlines()) or EOFError):
            code = cli.main(list(argv))
        return code, out.getvalue()

    def test_status_lists_models_with_reasons(self):
        code, out = self.run_cli("status")
        self.assertEqual(code, 0)
        self.assertIn("Tier: cpu", out)
        self.assertIn("Platform pack: linux-x64-vulkan", out)
        self.assertIn("[ ] tiny", out)
        self.assertIn("[-] huge", out)
        self.assertIn("needs 999 GB of RAM", out)

    def test_models_yes_downloads_recommended(self):
        code, out = self.run_cli("models", "--yes")
        self.assertEqual(code, 0, out)
        self.assertEqual((self.base / "models" / "tiny" / "t.gguf").read_bytes(), self.data)
        code, out = self.run_cli("status")
        self.assertIn("[x] tiny", out)

    def test_models_refuses_when_disk_short(self):
        """Review focus 2."""
        with mock.patch.object(hw, "detect", return_value=fake_profile(disk_free=10)):
            code, out = self.run_cli("models", "--add", "tiny")
        self.assertEqual(code, 1)
        self.assertIn("Not enough disk", out)
        self.assertFalse((self.base / "models" / "tiny").exists())

    def test_add_unknown_id(self):
        code, out = self.run_cli("models", "--add", "nope")
        self.assertEqual(code, 2)
        self.assertIn("tiny", out)

    def test_add_unrunnable_refused(self):
        code, out = self.run_cli("models", "--add", "huge")
        self.assertEqual(code, 1)
        self.assertIn("needs 999 GB of RAM", out)

    def test_remove(self):
        self.run_cli("models", "--add", "tiny")
        code, out = self.run_cli("models", "--remove", "tiny")
        self.assertEqual(code, 0)
        self.assertFalse((self.base / "models" / "tiny").exists())

    def test_interactive_default_is_recommended(self):
        code, out = self.run_cli("models", stdin="\n")
        self.assertEqual(code, 0, out)
        self.assertTrue((self.base / "models" / "tiny" / "t.gguf").exists())

    def test_setup_stops_on_unsupported_platform(self):
        mac_intel = hw.Profile(os="macos", arch="x86_64", wsl=False, rosetta=False, cpu_name="i7", cpu_cores=4,
                               ram_bytes=16 * GB, ram_available_bytes=None, disk_free_bytes=100 * GB,
                               gpus=[], vulkan_loader=False)
        with mock.patch.object(hw, "detect", return_value=mac_intel):
            code, out = self.run_cli("setup", "--yes")
        self.assertEqual(code, 1)
        self.assertIn("Intel Macs are not supported", out)

    def test_setup_yes(self):
        code, out = self.run_cli("setup", "--yes")
        self.assertEqual(code, 0, out)
        self.assertIn("Platform pack: linux-x64-vulkan", out)
        self.assertTrue((self.base / "models" / "tiny" / "t.gguf").exists())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3.14 -m unittest tests.test_cli -v`
Expected: FAIL with `ImportError: cannot import name 'cli'`

- [ ] **Step 3: Write the implementation**

`workbench/cli.py`:
```python
"""Commands. Stage 1 plan 1: status, models, setup."""

from __future__ import annotations

import argparse, pathlib

from workbench import catalog, fetch, hw, packs
from workbench.ui import ask, gb, say

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
WORK = ROOT / ".workbench" / "fetch"
CATALOG = ROOT / "catalog.toml"


def _profile_lines(p: hw.Profile) -> None:
    say(f"Machine: {p.os} {p.arch}{' (WSL)' if p.wsl else ''}, {p.cpu_name}, {p.cpu_cores} cores")
    say(f"RAM: {gb(p.ram_bytes)}   free disk for models: {gb(p.disk_free_bytes)}")
    for g in p.gpus:
        vram = gb(g.vram_bytes) if g.vram_bytes else "shared memory"
        say(f"GPU: {g.name} ({vram}{', driver ' + g.driver if g.driver else ''})")
    if not p.gpus:
        say("GPU: none found (models run on the CPU)")
    say(f"Tier: {p.tier}")
    choice = packs.choose_pack(p)
    say(f"Platform pack: {choice.pack or 'not supported'}")
    if choice.note:
        say(f"  {choice.note}")


def plan_downloads(chosen: list[catalog.Model], models_dir: pathlib.Path) -> int:
    return sum(m.total_bytes for m in chosen if not fetch.is_installed(m, models_dir))


def _download(chosen: list[catalog.Model], p: hw.Profile) -> int:
    need = plan_downloads(chosen, MODELS)
    if need * 1.02 > p.disk_free_bytes:
        say(f"Not enough disk: need {gb(need)}, free {gb(p.disk_free_bytes)}. "
            "Free up space or remove models, then run again.")
        return 1
    for m in chosen:
        if fetch.is_installed(m, MODELS):
            say(f"  {m.id}: already installed")
            continue
        say(f"  {m.id}: downloading {gb(m.total_bytes)}")
        try:
            fetch.fetch_model(m, MODELS, WORK, progress=lambda name, done, total: print(
                f"\r    {name}: {done * 100 // total}%", end="", flush=True))
            print()
        except fetch.FetchError as e:
            print()
            say(f"  {m.id}: {e}")
            return 1
    say("Models ready.")
    return 0


def _remove(m: catalog.Model) -> None:
    folder = MODELS / m.folder
    for f in m.files:
        (folder / f.name).unlink(missing_ok=True)
        (folder / f"{f.name}.ok").unlink(missing_ok=True)
    if folder.is_dir() and not any(folder.iterdir()):
        folder.rmdir()
    say(f"  removed {m.id}")


def cmd_status(a, p, models) -> int:
    _profile_lines(p)
    say("")
    say("Models  ([x] installed, [ ] can run here, [-] cannot run here):")
    for o in catalog.offered(models, p):
        mark = "x" if fetch.is_installed(o.model, MODELS) else (" " if o.runnable else "-")
        extra = f"  ({o.reason})" if o.reason else ""
        say(f"  [{mark}] {o.model.id:<18} {gb(o.model.total_bytes):>8}{extra}")
    return 0


def cmd_models(a, p, models) -> int:
    by_id = {m.id: m for m in models}
    offers = {o.model.id: o for o in catalog.offered(models, p)}
    for mid in (a.add or []) + (a.remove or []):
        if mid not in by_id:
            say(f"Unknown model {mid!r}. Known: {', '.join(by_id)}")
            return 2
    for mid in a.remove or []:
        _remove(by_id[mid])
    if a.add:
        bad = [o for o in (offers[m] for m in a.add) if not o.runnable]
        for o in bad:
            say(f"{o.model.id} cannot run on this machine: {o.reason}")
        if bad:
            return 1
        return _download([by_id[m] for m in a.add], p)
    if a.remove:
        return 0
    rec = catalog.recommended(list(offers.values()), p.tier)
    if a.yes:
        return _download(rec, p)
    say("Models this machine can run (* = recommended):")
    for o in offers.values():
        if o.runnable:
            star = "*" if o.model in rec else " "
            done = " installed" if fetch.is_installed(o.model, MODELS) else ""
            note = f" ({o.reason})" if o.reason else ""
            say(f"  {star} {o.model.id:<18} {gb(o.model.total_bytes):>8}{done}{note}")
    answer = ask("Models to install, comma-separated", ",".join(m.id for m in rec))
    picked = [s.strip() for s in answer.split(",") if s.strip()]
    unknown = [s for s in picked if s not in offers or not offers[s].runnable]
    if unknown:
        say(f"Not available here: {', '.join(unknown)}")
        return 2
    return _download([by_id[s] for s in picked], p)


def cmd_setup(a, p, models) -> int:
    _profile_lines(p)
    choice = packs.choose_pack(p)
    if choice.pack is None:
        return 1
    say("")
    return cmd_models(argparse.Namespace(add=None, remove=None, yes=a.yes), p, models)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="workbench.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    m = sub.add_parser("models")
    m.add_argument("--add", nargs="+")
    m.add_argument("--remove", nargs="+")
    m.add_argument("--yes", action="store_true")
    s = sub.add_parser("setup")
    s.add_argument("--yes", action="store_true")
    a = ap.parse_args(argv)
    p = hw.detect(MODELS)
    models = catalog.load(CATALOG)
    return {"status": cmd_status, "models": cmd_models, "setup": cmd_setup}[a.cmd](a, p, models)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3.14 -m unittest tests.test_cli -v`
Expected: 9 tests, OK

- [ ] **Step 5: Run the whole suite**

Run: `python3.14 -m unittest discover -s tests -v`
Expected: 55 tests OK, no network used.

- [ ] **Step 6: Exercise the real entry point**

Run from a fresh sparse clone, like a user:
```bash
cd /tmp && rm -rf /tmp/wb-try && git clone -q --filter=blob:none --sparse https://github.com/Xephex/workbench /tmp/wb-try \
  && cd /tmp/wb-try && python3.14 workbench.py status && python3.14 workbench.py models --add qwen3-embed-0.6b \
  && python3.14 workbench.py status | grep qwen3-embed
```
(Only after this plan's commits are pushed.) Expected: a profile for this laptop, a 0.6 GB download from GitHub, then `[x] qwen3-embed-0.6b`. Also run `python3.15 workbench.py status` and confirm it prints "switching to Python 3.14" and works.

- [ ] **Step 7: Commit and push**

```bash
git add workbench/cli.py tests/test_cli.py
git -c user.name=Xephex -c user.email=Xephex@users.noreply.github.com commit -m "feat: status, models and setup commands

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin main
```
