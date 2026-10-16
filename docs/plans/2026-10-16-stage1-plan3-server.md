# Stage 1, Plan 3: Server and engines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve installed Stage 1 models through one offline, OpenAI-compatible local API, with bounded model memory, reusable prompt caches, tracing, and thin terminal clients.

**Architecture:** Own FastAPI routes call a synchronous engine through a FIFO model manager; native inference runs in worker threads while the event loop sends SSE keep-alives. A llama.cpp adapter owns native handles, projectors and caches, and the unchanged llm-kit parser translates textual tool calls. OpenTelemetry records requests and their children locally, optionally exporting to a supervised loopback Phoenix process and an explicitly configured Langfuse endpoint; commands reuse an existing server or own a temporary server for their session.

**Tech Stack:** Python 3.14; stdlib bootstrap, HTTP clients and unittest; installed pack: FastAPI 0.143.2, uvicorn 0.54.0, sse-starlette 3.5.0, llama-cpp-python 0.4.2 (JamePeng), OpenTelemetry SDK/OTLP HTTP 1.45.1, OpenInference semantic conventions 0.1.41, optional Arize Phoenix 20.22.0. Gradio 6.30.0 is installed but is consumed by Plan 4.

**Spec:** `docs/specs/2026-10-15-workbench-design.md` (A9, A11, server-side A12, B7, Stage 1 tracing in B6; A4 client lifecycle, B2 device reporting, B3 thinking defaults and B4 native-library loading).

**This is plan 3 of 4 for Stage 1:**
1. Install foundation (done)
2. Packs and publishing (done; this plan starts at main `f9f2e43`)
3. Server and engines (this plan)
4. App, Pi, docs, CI: Gradio tabs, Pi, n8n, tune, monitor, selftest, `update`, three-OS CI

**Scope cut (operator, 2026-10-16):** Windows and Linux only. This is a plan document, not an instruction to implement during the writing lane. Audio requests and the `image`/`docs` commands belong to later work; image parts on the chat API are included here.

## Global Constraints

- Python 3.14 exactly. Users install only Python and git; every third-party import follows `install.activate(root)`. `status`, `models`, `setup` and their help still work without a pack.
- Default address `127.0.0.1:8080`. Any non-loopback bind requires a bearer API key, including direct `create_app` use. `--share` chooses `0.0.0.0` and creates/reuses a private key file.
- Only git downloads packages/models, from GitHub. HTTP to the local server/Phoenix is necessary local IPC. Remote image URLs are rejected. Langfuse is the spec's explicit, opt-in exception to offline operation; disabled by default.
- Windows starts only git or `sys.executable`; no executable launchers, shell, WMI, Node, CUDA toolkit or bundled llama.cpp server. Phoenix is started from a real `.py` module, never `python -` (strawberry inspects source files).
- RAM budget = `max(0, Profile.ram_bytes - 4_000_000_000)`; VRAM budget = `max(0, Profile.gpu_budget_bytes - 800_000_000)`. Bytes are decimal GB. Cache reservations are additional RAM charges, never free memory.
- One model instance per catalog ID, one instance lock, FIFO tickets assigned at admission, no eviction of a leased instance. Native calls and iterator finalization stay on the same worker thread.
- `n_threads` and `n_threads_batch` use physical cores. Preserve resident llama.cpp KV state and attach a bounded `LlamaRAMCache`; never reset between successful turns. Eviction, changed settings and cancelled inference invalidate that instance's cache.
- No real model in unit tests. The only optional real-model test reads `WORKBENCH_TEST_MODEL`; it skips when unset and never downloads. Run tests with the installed pack activated, from a file/module.
- llm-kit's tool parser and its 13 test bodies move unchanged from `e85c04a356d2ff041911dc4af52140c6613f5585`; own routes do not invoke its bundled-server monkey patches.
- User data stays under `data/`; models under `models/` or the configured directory. Neither is tracked. Keys, request bodies and authorization headers never enter access logs.
- Commits: author `Xephex <Xephex@users.noreply.github.com>`, trailer `Co-Authored-By: GPT-6 Astra (Codex) <noreply@openai.com>`. Commands below stage literal paths. Do not push or merge as part of this plan.

## Review Focus

1. **An agent's second turn replays list-valued system/tool content and assistant tool calls:** normalize text parts for every role and return proper tool calls in both response modes. Task 5 `test_agent_history_and_nonstream_tools`, Task 6 `test_streamed_tools_and_done`.
2. **A long CPU prompt appears hung, then the client disconnects:** send comments before the first token; cancel queued work or abort running inference without releasing its model early. Task 6 `test_keepalive_during_prefill_and_disconnect`.
3. **Two requests compete while a third model exceeds available memory:** retain FIFO order, never unload an active handle, fall back to CPU for small models and recover after a failed load. Task 3 `test_fifo_busy_lru_and_failed_load`, `test_small_model_cpu_fallback`.
4. **An unchanged long prefix is submitted on the next turn:** reuse the same native handle/cache, finish native iteration after the parser's finish chunk, and do not claim a cache hit merely because `set_cache` was called. Task 2 `test_prefix_cache_reused_after_stream_finish`; Task 9 `test_real_small_gguf` measures actual prompt-evaluation counts.
5. **Sharing or privacy is misconfigured:** reject every unauthenticated route on non-loopback binds, and keep disabled content out of local spans, OTLP and exception events. Task 5 `test_nonloopback_requires_key_on_all_routes`, Task 4 `test_private_trace_contains_no_content`.

---

## File Structure

```text
workbench/hw.py                       physical core count, preserving stdlib detection
workbench/catalog.py                  optional validated engine settings
tools/catalog_write.py                round-trip optional settings without rewriting existing entries
workbench/runtime.py                  stdlib settings, paths, footprint contracts
workbench/engines/__init__.py          package marker
workbench/engines/toolcalls.py         unchanged llm-kit parser
workbench/engines/llama.py             native loading, cache, vision, grammar, embed/rank, device report
workbench/server/__init__.py           package marker; no eager imports
workbench/server/manager.py            FIFO leases, separate memory budgets, LRU, tuning input
workbench/trace.py                    providers, file exporter, privacy, ASGI request spans
workbench/phoenix_runner.py            activate then start optional offline Phoenix from a file
workbench/server/schemas.py            owned request schemas and normalization
workbench/server/app.py                lifecycle, auth, JSON routes, status and models
workbench/server/streaming.py          bounded worker bridge, comments, disconnect and shutdown
workbench/server/jobs.py               durable model fetch/remove/cancel jobs
workbench/client.py                    stdlib HTTP/SSE and temporary-server ownership
workbench/tools/__init__.py            package marker
workbench/tools/commands.py            serve/chat/ask/embed/rank/status client formatting
workbench/cli.py                       lazy dispatch; existing setup/models remain bootstrap-safe
tests/server_support.py                deterministic fake engine and temporary runtime
tests/test_runtime.py, test_engine.py, test_toolcalls.py, test_manager.py
tests/test_trace.py, test_server.py, test_streaming.py, test_jobs.py, test_commands.py
tests/test_server_integration.py        one optional real-GGUF test
tools/test_server.py                   activate pack, then unittest (a real file on Windows)
docs/server.md                         API, commands, data, privacy, progress and ownership contract
```

Existing tests remain `python3.14 -m unittest discover -s tests -v`. New pack-dependent tests are run with `python3.14 tools/test_server.py`; do not silently skip them when a pack is missing. The optional integration test is the sole model-dependent skip. In task commands, `python3.14` means `py -3.14` on Windows.

---

### Task 1: Runtime inputs, physical cores and the test entry point

**Files:**
- Create: `workbench/runtime.py`, `tools/test_server.py`, `tests/test_runtime.py`, `workbench/engines/__init__.py`, `workbench/server/__init__.py`, `workbench/tools/__init__.py`
- Modify: `workbench/hw.py` (`detect` core count), `workbench/catalog.py` (`Model`, `load`), `tools/catalog_write.py` (`_value`, `dump`)

**Interfaces:**
- Consumes: `hw.Profile`, `catalog.Model`, `install.activate(root: Path) -> bool`, `install.site_dir(root: Path) -> Path`.
- Produces: `hw.physical_cores(plat: str | None = None, sysfs: Path = Path('/sys/devices/system/cpu')) -> int`; `Model.settings: dict[str, int]` (default `{}`); `runtime.Config(root: Path, models: Path, host: str = '127.0.0.1', port: int = 8080, key: str | None = None, capture: bool = True, phoenix: bool = True, default_model: str | None = None)`; `runtime.Footprint(ram: int, vram: int)`; `runtime.load_tuning(root: Path, model: Model, placement: str) -> dict`; `runtime.is_loopback(host: str) -> bool`.

- [ ] **Step 1: Write the failing tests**

`tests/test_runtime.py`:
```python
import pathlib, tempfile, unittest, tomllib
from unittest import mock
from workbench import catalog, hw
from workbench.runtime import Config, is_loopback, load_tuning
from tools import catalog_write

class RuntimeTest(unittest.TestCase):
    def test_physical_cores_not_smt_threads(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            for cpu, core in enumerate((0, 0, 1, 1)):
                p = root / f'cpu{cpu}' / 'topology'
                p.mkdir(parents=True)
                (p / 'physical_package_id').write_text('0')
                (p / 'core_id').write_text(str(core))
            with mock.patch('os.sched_getaffinity', return_value={0, 1, 2, 3}, create=True):
                self.assertEqual(hw.physical_cores('linux', root), 2)

    def test_windows_core_records(self):
        from workbench.hw import _count_core_records
        import struct
        data = struct.pack('<II', 0, 16) + bytes(8)
        data += struct.pack('<II', 1, 16) + bytes(8)
        data += struct.pack('<II', 0, 16) + bytes(8)
        self.assertEqual(_count_core_records(data), 2)
        self.assertRaises(ValueError, _count_core_records, b'bad')

    def test_settings_roundtrip_and_defaults(self):
        source = pathlib.Path('tests/fixtures/catalog-small.toml').read_text()
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / 'catalog.toml'
            path.write_text(source.replace('engine = "llama.cpp"',
                'engine = "llama.cpp"\nsettings = { n_ctx = 12288 }', 1))
            self.assertEqual(catalog.load(path)[0].settings, {'n_ctx': 12288})
            self.assertEqual(catalog.load(path)[1].settings, {})
            data = tomllib.loads(path.read_text())
            out = catalog_write.dump(data, '# test\n')
            self.assertEqual(tomllib.loads(out)['model'][0]['settings'], {'n_ctx': 12288})

    def test_bind_validation(self):
        self.assertTrue(is_loopback('::1'))
        self.assertTrue(is_loopback('127.0.0.2'))
        self.assertFalse(is_loopback('localhost.example.org'))
        self.assertRaises(ValueError, Config, pathlib.Path('.'), pathlib.Path('models'), '0.0.0.0')
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3.14 -m unittest tests.test_runtime -v`
Expected: FAIL (missing `workbench.runtime`).

- [ ] **Step 3: Implement runtime inputs and the pack-aware runner**

`workbench/runtime.py`:
```python
from __future__ import annotations
import dataclasses, ipaddress, json, pathlib
from workbench.catalog import Model

GB = 1_000_000_000
CACHE_BYTES = 512 * 1024**2
LOAD_KEYS = {'n_ctx', 'n_batch', 'n_ubatch', 'n_cpu_moe'}

def is_loopback(host: str) -> bool:
    if host == 'localhost':
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False

@dataclasses.dataclass(frozen=True)
class Footprint:
    ram: int
    vram: int

@dataclasses.dataclass(frozen=True)
class Config:
    root: pathlib.Path
    models: pathlib.Path
    host: str = '127.0.0.1'
    port: int = 8080
    key: str | None = None
    capture: bool = True
    phoenix: bool = True
    default_model: str | None = None

    def __post_init__(self):
        if self.host == 'localhost':
            object.__setattr__(self, 'host', '127.0.0.1')
        if not is_loopback(self.host) and not self.key:
            raise ValueError('A non-loopback bind requires an API key; use serve --share.')
        if not 0 < self.port < 65536:
            raise ValueError('Port must be between 1 and 65535.')

def load_tuning(root: pathlib.Path, model: Model, placement: str) -> dict:
    try:
        row = json.loads((root / 'data' / 'tuning.json').read_text())[model.id][placement]
        hashes = {f.name: f.sha256 for f in model.files}
        if row['sha256'] != hashes:
            return {}
        settings = row.get('settings', {})
        footprint = row.get('footprint', {})
        if set(settings) - LOAD_KEYS or any(type(v) is not int or v < 0 for v in settings.values()):
            return {}
        if any(type(footprint.get(k)) is not int or footprint[k] < 0 for k in ('ram', 'vram')):
            return {}
        return row
    except (OSError, ValueError, KeyError, TypeError):
        return {}
```

Append these helpers to `workbench/hw.py`; add `import struct`. In `detect`, replace `cpu_cores=os.cpu_count() or 1` with `cpu_cores=physical_cores(plat)`.
```python
def _count_core_records(data: bytes) -> int:
    offset = count = 0
    while offset < len(data):
        if len(data) - offset < 8:
            raise ValueError('Truncated processor topology')
        relation, size = struct.unpack_from('<II', data, offset)
        if size < 8 or offset + size > len(data):
            raise ValueError('Invalid processor topology')
        count += relation == 0
        offset += size
    return count

def physical_cores(plat=None, sysfs=pathlib.Path('/sys/devices/system/cpu')) -> int:
    plat = plat or ('windows' if sys.platform == 'win32' else 'linux')
    try:
        if plat == 'windows':
            fn = ctypes.WinDLL('kernel32', use_last_error=True).GetLogicalProcessorInformationEx
            fn.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
            fn.restype = ctypes.c_int
            size = ctypes.c_uint32()
            fn(0, None, ctypes.byref(size))
            if not size.value:
                raise OSError('No processor topology')
            buf = ctypes.create_string_buffer(size.value)
            if not fn(0, buf, ctypes.byref(size)):
                raise OSError('Cannot read processor topology')
            return max(1, _count_core_records(buf.raw[:size.value]))
        allowed = os.sched_getaffinity(0) if hasattr(os, 'sched_getaffinity') else None
        cores = set()
        for cpu in sysfs.glob('cpu[0-9]*'):
            if not cpu.name[3:].isdigit() or (allowed is not None and int(cpu.name[3:]) not in allowed):
                continue
            t = cpu / 'topology'
            cores.add(((t / 'physical_package_id').read_text().strip(),
                       (t / 'core_id').read_text().strip()))
        return len(cores) or 1
    except (OSError, ValueError, AttributeError):
        return 1
```

Unknown topology conservatively uses one thread; it does not pretend logical CPUs are physical cores. Add to `Model` after `source`:
```python
    settings: dict[str, int] = dataclasses.field(default_factory=dict)
```
In `catalog.load`, before constructing each `Model`, validate and pass the field:
```python
        settings = raw.get('settings', {})
        allowed = {'n_ctx', 'n_batch', 'n_ubatch', 'n_cpu_moe'}
        if not isinstance(settings, dict) or set(settings) - allowed:
            raise CatalogError(f'model {mid}: unsupported settings')
        if any(type(v) is not int or v < 0 for v in settings.values()):
            raise CatalogError(f'model {mid}: settings must be nonnegative integers')
```
Add `settings=dict(settings),` to its `Model(...)` keyword arguments. In `catalog_write._value`, before the final `raise`, add:
```python
    if isinstance(v, dict):
        return '{ ' + ', '.join(f'{k} = {_value(x)}' for k, x in v.items()) + ' }'
```
In `catalog_write.dump`, after writing `MODEL_KEYS` and before `for f in m['files']`, add:
```python
        if 'settings' in m:
            out.append(f"settings = {_value(m['settings'])}\n")
```

Each new package `__init__.py` contains:
```python
"""Workbench components; imports are deliberately lazy."""
```

`tools/test_server.py` (runner arguments are ordinary unittest names):
```python
import pathlib, sys, unittest
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from workbench import install
if not install.activate(ROOT):
    raise SystemExit('Install the pack first: python workbench.py setup --packages-only --yes')
if __name__ == '__main__':
    args = sys.argv[1:] or ['discover', '-s', 'tests', '-v']
    unittest.main(module=None, argv=[sys.argv[0], *args])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3.14 -m unittest tests.test_runtime tests.test_hw tests.test_catalog tests.test_catalog_write -v`
Expected: PASS, including the existing byte-identical catalog round trip.

- [ ] **Step 5: Commit**
```bash
git add workbench/runtime.py workbench/hw.py workbench/catalog.py tools/catalog_write.py tools/test_server.py tests/test_runtime.py workbench/engines/__init__.py workbench/server/__init__.py workbench/tools/__init__.py
git -c user.name=Xephex -c user.email=Xephex@users.noreply.github.com commit -m "feat: server runtime inputs and physical core detection" -m "Co-Authored-By: GPT-6 Astra (Codex) <noreply@openai.com>"
```

---

### Task 2: llama.cpp adapter, unchanged tool parser and reusable prefix cache

**Files:**
- Create: `workbench/engines/llama.py`, `workbench/engines/toolcalls.py`, `tests/test_engine.py`, `tests/test_toolcalls.py`

**Interfaces:**
- Consumes: `catalog.Model`, `runtime.Footprint`, `runtime.CACHE_BYTES`, the installed pack, `install.site_dir(root) -> Path`.
- Produces: `LlamaEngine(root: Path, native: object | None = None)` with `load(entry: Model, settings: dict) -> Handle`, `unload(handle: Handle) -> None`, `chat(handle: Handle, request: dict, phase: Callable[[str], None]) -> Iterator[dict]`, `embed(handle: Handle, texts: list[str]) -> tuple[list[list[float]], int]`, `rank(handle: Handle, query: str, docs: list[str]) -> list[float]`, `stats(handle: Handle) -> dict`, `abort(handle: Handle) -> None`.
- `settings` includes `models_dir: Path`, `placement: 'cpu' | 'gpu'`, `n_threads: int`, `cache_bytes: int`, `think: bool`, plus the validated catalog/tuning load keys. `Handle` exposes `entry`, `native`, `cache`, `placement`, `usage`, `phase`, `devices` and `handler`.
- `chat` always emits OpenAI chunks internally, regardless of the external response mode. The manager owns serialization; adapter methods never acquire manager locks. `abort` is the only method callable concurrently with inference.

- [ ] **Step 1: Write the failing tests and import the unchanged parser tests**

Copy the 13 existing test bodies using the exact extraction below. The source checkout is read-only; set `WB_LLM_KIT_REPO` to its location if it is not the operator's existing checkout. The only change to the test file is its import bootstrap. Do not run the legacy server's tests against its bundled server; the two equivalent HTTP assertions are in Tasks 5-6.

```bash
python3.14 -c 'import os,pathlib,subprocess; repo=os.environ.get("WB_LLM_KIT_REPO","/home/operator/Work/repos/github.com/Xephex/playground_stuff"); text=subprocess.check_output(["git","-C",repo,"show","e85c04a356d2ff041911dc4af52140c6613f5585:llm-kit/tests/test_toolcalls.py"]).decode(); old="sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / \"run\"))\nimport toolcalls"; assert old in text; pathlib.Path("tests/test_toolcalls.py").write_text(text.replace(old,"from workbench.engines import toolcalls"),encoding="utf-8")'
```

`tests/test_engine.py`:
```python
import json, pathlib, tempfile, types, unittest
from unittest import mock
from workbench.catalog import Model, ModelFile
from workbench.engines.llama import LlamaEngine

class Cache:
    def __init__(self, capacity_bytes):
        self.capacity_bytes = capacity_bytes
        self.prefix = ''

class Native:
    def __init__(self, **kwargs):
        self.kwargs, self.cache, self.closed = kwargs, None, False
        self.evaluated = []
    def set_cache(self, cache):
        self.cache = cache
    def generate(self, tokens, **kwargs):
        text = ''.join(tokens)
        shared = 0
        if self.cache:
            for a, b in zip(text, self.cache.prefix):
                if a != b:
                    break
                shared += 1
        self.evaluated.append(len(text) - shared)
        yield from getattr(self, 'output', 'OK')
    def create_chat_completion(self, **kwargs):
        self.last = kwargs
        prompt = ''.join(m.get('content') if isinstance(m.get('content'), str) else
                         json.dumps(m.get('content')) for m in kwargs['messages'])
        for token in self.generate(list(prompt)):
            yield {'id': 'c', 'object': 'chat.completion.chunk', 'created': 0,
                   'model': 'm', 'choices': [{'index': 0, 'delta': {'content': token},
                                              'finish_reason': None}]}
        yield {'id': 'c', 'object': 'chat.completion.chunk', 'created': 0,
               'model': 'm', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]}
        if self.cache:
            self.cache.prefix = prompt
    def embed(self, texts, **kwargs):
        return [[1.0, 0.0] for text in texts], len(texts)
    def rank(self, query, docs):
        return [float(query in doc) for doc in docs]
    def close(self):
        self.closed = True
    def abort(self):
        self.aborted = True

class EngineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)
        f = ModelFile('m.gguf', '0'*64, 1, 'x/y', (), 'main')
        self.entry = Model('m', ('chat', 'tools'), 'llama.cpp', (f,), 1, 0, ('cpu',), 'Apache-2.0', 'x')
        (self.root/'m').mkdir()
        (self.root/'m'/'m.gguf').write_bytes(b'x')
        native = types.SimpleNamespace(Llama=Native, LlamaRAMCache=Cache,
            LlamaEmbedding=Native, grammar=lambda schema: ('grammar', schema),
            handler=lambda entry, path, think: None)
        self.engine = LlamaEngine(self.root, native=native)

    def test_prefix_cache_reused_after_stream_finish(self):
        h = self.engine.load(self.entry, {'models_dir': self.root, 'placement': 'cpu',
            'n_threads': 4, 'cache_bytes': 1024, 'think': False})
        body = {'messages': [{'role': 'system', 'content': 'long prompt '*100}],
                'tools': [{'type': 'function', 'function': {'name': 'f'}}]}
        list(self.engine.chat(h, body, lambda phase: None))
        body['messages'].append({'role': 'user', 'content': 'next'})
        list(self.engine.chat(h, body, lambda phase: None))
        self.assertEqual(h.native.evaluated, [1200, 4])
        self.assertEqual(h.native.kwargs['n_threads'], 4)
        self.assertEqual(h.native.kwargs['n_threads_batch'], 4)
        self.assertEqual(h.cache.capacity_bytes, 1024)
        self.engine.unload(h)
        self.assertTrue(h.native.closed)
        self.assertIsNone(h.cache)

    def test_oversized_snapshot_is_not_copied(self):
        self.engine.api.state_bytes = lambda native: 2048
        h = self.engine.load(self.entry, {'models_dir': self.root, 'placement': 'cpu',
            'n_threads': 2, 'cache_bytes': 1024, 'think': False})
        list(self.engine.chat(h, {'messages': [{'role': 'user', 'content': 'x'}]}, lambda phase: None))
        self.assertEqual(h.cache.prefix, '')
        self.assertFalse(h.cache.snapshot_fits)
        self.assertFalse(h.native.closed)

    def test_schema_is_grammar_and_thinking_is_off(self):
        h = self.engine.load(self.entry, {'models_dir': self.root, 'placement': 'cpu',
            'n_threads': 2, 'cache_bytes': 1024, 'think': False})
        schema = {'type': 'object', 'properties': {'ok': {'type': 'boolean'}}}
        list(self.engine.chat(h, {'messages': [{'role': 'user', 'content': 'x'}],
            'response_format': {'type': 'json_schema', 'json_schema': {'name': 'answer', 'schema': schema}}},
            lambda phase: None))
        self.assertEqual(h.native.last['grammar'], ('grammar', schema))
        self.assertEqual(h.native.last['reasoning_budget'], 0)
        self.assertEqual(h.usage['prompt_tokens'], 1)

    def test_required_and_named_choices_return_tool_calls(self):
        h = self.engine.load(self.entry, {'models_dir': self.root, 'placement': 'cpu',
            'n_threads': 2, 'cache_bytes': 1024, 'think': False})
        tool = {'type': 'function', 'function': {'name': 'f', 'parameters': {'type': 'object'}}}
        for choice in ('required', {'type': 'function', 'function': {'name': 'f'}}):
            with mock.patch.object(Native, 'output', '{"name":"f","arguments":{}}', create=True):
                chunks = list(self.engine.chat(h, {'messages': [{'role': 'user', 'content': 'do it'}],
                    'tools': [tool], 'tool_choice': choice}, lambda phase: None))
            self.assertEqual(chunks[0]['choices'][0]['delta']['tool_calls'][0]['function']['name'], 'f')
            self.assertEqual(chunks[-1]['choices'][0]['finish_reason'], 'tool_calls')
            self.assertNotIn('tools', h.native.last)

    def test_mmproj_and_vector_adapter(self):
        import dataclasses
        projector = ModelFile('mmproj.gguf', '0'*64, 1, 'x/y', (), 'mmproj')
        (self.root/'m/mmproj.gguf').write_bytes(b'p')
        entry = dataclasses.replace(self.entry, kinds=('chat', 'vision'), files=self.entry.files+(projector,))
        handler = mock.Mock()
        self.engine.api.handler = mock.Mock(return_value=handler)
        settings = {'models_dir': self.root, 'placement': 'cpu', 'n_threads': 2, 'cache_bytes': 1024, 'think': False}
        h = self.engine.load(entry, settings)
        self.engine.api.handler.assert_called_once_with(entry, self.root/'m/mmproj.gguf', False)
        image = {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,eA=='}}
        list(self.engine.chat(h, {'messages': [{'role': 'user', 'content': [image]}]}, lambda phase: None))
        self.assertEqual(h.native.last['messages'][0]['content'], [image])
        self.engine.unload(h)
        handler.close.assert_called_once()
        embed = self.engine.load(dataclasses.replace(self.entry, kinds=('embeddings',)), settings)
        self.assertEqual(self.engine.embed(embed, ['a', 'b']), ([[1.0, 0.0], [1.0, 0.0]], 2))
        rank = self.engine.load(dataclasses.replace(self.entry, kinds=('rerank',)), settings)
        self.assertEqual(rank.native.kwargs['pooling_type'], 4)
        self.assertEqual(self.engine.rank(rank, 'cat', ['dog', 'cat']), [0.0, 1.0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3.14 -m unittest tests.test_engine tests.test_toolcalls -v`
Expected: FAIL (adapter and parser absent).

- [ ] **Step 3: Import the parser byte-for-byte, then implement the adapter**

```bash
python3.14 -c 'import hashlib,os,pathlib,subprocess; repo=os.environ.get("WB_LLM_KIT_REPO","/home/operator/Work/repos/github.com/Xephex/playground_stuff"); data=subprocess.check_output(["git","-C",repo,"show","e85c04a356d2ff041911dc4af52140c6613f5585:llm-kit/run/toolcalls.py"]); assert hashlib.sha256(data).hexdigest()=="ee28dd10733e428d8da536d685be4676a4e74b10d095309027eb0e6a491e8856"; pathlib.Path("workbench/engines/toolcalls.py").write_bytes(data)'
```

`workbench/engines/llama.py`:
```python
from __future__ import annotations
import ctypes, dataclasses, json, os, pathlib, types, uuid
from workbench import install
from workbench.engines import toolcalls

_DLL_HANDLES = []
_DLL_DIRS = []

def preload(root):
    site = install.site_dir(root)
    if os.name == 'nt':
        dirs = sorted((site / 'nvidia').glob('*/bin'))
        for directory in dirs:
            _DLL_DIRS.append(os.add_dll_directory(str(directory)))
            if str(directory) not in os.environ.get('PATH', '').split(os.pathsep):
                os.environ['PATH'] = str(directory) + os.pathsep + os.environ.get('PATH', '')
        for name in ('cudart64_12.dll', 'cublasLt64_12.dll', 'cublas64_12.dll'):
            for directory in dirs:
                path = directory / name
                if path.is_file():
                    _DLL_HANDLES.append(ctypes.WinDLL(str(path)))
    else:
        for pattern in ('*/lib/libcudart.so*', '*/lib/libcublasLt.so*', '*/lib/libcublas.so*'):
            for path in sorted((site / 'nvidia').glob(pattern)):
                _DLL_HANDLES.append(ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL))

def dependencies(root):
    preload(root)
    from llama_cpp import Llama, LlamaRAMCache, LlamaGrammar
    from llama_cpp.llama_embedding import LlamaEmbedding
    from llama_cpp import llama_multimodal as mm
    def handler(entry, path, think):
        cls = (mm.Gemma4ChatHandler if entry.id.startswith('gemma4') else
               mm.Qwen35ChatHandler if entry.id.startswith(('qwen3.5', 'qwen3.6')) else
               mm.GenericMTMDChatHandler)
        kw = {'mmproj_path': str(path), 'verbose': False}
        if cls in (mm.Gemma4ChatHandler, mm.Qwen35ChatHandler):
            kw['enable_thinking'] = think
        else:
            kw['chat_format'] = None
        return cls(**kw)
    return types.SimpleNamespace(Llama=Llama, LlamaRAMCache=LlamaRAMCache,
        LlamaEmbedding=LlamaEmbedding,
        grammar=lambda schema: LlamaGrammar.from_json_schema(json.dumps(schema), verbose=False, allow_fetch=False),
        state_bytes=lambda native: __import__('llama_cpp.llama_cpp', fromlist=['llama_state_get_size']).llama_state_get_size(native._ctx.ctx),
        handler=handler)

def device_report(llm):
    try:
        from llama_cpp import llama_cpp as api, _ggml as ggml
        # Model-bound devices, not just the list of backends present on the machine.
        ptr = llm._model.model
        count = api.llama_model_n_devices(ptr)
        rows = []
        for i in range(count):
            dev = api.llama_model_get_device(ptr, i)
            free, total = ctypes.c_size_t(), ctypes.c_size_t()
            ggml.ggml_backend_dev_memory(dev, ctypes.byref(free), ctypes.byref(total))
            rows.append({'name': ggml.ggml_backend_dev_name(dev).decode(),
                         'type': int(ggml.ggml_backend_dev_type(dev)),
                         'free': free.value, 'total': total.value})
        return rows or [{'name': 'CPU', 'type': 0, 'free': 0, 'total': 0}]
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return []

@dataclasses.dataclass
class Handle:
    entry: object
    native: object
    cache: object
    placement: str
    usage: dict
    phase: object
    devices: list
    handler: object = None
    think: bool = False
    metrics: dict = dataclasses.field(default_factory=dict)

class LlamaEngine:
    def __init__(self, root, native=None):
        self.root, self.api = root, native
        self.injected = native is not None

    def load(self, entry, settings):
        if entry.engine != 'llama.cpp':
            raise ValueError(f'Unsupported engine {entry.engine}')
        if self.api is None:
            self.api = dependencies(self.root)
        folder = pathlib.Path(settings['models_dir']) / entry.folder
        main = next(f for f in entry.files if f.role == 'main')
        if not (folder / main.name).is_file():
            raise ValueError(f'{entry.id} is not installed; run models --add {entry.id}')
        small = bool({'embeddings', 'rerank'} & set(entry.kinds))
        defaults = {'n_ctx': 16384, 'flash_attn_type': 1, 'type_k': 8, 'type_v': 8}
        if entry.id in ('gpt-oss-20b', 'qwen3.6-35b-a3b'):
            defaults.update(n_ctx=32768, n_cpu_moe=99)
        if entry.id == 'qwen3.5-9b':
            defaults['n_ctx'] = 12288
        if small:
            defaults = {'n_ctx': 2048, 'n_batch': 2048, 'n_ubatch': 2048,
                        'pooling_type': 4 if 'rerank' in entry.kinds else -1}
        defaults.update({k: settings[k] for k in ('n_ctx', 'n_batch', 'n_ubatch', 'n_cpu_moe') if k in settings})
        defaults.update(n_threads=settings['n_threads'], n_threads_batch=settings['n_threads'],
                        n_gpu_layers=-1 if settings['placement'] == 'gpu' else 0,
                        verbose=False, no_perf=False)
        think = settings['think']
        handler = None
        projector = next((f for f in entry.files if f.role == 'mmproj'), None)
        if projector and not small:
            path = folder / projector.name
            if not path.is_file():
                raise ValueError(f'{entry.id}: missing vision projector {projector.name}')
            handler = self.api.handler(entry, path, think)
            defaults['chat_handler'] = handler
        base = self.api.LlamaEmbedding if small else self.api.Llama
        # Count actual sampled tokens and mark the first sampled token. Role-only
        # chunks arrive before prefill, so they cannot mark generation start.
        class Metered(base):
            def generate(native_self, tokens, *args, **kwargs):
                h.usage['prompt_tokens'] = len(tokens)
                h.phase('prompt')
                first = True
                for token in super().generate(tokens, *args, **kwargs):
                    if first:
                        h.phase('generation')
                        first = False
                    h.usage['completion_tokens'] += 1
                    yield token
        try:
            native = Metered(model_path=str(folder / main.name), **defaults)
        except BaseException:
            if handler is not None:
                handler.close()
            raise
        adapter = self
        class BudgetCache(self.api.LlamaRAMCache):
            def __bool__(cache_self):
                size = getattr(adapter.api, 'state_bytes', lambda model: 0)(native)
                cache_self.snapshot_fits = size <= settings['cache_bytes']
                return cache_self.snapshot_fits
        cache = None if small else BudgetCache(capacity_bytes=settings['cache_bytes'])
        if cache is not None:
            native.set_cache(cache)
        devices = [] if self.injected else device_report(native)
        placement = settings['placement']
        if not self.injected and placement == 'gpu' and devices:
            placement = 'gpu' if any(d['type'] in (1, 2) for d in devices) else 'cpu'
        h = Handle(entry, native, cache, placement, {}, lambda name: None, devices, handler, think)
        return h

    def unload(self, h):
        if h.cache is not None:
            h.native.set_cache(None)
            h.cache = None
        try:
            if h.handler is not None:
                h.handler.close()
        finally:
            h.native.close()

    def abort(self, h):
        h.native.abort()

    def chat(self, h, request, phase):
        h.usage = {'prompt_tokens': 0, 'completion_tokens': 0}
        h.phase = phase
        phase('prompt')
        think = request.get('think', h.think)
        if h.handler is not None and hasattr(h.handler, 'enable_thinking'):
            h.handler.enable_thinking = think
        kw = {k: request[k] for k in ('messages', 'tools', 'tool_choice', 'temperature',
              'top_p', 'max_tokens', 'stop', 'seed', 'frequency_penalty', 'presence_penalty') if k in request}
        if kw.get('tool_choice') == 'none':
            kw.pop('tools', None)
        fmt = request.get('response_format')
        if fmt and fmt['type'] != 'text':
            schema = fmt['json_schema']['schema'] if fmt['type'] == 'json_schema' else {'type': 'object'}
            kw['grammar'] = self.api.grammar(schema)
        choice = request.get('tool_choice')
        forced = choice == 'required' or isinstance(choice, dict)
        if forced:
            tools = request['tools']
            if isinstance(choice, dict):
                tools = [t for t in tools if t['function']['name'] == choice['function']['name']]
            alternatives = []
            for index, tool in enumerate(tools):
                parameters = json.loads(json.dumps(tool['function'].get('parameters', {'type': 'object'})))
                def rebase(node):
                    if isinstance(node, dict):
                        ref = node.get('$ref')
                        if isinstance(ref, str):
                            if not ref.startswith('#'):
                                raise ValueError('External schema references are not supported offline')
                            node['$ref'] = f'#/anyOf/{index}/properties/arguments'+ref[1:]
                        for key, value in node.items():
                            if key not in ('const', 'enum'):
                                rebase(value)
                    elif isinstance(node, list):
                        for value in node:
                            rebase(value)
                rebase(parameters)
                alternatives.append({'type': 'object', 'properties': {
                    'name': {'const': tool['function']['name']}, 'arguments': parameters},
                    'required': ['name', 'arguments'], 'additionalProperties': False})
            kw['grammar'] = self.api.grammar({'anyOf': alternatives})
            kw.pop('tools', None)
            kw.pop('tool_choice', None)
            kw['messages'] = [{'role': 'system', 'content':
                'Select one tool and return a JSON object with name and arguments. Tools: '+json.dumps(tools)}] + kw['messages']
        kw.update(stream=True, model=h.entry.id, reasoning_budget=-1 if think else 0)
        raw = iter(h.native.create_chat_completion(**kw))
        fixed = toolcalls._fix_stream(raw, kw.get('tools')) if kw.get('tools') else raw
        try:
            if forced:
                pieces, base = [], None
                for chunk in raw:
                    base = chunk
                    pieces.extend(c.get('delta', {}).get('content') or '' for c in chunk.get('choices', []))
                selected = json.loads(''.join(pieces))
                if selected.get('name') not in {t['function']['name'] for t in tools} or not isinstance(selected.get('arguments'), dict):
                    raise ValueError('Model did not return the required tool')
                call = {'index': 0, 'id': 'call_'+uuid.uuid4().hex[:24], 'type': 'function',
                        'function': {'name': selected['name'], 'arguments': json.dumps(selected['arguments'])}}
                yield dict(base, choices=[{'index': 0, 'delta': {'role': 'assistant', 'tool_calls': [call]}, 'finish_reason': None}])
                yield dict(base, choices=[{'index': 0, 'delta': {}, 'finish_reason': 'tool_calls'}])
            else:
                yield from fixed
            # The unchanged parser stops on finish_reason. llama.cpp saves RAM
            # state after that yield; exhaust the native iterator to commit it.
            for unused in raw:
                pass
        finally:
            close = getattr(raw, 'close', None)
            if close:
                close()
            h.usage['total_tokens'] = sum(h.usage.get(k, 0) for k in ('prompt_tokens', 'completion_tokens'))
            self.measure(h)

    def embed(self, h, texts):
        vectors, count = h.native.embed(texts, return_count=True, truncate=False)
        h.usage = {'prompt_tokens': count, 'total_tokens': count}
        self.measure(h)
        return vectors, count

    def rank(self, h, query, docs):
        scores = h.native.rank(query, docs)
        self.measure(h)
        if 'prompt_evaluated' in h.metrics:
            count = h.metrics['prompt_evaluated']
            h.usage = {'prompt_tokens': count, 'total_tokens': count}
        return scores

    def measure(self, h):
        if not self.injected:
            from llama_cpp import llama_cpp as api
            perf = api.llama_perf_context(h.native._ctx.ctx)
            h.metrics = dict(prompt_evaluated=int(perf.n_p_eval), prompt_ms=float(perf.t_p_eval),
                             generation_ms=float(perf.t_eval))

    def stats(self, h):
        # Status may run during inference. Read cached Python values only.
        return {'usage': dict(h.usage), 'placement': h.placement, 'devices': h.devices,
                'device_verified': bool(h.devices), 'cache_bytes': getattr(h.cache, 'cache_size', 0),
                'cache_snapshot_fits': getattr(h.cache, 'snapshot_fits', True), **h.metrics}
```

`BudgetCache` skips a snapshot larger than its allocation before `save_state` creates the transient copies. A skipped snapshot does not reset the resident native KV state; consecutive turns still reuse its prefix. Status distinguishes that case with `cache_snapshot_fits`. Embedding/rank methods retain the fork's verified `LlamaEmbedding` API. Embeddings reject over-context input instead of silently truncating. Device information can be unavailable; status must say "unverified" instead of claiming GPU acceleration. Task 3 reaccounts a verified CPU fallback before exposing a handle. The native `generate` counter includes sampled EOS/stop tokens; report it consistently as sampled completion tokens, not the number of SSE chunks.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3.14 -m unittest tests.test_engine tests.test_toolcalls -v`
Expected: PASS (13 unchanged parser tests and the adapter tests, with no native model loaded).

- [ ] **Step 5: Commit**
```bash
 git add workbench/engines/llama.py workbench/engines/toolcalls.py tests/test_engine.py tests/test_toolcalls.py
 git -c user.name=Xephex -c user.email=Xephex@users.noreply.github.com commit -m "feat: llama adapter with retained prefix cache and tool-call translation" -m "Co-Authored-By: GPT-6 Astra (Codex) <noreply@openai.com>"
```

---

### Task 3: Model memory admission, FIFO leases and LRU eviction

**Files:**
- Create: `workbench/server/manager.py`, `tests/server_support.py`, `tests/test_manager.py`

**Interfaces:**
- Consumes: the Task 2 engine interface; `hw.Profile`; `runtime.load_tuning`, `Footprint`, `CACHE_BYTES`.
- Produces: `Manager(config: Config, profile: Profile, models: list[Model], engine: object)`; `submit(mid: str) -> Ticket`; `lease(ticket: Ticket, cancel: threading.Event, loading: Callable[[], ContextManager] = nullcontext) -> ContextManager[Handle]`; `snapshot() -> dict`; `reserve_change(mid: str) -> None`; `release_change(mid: str) -> None`; `close() -> None`; `ManagerError(status: int, message: str)`.
- `Ticket` is assigned synchronously when a validated request enters the server, before thread scheduling. A ticket is consumed exactly once. Busy and waiting instances cannot be removed; queued cancelled tickets are removed without blocking followers.
- `data/tuning.json` is an input contract for Plan 4: `{model_id: {cpu_or_gpu: {sha256: {filename: hash}, settings: {n_ctx: int}, footprint: {ram: bytes, vram: bytes}}}}`. Stale hashes are ignored. Estimates remain labeled estimates until a matching tuned record exists.

- [ ] **Step 1: Write the failing tests and fake engine**

`tests/server_support.py`:
```python
import dataclasses, pathlib, tempfile, threading, types
from workbench.catalog import Model, ModelFile
from workbench.runtime import Config
from tests.test_catalog import profile

CALL = '<tool_call>{"name":"write_file","arguments":{"file_path":"hello.py","content":"print(2 + 3)"}}</tool_call>'
TOOLS = [{'type': 'function', 'function': {'name': 'write_file', 'parameters': {
    'type': 'object', 'properties': {'file_path': {'type': 'string'}, 'content': {'type': 'string'}}}}}]

class FakeEngine:
    def __init__(self):
        self.loads, self.unloads, self.requests = [], [], []
        self.fail = set()
        self.prefill = threading.Event()
        self.prefill.set()
        self.started = threading.Event()
    def load(self, entry, settings):
        self.loads.append((entry.id, dict(settings)))
        if entry.id in self.fail:
            raise RuntimeError('load failed')
        return types.SimpleNamespace(entry=entry, placement=settings['placement'],
            usage={}, devices=[], aborted=threading.Event())
    def unload(self, h):
        self.unloads.append(h.entry.id)
    def abort(self, h):
        h.aborted.set()
    def chat(self, h, request, phase):
        from workbench.engines.toolcalls import _fix_stream
        self.requests.append(request)
        self.started.set()
        phase('prompt')
        while not self.prefill.wait(.01):
            if h.aborted.is_set():
                return
        phase('generation')
        text = CALL if request.get('tools') and request.get('tool_choice') != 'none' else 'hello'
        def raw():
            for piece in (text[i:i+3] for i in range(0, len(text), 3)):
                if h.aborted.is_set():
                    return
                yield {'id': 'c', 'object': 'chat.completion.chunk', 'created': 0,
                       'model': h.entry.id, 'choices': [{'index': 0, 'delta': {'content': piece}, 'finish_reason': None}]}
            yield {'id': 'c', 'object': 'chat.completion.chunk', 'created': 0,
                   'model': h.entry.id, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]}
        yield from _fix_stream(raw(), request.get('tools'))
        h.usage = {'prompt_tokens': 12, 'completion_tokens': 3, 'total_tokens': 15}
    def embed(self, h, texts):
        return [[float(len(t)), 1.0] for t in texts], sum(len(t) for t in texts)
    def rank(self, h, query, docs):
        return [float(query in d) for d in docs]
    def stats(self, h):
        return {'usage': h.usage, 'placement': h.placement, 'devices': [], 'device_verified': False}

def model(mid, kinds=('chat', 'tools'), ram=1, vram=0):
    f = ModelFile('model.gguf', '0'*64, 1, 'Xephex/models', (), 'main')
    return Model(mid, kinds, 'llama.cpp', (f,), ram, vram, ('cpu', 'gpu-8'), 'Apache-2.0', 'test')

class Runtime:
    def __init__(self, models=None, ram=16, vram=0, **config):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)
        self.models = models or [model('chat'), model('embed', ('embeddings',)), model('rank', ('rerank',))]
        self.config = Config(self.root, self.root/'models', phoenix=False, **config)
        self.profile = profile(ram, vram)
        self.engine = FakeEngine()
        for m in self.models:
            folder = self.config.models/m.folder
            folder.mkdir(parents=True)
            (folder/'model.gguf').write_bytes(b'x')
            (folder/'model.gguf.ok').write_text('0'*64)
    def close(self):
        self.tmp.cleanup()
```

`tests/test_manager.py`:
```python
import threading, unittest
from concurrent.futures import ThreadPoolExecutor
from tests.server_support import Runtime, model
from workbench.server.manager import Manager, ManagerError

class ManagerTest(unittest.TestCase):
    def test_fifo_busy_lru_and_failed_load(self):
        r = Runtime([model('a', ram=3), model('b', ram=3), model('c', ram=99)], ram=10)
        self.addCleanup(r.close)
        mgr = Manager(r.config, r.profile, r.models, r.engine)
        self.addCleanup(mgr.close)
        cancel = threading.Event()
        first = mgr.submit('a')
        second, third = mgr.submit('a'), mgr.submit('a')
        order = []
        def run(ticket):
            with mgr.lease(ticket, cancel):
                order.append(ticket.number)
        with mgr.lease(first, cancel):
            with self.assertRaises(ManagerError):
                mgr.reserve_change('a')
            pool = ThreadPoolExecutor(2)
            f3 = pool.submit(run, third)
            f2 = pool.submit(run, second)
            self.assertEqual(r.engine.unloads, [])
        f2.result(2); f3.result(2); pool.shutdown()
        self.assertEqual(order, [second.number, third.number])
        with self.assertRaisesRegex(ManagerError, 'budget'):
            with mgr.lease(mgr.submit('c'), cancel):
                self.fail('oversized model admitted')
        r.engine.fail.add('b')
        with self.assertRaisesRegex(RuntimeError, 'load failed'):
            with mgr.lease(mgr.submit('b'), cancel):
                self.fail('failed model admitted')
        r.engine.fail.clear()
        with mgr.lease(mgr.submit('b'), cancel):
            self.assertIn('a', r.engine.unloads)
        self.assertEqual(mgr.snapshot()['queued'], 0)

    def test_cancelled_ticket_does_not_block_followers(self):
        r = Runtime([model('a')])
        self.addCleanup(r.close)
        mgr = Manager(r.config, r.profile, r.models, r.engine)
        self.addCleanup(mgr.close)
        first, second, third = mgr.submit('a'), mgr.submit('a'), mgr.submit('a')
        cancelled = threading.Event(); cancelled.set()
        with mgr.lease(first, threading.Event()):
            with self.assertRaises(ManagerError):
                with mgr.lease(second, cancelled):
                    self.fail('cancelled ticket ran')
        with mgr.lease(third, threading.Event()):
            pass
        self.assertEqual(len(r.engine.loads), 1)
        self.assertEqual(mgr.snapshot()['queued'], 0)

    def test_lru_uses_last_completed_lease(self):
        r = Runtime([model('a'), model('b'), model('c')], ram=10)
        self.addCleanup(r.close)
        mgr = Manager(r.config, r.profile, r.models, r.engine)
        self.addCleanup(mgr.close)
        for mid in ('a', 'b', 'a', 'c'):
            with mgr.lease(mgr.submit(mid), threading.Event()):
                pass
        self.assertEqual(r.engine.unloads, ['b'])

    def test_status_and_admission_do_not_block_loading(self):
        r = Runtime([model('a')])
        self.addCleanup(r.close)
        mgr = Manager(r.config, r.profile, r.models, r.engine)
        self.addCleanup(mgr.close)
        started, release = threading.Event(), threading.Event()
        original = r.engine.load
        def delayed(entry, settings):
            started.set()
            release.wait(2)
            return original(entry, settings)
        r.engine.load = delayed
        def use():
            with mgr.lease(mgr.submit('a'), threading.Event()):
                pass
        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(use)
            try:
                self.assertTrue(started.wait(1))
                self.assertEqual(mgr.snapshot()['loading'], ['a'])
                ticket = mgr.submit('a')
                cancelled = threading.Event(); cancelled.set()
                with self.assertRaises(ManagerError):
                    with mgr.lease(ticket, cancelled):
                        self.fail('cancelled ticket ran')
            finally:
                release.set()
            future.result(2)

    def test_verified_backend_fallback_reloads_with_cpu_budget(self):
        r = Runtime([model('chat', ram=2, vram=6)], vram=8)
        self.addCleanup(r.close)
        original = r.engine.load
        def fallback(entry, settings):
            handle = original(entry, settings)
            handle.placement = 'cpu'
            return handle
        r.engine.load = fallback
        mgr = Manager(r.config, r.profile, r.models, r.engine)
        self.addCleanup(mgr.close)
        with mgr.lease(mgr.submit('chat'), threading.Event()) as handle:
            self.assertEqual(handle.placement, 'cpu')
            self.assertGreater(mgr.slots['chat'].footprint.ram, 8_000_000_000)
        self.assertEqual([settings['placement'] for mid, settings in r.engine.loads], ['gpu', 'cpu'])
        self.assertEqual(r.engine.unloads, ['chat'])

    def test_small_model_cpu_fallback(self):
        r = Runtime([model('chat', ram=2, vram=6), model('embed', ('embeddings',), ram=2, vram=2)], vram=8)
        self.addCleanup(r.close)
        mgr = Manager(r.config, r.profile, r.models, r.engine)
        self.addCleanup(mgr.close)
        with mgr.lease(mgr.submit('chat'), threading.Event()):
            with mgr.lease(mgr.submit('embed'), threading.Event()) as h:
                self.assertEqual(h.placement, 'cpu')
        self.assertEqual(r.engine.unloads, [])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3.14 -m unittest tests.test_manager -v`
Expected: FAIL (manager absent).

- [ ] **Step 3: Implement the manager**

`workbench/server/manager.py`:
```python
from __future__ import annotations
import collections, contextlib, dataclasses, threading, time
from workbench import fetch
from workbench.runtime import GB, CACHE_BYTES, Footprint, load_tuning

class ManagerError(RuntimeError):
    def __init__(self, status, message):
        self.status = status
        super().__init__(message)

@dataclasses.dataclass(frozen=True)
class Ticket:
    number: int
    mid: str

@dataclasses.dataclass
class Slot:
    handle: object
    footprint: Footprint
    lock: object = dataclasses.field(default_factory=threading.Lock)
    active: bool = False
    used: float = 0
    source: str = 'catalog estimate'

class Manager:
    def __init__(self, config, profile, models, engine):
        self.config, self.profile, self.engine = config, profile, engine
        self.models = {m.id: m for m in models}
        self.budget = Footprint(max(0, profile.ram_bytes-4*GB), max(0, profile.gpu_budget_bytes-800_000_000))
        self.cv = threading.Condition()
        self.slots = {}
        self.loading = {}
        self.loader = threading.Lock()
        self.queues = collections.defaultdict(collections.deque)
        self.blocked = set()
        self.cpu_only = set()
        self.serial = 0
        self.closed = False

    def submit(self, mid):
        with self.cv:
            if self.closed:
                raise ManagerError(503, 'Server is shutting down')
            if mid not in self.models:
                raise ManagerError(404, f'Unknown model {mid}')
            if mid in self.blocked:
                raise ManagerError(409, 'Model installation is changing')
            if not fetch.is_installed(self.models[mid], self.config.models):
                raise ManagerError(409, f'{mid} is not installed; run models --add {mid}')
            self.serial += 1
            ticket = Ticket(self.serial, mid)
            self.queues[mid].append(ticket)
            self.cv.notify_all()
            return ticket

    def cache_capacity(self, m, placement, row):
        if {'embeddings', 'rerank'} & set(m.kinds):
            return 0
        base = row['footprint']['ram'] if row else int((m.ram_gb + (m.vram_gb if placement == 'cpu' else 0))*GB)
        return min(CACHE_BYTES, max(0, (self.budget.ram-base)//3))

    def cost(self, m, placement):
        row = load_tuning(self.config.root, m, placement)
        # Reserve resident cache plus two transient copies made by save_state.
        cache = 3*self.cache_capacity(m, placement, row)
        if row:
            f = row['footprint']
            return Footprint(f['ram']+cache, f['vram']), row
        gpu = int((m.vram_gb*GB) if m.vram_gb > 0 else m.total_bytes*1.2) if placement == 'gpu' else 0
        ram = int((m.ram_gb + (m.vram_gb if placement == 'cpu' else 0))*GB)
        return Footprint(ram+cache, gpu), {}

    def fits(self, cost):
        return (sum(s.footprint.ram for s in self.slots.values())+sum(f.ram for f in self.loading.values())+cost.ram <= self.budget.ram and
                sum(s.footprint.vram for s in self.slots.values())+sum(f.vram for f in self.loading.values())+cost.vram <= self.budget.vram)

    def possible(self, cost):
        return cost.ram <= self.budget.ram and cost.vram <= self.budget.vram

    def evict(self, mid):
        slot = self.slots[mid]
        self.engine.unload(slot.handle)
        del self.slots[mid]

    def admit(self, m, loading):
        placement = 'gpu' if self.budget.vram and m.id not in self.cpu_only else 'cpu'
        cost, row = self.cost(m, placement)
        small = bool({'embeddings', 'rerank'} & set(m.kinds))
        if placement == 'gpu' and (not self.possible(cost) or (small and not self.fits(cost))):
            placement = 'cpu'
            cost, row = self.cost(m, placement)
        if not self.possible(cost):
            raise ManagerError(503, f'{m.id} exceeds the RAM/VRAM budget')
        while not self.fits(cost):
            candidates = [(s.used, mid) for mid, s in self.slots.items() if not s.active and not self.queues[mid]]
            if not candidates:
                return None
            self.evict(min(candidates)[1])
        settings = {**m.settings, **row.get('settings', {}), 'models_dir': self.config.models,
                    'placement': placement, 'n_threads': self.profile.cpu_cores,
                    'cache_bytes': self.cache_capacity(m, placement, row), 'think': self.profile.tier != 'cpu'}
        self.loading[m.id] = cost
        self.cv.release()
        try:
            with self.loader, loading():
                handle = self.engine.load(m, settings)
        finally:
            self.cv.acquire()
            del self.loading[m.id]
            self.cv.notify_all()
        if handle.placement != placement:
            self.engine.unload(handle)
            self.cpu_only.add(m.id)
            # Reload with CPU-specific settings, footprint and cache allowance.
            # A GPU tuning record cannot describe the silently fallen-back load.
            return self.admit(m, loading)
        slot = Slot(handle, cost, source='tuned' if row else 'catalog estimate')
        self.slots[m.id] = slot
        return slot

    @contextlib.contextmanager
    def lease(self, ticket, cancel, loading=contextlib.nullcontext):
        slot = None
        acquired = False
        failed = True
        try:
            with self.cv:
                while True:
                    if cancel.is_set() or self.closed:
                        raise ManagerError(499, 'Request cancelled')
                    queue = self.queues[ticket.mid]
                    if not queue or ticket not in queue:
                        raise ManagerError(409, 'Ticket has already been consumed')
                    current = self.slots.get(ticket.mid)
                    if queue[0] == ticket and (current is None or not current.active):
                        slot = current or self.admit(self.models[ticket.mid], loading)
                        if slot is not None:
                            if cancel.is_set() or self.closed:
                                self.evict(ticket.mid)
                                raise ManagerError(499, 'Request cancelled')
                            slot.active = True
                            slot.lock.acquire()
                            acquired = True
                            break
                    self.cv.wait(.05)
            yield slot.handle
            failed = False
        finally:
            with self.cv:
                queue = self.queues[ticket.mid]
                if ticket in queue:
                    queue.remove(ticket)
                if acquired:
                    slot.active = False
                    slot.used = time.monotonic()
                    slot.lock.release()
                    if cancel.is_set() or failed:
                        self.evict(ticket.mid)
                self.cv.notify_all()

    def reserve_change(self, mid):
        with self.cv:
            if mid not in self.models:
                raise ManagerError(404, f'Unknown model {mid}')
            if mid in self.blocked or self.queues[mid] or (mid in self.slots and self.slots[mid].active):
                raise ManagerError(409, 'Model is busy')
            if mid in self.slots:
                self.evict(mid)
            self.blocked.add(mid)

    def release_change(self, mid):
        with self.cv:
            self.blocked.discard(mid)
            self.cv.notify_all()

    def snapshot(self):
        with self.cv:
            return {'budget': dataclasses.asdict(self.budget),
                'queued': max(0, sum(map(len, self.queues.values()))-sum(s.active for s in self.slots.values())-len(self.loading)),
                'loading': list(self.loading), 'loaded': [{'id': mid, 'busy': s.active, 'footprint': dataclasses.asdict(s.footprint),
                    'footprint_source': s.source, **self.engine.stats(s.handle)} for mid, s in self.slots.items()]}

    def close(self):
        with self.cv:
            self.closed = True
            self.cv.notify_all()
            while self.loading or any(s.active for s in self.slots.values()):
                self.cv.wait(.05)
            for mid in list(self.slots):
                self.evict(mid)
```

Model loading is serialized by a separate loader lock. Admission reserves bytes under the condition, then releases that condition during the slow native load; status, ticket admission and SSE remain responsive. Reservations count toward both budgets so two loads cannot spend the same free bytes. Inference on different resident models is concurrent. A load failure leaves no phantom reservation or ticket. Cache capacity is at most 512 MiB and shrinks when a model's base footprint nearly fills the RAM budget; a zero-capacity snapshot cache still leaves native KV prefix reuse enabled. This is admission control from estimates/tuning, not a claim that OS/GPU allocations are exactly predictable. Plan 4's measured footprint must include context and projector allocations and exclude the separately reserved cache.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3.14 -m unittest tests.test_manager -v`
Expected: PASS; the fake engine sees FIFO execution, CPU fallback and one LRU unload.

- [ ] **Step 5: Commit**
```bash
 git add workbench/server/manager.py tests/server_support.py tests/test_manager.py
 git -c user.name=Xephex -c user.email=Xephex@users.noreply.github.com commit -m "feat: budgeted model manager with FIFO leases and LRU eviction" -m "Co-Authored-By: GPT-6 Astra (Codex) <noreply@openai.com>"
```

---

### Task 4: Private OpenInference traces and optional offline Phoenix

**Files:**
- Create: `workbench/trace.py`, `workbench/phoenix_runner.py`, `tests/test_trace.py`

**Interfaces:**
- Consumes: `Config`; installed OTel SDK and OTLP HTTP exporter; `install.activate(root)` in the Phoenix child.
- Produces: `Tracing(config: Config, exporters: list | None = None)`; `child(name: str, attributes: dict | None = None) -> ContextManager[Span]`; `content(span: Span, key: str, value: object, capture: bool) -> None`; `close() -> None`; `RequestTracing(app: ASGIApp, tracing: Tracing)`; `Phases(tracing: Tracing)` with `change(name: str) -> None`, `close() -> None`.
- File format: one complete OTel span JSON object per line in `data/traces/spans.jsonl`; append-only with a process-local lock. Always retain the local copy, including when a viewer/exporter fails. Historical file replay is not performed automatically; this prevents duplicate historical spans.
- Optional `data/trace.json`: `{"langfuse": {"enabled": true, "host": "https://traces.example.org", "public_key": "pk", "secret_key": "sk"}}`. Missing/disabled means zero external trace traffic. Phoenix always binds to loopback, separate from `--share`.

- [ ] **Step 1: Write the failing tests**

`tests/test_trace.py`:
```python
import json, pathlib, tempfile, unittest
from unittest import mock
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from workbench.runtime import Config
from workbench.trace import Tracing, Phases

class Capture(SpanExporter):
    def __init__(self):
        self.spans = []
    def export(self, spans):
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS
    def shutdown(self):
        return None

class TraceTest(unittest.TestCase):
    def test_private_trace_contains_no_content(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            sink = Capture()
            t = Tracing(Config(root, root/'models', capture=False, phoenix=False), [sink])
            with t.child('request') as parent:
                t.content(parent, 'input.value', 'SECRET PROMPT', True)
                with t.child('model.load'):
                    pass
                phases = Phases(t)
                phases.change('prompt'); phases.change('generation'); phases.close()
                try:
                    with t.child('failed'):
                        raise ValueError('SECRET PROMPT')
                except ValueError:
                    pass
            t.close()
            saved = (root/'data/traces/spans.jsonl').read_text()
            self.assertNotIn('SECRET PROMPT', saved)
            self.assertNotIn('SECRET PROMPT', str([dict(s.attributes) for s in sink.spans]))
            child = next(s for s in sink.spans if s.name == 'model.load')
            parent = next(s for s in sink.spans if s.name == 'request')
            self.assertEqual(child.parent.span_id, parent.context.span_id)
            self.assertEqual({s.name for s in sink.spans}, {'request', 'model.load', 'prompt', 'generation', 'failed'})
            self.assertTrue(all(json.loads(line) for line in saved.splitlines()))

    def test_phoenix_file_runner_and_offline_environment(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            process = mock.Mock()
            process.poll.return_value = None
            opener = mock.MagicMock()
            opener.open.return_value.__enter__.return_value.status = 200
            with mock.patch('workbench.trace.importlib.util.find_spec', return_value=object()), \
                 mock.patch('workbench.trace.subprocess.Popen', return_value=process) as spawn, \
                 mock.patch('workbench.trace.urllib.request.build_opener', return_value=opener), \
                 mock.patch('workbench.trace.OTLPSpanExporter'):
                t = Tracing(Config(root, root/'models'))
                self.assertIsNotNone(t.viewer_url)
                argv = spawn.call_args.args[0]
                self.assertEqual(argv[1:], ['-m', 'workbench.phoenix_runner'])
                env = spawn.call_args.kwargs['env']
                self.assertEqual(env['PHOENIX_ALLOW_EXTERNAL_RESOURCES'], 'false')
                self.assertEqual(env['PHOENIX_TELEMETRY_ENABLED'], 'false')
                self.assertEqual(env['PHOENIX_HOST'], '127.0.0.1')
                t.close()
                process.terminate.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3.14 tools/test_server.py tests.test_trace -v`
Expected: FAIL (tracing module absent).

- [ ] **Step 3: Implement trace lifetime, exporters and the Phoenix file entry point**

`workbench/trace.py`:
```python
from __future__ import annotations
import base64, contextlib, importlib.util, json, os, pathlib, socket, subprocess, sys, threading, time
import urllib.request
from opentelemetry import trace as otel
from opentelemetry.propagate import extract
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult, SimpleSpanProcessor, BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.trace import SpanKind, Status, StatusCode
from openinference.semconv.trace import SpanAttributes

class FileExporter(SpanExporter):
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open('a', encoding='utf-8')
        self.lock = threading.Lock()
    def export(self, spans):
        with self.lock:
            for span in spans:
                self.file.write(json.dumps(json.loads(span.to_json()), separators=(',', ':'))+'\n')
            self.file.flush()
        return SpanExportResult.SUCCESS
    def shutdown(self):
        with self.lock:
            self.file.close()

class Tracing:
    def __init__(self, config, exporters=None):
        self.config, self.viewer, self.viewer_log = config, None, None
        self.viewer_url = None
        self.provider = TracerProvider(resource=Resource.create({'service.name': 'workbench'}))
        self.provider.add_span_processor(SimpleSpanProcessor(FileExporter(config.root/'data/traces/spans.jsonl')))
        self.tracer = self.provider.get_tracer('workbench')
        for exporter in exporters or []:
            self.provider.add_span_processor(SimpleSpanProcessor(exporter))
        if exporters is None:
            try:
                self.add_langfuse()
                if config.phoenix and importlib.util.find_spec('phoenix') is not None:
                    self.start_viewer()
            except BaseException:
                self.close()
                raise

    def start_viewer(self):
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            port = probe.getsockname()[1]
        folder = self.config.root/'data/phoenix'
        folder.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, PHOENIX_ALLOW_EXTERNAL_RESOURCES='false', PHOENIX_TELEMETRY_ENABLED='false',
                   PHOENIX_WORKING_DIR=str(folder), PHOENIX_HOST='127.0.0.1', PHOENIX_PORT=str(port))
        self.viewer_log = (folder/'server.log').open('ab')
        self.viewer = subprocess.Popen([sys.executable, '-m', 'workbench.phoenix_runner'],
            cwd=self.config.root, env=env, stdin=subprocess.DEVNULL, stdout=self.viewer_log, stderr=subprocess.STDOUT)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        url = f'http://127.0.0.1:{port}'
        for attempt in range(100):
            if self.viewer.poll() is not None:
                break
            try:
                with opener.open(url+'/healthz', timeout=.2) as reply:
                    if reply.status == 200:
                        self.viewer_url = url
                        self.provider.add_span_processor(BatchSpanProcessor(
                            OTLPSpanExporter(endpoint=url+'/v1/traces', timeout=2)))
                        return
            except OSError:
                time.sleep(.1)
        self.stop_viewer()

    def add_langfuse(self):
        path = self.config.root/'data/trace.json'
        if not path.exists():
            return
        row = json.loads(path.read_text()).get('langfuse', {})
        if not row.get('enabled', False):
            return
        host = row['host'].rstrip('/')
        if not host.startswith(('http://', 'https://')):
            raise ValueError('Langfuse host must be an HTTP(S) URL')
        auth = base64.b64encode((row['public_key']+':'+row['secret_key']).encode()).decode()
        self.provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(
            endpoint=host+'/api/public/otel/v1/traces', timeout=2,
            headers={'Authorization': 'Basic '+auth, 'x-langfuse-ingestion-version': '4'})))

    @contextlib.contextmanager
    def child(self, name, attributes=None):
        with self.tracer.start_as_current_span(name, attributes=attributes or {},
                record_exception=False, set_status_on_exception=False) as span:
            try:
                yield span
            except BaseException as error:
                span.set_attribute('error.type', type(error).__name__)
                span.set_status(Status(StatusCode.ERROR))
                raise

    def content(self, span, key, value, capture):
        if self.config.capture and capture:
            span.set_attribute(key, json.dumps(value, ensure_ascii=False))
            if key in ('input.value', 'output.value'):
                span.set_attribute(key.replace('.value', '.mime_type'), 'application/json')

    def stop_viewer(self):
        if self.viewer is not None:
            if self.viewer.poll() is None:
                self.viewer.terminate()
                try:
                    self.viewer.wait(5)
                except subprocess.TimeoutExpired:
                    self.viewer.kill(); self.viewer.wait()
            self.viewer = None
        if self.viewer_log is not None:
            self.viewer_log.close()
            self.viewer_log = None

    def close(self):
        self.provider.force_flush(timeout_millis=5000)
        self.provider.shutdown()
        self.stop_viewer()

class Phases:
    def __init__(self, tracing):
        self.tracing, self.span, self.name = tracing, None, None
    def change(self, name):
        if name == self.name:
            return
        self.close()
        self.span = self.tracing.tracer.start_span(name)
        self.name = name
    def close(self):
        if self.span is not None:
            self.span.end()
            self.span = None
        self.name = None

class RequestTracing:
    def __init__(self, app, tracing):
        self.app, self.tracing = app, tracing
    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        headers = {k.decode().lower(): v.decode() for k, v in scope.get('headers', [])}
        attrs = {'http.request.method': scope['method'], 'url.path': scope['path'],
                 SpanAttributes.OPENINFERENCE_SPAN_KIND: 'CHAIN'}
        with self.tracing.tracer.start_as_current_span(scope['method']+' '+scope['path'],
                context=extract(headers), kind=SpanKind.SERVER, attributes=attrs,
                record_exception=False, set_status_on_exception=False) as span:
            async def traced_send(message):
                if message['type'] == 'http.response.start':
                    span.set_attribute('http.response.status_code', message['status'])
                    if message['status'] >= 400:
                        span.set_status(Status(StatusCode.ERROR))
                await send(message)
            try:
                await self.app(scope, receive, traced_send)
            except BaseException as error:
                span.set_attribute('error.type', type(error).__name__)
                span.set_status(Status(StatusCode.ERROR))
                raise
```

`workbench/phoenix_runner.py`:
```python
import os, pathlib, runpy
from workbench import install

def main():
    root = pathlib.Path(__file__).resolve().parents[1]
    if not install.activate(root):
        raise SystemExit('Phoenix requires the installed tracing pack.')
    os.environ['PHOENIX_ALLOW_EXTERNAL_RESOURCES'] = 'false'
    os.environ['PHOENIX_TELEMETRY_ENABLED'] = 'false'
    import sys
    sys.argv = ['phoenix', 'serve']
    runpy.run_module('phoenix.server.main', run_name='__main__')

if __name__ == '__main__':
    main()
```

No automatic exception recording is used: third-party exception strings can contain prompts, tool arguments or images. Privacy-off requests retain timing, model, counts, memory, status and exception class only. A caller can opt out per request with `X-Workbench-Trace-Content: false`; app tabs in Plan 4 use that header. Parent `traceparent` is extracted for future agent spans. Generated tool calls get child spans in Task 5; executing tools is the client's responsibility.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3.14 tools/test_server.py tests.test_trace -v`
Expected: PASS; a private request produces linked child spans and no prompt in either exporter.

- [ ] **Step 5: Commit**
```bash
 git add workbench/trace.py workbench/phoenix_runner.py tests/test_trace.py
 git -c user.name=Xephex -c user.email=Xephex@users.noreply.github.com commit -m "feat: private local OpenInference traces and optional offline Phoenix" -m "Co-Authored-By: GPT-6 Astra (Codex) <noreply@openai.com>"
```

---

### Task 5: Owned API schemas, authenticated JSON routes and request tracing

**Files:**
- Create: `workbench/server/schemas.py`, `workbench/server/app.py`, `tests/test_server.py`

**Interfaces:**
- Consumes: Tasks 1-4; `fetch.is_installed(model, models_dir) -> bool`; `catalog.offered(models, profile) -> list[Offer]`.
- Produces: `create_app(config: Config, profile: Profile | None = None, models: list[Model] | None = None, engine: object | None = None, tracing: Tracing | None = None) -> FastAPI`; `Service.chat(ticket: Ticket, body: dict, emit: Callable[[dict], None], run: Run, capture: bool) -> dict`; `Run.cancel() -> None`; `Run.event: threading.Event`; `merge_chunks(chunks: list[dict], model: str, usage: dict) -> dict`.
- Own request types: `ChatRequest`, `EmbeddingRequest`, `RerankRequest`. OpenAI error envelope is `{"error":{"message":str,"type":str,"code":str}}`. Invalid input = 400; unknown model = 404; absent/wrong key = 401; not installed/busy = 409; insufficient budget = 503. Engine failures expose a generic 500, never raw prompts.
- `/v1/models` lists installed, runnable entries with `object: model`, `owned_by: workbench`, `created: 0`, plus `capabilities`. `/wb/models` includes all catalog entries and install/runnable reasons. `/wb/status` reports budgets, actual/unverified device information, queue depth and trace viewer URL.

- [ ] **Step 1: Write the failing HTTP tests**

`tests/test_server.py`:
```python
import json, unittest
from fastapi.testclient import TestClient
from tests.server_support import Runtime, TOOLS
from workbench.server.app import create_app
from workbench.runtime import Config

class ServerTest(unittest.TestCase):
    def setUp(self):
        self.runtime = Runtime()
        self.addCleanup(self.runtime.close)
        r = self.runtime
        self.app = create_app(r.config, r.profile, r.models, r.engine)
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def test_agent_history_and_nonstream_tools(self):
        history = [{'role': 'system', 'content': [{'type': 'text', 'text': 'Be useful.'}]},
            {'role': 'user', 'content': [{'type': 'text', 'text': 'Write a file.'}]},
            {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'old', 'type': 'function',
                'function': {'name': 'write_file', 'arguments': '{}'}}]},
            {'role': 'tool', 'tool_call_id': 'old', 'content': [{'type': 'text', 'text': 'done'}]}]
        reply = self.client.post('/v1/chat/completions', json={'model': 'chat', 'messages': history, 'tools': TOOLS})
        self.assertEqual(reply.status_code, 200, reply.text)
        choice = reply.json()['choices'][0]
        self.assertEqual(choice['finish_reason'], 'tool_calls')
        call = choice['message']['tool_calls'][0]
        self.assertEqual(call['function']['name'], 'write_file')
        self.assertEqual(json.loads(call['function']['arguments'])['content'], 'print(2 + 3)')
        seen = self.runtime.engine.requests[-1]['messages']
        self.assertEqual(seen[0]['content'], 'Be useful.')
        self.assertEqual(seen[-1]['content'], 'done')
        self.assertEqual(seen[2]['tool_calls'][0]['id'], 'old')

    def test_embeddings_rerank_and_unknown_model(self):
        r = self.client.post('/v1/embeddings', json={'model': 'embed', 'input': ['a', 'bb']})
        self.assertEqual(r.status_code, 200)
        self.assertEqual([x['index'] for x in r.json()['data']], [0, 1])
        r = self.client.post('/v1/rerank', json={'model': 'rank', 'query': 'cat',
            'documents': ['dog', {'text': 'cat'}, 'cat again'], 'top_n': 2})
        self.assertEqual([x['index'] for x in r.json()['results']], [1, 2])
        r = self.client.post('/v1/embeddings', json={'model': 'missing', 'input': 'a'})
        self.assertEqual(r.status_code, 404)
        r = self.client.post('/v1/embeddings', json={'model': 'chat', 'input': 'a'})
        self.assertEqual(r.status_code, 400)

    def test_images_and_schema_rejected_before_inference_when_invalid(self):
        for content in ([{'type': 'image_url', 'image_url': {'url': 'https://example.org/a.png'}}],
                        [{'type': 'input_audio', 'input_audio': {'data': 'x', 'format': 'wav'}}]):
            r = self.client.post('/v1/chat/completions', json={'model': 'chat',
                'messages': [{'role': 'user', 'content': content}]})
            self.assertEqual(r.status_code, 400)
        self.assertEqual(self.runtime.engine.loads, [])

    def test_per_request_privacy_hides_embedding_and_chat_content(self):
        headers = {'X-Workbench-Trace-Content': 'false'}
        reply = self.client.post('/v1/chat/completions', headers=headers, json={
            'model': 'chat', 'messages': [{'role': 'user', 'content': 'PRIVATE_SENTINEL'}]})
        self.assertEqual(reply.status_code, 200)
        reply = self.client.post('/v1/embeddings', headers=headers,
                                 json={'model': 'embed', 'input': 'PRIVATE_SENTINEL'})
        self.assertEqual(reply.status_code, 200)
        saved = (self.runtime.root/'data/traces/spans.jsonl').read_text()
        self.assertNotIn('PRIVATE_SENTINEL', saved)
        self.assertIn('workbench.memory.ram_bytes', saved)

    def test_nonloopback_requires_key_on_all_routes(self):
        r = self.runtime
        with self.assertRaises(ValueError):
            Config(r.root, r.config.models, host='0.0.0.0')
        cfg = Config(r.root, r.config.models, host='::', key='secret', phoenix=False)
        with TestClient(create_app(cfg, r.profile, r.models, r.engine)) as client:
            for path in ('/v1/models', '/v1/chat/completions', '/v1/embeddings', '/v1/rerank',
                         '/wb/status', '/wb/models', '/wb/jobs', '/docs', '/openapi.json'):
                self.assertEqual(client.get(path).status_code, 401, path)
            self.assertEqual(client.post('/wb/models', json={}).status_code, 401)
            self.assertEqual(client.get('/wb/status', headers={'Authorization': 'Bearer wrong'}).status_code, 401)
            self.assertEqual(client.get('/wb/status', headers={'Authorization': 'Bearer secret'}).status_code, 200)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3.14 tools/test_server.py tests.test_server -v`
Expected: FAIL (app absent).

- [ ] **Step 3: Implement schemas, service and JSON routes**

`workbench/server/schemas.py`:
```python
from __future__ import annotations
import base64
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator, field_validator

class Input(BaseModel):
    model_config = ConfigDict(extra='ignore', strict=True)

class TextPart(Input):
    type: Literal['text']
    text: str

class ImageURL(Input):
    url: str
    detail: Literal['auto', 'low', 'high'] = 'auto'
    @field_validator('url')
    @classmethod
    def local_image(cls, url):
        head, sep, data = url.partition(',')
        if not sep or head not in ('data:image/png;base64', 'data:image/jpeg;base64', 'data:image/webp;base64'):
            raise ValueError('Images must be inline PNG, JPEG or WebP base64 data URLs')
        if len(data) > 28_000_000:
            raise ValueError('Image exceeds 20 MB')
        base64.b64decode(data, validate=True)
        return url

class ImagePart(Input):
    type: Literal['image_url']
    image_url: ImageURL

class FunctionCall(Input):
    name: str
    arguments: str

class ToolCall(Input):
    id: str
    type: Literal['function'] = 'function'
    function: FunctionCall

class Message(Input):
    role: Literal['system', 'developer', 'user', 'assistant', 'tool', 'function']
    content: str | list[TextPart | ImagePart] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None
    @model_validator(mode='after')
    def valid(self):
        if self.role == 'tool' and not self.tool_call_id:
            raise ValueError('Tool messages need tool_call_id')
        if self.content is None and not (self.role == 'assistant' and self.tool_calls):
            raise ValueError('Content is required')
        if self.tool_calls and self.role != 'assistant':
            raise ValueError('Only assistant messages contain tool_calls')
        if isinstance(self.content, list) and any(isinstance(x, ImagePart) for x in self.content) and self.role != 'user':
            raise ValueError('Image parts require the user role')
        return self

class Function(Input):
    name: str = Field(min_length=1)
    description: str | None = None
    parameters: dict = Field(default_factory=lambda: {'type': 'object', 'properties': {}})
    strict: bool | None = None

class Tool(Input):
    type: Literal['function'] = 'function'
    function: Function

class FunctionName(Input):
    name: str

class NamedChoice(Input):
    type: Literal['function']
    function: FunctionName

class SchemaSpec(Input):
    name: str
    schema_: dict = Field(alias='schema')
    strict: bool = True

class ResponseFormat(Input):
    type: Literal['text', 'json_object', 'json_schema']
    json_schema: SchemaSpec | None = None
    @model_validator(mode='after')
    def valid(self):
        if self.type == 'json_schema' and self.json_schema is None:
            raise ValueError('json_schema requires a schema')
        return self

class StreamOptions(Input):
    include_usage: bool = False

class ChatRequest(Input):
    model: str | None = None
    messages: list[Message] = Field(min_length=1)
    stream: bool = False
    stream_options: StreamOptions | None = None
    tools: list[Tool] | None = None
    tool_choice: Literal['auto', 'none', 'required'] | NamedChoice | None = None
    response_format: ResponseFormat | None = None
    max_tokens: int = Field(default=1024, ge=1, le=131072)
    temperature: float = Field(default=.7, ge=0, le=2)
    top_p: float = Field(default=.95, gt=0, le=1)
    seed: int | None = None
    stop: str | list[str] | None = None
    frequency_penalty: float = Field(default=0, ge=-2, le=2)
    presence_penalty: float = Field(default=0, ge=-2, le=2)
    think: bool | None = None
    n: Literal[1] = 1
    @model_validator(mode='after')
    def valid(self):
        if self.tool_choice not in (None, 'none') and not self.tools:
            raise ValueError('tool_choice requires tools')
        if isinstance(self.tool_choice, NamedChoice):
            if self.tool_choice.function.name not in {t.function.name for t in self.tools or []}:
                raise ValueError('Named tool_choice is not declared in tools')
        if self.tools and self.response_format and self.response_format.type != 'text':
            raise ValueError('Use tools or response_format in a request, not both')
        return self
    def payload(self):
        result = self.model_dump(exclude_none=True, by_alias=True)
        for message in result['messages']:
            content = message.get('content')
            if isinstance(content, list) and all(p['type'] == 'text' for p in content):
                message['content'] = ''.join(p['text'] for p in content)
        return result

class EmbeddingRequest(Input):
    model: str | None = None
    input: str | list[str]
    encoding_format: Literal['float', 'base64'] = 'float'
    @model_validator(mode='after')
    def valid(self):
        if self.input == '' or self.input == [] or (isinstance(self.input, list) and any(not t for t in self.input)):
            raise ValueError('Embedding input must contain nonempty text')
        return self

class Document(Input):
    text: str

class RerankRequest(Input):
    model: str | None = None
    query: str = Field(min_length=1)
    documents: list[str | Document] = Field(min_length=1)
    top_n: int | None = Field(default=None, ge=1)
    return_documents: bool = False
```

`workbench/server/app.py`:
```python
from __future__ import annotations
import asyncio, base64, contextlib, dataclasses, hmac, json, math, struct, threading, time, uuid
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from opentelemetry import trace as otel
from opentelemetry.trace import Status, StatusCode
from openinference.semconv.trace import SpanAttributes
from workbench import catalog, fetch, hw
from workbench.engines.llama import LlamaEngine
from workbench.runtime import Config
from workbench.server.manager import Manager, ManagerError
from workbench.server.schemas import ChatRequest, EmbeddingRequest, RerankRequest
from workbench.trace import Tracing, RequestTracing, Phases

def error(status, message, kind='invalid_request_error'):
    return JSONResponse({'error': {'message': message, 'type': kind, 'code': str(status)}}, status_code=status)

class Auth:
    def __init__(self, app, key):
        self.app, self.key = app, key
    async def __call__(self, scope, receive, send):
        if scope['type'] == 'http' and self.key:
            supplied = dict(scope.get('headers', [])).get(b'authorization', b'')
            if not hmac.compare_digest(supplied, ('Bearer '+self.key).encode()):
                return await error(401, 'A valid bearer API key is required')(scope, receive, send)
        return await self.app(scope, receive, send)

class Run:
    def __init__(self, engine):
        self.engine, self.event, self.lock, self.handle = engine, threading.Event(), threading.Lock(), None
        self.stage = 'queued'
    def cancel(self):
        with self.lock:
            self.event.set()
            if self.handle is not None:
                self.engine.abort(self.handle)
    @contextlib.contextmanager
    def bind(self, handle):
        with self.lock:
            if self.event.is_set():
                raise ManagerError(499, 'Request cancelled')
            self.handle = handle
        try:
            yield
        finally:
            with self.lock:
                self.handle = None

def merge_chunks(chunks, model, usage):
    text, calls, finish = [], {}, 'stop'
    ident, created = 'chatcmpl-'+uuid.uuid4().hex, int(time.time())
    for chunk in chunks:
        ident, created = chunk.get('id', ident), chunk.get('created', created)
        for choice in chunk.get('choices', []):
            delta = choice.get('delta', {})
            if delta.get('content'):
                text.append(delta['content'])
            for call in delta.get('tool_calls') or []:
                target = calls.setdefault(call['index'], {'id': '', 'type': 'function',
                                                         'function': {'name': '', 'arguments': ''}})
                if call.get('id'):
                    target['id'] = call['id']
                for key in ('name', 'arguments'):
                    target['function'][key] += call.get('function', {}).get(key, '')
            finish = choice.get('finish_reason') or finish
    msg = {'role': 'assistant', 'content': ''.join(text) or None}
    if calls:
        msg['tool_calls'] = [calls[i] for i in sorted(calls)]
    return {'id': ident, 'object': 'chat.completion', 'created': created, 'model': model,
            'choices': [{'index': 0, 'message': msg, 'finish_reason': finish, 'logprobs': None}], 'usage': usage}

class Service:
    def __init__(self, manager, tracing):
        self.manager, self.engine, self.tracing = manager, manager.engine, tracing
    def chat(self, ticket, body, emit, run, capture):
        chunks = []
        phases = Phases(self.tracing)
        span = otel.get_current_span()
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, 'LLM')
        span.set_attribute(SpanAttributes.LLM_MODEL_NAME, ticket.mid)
        self.tracing.content(span, 'input.value', body['messages'], capture)
        def phase(name):
            run.stage = name
            phases.change(name)
        @contextlib.contextmanager
        def loading():
            run.stage = 'loading'
            with self.tracing.child('model.load'):
                yield
        try:
            with self.manager.lease(ticket, run.event, loading) as handle:
                with run.bind(handle):
                    for chunk in self.engine.chat(handle, body, phase):
                        if run.event.is_set():
                            raise ManagerError(499, 'Request cancelled')
                        chunks.append(chunk)
                        emit(chunk)
                    if run.event.is_set():
                        raise ManagerError(499, 'Request cancelled')
                    stats = self.engine.stats(handle)
                    usage = stats.get('usage', {})
                    result = merge_chunks(chunks, ticket.mid, usage)
                    span.set_attribute('workbench.placement', handle.placement)
                    footprint = self.manager.slots[ticket.mid].footprint
                    span.set_attribute('workbench.memory.ram_bytes', footprint.ram)
                    span.set_attribute('workbench.memory.vram_bytes', footprint.vram)
                    span.set_attribute('workbench.memory.source', self.manager.slots[ticket.mid].source)
                    self.tracing.content(span, 'output.value', result['choices'][0]['message'], capture)
                    for key, attr in (('prompt_tokens', SpanAttributes.LLM_TOKEN_COUNT_PROMPT),
                                      ('completion_tokens', SpanAttributes.LLM_TOKEN_COUNT_COMPLETION),
                                      ('total_tokens', SpanAttributes.LLM_TOKEN_COUNT_TOTAL)):
                        if key in usage:
                            span.set_attribute(attr, usage[key])
                    for call in result['choices'][0]['message'].get('tool_calls', []):
                        with self.tracing.child('tool.call', {SpanAttributes.OPENINFERENCE_SPAN_KIND: 'TOOL'}) as child:
                            if capture and self.tracing.config.capture:
                                child.set_attribute('tool.name', call['function']['name'])
                            self.tracing.content(child, 'input.value', call['function']['arguments'], capture)
                    return result
        except BaseException as exc:
            span.set_attribute('error.type', type(exc).__name__)
            span.set_status(Status(StatusCode.ERROR))
            raise
        finally:
            phases.close()

def create_app(config, profile=None, models=None, engine=None, tracing=None):
    profile = profile or hw.detect(config.models)
    if profile.os not in ('windows', 'linux'):
        raise ValueError('The Stage 1 server supports Windows and Linux.')
    models = models if models is not None else catalog.load(config.root/'catalog.toml')
    if config.default_model is not None:
        selected = next((m for m in models if m.id == config.default_model), None)
        if selected is None or not ({'chat', 'reasoning'} & set(selected.kinds)):
            raise ValueError('The selected default model must be a catalog chat model')
        if not fetch.is_installed(selected, config.models):
            raise ValueError(f'{selected.id} is not installed; run models --add {selected.id}')
    engine = engine or LlamaEngine(config.root)
    tracing = tracing or Tracing(config)
    manager = Manager(config, profile, models, engine)
    service = Service(manager, tracing)
    closed = False
    async def cleanup():
        nonlocal closed
        if closed:
            return
        closed = True
        await asyncio.to_thread(manager.close)
        await asyncio.to_thread(tracing.close)
    @contextlib.asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            await cleanup()
    app = FastAPI(lifespan=lifespan)
    app.state.cleanup = cleanup
    app.state.manager, app.state.service, app.state.tracing = manager, service, tracing
    app.state.instance = uuid.uuid4().hex
    admission = asyncio.Lock()
    async def submit(mid):
        async with admission:
            return await asyncio.to_thread(manager.submit, mid)
    app.add_middleware(Auth, key=config.key)
    app.add_middleware(RequestTracing, tracing=tracing)

    @app.exception_handler(ManagerError)
    async def manager_error(request, exc):
        return error(exc.status, str(exc))
    @app.exception_handler(RequestValidationError)
    async def invalid(request, exc):
        locations = ', '.join('.'.join(map(str, x['loc'])) for x in exc.errors())
        return error(400, 'Invalid request fields: '+locations)
    @app.exception_handler(ValueError)
    async def engine_input(request, exc):
        return error(400, 'Input is invalid or exceeds the model context; shorten it or choose another model.')
    @app.exception_handler(Exception)
    async def failed(request, exc):
        return error(500, 'Inference failed; inspect local status and trace error type.', 'server_error')

    def choose(mid, kind):
        choices = [m for m in models if kind in m.kinds or (kind == 'chat' and 'reasoning' in m.kinds)]
        if mid is None and kind == 'chat' and config.default_model is not None:
            mid = config.default_model
        if mid is None:
            runnable = {o.model.id for o in catalog.offered(choices, profile) if o.runnable}
            ready = [m for m in choices if m.id in runnable and fetch.is_installed(m, config.models)
                     and (manager.possible(manager.cost(m, 'cpu')[0]) or
                          (manager.budget.vram and manager.possible(manager.cost(m, 'gpu')[0])))]
            ready.sort(key=lambda m: profile.tier not in m.recommended_for)
            if not ready:
                raise ManagerError(409, f'No installed {kind} model; run models first')
            mid = ready[0].id
        if mid not in manager.models:
            raise ManagerError(404, f'Unknown model {mid}')
        if mid not in {m.id for m in choices}:
            raise ManagerError(400, f'{mid} does not support {kind}')
        return mid

    async def execute_safe(operation):
        try:
            return await asyncio.to_thread(operation)
        except ManagerError:
            raise
        except ValueError:
            raise ManagerError(400, 'Input is invalid or exceeds the model context') from None
        except Exception:
            raise ManagerError(500, 'Inference failed; inspect the trace error type') from None

    @app.get('/v1/models')
    async def list_models():
        return {'object': 'list', 'data': [{'id': o.model.id, 'object': 'model', 'owned_by': 'workbench',
            'created': 0, 'capabilities': list(o.model.kinds)} for o in catalog.offered(models, profile)
            if o.runnable and fetch.is_installed(o.model, config.models)]}
    @app.get('/wb/models')
    async def model_state():
        return {'models': [{'id': o.model.id, 'kinds': list(o.model.kinds), 'recommended': profile.tier in o.model.recommended_for,
            'installed': fetch.is_installed(o.model, config.models), 'runnable': o.runnable, 'reason': o.reason}
            for o in catalog.offered(models, profile)]}
    @app.get('/wb/status')
    async def status():
        return {'service': 'workbench', 'instance': app.state.instance, 'tier': profile.tier,
                'physical_cores': profile.cpu_cores, 'trace_viewer': tracing.viewer_url,
                **(await asyncio.to_thread(manager.snapshot))}
    @app.get('/wb/jobs')
    async def jobs():
        return {'jobs': []}
    @app.post('/v1/chat/completions')
    async def chat(body: ChatRequest, request: Request):
        mid = choose(body.model, 'chat')
        payload = body.payload()
        if payload.get('tools') and 'tools' not in manager.models[mid].kinds:
            raise ManagerError(400, f'{mid} does not support tools')
        images = any(isinstance(m.get('content'), list) and any(p['type'] == 'image_url' for p in m['content'])
                     for m in payload['messages'])
        if images and ('vision' not in manager.models[mid].kinds or
                       not any(f.role == 'mmproj' for f in manager.models[mid].files)):
            raise ManagerError(400, f'{mid} does not support images')
        if body.stream:
            raise ManagerError(400, 'This route currently accepts JSON responses only')
        ticket = await submit(mid)
        run = Run(engine)
        capture = request.headers.get('x-workbench-trace-content', 'true').lower() != 'false'
        try:
            return await asyncio.to_thread(service.chat, ticket, payload, lambda chunk: None, run, capture)
        except asyncio.CancelledError:
            run.cancel()
            raise
    @app.post('/v1/embeddings')
    async def embeddings(body: EmbeddingRequest, request: Request):
        mid = choose(body.model, 'embeddings')
        ticket = await submit(mid)
        texts = [body.input] if isinstance(body.input, str) else body.input
        capture = request.headers.get('x-workbench-trace-content', 'true').lower() != 'false'
        def execute():
            with manager.lease(ticket, threading.Event(), lambda: tracing.child('model.load')) as handle:
                with tracing.child('embedding', {SpanAttributes.OPENINFERENCE_SPAN_KIND: 'EMBEDDING',
                                                SpanAttributes.LLM_MODEL_NAME: mid}) as span:
                    tracing.content(span, 'input.value', texts, capture)
                    vectors, count = engine.embed(handle, texts)
                    tracing.content(span, 'output.value', vectors, capture)
                    span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, count)
                    span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_TOTAL, count)
                    footprint = manager.slots[mid].footprint
                    span.set_attribute('workbench.memory.ram_bytes', footprint.ram)
                    span.set_attribute('workbench.memory.vram_bytes', footprint.vram)
                    span.set_attribute('workbench.memory.source', manager.slots[mid].source)
            if len(vectors) != len(texts) or any(not math.isfinite(x) for v in vectors for x in v):
                raise RuntimeError('Invalid embedding output')
            values = vectors if body.encoding_format == 'float' else [
                base64.b64encode(struct.pack('<'+'f'*len(v), *v)).decode() for v in vectors]
            return {'object': 'list', 'model': mid, 'data': [{'object': 'embedding', 'index': i, 'embedding': v}
                for i, v in enumerate(values)], 'usage': {'prompt_tokens': count, 'total_tokens': count}}
        return await execute_safe(execute)
    @app.post('/v1/rerank')
    async def rerank(body: RerankRequest, request: Request):
        mid = choose(body.model, 'rerank')
        ticket = await submit(mid)
        docs = [d if isinstance(d, str) else d.text for d in body.documents]
        capture = request.headers.get('x-workbench-trace-content', 'true').lower() != 'false'
        def execute():
            with manager.lease(ticket, threading.Event(), lambda: tracing.child('model.load')) as handle:
                with tracing.child('rerank', {SpanAttributes.OPENINFERENCE_SPAN_KIND: 'RERANKER',
                                             SpanAttributes.LLM_MODEL_NAME: mid}) as span:
                    tracing.content(span, 'input.value', {'query': body.query, 'documents': docs}, capture)
                    scores = engine.rank(handle, body.query, docs)
                    tracing.content(span, 'output.value', scores, capture)
                    footprint = manager.slots[mid].footprint
                    span.set_attribute('workbench.memory.ram_bytes', footprint.ram)
                    span.set_attribute('workbench.memory.vram_bytes', footprint.vram)
                    span.set_attribute('workbench.memory.source', manager.slots[mid].source)
                    count = engine.stats(handle).get('usage', {}).get('prompt_tokens')
                    if count is not None:
                        span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, count)
            if len(scores) != len(docs) or any(not math.isfinite(s) for s in scores):
                raise RuntimeError('Invalid reranker output')
            order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:body.top_n]
            return {'model': mid, 'results': [dict(index=i, relevance_score=scores[i],
                **({'document': {'text': docs[i]}} if body.return_documents else {})) for i in order]}
        return await execute_safe(execute)
    return app
```

The empty jobs list is the accurate state before Task 7 adds model mutation jobs. Task 6 replaces the JSON-only branch and cancellation wrapper with the common worker bridge. No client can ask the server to fetch arbitrary paths or URLs. Rerank ties preserve input index; `top_n > len(documents)` naturally returns all documents.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3.14 tools/test_server.py tests.test_server tests.test_trace -v`
Expected: PASS; non-streamed tool calls survive the owned route response without a llama.cpp response-model patch.

- [ ] **Step 5: Commit**
```bash
 git add workbench/server/schemas.py workbench/server/app.py tests/test_server.py
 git -c user.name=Xephex -c user.email=Xephex@users.noreply.github.com commit -m "feat: owned authenticated chat embedding rerank and status routes" -m "Co-Authored-By: GPT-6 Astra (Codex) <noreply@openai.com>"
```

---

### Task 6: SSE keep-alives, bounded streaming and cancellation

**Files:**
- Create: `workbench/server/streaming.py`, `tests/test_streaming.py`
- Modify: `workbench/server/app.py` (`Run`, `Service.chat`, chat handler, lifespan)

**Interfaces:**
- Consumes: `Service.chat`, `Run`, `Ticket`, `ChatRequest`; sse-starlette `EventSourceResponse`, `ServerSentEvent`.
- Produces: `Worker(service: Service, ticket: Ticket, body: dict, capture: bool)` with `start() -> None`, `cancel() -> None`, `join() -> None`, `done: Event`, `result: dict | None`, `failure: Exception | None`; `respond(request: Request, worker: Worker, stream: bool, include_usage: bool, ping: float = 10) -> Response | dict`; `shutdown(workers: set[Worker]) -> Awaitable[None]`.
- SSE frames use `data: <JSON>`, then exactly one `data: [DONE]`. Optional terminal usage has `choices: []`. Keep-alive comments report `queued`, `loading`, `prompt` or `generation`; no fabricated percentage. Transport comments may not reset an agent framework's own chunk timer, so document the dcode timeout caveat.

- [ ] **Step 1: Write the failing tests**

`tests/test_streaming.py`:
```python
import asyncio, json, unittest
from fastapi.testclient import TestClient
from tests.server_support import Runtime, TOOLS
from workbench.server.app import create_app

class StreamingTest(unittest.IsolatedAsyncioTestCase):
    async def test_keepalive_during_prefill_and_disconnect(self):
        r = Runtime()
        self.addCleanup(r.close)
        r.engine.prefill.clear()
        app = create_app(r.config, r.profile, r.models, r.engine)
        app.state.ping = .02
        output, incoming = asyncio.Queue(), asyncio.Queue()
        body = json.dumps({'model': 'chat', 'stream': True,
            'messages': [{'role': 'user', 'content': 'long prompt'}]}).encode()
        await incoming.put({'type': 'http.request', 'body': body, 'more_body': False})
        scope = {'type': 'http', 'asgi': {'version': '3.0'}, 'http_version': '1.1',
            'method': 'POST', 'scheme': 'http', 'path': '/v1/chat/completions', 'raw_path': b'/v1/chat/completions',
            'query_string': b'', 'root_path': '', 'headers': [(b'content-type', b'application/json')],
            'client': ('127.0.0.1', 1234), 'server': ('127.0.0.1', 8080)}
        async with app.router.lifespan_context(app):
            running = asyncio.create_task(app(scope, incoming.get, output.put))
            seen = b''
            for attempt in range(20):
                event = await asyncio.wait_for(output.get(), 1)
                seen += event.get('body', b'')
                if b': prompt' in seen:
                    break
            self.assertIn(b': prompt', seen)
            self.assertNotIn(b'data: {', seen)
            self.assertTrue(r.engine.started.is_set())
            await incoming.put({'type': 'http.disconnect'})
            await asyncio.wait_for(running, 2)
            self.assertEqual(app.state.manager.snapshot()['queued'], 0)
            self.assertEqual(app.state.manager.snapshot()['loaded'], [])
            self.assertIn('chat', r.engine.unloads)

class ToolStreamTest(unittest.TestCase):
    def test_streamed_tools_and_done(self):
        r = Runtime()
        self.addCleanup(r.close)
        with TestClient(create_app(r.config, r.profile, r.models, r.engine)) as client:
            reply = client.post('/v1/chat/completions', json={'model': 'chat', 'stream': True,
                'stream_options': {'include_usage': True}, 'tools': TOOLS,
                'messages': [{'role': 'user', 'content': 'write hello.py'}]})
            self.assertEqual(reply.status_code, 200)
            values = [line[6:] for line in reply.text.splitlines() if line.startswith('data: ')]
            self.assertEqual(values.count('[DONE]'), 1)
            data = [json.loads(v) for v in values if v != '[DONE]']
            calls = [call for ch in data for c in ch['choices'] for call in c['delta'].get('tool_calls', [])]
            self.assertEqual(calls[0]['function']['name'], 'write_file')
            self.assertTrue(any(c.get('finish_reason') == 'tool_calls' for ch in data for c in ch['choices']))
            self.assertEqual(data[-1]['usage']['total_tokens'], 15)
            text = ''.join(c['delta'].get('content', '') for ch in data for c in ch['choices'])
            self.assertNotIn('<tool_call>', text)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3.14 tools/test_server.py tests.test_streaming -v`
Expected: FAIL (stream requests currently return 400).

- [ ] **Step 3: Implement the worker bridge and wire chat lifetime to it**

`workbench/server/streaming.py`:
```python
from __future__ import annotations
import asyncio, contextvars, json, queue, threading
import anyio
from sse_starlette.sse import EventSourceResponse, ServerSentEvent
from workbench.server.manager import ManagerError

class Worker:
    def __init__(self, service, ticket, body, capture):
        from workbench.server.app import Run
        self.service, self.ticket, self.body, self.capture = service, ticket, body, capture
        self.run = Run(service.engine)
        self.events = queue.Queue(maxsize=32)
        self.done = threading.Event()
        self.result, self.failure = None, None
        self.context = contextvars.copy_context()
        self.thread = threading.Thread(target=lambda: self.context.run(self.execute), daemon=False)
    def emit(self, chunk):
        while not self.run.event.is_set():
            try:
                self.events.put(chunk, timeout=.05)
                return
            except queue.Full:
                continue
        raise ManagerError(499, 'Request cancelled')
    def execute(self):
        try:
            self.result = self.service.chat(self.ticket, self.body, self.emit, self.run, self.capture)
        except Exception as exc:
            self.failure = exc
        finally:
            self.done.set()
    def start(self):
        self.thread.start()
    def cancel(self):
        self.run.cancel()
    def join(self):
        self.thread.join()

async def finish(worker):
    # AnyIO cancels the SSE generator on disconnect. Shield cleanup until the
    # native worker has stopped; otherwise a subsequent lease could race it.
    with anyio.CancelScope(shield=True):
        while not worker.done.is_set():
            if worker.run.event.is_set():
                # The native generator clears its abort flag at entry. Repeat
                # the signal until it observes cancellation or finishes.
                worker.cancel()
            await asyncio.sleep(.05)
        await asyncio.to_thread(worker.join)

def failure_payload(exc):
    if isinstance(exc, ManagerError):
        return {'error': {'message': str(exc), 'type': 'server_error', 'code': str(exc.status)}}
    if isinstance(exc, ValueError):
        return {'error': {'message': 'Input is invalid or exceeds the model context',
                          'type': 'invalid_request_error', 'code': '400'}}
    return {'error': {'message': 'Inference failed', 'type': 'server_error', 'code': '500'}}

async def respond(request, worker, stream, include_usage, ping=10):
    workers = request.app.state.workers
    workers.add(worker)
    worker.start()
    async def pieces():
        try:
            yield ServerSentEvent(comment=worker.run.stage)
            while not worker.done.is_set() or not worker.events.empty():
                try:
                    item = worker.events.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(.01)
                    continue
                yield {'data': json.dumps(item, ensure_ascii=False)}
            if worker.failure:
                yield {'data': json.dumps(failure_payload(worker.failure))}
            elif include_usage:
                result = worker.result
                yield {'data': json.dumps({'id': result['id'], 'object': 'chat.completion.chunk',
                    'created': result['created'], 'model': result['model'], 'choices': [], 'usage': result['usage']})}
            yield {'data': '[DONE]'}
        finally:
            if not worker.done.is_set():
                worker.cancel()
            await finish(worker)
            workers.discard(worker)
    if stream:
        return EventSourceResponse(pieces(), ping=ping, send_timeout=30,
            ping_message_factory=lambda: ServerSentEvent(comment=worker.run.stage),
            headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})
    try:
        while not worker.done.is_set():
            # Drain the same bounded queue for a JSON caller; Service retains
            # the final answer, and backpressure must not deadlock the producer.
            while not worker.events.empty():
                worker.events.get_nowait()
            if await request.is_disconnected():
                worker.cancel()
                raise ManagerError(499, 'Request cancelled')
            await asyncio.sleep(.01)
        if worker.failure:
            failure = failure_payload(worker.failure)['error']
            raise ManagerError(int(failure['code']), failure['message']) from None
        return worker.result
    finally:
        if not worker.done.is_set():
            worker.cancel()
        await finish(worker)
        workers.discard(worker)

async def shutdown(workers):
    current = list(workers)
    for worker in current:
        worker.cancel()
    for worker in current:
        await finish(worker)
    workers.clear()
```

In `app.py`, add this import:
```python
from workbench.server.streaming import Worker, respond, shutdown
```
Replace the `cleanup` function with:
```python
    async def cleanup():
        nonlocal closed
        if closed:
            return
        closed = True
        await shutdown(app.state.workers)
        await asyncio.to_thread(manager.close)
        await asyncio.to_thread(tracing.close)
```
After creating the FastAPI object, add:
```python
    app.state.workers = set()
    app.state.ping = 10
```
Replace the chat handler's code starting at `if body.stream:` through its cancellation handler with:
```python
        ticket = await submit(mid)
        capture = request.headers.get('x-workbench-trace-content', 'true').lower() != 'false'
        worker = Worker(service, ticket, payload, capture)
        return await respond(request, worker, body.stream,
            bool(body.stream_options and body.stream_options.include_usage), app.state.ping)
```

Errors after SSE headers are sent become an OpenAI error frame followed by `[DONE]`; they cannot retroactively change HTTP 200. Validation and model lookup happen before headers. A disconnect requests native abort, waits for native finalization, and then evicts the interrupted handle so a partial prefix cannot pollute the next turn. An already-complete turn preserves its cache even if the consumer closes after receiving the answer.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3.14 tools/test_server.py tests.test_streaming tests.test_server tests.test_manager -v`
Expected: PASS, including actual ASGI comment delivery while the fake native engine is blocked.

- [ ] **Step 5: Commit**
```bash
 git add workbench/server/streaming.py workbench/server/app.py tests/test_streaming.py
 git -c user.name=Xephex -c user.email=Xephex@users.noreply.github.com commit -m "feat: responsive SSE with keep-alives and safe cancellation" -m "Co-Authored-By: GPT-6 Astra (Codex) <noreply@openai.com>"
```

---

### Task 7: Durable model jobs and `/wb/models` mutations

**Files:**
- Create: `workbench/server/jobs.py`, `tests/test_jobs.py`
- Modify: `workbench/server/app.py` (jobs lifecycle, `GET /wb/jobs`, `POST /wb/models`)

**Interfaces:**
- Consumes: `Manager.reserve_change/release_change`, `fetch.fetch_model(model, models_dir, work_dir, progress) -> Path`, `fetch.remove_file(file, folder) -> None`, `fetch.remaining_bytes(file, folder) -> int`.
- Produces: `Jobs(config: Config, manager: Manager, tracing: Tracing, download: Callable = fetch.fetch_model)`; `submit(action: str, mid: str) -> dict`; `cancel(job_id: str) -> dict`; `snapshot() -> list[dict]`; `close() -> None`.
- POST bodies: `{"action":"install"|"remove","model":"catalog-id"}` or `{"action":"cancel","job_id":"id"}`. Return 202 with the job record. Cancellation is cooperative at chunk boundaries, preserving `.partial`/`.fetching` for a retry. It never deletes another job's files.
- Jobs use one FIFO background thread and `data/jobs/<id>.json`, updated by atomic replace. States: `queued`, `running`, `succeeded`, `failed`, `cancelled`, `interrupted`. Closing a browser has no effect. Server shutdown requests cancellation and waits for the current git chunk; restart marks unfinished jobs interrupted and a new install request resumes through the existing fetcher.

- [ ] **Step 1: Write the failing tests**

`tests/test_jobs.py`:
```python
import json, threading, time, unittest
from tests.server_support import Runtime
from workbench.server.manager import Manager, ManagerError
from workbench.server.jobs import Jobs
from workbench.trace import Tracing

class JobsTest(unittest.TestCase):
    def test_cancel_and_restart_preserve_results_and_partial(self):
        r = Runtime()
        self.addCleanup(r.close)
        mgr = Manager(r.config, r.profile, r.models, r.engine)
        self.addCleanup(mgr.close)
        trace = Tracing(r.config, [])
        self.addCleanup(trace.close)
        started, release = threading.Event(), threading.Event()
        def download(model, models, work, progress):
            (models/model.folder/'model.gguf.partial').write_bytes(b'partial')
            started.set()
            release.wait(2)
            progress('model.gguf', 1, 2)
            return models/model.folder
        jobs = Jobs(r.config, mgr, trace, download)
        row = jobs.submit('install', 'chat')
        self.assertTrue(started.wait(1))
        with self.assertRaises(ManagerError):
            mgr.submit('chat')
        jobs.cancel(row['id'])
        release.set(); jobs.close()
        self.assertEqual(jobs.snapshot()[0]['state'], 'cancelled')
        self.assertTrue((r.config.models/'chat/model.gguf.partial').exists())
        path = r.root/'data/jobs/crashed.json'
        path.write_text(json.dumps({'id': 'crashed', 'state': 'running', 'model': 'chat', 'action': 'install'}))
        restarted = Jobs(r.config, mgr, trace, download)
        self.addCleanup(restarted.close)
        states = {j['id']: j['state'] for j in restarted.snapshot()}
        self.assertEqual(states[row['id']], 'cancelled')
        self.assertEqual(states['crashed'], 'interrupted')

    def test_remove_and_failed_download_release_model(self):
        r = Runtime()
        self.addCleanup(r.close)
        mgr = Manager(r.config, r.profile, r.models, r.engine)
        self.addCleanup(mgr.close)
        trace = Tracing(r.config, [])
        self.addCleanup(trace.close)
        def broken(*args, **kwargs):
            raise OSError('disk full')
        jobs = Jobs(r.config, mgr, trace, broken)
        bad = jobs.submit('install', 'chat')
        deadline = time.monotonic()+2
        while jobs.snapshot()[0]['state'] in ('queued', 'running'):
            self.assertLess(time.monotonic(), deadline)
            time.sleep(.01)
        jobs.close()
        self.assertEqual(jobs.snapshot()[0]['state'], 'failed')
        again = Jobs(r.config, mgr, trace, broken)
        row = again.submit('remove', 'chat')
        deadline = time.monotonic()+2
        while next(j for j in again.snapshot() if j['id'] == row['id'])['state'] in ('queued', 'running'):
            self.assertLess(time.monotonic(), deadline)
            time.sleep(.01)
        again.close()
        self.assertFalse((r.config.models/'chat/model.gguf').exists())
        self.assertNotIn('chat', mgr.blocked)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3.14 tools/test_server.py tests.test_jobs -v`
Expected: FAIL (jobs module absent).

- [ ] **Step 3: Implement durable jobs and attach the routes**

`workbench/server/jobs.py`:
```python
from __future__ import annotations
import contextvars, json, pathlib, shutil, threading, time, uuid
from concurrent.futures import ThreadPoolExecutor
from pydantic import model_validator
from typing import Literal
from workbench import fetch
from workbench.server.manager import ManagerError
from workbench.server.schemas import Input

class ModelAction(Input):
    action: Literal['install', 'remove', 'cancel']
    model: str | None = None
    job_id: str | None = None
    @model_validator(mode='after')
    def valid(self):
        if self.action == 'cancel' and (not self.job_id or self.model):
            raise ValueError('cancel requires only job_id')
        if self.action != 'cancel' and (not self.model or self.job_id):
            raise ValueError('install/remove require only model')
        return self

class Cancelled(Exception):
    pass

class Jobs:
    def __init__(self, config, manager, tracing, download=fetch.fetch_model):
        self.config, self.manager, self.tracing, self.download = config, manager, tracing, download
        self.folder = config.root/'data/jobs'
        self.folder.mkdir(parents=True, exist_ok=True)
        self.lock, self.rows, self.events = threading.RLock(), {}, {}
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='workbench-jobs')
        self.closed = False
        for path in sorted(self.folder.glob('*.json')):
            row = json.loads(path.read_text())
            self.rows[row['id']] = row
            if row['state'] in ('queued', 'running'):
                row['state'] = 'interrupted'
                row['error'] = 'Server stopped; repeat the model action to resume.'
                self.save(row)

    def save(self, row):
        path = self.folder/(row['id']+'.json')
        temp = path.with_suffix('.json.tmp')
        temp.write_text(json.dumps(row, indent=2), encoding='utf-8')
        temp.replace(path)

    def submit(self, action, mid):
        if action not in ('install', 'remove'):
            raise ManagerError(400, 'Unsupported model action')
        with self.lock:
            if self.closed:
                raise ManagerError(503, 'Job worker is shutting down')
            self.manager.reserve_change(mid)
            try:
                model = self.manager.models[mid]
                if action == 'install':
                    self.config.models.mkdir(parents=True, exist_ok=True)
                    need = sum(fetch.remaining_bytes(f, self.config.models/model.folder) for f in model.files)
                    scratch = max((c.bytes for f in model.files for c in f.chunks), default=0)
                    if need and need*1.02+scratch > shutil.disk_usage(self.config.models).free:
                        raise ManagerError(409, 'Not enough disk for model and one scratch chunk')
                ident = uuid.uuid4().hex
                row = {'id': ident, 'action': action, 'model': mid, 'state': 'queued',
                       'created': time.time(), 'progress': {}, 'result': None, 'error': None}
                self.rows[ident], self.events[ident] = row, threading.Event()
                self.save(row)
                context = contextvars.copy_context()
                self.pool.submit(context.run, self.execute, ident)
                return dict(row)
            except BaseException:
                self.manager.release_change(mid)
                raise

    def execute(self, ident):
        with self.lock:
            row, event = self.rows[ident], self.events[ident]
        model = self.manager.models[row['model']]
        def progress(name, done, total):
            if event.is_set():
                raise Cancelled()
            with self.lock:
                row['progress'] = {'file': name, 'done': done, 'total': total}
                self.save(row)
        try:
            with self.lock:
                row['state'] = 'running'
                self.save(row)
            if event.is_set():
                raise Cancelled()
            with self.tracing.child('job.'+row['action'], {'workbench.model': model.id}):
                if row['action'] == 'install':
                    self.download(model, self.config.models, self.config.root/'.workbench/fetch', progress=progress)
                else:
                    folder = self.config.models/model.folder
                    for file in model.files:
                        if event.is_set():
                            raise Cancelled()
                        fetch.remove_file(file, folder)
                    if folder.is_dir() and not any(folder.iterdir()):
                        folder.rmdir()
            state, failure = 'succeeded', None
        except Cancelled:
            state, failure = 'cancelled', None
        except Exception as exc:
            state, failure = 'failed', type(exc).__name__+': model action failed; retry to resume'
        finally:
            self.manager.release_change(model.id)
        with self.lock:
            row.update(state=state, error=failure, finished=time.time(),
                       result={'installed': fetch.is_installed(model, self.config.models)})
            try:
                self.save(row)
            except OSError:
                row.update(state='failed', error='Unable to persist the job result; check free disk and permissions')

    def cancel(self, ident):
        with self.lock:
            if ident not in self.rows:
                raise ManagerError(404, 'Unknown job')
            if self.rows[ident]['state'] in ('queued', 'running'):
                self.events[ident].set()
            return dict(self.rows[ident])

    def snapshot(self):
        with self.lock:
            return json.loads(json.dumps(list(self.rows.values())))

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            for event in self.events.values():
                event.set()
        self.pool.shutdown(wait=True)
```

In `app.py`, import jobs and construct the worker immediately after `service`:
```python
from workbench.server.jobs import Jobs, ModelAction
```
```python
    jobs_worker = Jobs(config, manager, tracing)
```
In `cleanup`, after shutting down inference workers and before closing the manager:
```python
        await asyncio.to_thread(jobs_worker.close)
```
Replace the `GET /wb/jobs` function and add POST:
```python
    @app.get('/wb/jobs')
    async def jobs():
        return {'jobs': jobs_worker.snapshot()}
    @app.post('/wb/models', status_code=202)
    async def change_model(body: ModelAction):
        if body.action == 'cancel':
            return jobs_worker.cancel(body.job_id)
        return await asyncio.to_thread(jobs_worker.submit, body.action, body.model)
```

Install/remove target a catalog entry, never a client-provided file path. The model manager holds its change reservation until the job finishes, so inference cannot race an unlink or incomplete install. Plan 4 can poll the records without keeping a browser connection alive.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3.14 tools/test_server.py tests.test_jobs tests.test_server tests.test_streaming -v`
Expected: PASS; cancellation preserves partial downloads, terminal results survive restart, and model locks are released after failure.

- [ ] **Step 5: Commit**
```bash
 git add workbench/server/jobs.py workbench/server/app.py tests/test_jobs.py
 git -c user.name=Xephex -c user.email=Xephex@users.noreply.github.com commit -m "feat: durable model installation jobs and safe model mutations" -m "Co-Authored-By: GPT-6 Astra (Codex) <noreply@openai.com>"
```

---

### Task 8: Thin commands and explicit temporary-server ownership

**Files:**
- Create: `workbench/client.py`, `workbench/tools/commands.py`, `tests/test_commands.py`
- Modify: `workbench/cli.py` (parser, early lazy dispatch, status extension)

**Interfaces:**
- Consumes: `install.activate`, `create_app`, existing `cli.ROOT/MODELS/CATALOG`, `ui.say`, all HTTP routes.
- Produces: stdlib-only `Client(base: str = 'http://127.0.0.1:8080', key: str | None = None)` with `request(method: str, path: str, body: dict | None = None, timeout: float = 3600) -> dict`, `stream(path: str, body: dict, progress: Callable[[str], None] | None = None) -> Iterator[dict]`, `peek() -> dict | None`; `session(root: Path, model: str | None = None, base: str = DEFAULT_BASE) -> ContextManager[Client]`; `commands.add_parsers(subparsers) -> None`; `commands.dispatch(args: Namespace, root: Path, models_dir: Path) -> int`; `commands.status(args, profile, models, root, models_dir) -> int`.
- Commands: `serve [--host HOST] [--port 8080] [--share] [--model ID] [--json] [--no-trace-content] [--no-trace-viewer]`; `chat [--system TEXT] [--think]`; `ask TEXT [--think]`; `embed TEXT...`; `rank QUERY DOCUMENT... [--top-n N]`; `status`. All accept `--model` and `--json`; clients accept `--server URL`. Keys come from `WORKBENCH_API_KEY` or `.workbench/api-key.txt`, never a subprocess argument.
- `status` probes an existing server and never auto-starts one. Other inference clients auto-start only at the default loopback URL. Explicit remote/server URLs must already be running. A session stops only the `Popen` object it created; it never kills a PID read from disk.

- [ ] **Step 1: Write the failing tests**

`tests/test_commands.py`:
```python
import argparse, contextlib, io, json, pathlib, tempfile, unittest
from unittest import mock
from workbench import cli
from workbench.client import Client, session, ClientError
from workbench.tools import commands

class CommandsTest(unittest.TestCase):
    def test_existing_server_is_never_started_or_stopped(self):
        with mock.patch.object(Client, 'peek', return_value={'service': 'workbench', 'instance': 'existing'}), \
             mock.patch('workbench.client.subprocess.Popen') as popen:
            with session(pathlib.Path('.')) as client:
                self.assertIsInstance(client, Client)
            popen.assert_not_called()

    def test_owned_server_stops_on_error(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            process = mock.Mock()
            process.poll.return_value = None
            with mock.patch.object(Client, 'peek', side_effect=[None, {'service': 'workbench', 'instance': 'owned'}]), \
                 mock.patch('workbench.client.uuid.uuid4', return_value=mock.Mock(hex='owned')), \
                 mock.patch('workbench.client.subprocess.Popen', return_value=process):
                with self.assertRaisesRegex(ClientError, 'consumer failed'):
                    with session(root):
                        raise ClientError('consumer failed')
            process.stdin.write.assert_called_once_with('stop\n')
            process.wait.assert_called_once_with(timeout=30)
            process.terminate.assert_not_called()

    def test_ask_uses_http_and_json_has_no_progress(self):
        fake = mock.Mock()
        fake.request.return_value = {'choices': [{'message': {'content': 'hello'}}]}
        @contextlib.contextmanager
        def connected(*args, **kwargs):
            yield fake
        args = argparse.Namespace(cmd='ask', text='hi', model='chat', json=True,
            think=False, server='http://127.0.0.1:8080')
        out = io.StringIO()
        with mock.patch.object(commands, 'session', connected), mock.patch('sys.stdout', out):
            self.assertEqual(commands.dispatch(args, pathlib.Path('.'), pathlib.Path('models')), 0)
        self.assertEqual(json.loads(out.getvalue()), fake.request.return_value)
        self.assertEqual(fake.request.call_args.args[:2], ('POST', '/v1/chat/completions'))
        self.assertEqual(fake.request.call_args.args[2]['messages'][-1]['content'], 'hi')

    def test_bootstrap_help_does_not_import_server(self):
        import builtins
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.startswith(('fastapi', 'llama_cpp', 'opentelemetry', 'workbench.server')):
                raise AssertionError('Bootstrap imported '+name)
            return original(name, *args, **kwargs)
        with mock.patch('builtins.__import__', side_effect=guarded), mock.patch('sys.stdout', io.StringIO()):
            for argv in (['status', '--help'], ['models', '--help'], ['setup', '--help']):
                with self.assertRaises(SystemExit) as caught:
                    cli.main(argv)
                self.assertEqual(caught.exception.code, 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3.14 -m unittest tests.test_commands -v`
Expected: FAIL (client module absent).

- [ ] **Step 3: Implement stdlib HTTP and server ownership**

`workbench/client.py`:
```python
from __future__ import annotations
import contextlib, json, os, pathlib, subprocess, sys, time, urllib.error, urllib.request, uuid

DEFAULT_BASE = 'http://127.0.0.1:8080'
class ClientError(RuntimeError):
    pass

class Client:
    def __init__(self, base=DEFAULT_BASE, key=None):
        self.base, self.key = base.rstrip('/'), key
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def open(self, method, path, body=None, timeout=3600):
        headers = {'Content-Type': 'application/json'}
        if self.key:
            headers['Authorization'] = 'Bearer '+self.key
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base+path, data=data, headers=headers, method=method)
        try:
            return self.opener.open(req, timeout=timeout)
        except urllib.error.HTTPError as exc:
            try:
                message = json.loads(exc.read()).get('error', {}).get('message', f'HTTP {exc.code}')
            except ValueError:
                message = f'HTTP {exc.code}'
            raise ClientError(message) from None
    def request(self, method, path, body=None, timeout=3600):
        with self.open(method, path, body, timeout) as response:
            return json.load(response)
    def peek(self):
        try:
            result = self.request('GET', '/wb/status', timeout=1)
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, ConnectionRefusedError):
                return None
            raise ClientError('The server is unreachable or responding too slowly') from None
        except ConnectionRefusedError:
            return None
        except (TimeoutError, ConnectionError):
            raise ClientError('The server is responding too slowly; no second server was started') from None
        if result.get('service') != 'workbench':
            raise ClientError('The selected port is serving another application')
        return result
    def stream(self, path, body, progress=None):
        with self.open('POST', path, dict(body, stream=True), timeout=60) as response:
            data = []
            for raw in response:
                line = raw.decode('utf-8').rstrip('\r\n')
                if line.startswith(':'):
                    if progress:
                        progress(line[1:].strip())
                elif line.startswith('data:'):
                    data.append(line[5:].lstrip())
                elif not line and data:
                    value = '\n'.join(data)
                    data.clear()
                    if value == '[DONE]':
                        return
                    chunk = json.loads(value)
                    if 'error' in chunk:
                        raise ClientError(chunk['error']['message'])
                    yield chunk
            raise ClientError('Server disconnected before [DONE]')

def read_key(root):
    key = os.environ.get('WORKBENCH_API_KEY')
    if key:
        return key
    try:
        return (root/'.workbench/api-key.txt').read_text().strip() or None
    except FileNotFoundError:
        return None

def share_key(root):
    key = read_key(root)
    if key:
        return key
    path = root/'.workbench/api-key.txt'
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return read_key(root)
    with os.fdopen(fd, 'w') as file:
        key = uuid.uuid4().hex+uuid.uuid4().hex
        file.write(key+'\n')
    return key

@contextlib.contextmanager
def session(root, model=None, base=DEFAULT_BASE):
    client = Client(base, read_key(root))
    if client.peek() is not None:
        yield client
        return
    if base.rstrip('/') != DEFAULT_BASE:
        raise ClientError('The requested server is not running')
    token = uuid.uuid4().hex
    folder = root/'data/server'
    folder.mkdir(parents=True, exist_ok=True)
    argv = [sys.executable, str(root/'workbench.py'), 'serve', '--host', '127.0.0.1', '--port', '8080']
    if model:
        argv += ['--model', model]
    log = (folder/(token+'.log')).open('ab')
    env = dict(os.environ, WORKBENCH_MANAGED_SERVER='1', WORKBENCH_SERVER_INSTANCE=token)
    try:
        process = subprocess.Popen(argv, cwd=root, env=env, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, text=True)
    except BaseException:
        log.close()
        raise
    try:
        deadline = time.monotonic()+60
        while time.monotonic() < deadline:
            state = client.peek()
            if state and state['instance'] == token:
                break
            if process.poll() is not None:
                # A concurrent client may have won the bind. Reuse only a
                # verified Workbench server; this process owns no other PID.
                if state:
                    break
                raise ClientError(f'Server failed to start; see {folder/(token+".log")}')
            time.sleep(.1)
        else:
            raise ClientError('Server startup timed out')
        yield client
    finally:
        if process.poll() is None:
            try:
                process.stdin.write('stop\n'); process.stdin.flush()
                process.wait(timeout=30)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
        if process.stdin:
            process.stdin.close()
        log.close()
```

- [ ] **Step 4: Implement command formatting and lazy dispatch**

`workbench/tools/commands.py`:
```python
from __future__ import annotations
import dataclasses, json, os, pathlib, sys, threading, uuid
from workbench import catalog, fetch, install
from workbench.client import Client, ClientError, DEFAULT_BASE, read_key, share_key, session
from workbench.runtime import Config

def add_parsers(sub):
    serve = sub.add_parser('serve')
    serve.add_argument('--host', default='127.0.0.1')
    serve.add_argument('--port', type=int, default=8080)
    serve.add_argument('--share', action='store_true')
    serve.add_argument('--no-trace-content', action='store_true')
    serve.add_argument('--no-trace-viewer', action='store_true')
    parsers = [serve]
    for name in ('chat', 'ask', 'embed', 'rank'):
        parser = sub.add_parser(name)
        parser.add_argument('--server', default=DEFAULT_BASE)
        parsers.append(parser)
        if name == 'chat':
            parser.add_argument('--system', default='You are a helpful assistant.')
        if name == 'ask':
            parser.add_argument('text')
        if name == 'embed':
            parser.add_argument('texts', nargs='+')
        if name == 'rank':
            parser.add_argument('query')
            parser.add_argument('documents', nargs='+')
            parser.add_argument('--top-n', type=int)
        if name in ('chat', 'ask'):
            parser.add_argument('--think', action='store_true', default=None)
    for parser in parsers:
        parser.add_argument('--model')
        parser.add_argument('--json', action='store_true')

def serve(a, root, models_dir):
    if not install.activate(root):
        raise ClientError('Packages are not installed. Run setup first.')
    # Third-party imports occur only after activation, including on Windows.
    from workbench.server.app import create_app
    import uvicorn
    key = share_key(root) if a.share else read_key(root)
    config = Config(root, models_dir, host='0.0.0.0' if a.share else a.host,
        port=a.port, key=key, capture=not a.no_trace_content, phoenix=not a.no_trace_viewer,
        default_model=a.model)
    app = create_app(config)
    app.state.instance = os.environ.get('WORKBENCH_SERVER_INSTANCE', app.state.instance)
    server = uvicorn.Server(uvicorn.Config(app, host=config.host, port=config.port,
        log_level='warning', access_log=False))
    if os.environ.get('WORKBENCH_MANAGED_SERVER') == '1':
        def watch_owner():
            sys.stdin.readline()
            server.should_exit = True
        threading.Thread(target=watch_owner, daemon=True).start()
    host = f'[{config.host}]' if ':' in config.host else config.host
    address = f'http://{host}:{config.port}'
    if a.json:
        print(json.dumps({'address': address, 'authenticated': bool(key)}), flush=True)
    else:
        print(f'Serving at {address}; Ctrl+C stops the server.', flush=True)
        if a.share:
            print('Sharing requires the key in WORKBENCH_API_KEY or .workbench/api-key.txt.', flush=True)
    try:
        server.run()
    finally:
        import asyncio
        asyncio.run(app.state.cleanup())
    return 0

def answer(client, a, messages):
    body = {'messages': messages}
    if a.model:
        body['model'] = a.model
    if a.think is not None:
        body['think'] = a.think
    if a.json:
        result = client.request('POST', '/v1/chat/completions', body)
        print(json.dumps(result, ensure_ascii=False))
        return result['choices'][0]['message']
    chunks = []
    last_phase = None
    def progress(phase):
        nonlocal last_phase
        if phase != last_phase:
            label = {'prompt': 'Reading the prompt', 'queued': 'Waiting for model',
                     'loading': 'Loading model', 'generation': 'Generating'}
            print(label.get(phase, phase), file=sys.stderr, flush=True)
            last_phase = phase
    for chunk in client.stream('/v1/chat/completions', body, progress):
        for choice in chunk.get('choices', []):
            piece = choice['delta'].get('content') or ''
            chunks.append(piece)
            print(piece, end='', flush=True)
            for call in choice['delta'].get('tool_calls') or []:
                print(json.dumps({'tool_call': call}), flush=True)
    print()
    return {'role': 'assistant', 'content': ''.join(chunks)}

def dispatch(a, root, models_dir):
    try:
        if a.cmd == 'serve':
            return serve(a, root, models_dir)
        with session(root, a.model, a.server) as client:
            if a.cmd == 'ask':
                answer(client, a, [{'role': 'user', 'content': a.text}])
            elif a.cmd == 'chat':
                messages = [{'role': 'system', 'content': a.system}]
                try:
                    while True:
                        try:
                            prompt = input('you> ' if not a.json else '').strip()
                        except EOFError:
                            break
                        if prompt in ('exit', 'quit'):
                            break
                        if prompt == '/reset':
                            messages = messages[:1]
                        elif prompt:
                            messages.append({'role': 'user', 'content': prompt})
                            messages.append(answer(client, a, messages))
                finally:
                    folder = root/'data/chats'
                    folder.mkdir(parents=True, exist_ok=True)
                    (folder/(uuid.uuid4().hex+'.json')).write_text(json.dumps(
                        {'model': a.model, 'messages': messages}, ensure_ascii=False, indent=2), encoding='utf-8')
            elif a.cmd == 'embed':
                body = {'input': a.texts}
                if a.model:
                    body['model'] = a.model
                result = client.request('POST', '/v1/embeddings', body)
                if a.json:
                    print(json.dumps(result))
                else:
                    for row in result['data']:
                        print(json.dumps(row['embedding']))
            elif a.cmd == 'rank':
                body = {'query': a.query, 'documents': a.documents}
                if a.model:
                    body['model'] = a.model
                if a.top_n is not None:
                    body['top_n'] = a.top_n
                result = client.request('POST', '/v1/rerank', body)
                if a.json:
                    print(json.dumps(result))
                else:
                    for row in result['results']:
                        print(f"{row['index']}\t{row['relevance_score']:.6f}\t{a.documents[row['index']]}")
        return 0
    except (ClientError, OSError, ValueError) as exc:
        if a.json:
            print(json.dumps({'error': str(exc)}))
        else:
            print(str(exc), file=sys.stderr)
        return 1

def status(a, profile, models, root, models_dir):
    from workbench.cli import cmd_status
    server = Client(a.server, read_key(root)).peek()
    chosen = [m for m in models if not a.model or m.id == a.model]
    if a.json:
        print(json.dumps({'profile': dataclasses.asdict(profile), 'server': server,
            'models': [{'id': m.id, 'installed': fetch.is_installed(m, models_dir)} for m in chosen]}))
    else:
        cmd_status(a, profile, chosen)
        print('Server: '+('not running' if server is None else json.dumps(server)))
    return 0
```

In `cli.main`, replace `sub.add_parser('status')` with:
```python
    status = sub.add_parser('status')
    status.add_argument('--model')
    status.add_argument('--json', action='store_true')
    status.add_argument('--server', default='http://127.0.0.1:8080')
    from workbench.tools import commands
    commands.add_parsers(sub)
```
Immediately after parsing arguments and before hardware detection:
```python
    if a.cmd in ('serve', 'chat', 'ask', 'embed', 'rank'):
        return commands.dispatch(a, ROOT, MODELS)
```
After loading the catalog, before the existing command dictionary dispatch:
```python
    if a.cmd == 'status':
        try:
            return commands.status(a, p, models, ROOT, MODELS)
        except (OSError, ValueError, commands.ClientError) as exc:
            say(json.dumps({'error': str(exc)}) if a.json else str(exc))
            return 1
```
Add `json` to the stdlib imports in `cli.py`. The `models` and `setup` handlers remain available with no pack. Their live-server safety guard is added in Task 9.

- [ ] **Step 5: Run test to verify it passes**

Run: `python3.14 -m unittest tests.test_commands tests.test_cli -v`
Expected: PASS; commands send HTTP requests and bootstrap commands import no server dependencies.

- [ ] **Step 6: Commit**
```bash
 git add workbench/client.py workbench/tools/commands.py workbench/cli.py tests/test_commands.py
 git -c user.name=Xephex -c user.email=Xephex@users.noreply.github.com commit -m "feat: thin server clients and owned temporary server lifecycle" -m "Co-Authored-By: GPT-6 Astra (Codex) <noreply@openai.com>"
```

---

### Task 9: User-entry verification, live model safety and the optional real-model check

**Files:**
- Create: `tests/test_server_integration.py`, `docs/server.md`
- Modify: `workbench/tools/commands.py` (live model operations), `workbench/cli.py` (bootstrap mutation guard), `tests/test_cli.py` (isolate probes), `tests/test_commands.py` (live model delegation)

**Interfaces:**
- Consumes: all previous task interfaces; `WORKBENCH_TEST_MODEL` points at a small chat GGUF already on disk.
- Produces: `commands.models_running(args: Namespace, root: Path) -> int | None` (`None` means no server, so the existing bootstrap model flow runs); CLI/API documentation; one optional real-model test and model-free real-entry tests.
- No CI workflow is added. Plan 4 consumes the file-based runner and documented commands on Windows/Linux; the three-OS CI remains out of this plan's scope.

- [ ] **Step 1: Write the failing end-to-end tests**

`tests/test_commands.py`, append:
```python
class LiveModelsTest(unittest.TestCase):
    def test_models_mutation_goes_through_running_server(self):
        args = argparse.Namespace(add=['chat'], remove=None, yes=False)
        fake = mock.Mock()
        fake.peek.return_value = {'service': 'workbench'}
        fake.request.side_effect = [
            {'models': [{'id': 'chat', 'installed': False, 'runnable': True, 'recommended': True}]},
            {'id': 'j'}, {'jobs': [{'id': 'j', 'state': 'succeeded'}]}]
        with mock.patch.object(commands, 'Client', return_value=fake), mock.patch('sys.stdout', io.StringIO()):
            self.assertEqual(commands.models_running(args, pathlib.Path('.')), 0)
        self.assertEqual(fake.request.call_args_list[1].args,
                         ('POST', '/wb/models', {'action': 'install', 'model': 'chat'}))
```

`tests/test_server_integration.py`:
```python
import dataclasses, os, pathlib, shutil, socket, subprocess, sys, tempfile, threading, time, unittest
import uvicorn
from fastapi.testclient import TestClient
from tests.server_support import Runtime, model
from workbench.catalog import ModelFile
from workbench.engines.llama import LlamaEngine
from workbench.server.app import create_app
from workbench.runtime import Config
from workbench import hw

ROOT = pathlib.Path(__file__).resolve().parents[1]

class EntryPointTest(unittest.TestCase):
    def test_cli_ask_embed_rank_status_over_real_http(self):
        r = Runtime()
        self.addCleanup(r.close)
        sock = socket.socket()
        sock.bind(('127.0.0.1', 0)); sock.listen(128)
        self.addCleanup(sock.close)
        port = sock.getsockname()[1]
        app = create_app(r.config, r.profile, r.models, r.engine)
        server = uvicorn.Server(uvicorn.Config(app, log_level='error', access_log=False))
        thread = threading.Thread(target=lambda: server.run(sockets=[sock]))
        thread.start()
        def stop():
            server.should_exit = True
            thread.join(10)
            self.assertFalse(thread.is_alive())
        self.addCleanup(stop)
        deadline = time.monotonic()+5
        while not server.started:
            self.assertLess(time.monotonic(), deadline)
            time.sleep(.01)
        base = f'http://127.0.0.1:{port}'
        for args in (['ask', 'hello'], ['embed', 'a', 'bb'], ['rank', 'cat', 'dog', 'cat'], ['status']):
            result = subprocess.run([sys.executable, str(ROOT/'workbench.py'), *args, '--json', '--server', base],
                cwd=ROOT, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr+result.stdout)
            import json
            self.assertIsInstance(json.loads(result.stdout), dict)
        self.assertTrue(server.started)
        self.assertTrue(thread.is_alive())
        # The CLI reused this server; it did not stop a server it did not own.

class RealModelTest(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('WORKBENCH_TEST_MODEL'), 'Set WORKBENCH_TEST_MODEL to a small chat GGUF')
    def test_real_small_gguf(self):
        source = pathlib.Path(os.environ['WORKBENCH_TEST_MODEL']).resolve()
        self.assertTrue(source.is_file(), source)
        r = Runtime([model('real', ram=2)], ram=16)
        self.addCleanup(r.close)
        folder = r.config.models/'real'
        target = folder/'small.gguf'
        try:
            os.link(source, target)
        except OSError:
            shutil.copyfile(source, target)
        (folder/'small.gguf.ok').write_text('integration-fixture')
        entry = dataclasses.replace(r.models[0], kinds=('chat',),
            files=(ModelFile('small.gguf', '0'*64, source.stat().st_size, 'test/local', (), 'main'),),
            settings={'n_ctx': 1024})
        engine = LlamaEngine(ROOT)
        cfg = Config(ROOT, r.config.models, capture=False, phoenix=False)
        # Route trace data to the temporary root while native packages resolve
        # from the activated repository pack through LlamaEngine(ROOT).
        cfg = dataclasses.replace(cfg, root=r.root)
        profile = dataclasses.replace(hw.detect(r.config.models), gpus=[])
        app = create_app(cfg, profile, [entry], engine)
        with TestClient(app) as client:
            messages = [{'role': 'system', 'content': 'Remember these facts. '+('The sky is blue. '*50)},
                        {'role': 'user', 'content': 'Say OK.'}]
            first = client.post('/v1/chat/completions', json={'model': 'real', 'messages': messages,
                'max_tokens': 4, 'temperature': 0, 'think': False})
            self.assertEqual(first.status_code, 200, first.text)
            before = client.get('/wb/status').json()['loaded'][0]['prompt_evaluated']
            messages += [first.json()['choices'][0]['message'], {'role': 'user', 'content': 'Say OK again.'}]
            second = client.post('/v1/chat/completions', json={'model': 'real', 'messages': messages,
                'max_tokens': 4, 'temperature': 0, 'think': False})
            self.assertEqual(second.status_code, 200, second.text)
            after = client.get('/wb/status').json()['loaded'][0]['prompt_evaluated']
            self.assertGreater(before, 0)
            self.assertLess(after, before)
            self.assertEqual(len(app.state.manager.slots), 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3.14 tools/test_server.py tests.test_commands.LiveModelsTest tests.test_server_integration -v`
Expected: FAIL (missing `models_running`); the real-GGUF test skips unless the environment variable is set. The HTTP fake test must run, not skip.

- [ ] **Step 3: Route live mutations through jobs and write the user contract**

Add `import time` to `commands.py`, then append:
```python
def models_running(a, root):
    client = Client(DEFAULT_BASE, read_key(root))
    if client.peek() is None:
        return None
    rows = client.request('GET', '/wb/models')['models']
    known = {r['id']: r for r in rows}
    remove = a.remove or []
    add = a.add or []
    if not remove and not add:
        recommended = [r['id'] for r in rows if r['runnable'] and r['recommended']]
        if a.yes:
            add = recommended
        else:
            from workbench.ui import ask
            for row in rows:
                print(f"{row['id']}: {'installed' if row['installed'] else 'not installed'}"
                      + (f" ({row.get('reason')})" if row.get('reason') else ''))
            value = ask('Models to install, comma-separated', ','.join(recommended))
            add = [x.strip() for x in value.split(',') if x.strip()]
    for action, ids in (('remove', remove), ('install', add)):
        for mid in ids:
            if mid not in known:
                raise ClientError(f'Unknown model {mid}')
            if action == 'install' and not known[mid]['runnable']:
                raise ClientError(known[mid].get('reason') or 'Model cannot run here')
            row = client.request('POST', '/wb/models', {'action': action, 'model': mid})
            while True:
                result = next(j for j in client.request('GET', '/wb/jobs')['jobs'] if j['id'] == row['id'])
                if result['state'] not in ('queued', 'running'):
                    if result['state'] != 'succeeded':
                        raise ClientError(result.get('error') or result['state'])
                    print(f'{mid}: {action} succeeded')
                    break
                time.sleep(.2)
    return 0
```

In `cli.main`, after inference early dispatch and before hardware detection, add:
```python
    if a.cmd in ('models', 'setup'):
        try:
            if a.cmd == 'models':
                result = commands.models_running(a, ROOT)
                if result is not None:
                    return result
            elif commands.Client(key=commands.read_key(ROOT)).peek() is not None:
                say('Stop the running server before setup changes its installed packages.')
                return 1
        except (commands.ClientError, OSError, ValueError) as exc:
            say(str(exc))
            return 1
```

In `tests/test_cli.py`, append this patch to the existing `self.patches` list in `setUp`, so bootstrap tests cannot contact an unrelated server on the developer's laptop:
```python
            mock.patch('workbench.client.Client.peek', return_value=None),
```

`docs/server.md`:
```markdown
# Local server

Run `python workbench.py setup --yes`, then `python workbench.py serve`.
On Windows use `py -3.14`. The server binds to `127.0.0.1:8080` and loads a
model on its first request. Only Windows and Linux are supported here.

Commands use the same HTTP API:

    python workbench.py chat --model qwen3.5-4b
    python workbench.py ask "Explain a binary search" --json
    python workbench.py embed "a short document" "another document" --json
    python workbench.py rank "bicycles" "a car" "a bicycle" --top-n 1 --json
    python workbench.py status --json

Add `--server http://127.0.0.1:8080` to use an explicit server. Inference
commands reuse a running server; otherwise they start a temporary loopback
server and stop only that process on exit. Status never starts a server.
Chat retains history for the session and saves it under `data/chats`.
`/reset` starts a fresh conversation. `--think` enables reasoning; it is off
by default on CPU-tier machines. `--json` emits complete JSON (chat emits one
JSON object per reply); progress goes to stderr in normal text mode.

A cold long agent prompt can take many minutes on CPU. The API sends SSE
comments for queued/loading/prompt/generation states every ten seconds.
"Reading the prompt" is activity feedback, not a percentage. The resident
model keeps KV state and a bounded RAM prefix cache between turns. LRU
unloading or cancellation drops that cache. A framework that times only data
chunks may still need a longer timeout: dcode uses
`LANGCHAIN_OPENAI_STREAM_CHUNK_TIMEOUT_S`. Prefer a GPU or a shorter system
prompt for dcode/Forge; the recorded 11.4k-token CPU prompt took 26 minutes.

`serve --share` binds all interfaces and generates/reuses
`.workbench/api-key.txt`; `WORKBENCH_API_KEY` overrides the file. Clients send
`Authorization: Bearer KEY`. Every route requires the key when one is set.
Any non-loopback `--host` without a key is refused. Keys are never printed.
Loopback binding remains the default; sharing does not enable downloads.

API routes:

- GET `/v1/models`: installed, runnable model IDs and capabilities.
- POST `/v1/chat/completions`: messages, tools/tool_choice, stream,
  stream_options.include_usage, max_tokens, sampling settings, think,
  response_format (text, json_object or json_schema).
- POST `/v1/embeddings`: model, input (string or list), encoding_format
  (float or base64). Token arrays and dimension reduction are not supported.
- POST `/v1/rerank`: model, query, documents (strings or objects with text),
  optional top_n and return_documents. Results retain original indexes and
  sort by descending relevance_score; ties retain input order.
- GET `/wb/status`: loaded models, memory reservations, device verification,
  FIFO queue depth, physical core count and optional trace viewer URL.
- GET `/wb/models`: all catalog entries and installation/runnable state.
- POST `/wb/models`: action install/remove plus model, or cancel plus job_id.
- GET `/wb/jobs`: durable progress, terminal results and interrupted jobs.

Messages accept strings or lists of text parts for every role. Assistant
tool-call history may have null content; tool replies require tool_call_id.
Vision models accept user image_url parts containing PNG/JPEG/WebP base64
data URLs; the catalog mmproj is loaded with the model. Remote URLs, local
file URLs and audio parts are rejected. JSON schema uses native grammar.
Tools and non-text response_format cannot be combined. Auto choice uses the
model template and parser; none disables tools. Required or a named choice
uses a grammar-constrained single-call envelope, held until complete before
emitting tool_calls. Tools are returned,
never executed by the server. A streamed error is an error data frame then
[DONE]; clients must inspect it even when HTTP status is 200.

RAM reserves 4 GB for the system and VRAM reserves 0.8 GB. Cache memory is
charged separately. Memory numbers are catalog estimates until matching
measurements are supplied by tune. Small embedding/reranker models use GPU
space when available alongside chat, otherwise CPU. Unknown native device
information is reported as unverified, never as proven acceleration.

Model changes are jobs. Closing a browser leaves jobs running. Cancelling a
fetch takes effect at the next chunk boundary and preserves resume state.
A server restart marks unfinished jobs interrupted; repeat the install to
resume. A model with active or queued requests cannot be removed. The models
command delegates to the server when it is running. Stop the server before
setup replaces packages.

Every request writes OTel spans to `data/traces/spans.jsonl`. If the tracing
pack is installed, the server starts a local Phoenix viewer and exports new
spans to it. Its URL appears in status. Phoenix telemetry/external resources
are disabled, and the viewer binds only to loopback. Startup failure keeps
file tracing available; inspect `data/phoenix/server.log`. No historical
file replay occurs automatically. `serve --no-trace-viewer` skips the viewer.

Inputs/outputs are recorded by default. `serve --no-trace-content` disables
content for all requests. `X-Workbench-Trace-Content: false` disables it for
one request/tab. This does not disable chat transcript saving by the chat
command. Timing, model, sampled token counts, memory and error classes remain.
No exception message or authorization header is recorded in spans.

Langfuse is optional and off by default. Put an enabled langfuse object with
host, public_key and secret_key in `data/trace.json` to add OTLP/HTTP export to
`<host>/api/public/otel/v1/traces`. Keep this user-data file private. This is an
explicit external network opt-in; local traces and Phoenix remain enabled.

Developer checks: `python3.14 tools/test_server.py`. The runner activates the
pack before importing dependencies. It is a file so Windows Phoenix and
strawberry source inspection works. Unit tests use a fake engine. Set
WORKBENCH_TEST_MODEL to an existing small chat GGUF to enable the single
optional real-model test; it checks two API turns and reduced prompt work on
the second turn and never downloads a model.
```

- [ ] **Step 4: Run test to verify it passes and exercise the entry points**

Run: `python3.14 tools/test_server.py`
Expected: PASS; exactly the real-GGUF test skips when its environment variable is absent. Existing unrelated environment-dependent tests retain their existing skip rules.

Run the optional test, with `WORKBENCH_TEST_MODEL` already set by the operator:
`python3.14 tools/test_server.py tests.test_server_integration.RealModelTest -v`.
Expected: PASS with two successful HTTP turns and fewer evaluated prompt tokens on turn two. If unset, Expected: SKIP; do not describe real-model inference as verified.

Exercise a real installed entry point with a chosen installed chat model:
```bash
python3.14 workbench.py serve --no-trace-content
python3.14 workbench.py status --json
python3.14 workbench.py ask "Say hello in one sentence" --json
python3.14 workbench.py chat
```

Use separate dedicated terminals for the foreground server and clients. Enter two chat turns, `/reset`, then `exit`; inspect `data/chats`. Confirm clients leave the foreground server running. Stop that server with Ctrl+C, run `ask` again and verify that its temporary server exits when the answer completes. With the optional tracing layer installed, repeat `serve` without `--no-trace-content`, open the status-reported Phoenix URL, and confirm the request and child spans appear offline. On Windows run these checks from `workbench.py` and the runner file, never stdin code. Hardware-specific results are evidence for that machine only; Plan 4 owns the CI matrix and full viewer/offline browser audit.

- [ ] **Step 5: Commit**
```bash
 git add tests/test_server_integration.py tests/test_commands.py tests/test_cli.py workbench/tools/commands.py workbench/cli.py docs/server.md
 git -c user.name=Xephex -c user.email=Xephex@users.noreply.github.com commit -m "test: server user workflows and optional real GGUF cache verification" -m "Co-Authored-By: GPT-6 Astra (Codex) <noreply@openai.com>"
```

---

## After this plan

Plan 4 owns the Gradio app tabs and mounting (`gr.mount_gradio_app`), the `image` and `docs` commands/document indexing, Pi, n8n workflows, tune, monitor, selftest, `update`, launcher-menu work and the three-OS CI. Its UI must set `GRADIO_ANALYTICS_ENABLED=False`, use local fonts/assets, avoid SSR/CDNs and never enable Gradio share mode. It consumes this plan's app factory, HTTP clients, model jobs, per-request privacy header and tuning-file contract. No Gradio imports are added here.

Audio and speech endpoints remain Stage 2; Jev/ONNX Stage 3; evals/datasets Stage 4; Deep Agents and Forge Stage 5; fine-tuning Stage 6; image generation Stage 7. macOS support, MTP models and measured pack/backend coverage remain outside this plan.

## Scope rulings and self-review

- **Network wording:** A2's offline rule allows the local HTTP transport mandated by A4; B6 explicitly authorizes optional Langfuse export. No implicit runtime downloads or remote image fetches are added.
- **Jobs wording:** The Stage 1 model installation/removal job worker is included because A9 names model fetch as a durable job. Cancellation uses `POST /wb/models` so the endpoint inventory stays within B7. Other long-job kinds are deferred to their stages.
- **Trace viewer wording:** B6's optional Phoenix runs as an owned loopback process, launched through a real module. Local file recording always remains available. Automatic historical replay is omitted to avoid inventing an import/deduplication protocol.
- **Memory wording:** Existing catalog `ram_gb`/`vram_gb` are estimates. CPU placement reserves their sum, GPU placement reserves both separately; zero-VRAM small-model entries still reserve estimated VRAM when GPU-loaded. Cache allocations are charged in addition and shrink to fit the base footprint's remaining RAM headroom. Matching tune measurements replace base estimates without changing the manager API.
- **Command ownership:** A4 requires temporary servers for inference clients; status remains a read-only probe that works before setup. Existing models commands use HTTP when a server is running and retain the original no-pack bootstrap path otherwise.
- **Platform wording:** The explicit operator scope overrides the spec's macOS statements. No macOS-specific implementation or CI is added; pre-existing detection is retained.

Coverage: A9/B7 routes and schemas (Tasks 5-7), engine/parser/cache (Task 2), FIFO/budgets/LRU (Task 3), A11/B6 traces and optional viewer/export (Task 4 plus request attributes in Task 5), A12 server commands and ownership (Tasks 8-9), physical cores/settings (Task 1). The optional real model is confined to Task 9. Plan 4 boundaries are enumerated above.

Self-review completed for the writing lane:

- Spec gaps found and closed: physical-core detection and optional catalog settings (Task 1); native cache finalization, bounded snapshot copying and forced tool choices (Task 2); nonblocking load admission and verified CPU fallback accounting (Task 3); privacy for error events and embedding/rerank content (Tasks 4-5); disconnect cleanup (Task 6); durable model mutations and live CLI routing (Tasks 7-9). No remaining in-scope requirement is intentionally deferred.
- Cross-task audit: the manager consumes `Profile.ram_bytes`, `Profile.gpu_budget_bytes`, `Model.kinds`, `Model.files` and `Model.folder` from the existing code; it does not invent a `Profile.backend` field. Adapter `chat` always returns an iterator; `Service.chat` returns the merged response; `respond` handles both external response modes. Ticket, cancellation, tuning and job-state names agree across producers and consumers.
- Document checks: nine tasks, each with Files, Interfaces, failing-test code, an Expected FAIL run, implementation code, an Expected PASS run and a commit; five Review Focus entries each name tests in their owning tasks. Python blocks are syntax-checked as documentation, and the parser's pinned source hash is checked against the source commit. No implementation or model tests were executed while writing this plan.
