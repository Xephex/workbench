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
        if not small:
            # Hybrid models need an intermediate checkpoint to reuse short prompts.
            interval = max(1, min(4096, defaults['n_ctx'] // 4))
            defaults['checkpoint_interval'] = interval
            defaults.setdefault('n_batch', min(512, interval))
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
        if not self.injected:
            # Reset performance counters only; retain resident KV/cache state.
            h.native._ctx.reset_timings()
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
            h.metrics = dict(prompt_evaluated=int(perf.n_p_eval), prompt_ms=float(perf.t_p_eval_ms),
                             generation_ms=float(perf.t_eval_ms))

    def stats(self, h):
        # Status may run during inference. Read cached Python values only.
        return {'usage': dict(h.usage), 'placement': h.placement, 'devices': h.devices,
                'device_verified': bool(h.devices), 'cache_bytes': getattr(h.cache, 'cache_size', 0),
                'cache_snapshot_fits': getattr(h.cache, 'snapshot_fits', True), **h.metrics}
