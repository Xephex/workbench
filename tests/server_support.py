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
