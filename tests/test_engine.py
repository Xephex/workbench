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
    def test_hybrid_checkpoint_interval_scales_to_short_context(self):
        for context, interval in ((1024, 256), (16384, 4096)):
            h = self.engine.load(self.entry, {'models_dir': self.root, 'placement': 'cpu',
                'n_threads': 2, 'cache_bytes': 1024, 'think': False, 'n_ctx': context})
            self.assertEqual(h.native.kwargs['checkpoint_interval'], interval)
            self.assertEqual(h.native.kwargs['n_batch'], min(512, interval))

    def test_each_native_chat_starts_a_new_performance_interval(self):
        h = self.engine.load(self.entry, {'models_dir': self.root, 'placement': 'cpu',
            'n_threads': 2, 'cache_bytes': 1024, 'think': False})
        h.native._ctx = mock.Mock()
        self.engine.injected = False
        with mock.patch.object(self.engine, 'measure'):
            for _ in range(2):
                list(self.engine.chat(h, {'messages': [{'role': 'user', 'content': 'OK'}]},
                                      lambda phase: None))
        self.assertEqual(h.native._ctx.reset_timings.call_count, 2)
        self.assertEqual(h.native.evaluated, [2, 0])

    def test_native_performance_fields_match_installed_binding(self):
        from llama_cpp import llama_cpp as api
        perf = api.llama_perf_context_data(n_p_eval=123, t_p_eval_ms=45.5, t_eval_ms=6.5)
        handle = types.SimpleNamespace(native=types.SimpleNamespace(
            _ctx=types.SimpleNamespace(ctx=object())), metrics={})
        engine = LlamaEngine(pathlib.Path('.'))
        with mock.patch.object(api, 'llama_perf_context', return_value=perf):
            engine.measure(handle)
        self.assertEqual(handle.metrics,
                         {'prompt_evaluated': 123, 'prompt_ms': 45.5, 'generation_ms': 6.5})

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
