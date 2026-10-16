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

    def test_engine_failures_are_generic_and_do_not_escape_to_access_logs(self):
        def failed(*args):
            raise RuntimeError('PRIVATE_FAILURE_SENTINEL')
        for operation, path, body in (
            ('chat', '/v1/chat/completions', {'model': 'chat',
                'messages': [{'role': 'user', 'content': 'PRIVATE_FAILURE_SENTINEL'}]}),
            ('embed', '/v1/embeddings', {'model': 'embed', 'input': 'PRIVATE_FAILURE_SENTINEL'}),
            ('rank', '/v1/rerank', {'model': 'rank', 'query': 'PRIVATE_FAILURE_SENTINEL',
                                  'documents': ['a']}),
        ):
            with self.subTest(operation=operation):
                setattr(self.runtime.engine, operation, failed)
                # TestClient re-raises uncaught ASGI errors, as uvicorn would log them.
                reply = self.client.post(path, json=body,
                    headers={'X-Workbench-Trace-Content': 'false'})
                self.assertEqual(reply.status_code, 500)
                self.assertEqual(reply.json()['error']['type'], 'server_error')
                self.assertEqual(reply.json()['error']['code'], '500')
                self.assertNotIn('PRIVATE_FAILURE_SENTINEL', reply.text)
        saved = (self.runtime.root/'data/traces/spans.jsonl').read_text()
        self.assertNotIn('PRIVATE_FAILURE_SENTINEL', saved)
        self.assertIn('RuntimeError', saved)
        status = self.client.get('/wb/status').json()
        self.assertEqual(status['queued'], 0)
        self.assertEqual(status['loaded'], [])
