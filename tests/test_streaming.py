import asyncio, json, threading, unittest
from fastapi.testclient import TestClient
from tests.server_support import Runtime, TOOLS
from workbench.server.app import create_app
from workbench.server.streaming import Worker, finish, shutdown

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

    async def test_disconnect_while_sending_headers_stops_worker(self):
        r = Runtime()
        self.addCleanup(r.close)
        r.engine.prefill.clear()
        app = create_app(r.config, r.profile, r.models, r.engine)
        incoming, output = asyncio.Queue(), asyncio.Queue()
        body = json.dumps({'model': 'chat', 'stream': True,
            'messages': [{'role': 'user', 'content': 'long prompt'}]}).encode()
        await incoming.put({'type': 'http.request', 'body': body, 'more_body': False})
        scope = {'type': 'http', 'asgi': {'version': '3.0'}, 'http_version': '1.1',
            'method': 'POST', 'scheme': 'http', 'path': '/v1/chat/completions',
            'raw_path': b'/v1/chat/completions', 'query_string': b'', 'root_path': '',
            'headers': [(b'content-type', b'application/json')],
            'client': ('127.0.0.1', 1234), 'server': ('127.0.0.1', 8080)}
        async def send(event):
            await output.put(event)
            if event['type'] == 'http.response.start':
                await asyncio.Event().wait()
        async with app.router.lifespan_context(app):
            running = asyncio.create_task(app(scope, incoming.get, send))
            try:
                start = await asyncio.wait_for(output.get(), 1)
                self.assertEqual(start['status'], 200)
                self.assertTrue(await asyncio.to_thread(r.engine.started.wait, 1))
                await incoming.put({'type': 'http.disconnect'})
                await asyncio.wait_for(running, 2)
                self.assertEqual(app.state.workers, set())
                self.assertEqual(app.state.manager.snapshot()['loaded'], [])
                self.assertIn('chat', r.engine.unloads)
            finally:
                if not running.done():
                    running.cancel()
                    await asyncio.gather(running, return_exceptions=True)

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

    def test_stream_failure_is_private_and_ends_once_without_usage(self):
        for exc, code, kind in ((RuntimeError('PRIVATE_FAILURE'), '500', 'server_error'),
                                (ValueError('PRIVATE_FAILURE'), '400', 'invalid_request_error')):
            with self.subTest(code=code):
                r = Runtime()
                self.addCleanup(r.close)
                def failed(*args):
                    raise exc
                r.engine.chat = failed
                with TestClient(create_app(r.config, r.profile, r.models, r.engine)) as client:
                    reply = client.post('/v1/chat/completions',
                        headers={'X-Workbench-Trace-Content': 'false'}, json={
                            'model': 'chat', 'stream': True, 'stream_options': {'include_usage': True},
                            'messages': [{'role': 'user', 'content': 'PRIVATE_FAILURE'}]})
                    self.assertEqual(reply.status_code, 200)
                    values = [line[6:] for line in reply.text.splitlines() if line.startswith('data: ')]
                    self.assertEqual(len(values), 2)
                    self.assertEqual(values[-1], '[DONE]')
                    failure = json.loads(values[0])['error']
                    self.assertEqual((failure['code'], failure['type']), (code, kind))
                    self.assertNotIn('PRIVATE_FAILURE', reply.text)
                    self.assertEqual(client.get('/wb/status').json()['loaded'], [])
                self.assertNotIn('PRIVATE_FAILURE', (r.root/'data/traces/spans.jsonl').read_text())

    def test_json_drains_bounded_queue_and_success_reuses_handle(self):
        r = Runtime()
        self.addCleanup(r.close)
        def many(h, body, phase):
            phase('generation')
            for i in range(100):
                yield {'id': 'many', 'created': 0, 'model': 'chat', 'object': 'chat.completion.chunk',
                    'choices': [{'index': 0, 'delta': {'content': 'x'}, 'finish_reason': None}]}
            h.usage = {'prompt_tokens': 1, 'completion_tokens': 100, 'total_tokens': 101}
        r.engine.chat = many
        app = create_app(r.config, r.profile, r.models, r.engine)
        with TestClient(app) as client:
            for stream in (False, True, False):
                reply = client.post('/v1/chat/completions', json={'model': 'chat', 'stream': stream,
                    'messages': [{'role': 'user', 'content': 'hello'}]})
                self.assertEqual(reply.status_code, 200)
                if stream:
                    self.assertEqual(reply.text.count('data: [DONE]'), 1)
                    self.assertNotIn('"usage"', reply.text)
                else:
                    self.assertEqual(reply.json()['choices'][0]['message']['content'], 'x'*100)
                self.assertEqual(app.state.workers, set())
            self.assertEqual(len(r.engine.loads), 1)
            self.assertEqual(r.engine.unloads, [])

class WorkerLifecycleTest(unittest.IsolatedAsyncioTestCase):
    async def test_backpressure_cancel_finalizes_before_unload_on_worker_thread(self):
        r = Runtime()
        self.addCleanup(r.close)
        produced, order = threading.Event(), []
        def many(h, body, phase):
            order.append(('start', threading.get_ident()))
            try:
                for i in range(100):
                    if i == 32:
                        produced.set()
                    yield {'choices': [{'delta': {'content': 'x'}, 'finish_reason': None}]}
            finally:
                order.append(('finalize', threading.get_ident()))
        unload = r.engine.unload
        def tracked_unload(h):
            order.append(('unload', threading.get_ident()))
            unload(h)
        r.engine.chat, r.engine.unload = many, tracked_unload
        app = create_app(r.config, r.profile, r.models, r.engine)
        async with app.router.lifespan_context(app):
            worker = Worker(app.state.service, app.state.manager.submit('chat'),
                {'messages': [{'role': 'user', 'content': 'hello'}]}, False)
            app.state.workers.add(worker)
            worker.start()
            self.assertTrue(await asyncio.to_thread(produced.wait, 1))
            self.assertFalse(worker.done.is_set())
            self.assertTrue(app.state.manager.snapshot()['loaded'][0]['busy'])
            worker.cancel()
            await asyncio.wait_for(finish(worker), 2)
            self.assertEqual([name for name, tid in order], ['start', 'finalize', 'unload'])
            self.assertEqual(len({tid for name, tid in order}), 1)
            self.assertEqual(app.state.manager.snapshot()['loaded'], [])
            self.assertEqual(app.state.manager.snapshot()['queued'], 0)
            self.assertEqual(worker.failure.status, 499)

    async def test_shutdown_cancels_queued_and_running_work(self):
        r = Runtime()
        self.addCleanup(r.close)
        r.engine.prefill.clear()
        app = create_app(r.config, r.profile, r.models, r.engine)
        async with app.router.lifespan_context(app):
            workers = []
            for i in range(2):
                worker = Worker(app.state.service, app.state.manager.submit('chat'),
                    {'messages': [{'role': 'user', 'content': 'hello'}]}, False)
                app.state.workers.add(worker)
                worker.start()
                workers.append(worker)
            self.assertTrue(await asyncio.to_thread(r.engine.started.wait, 1))
            self.assertEqual(app.state.manager.snapshot()['queued'], 1)
            await asyncio.wait_for(shutdown(app.state.workers), 2)
            self.assertEqual(app.state.workers, set())
            self.assertTrue(all(w.done.is_set() and not w.thread.is_alive() for w in workers))
            self.assertEqual(app.state.manager.snapshot()['queued'], 0)
            self.assertEqual(app.state.manager.snapshot()['loaded'], [])
            self.assertEqual(len(r.engine.loads), 1)
