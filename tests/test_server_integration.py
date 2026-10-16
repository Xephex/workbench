import dataclasses, os, pathlib, socket, subprocess, sys, threading, time, unittest
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
        # Reference the operator's model in place; never duplicate a large GGUF.
        try:
            os.link(source, target)
        except OSError:
            target.symlink_to(source)
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
