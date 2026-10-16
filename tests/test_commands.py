import argparse, contextlib, io, json, pathlib, tempfile, unittest
from unittest import mock
from workbench import cli
from workbench.client import Client, session, ClientError
from workbench.tools import commands

class CommandsTest(unittest.TestCase):
    def test_nonchat_model_is_selected_in_request_not_server_default(self):
        for cmd, fields, endpoint in (
            ('embed', {'texts': ['hello']}, '/v1/embeddings'),
            ('rank', {'query': 'hello', 'documents': ['hello'], 'top_n': 1}, '/v1/rerank'),
        ):
            with self.subTest(cmd=cmd):
                fake = mock.Mock()
                fake.request.return_value = {'data': [], 'results': []}
                @contextlib.contextmanager
                def connected(root, model, base):
                    self.assertIsNone(model, 'Embedding/rerank models cannot be a chat default')
                    yield fake
                args = argparse.Namespace(cmd=cmd, model=cmd, json=True,
                    server='http://127.0.0.1:8080', **fields)
                with mock.patch.object(commands, 'session', connected), \
                     mock.patch('sys.stdout', io.StringIO()):
                    self.assertEqual(commands.dispatch(args, pathlib.Path('.'), pathlib.Path('models')), 0)
                self.assertEqual(fake.request.call_args.args[:2], ('POST', endpoint))
                self.assertEqual(fake.request.call_args.args[2]['model'], cmd)

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
