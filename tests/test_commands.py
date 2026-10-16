import argparse, contextlib, email.message, io, json, pathlib, tempfile, unittest, urllib.error
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

    def test_cli_live_mutations_bypass_bootstrap_and_setup_is_blocked(self):
        fake = mock.Mock()
        fake.peek.return_value = {'service': 'workbench'}
        fake.request.side_effect = [
            {'models': [{'id': 'chat', 'installed': True, 'runnable': True}]},
            {'id': 'j'}, {'jobs': [{'id': 'j', 'state': 'succeeded'}]}]
        with mock.patch.object(commands, 'Client', return_value=fake), \
             mock.patch.object(cli.hw, 'detect') as detect, \
             mock.patch.object(cli, 'cmd_models') as bootstrap, \
             mock.patch.object(cli.install, 'install_pack') as install_pack, \
             mock.patch('sys.stdout', io.StringIO()) as out:
            self.assertEqual(cli.main(['models', '--remove', 'chat']), 0)
            self.assertEqual(cli.main(['setup', '--packages-only', '--yes']), 1)
            self.assertIn('Stop the running server', out.getvalue())
        detect.assert_not_called()
        bootstrap.assert_not_called()
        install_pack.assert_not_called()
        self.assertEqual(fake.request.call_args_list[1].args,
                         ('POST', '/wb/models', {'action': 'remove', 'model': 'chat'}))

    def test_foreign_404_is_not_a_running_workbench(self):
        err = urllib.error.HTTPError(
            'http://127.0.0.1:8080/wb/status', 404, 'Not Found', email.message.EmailMessage(),
            io.BytesIO(b'<html>404</html>'))
        client = Client()
        with mock.patch.object(client.opener, 'open', side_effect=err):
            self.assertIsNone(client.peek())
        err.close()

    def test_missing_server_falls_back_and_probe_errors_do_not(self):
        args = argparse.Namespace(add=['chat'], remove=None, yes=False)
        with mock.patch.object(commands.Client, 'peek', return_value=None):
            self.assertIsNone(commands.models_running(args, pathlib.Path('.')))
        with mock.patch.object(commands.Client, 'peek', side_effect=ClientError('Unauthorized')), \
             mock.patch.object(cli, 'cmd_models') as bootstrap, \
             mock.patch('sys.stdout', io.StringIO()):
            self.assertEqual(cli.main(['models', '--remove', 'chat']), 1)
        bootstrap.assert_not_called()

    def test_live_job_polling_reports_failure(self):
        fake = mock.Mock()
        fake.request.side_effect = [
            {'models': [{'id': 'chat', 'runnable': True, 'recommended': True}]},
            {'id': 'j'}, {'jobs': [{'id': 'j', 'state': 'running'}]},
            {'jobs': [{'id': 'j', 'state': 'failed', 'error': 'Model is busy'}]}]
        args = argparse.Namespace(add=None, remove=None, yes=True)
        with mock.patch.object(commands, 'Client', return_value=fake), \
             mock.patch.object(commands.time, 'sleep') as sleep:
            with self.assertRaisesRegex(ClientError, 'Model is busy'):
                commands.models_running(args, pathlib.Path('.'))
        sleep.assert_called_once_with(.2)
