import json, os, pathlib, tempfile, unittest
from unittest import mock
from opentelemetry import trace as otel
from opentelemetry.sdk.trace import TracerProvider
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
    def test_local_recording_ignores_ambient_sdk_disable(self):
        for disabled in ('true', ' TRUE '):
            with self.subTest(disabled=disabled), tempfile.TemporaryDirectory() as d, \
                 mock.patch.dict(os.environ, {'OTEL_SDK_DISABLED': disabled}):
                root = pathlib.Path(d)
                global_provider = otel.get_tracer_provider()
                sink = Capture()
                t = Tracing(Config(root, root/'models', capture=False, phoenix=False), [sink])
                try:
                    with t.child('request'):
                        with t.child('model.load'):
                            pass
                        phases = Phases(t)
                        phases.change('prompt'); phases.change('generation'); phases.close()
                    self.assertEqual(os.environ['OTEL_SDK_DISABLED'], disabled)
                    self.assertIs(otel.get_tracer_provider(), global_provider)
                    # Other SDK users must still honor the process-wide switch.
                    other = TracerProvider()
                    try:
                        with other.get_tracer('other').start_as_current_span('disabled') as span:
                            self.assertFalse(span.is_recording())
                    finally:
                        other.shutdown()
                finally:
                    t.close()
                rows = [json.loads(line) for line in (root/'data/traces/spans.jsonl').read_text().splitlines()]
                expected = {'request', 'model.load', 'prompt', 'generation'}
                self.assertEqual({row['name'] for row in rows}, expected)
                self.assertEqual({span.name for span in sink.spans}, expected)
                parent = next(span for span in sink.spans if span.name == 'request')
                for span in sink.spans:
                    if span.name != 'request':
                        self.assertEqual(span.parent.span_id, parent.context.span_id)

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

    def test_viewer_allows_first_database_startup(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            process = mock.Mock()
            process.poll.return_value = None
            opener = mock.MagicMock()
            reply = mock.MagicMock()
            reply.__enter__.return_value.status = 200
            opener.open.side_effect = [OSError('not ready')] * 120 + [reply]
            with mock.patch('workbench.trace.importlib.util.find_spec', return_value=object()), \
                 mock.patch('workbench.trace.subprocess.Popen', return_value=process), \
                 mock.patch('workbench.trace.urllib.request.build_opener', return_value=opener), \
                 mock.patch('workbench.trace.time.sleep'), \
                 mock.patch('workbench.trace.OTLPSpanExporter'):
                t = Tracing(Config(root, root/'models'))
                try:
                    self.assertIsNotNone(t.viewer_url)
                finally:
                    t.close()

    def test_viewer_spawn_failure_keeps_local_recording(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            with mock.patch('workbench.trace.importlib.util.find_spec', return_value=object()), \
                 mock.patch('workbench.trace.subprocess.Popen', side_effect=OSError('cannot spawn')):
                t = Tracing(Config(root, root/'models'))
                try:
                    self.assertIsNone(t.viewer_url)
                    with t.child('local.after.viewer.failure'):
                        pass
                finally:
                    t.close()
            rows = (root/'data/traces/spans.jsonl').read_text().splitlines()
            self.assertEqual([json.loads(row)['name'] for row in rows], ['local.after.viewer.failure'])
            self.assertIsNone(t.viewer_log)

    def test_runner_disables_wildcard_grpc_and_forces_loopback(self):
        import asyncio, os, sys, types
        from workbench import phoenix_runner

        # Phoenix has no gRPC host setting; its normal listener binds [::].
        class GrpcServer:
            def __init__(self, enqueue_span, port, disabled=False):
                self.listening = False
            async def __aenter__(self):
                self.listening = True
            async def __aexit__(self, *args):
                self.listening = False

        grpc_module = types.ModuleType('phoenix.server.grpc_server')
        grpc_module.GrpcServer = GrpcServer
        phoenix = types.ModuleType('phoenix')
        server = types.ModuleType('phoenix.server')
        phoenix.server = server
        server.grpc_server = grpc_module
        modules = {'phoenix': phoenix, 'phoenix.server': server,
                   'phoenix.server.grpc_server': grpc_module}

        def run_module(name, run_name):
            with self.subTest('http listener'):
                self.assertEqual(os.environ['PHOENIX_HOST'], '127.0.0.1')
            self.assertEqual(os.environ['PHOENIX_ALLOW_EXTERNAL_RESOURCES'], 'false')
            self.assertEqual(os.environ['PHOENIX_TELEMETRY_ENABLED'], 'false')
            async def start_listener():
                listener = grpc_module.GrpcServer(None, port=4317, disabled=False)
                async with listener:
                    self.assertFalse(listener.listening)
            asyncio.run(start_listener())

        with mock.patch.dict(sys.modules, modules), \
             mock.patch.dict(os.environ, {'PHOENIX_HOST': '0.0.0.0'}), \
             mock.patch.object(phoenix_runner.install, 'activate', return_value=True), \
             mock.patch.object(phoenix_runner.runpy, 'run_module', side_effect=run_module), \
             mock.patch.object(sys, 'argv', ['workbench.phoenix_runner']):
            phoenix_runner.main()
