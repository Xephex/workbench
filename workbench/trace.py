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
        try:
            self.viewer = subprocess.Popen([sys.executable, '-m', 'workbench.phoenix_runner'],
                cwd=self.config.root, env=env, stdin=subprocess.DEVNULL, stdout=self.viewer_log, stderr=subprocess.STDOUT)
        except OSError:
            self.stop_viewer()
            return
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        url = f'http://127.0.0.1:{port}'
        # First-launch database migrations can exceed ten seconds.
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
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
