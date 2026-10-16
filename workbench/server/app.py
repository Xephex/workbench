from __future__ import annotations
import asyncio, base64, contextlib, dataclasses, hmac, json, math, struct, threading, time, uuid
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from opentelemetry import trace as otel
from opentelemetry.trace import Status, StatusCode
from openinference.semconv.trace import SpanAttributes
from workbench import catalog, fetch, hw
from workbench.engines.llama import LlamaEngine
from workbench.runtime import Config
from workbench.server.manager import Manager, ManagerError
from workbench.server.jobs import Jobs, ModelAction
from workbench.server.schemas import ChatRequest, EmbeddingRequest, RerankRequest
from workbench.server.streaming import Worker, respond, shutdown
from workbench.trace import Tracing, RequestTracing, Phases

def error(status, message, kind='invalid_request_error'):
    return JSONResponse({'error': {'message': message, 'type': kind, 'code': str(status)}}, status_code=status)

class Auth:
    def __init__(self, app, key):
        self.app, self.key = app, key
    async def __call__(self, scope, receive, send):
        if scope['type'] == 'http' and self.key:
            supplied = dict(scope.get('headers', [])).get(b'authorization', b'')
            if not hmac.compare_digest(supplied, ('Bearer '+self.key).encode()):
                return await error(401, 'A valid bearer API key is required')(scope, receive, send)
        return await self.app(scope, receive, send)

class Run:
    def __init__(self, engine):
        self.engine, self.event, self.lock, self.handle = engine, threading.Event(), threading.Lock(), None
        self.stage = 'queued'
    def cancel(self):
        with self.lock:
            self.event.set()
            if self.handle is not None:
                self.engine.abort(self.handle)
    @contextlib.contextmanager
    def bind(self, handle):
        with self.lock:
            if self.event.is_set():
                raise ManagerError(499, 'Request cancelled')
            self.handle = handle
        try:
            yield
        finally:
            with self.lock:
                self.handle = None

def merge_chunks(chunks, model, usage):
    text, calls, finish = [], {}, 'stop'
    ident, created = 'chatcmpl-'+uuid.uuid4().hex, int(time.time())
    for chunk in chunks:
        ident, created = chunk.get('id', ident), chunk.get('created', created)
        for choice in chunk.get('choices', []):
            delta = choice.get('delta', {})
            if delta.get('content'):
                text.append(delta['content'])
            for call in delta.get('tool_calls') or []:
                target = calls.setdefault(call['index'], {'id': '', 'type': 'function',
                                                         'function': {'name': '', 'arguments': ''}})
                if call.get('id'):
                    target['id'] = call['id']
                for key in ('name', 'arguments'):
                    target['function'][key] += call.get('function', {}).get(key, '')
            finish = choice.get('finish_reason') or finish
    msg = {'role': 'assistant', 'content': ''.join(text) or None}
    if calls:
        msg['tool_calls'] = [calls[i] for i in sorted(calls)]
    return {'id': ident, 'object': 'chat.completion', 'created': created, 'model': model,
            'choices': [{'index': 0, 'message': msg, 'finish_reason': finish, 'logprobs': None}], 'usage': usage}

class Service:
    def __init__(self, manager, tracing):
        self.manager, self.engine, self.tracing = manager, manager.engine, tracing
    def chat(self, ticket, body, emit, run, capture):
        chunks = []
        phases = Phases(self.tracing)
        span = otel.get_current_span()
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, 'LLM')
        span.set_attribute(SpanAttributes.LLM_MODEL_NAME, ticket.mid)
        self.tracing.content(span, 'input.value', body['messages'], capture)
        def phase(name):
            run.stage = name
            phases.change(name)
        @contextlib.contextmanager
        def loading():
            run.stage = 'loading'
            with self.tracing.child('model.load'):
                yield
        try:
            with self.manager.lease(ticket, run.event, loading) as handle:
                with run.bind(handle):
                    for chunk in self.engine.chat(handle, body, phase):
                        if run.event.is_set():
                            raise ManagerError(499, 'Request cancelled')
                        chunks.append(chunk)
                        emit(chunk)
                    if run.event.is_set():
                        raise ManagerError(499, 'Request cancelled')
                    stats = self.engine.stats(handle)
                    usage = stats.get('usage', {})
                    result = merge_chunks(chunks, ticket.mid, usage)
                    span.set_attribute('workbench.placement', handle.placement)
                    footprint = self.manager.slots[ticket.mid].footprint
                    span.set_attribute('workbench.memory.ram_bytes', footprint.ram)
                    span.set_attribute('workbench.memory.vram_bytes', footprint.vram)
                    span.set_attribute('workbench.memory.source', self.manager.slots[ticket.mid].source)
                    self.tracing.content(span, 'output.value', result['choices'][0]['message'], capture)
                    for key, attr in (('prompt_tokens', SpanAttributes.LLM_TOKEN_COUNT_PROMPT),
                                      ('completion_tokens', SpanAttributes.LLM_TOKEN_COUNT_COMPLETION),
                                      ('total_tokens', SpanAttributes.LLM_TOKEN_COUNT_TOTAL)):
                        if key in usage:
                            span.set_attribute(attr, usage[key])
                    for call in result['choices'][0]['message'].get('tool_calls', []):
                        with self.tracing.child('tool.call', {SpanAttributes.OPENINFERENCE_SPAN_KIND: 'TOOL'}) as child:
                            if capture and self.tracing.config.capture:
                                child.set_attribute('tool.name', call['function']['name'])
                            self.tracing.content(child, 'input.value', call['function']['arguments'], capture)
                    return result
        except BaseException as exc:
            span.set_attribute('error.type', type(exc).__name__)
            span.set_status(Status(StatusCode.ERROR))
            raise
        finally:
            phases.close()

def create_app(config, profile=None, models=None, engine=None, tracing=None):
    profile = profile or hw.detect(config.models)
    if profile.os not in ('windows', 'linux'):
        raise ValueError('The Stage 1 server supports Windows and Linux.')
    models = models if models is not None else catalog.load(config.root/'catalog.toml')
    if config.default_model is not None:
        selected = next((m for m in models if m.id == config.default_model), None)
        if selected is None or not ({'chat', 'reasoning'} & set(selected.kinds)):
            raise ValueError('The selected default model must be a catalog chat model')
        if not fetch.is_installed(selected, config.models):
            raise ValueError(f'{selected.id} is not installed; run models --add {selected.id}')
    engine = engine or LlamaEngine(config.root)
    tracing = tracing or Tracing(config)
    manager = Manager(config, profile, models, engine)
    service = Service(manager, tracing)
    jobs_worker = Jobs(config, manager, tracing)
    closed = False
    async def cleanup():
        nonlocal closed
        if closed:
            return
        closed = True
        await shutdown(app.state.workers)
        await asyncio.to_thread(jobs_worker.close)
        await asyncio.to_thread(manager.close)
        await asyncio.to_thread(tracing.close)
    @contextlib.asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            await cleanup()
    app = FastAPI(lifespan=lifespan)
    app.state.workers = set()
    app.state.ping = 10
    app.state.cleanup = cleanup
    app.state.manager, app.state.service, app.state.tracing = manager, service, tracing
    app.state.instance = uuid.uuid4().hex
    admission = asyncio.Lock()
    async def submit(mid):
        async with admission:
            return await asyncio.to_thread(manager.submit, mid)
    app.add_middleware(Auth, key=config.key)
    app.add_middleware(RequestTracing, tracing=tracing)

    @app.exception_handler(ManagerError)
    async def manager_error(request, exc):
        return error(exc.status, str(exc), 'server_error' if exc.status >= 500 else 'invalid_request_error')
    @app.exception_handler(RequestValidationError)
    async def invalid(request, exc):
        locations = ', '.join('.'.join(map(str, x['loc'])) for x in exc.errors())
        return error(400, 'Invalid request fields: '+locations)
    @app.exception_handler(ValueError)
    async def engine_input(request, exc):
        return error(400, 'Input is invalid or exceeds the model context; shorten it or choose another model.')
    @app.exception_handler(Exception)
    async def failed(request, exc):
        return error(500, 'Inference failed; inspect local status and trace error type.', 'server_error')

    def choose(mid, kind):
        choices = [m for m in models if kind in m.kinds or (kind == 'chat' and 'reasoning' in m.kinds)]
        if mid is None and kind == 'chat' and config.default_model is not None:
            mid = config.default_model
        if mid is None:
            runnable = {o.model.id for o in catalog.offered(choices, profile) if o.runnable}
            ready = [m for m in choices if m.id in runnable and fetch.is_installed(m, config.models)
                     and (manager.possible(manager.cost(m, 'cpu')[0]) or
                          (manager.budget.vram and manager.possible(manager.cost(m, 'gpu')[0])))]
            ready.sort(key=lambda m: profile.tier not in m.recommended_for)
            if not ready:
                raise ManagerError(409, f'No installed {kind} model; run models first')
            mid = ready[0].id
        if mid not in manager.models:
            raise ManagerError(404, f'Unknown model {mid}')
        if mid not in {m.id for m in choices}:
            raise ManagerError(400, f'{mid} does not support {kind}')
        return mid

    async def execute_safe(operation):
        try:
            return await asyncio.to_thread(operation)
        except ManagerError:
            raise
        except ValueError:
            raise ManagerError(400, 'Input is invalid or exceeds the model context') from None
        except Exception:
            raise ManagerError(500, 'Inference failed; inspect the trace error type') from None

    @app.get('/v1/models')
    async def list_models():
        return {'object': 'list', 'data': [{'id': o.model.id, 'object': 'model', 'owned_by': 'workbench',
            'created': 0, 'capabilities': list(o.model.kinds)} for o in catalog.offered(models, profile)
            if o.runnable and fetch.is_installed(o.model, config.models)]}
    @app.get('/wb/models')
    async def model_state():
        return {'models': [{'id': o.model.id, 'kinds': list(o.model.kinds), 'recommended': o.runnable and o.reason is None,
            'installed': fetch.is_installed(o.model, config.models), 'runnable': o.runnable, 'reason': o.reason}
            for o in catalog.offered(models, profile)]}
    @app.get('/wb/status')
    async def status():
        return {'service': 'workbench', 'instance': app.state.instance, 'tier': profile.tier,
                'physical_cores': profile.cpu_cores, 'trace_viewer': tracing.viewer_url,
                **(await asyncio.to_thread(manager.snapshot))}
    @app.get('/wb/jobs')
    async def jobs():
        return {'jobs': jobs_worker.snapshot()}
    @app.post('/wb/models', status_code=202)
    async def change_model(body: ModelAction):
        if body.action == 'cancel':
            return jobs_worker.cancel(body.job_id)
        return await asyncio.to_thread(jobs_worker.submit, body.action, body.model)
    @app.post('/v1/chat/completions')
    async def chat(body: ChatRequest, request: Request):
        mid = choose(body.model, 'chat')
        payload = body.payload()
        if payload.get('tools') and 'tools' not in manager.models[mid].kinds:
            raise ManagerError(400, f'{mid} does not support tools')
        images = any(isinstance(m.get('content'), list) and any(p['type'] == 'image_url' for p in m['content'])
                     for m in payload['messages'])
        if images and ('vision' not in manager.models[mid].kinds or
                       not any(f.role == 'mmproj' for f in manager.models[mid].files)):
            raise ManagerError(400, f'{mid} does not support images')
        ticket = await submit(mid)
        capture = request.headers.get('x-workbench-trace-content', 'true').lower() != 'false'
        worker = Worker(service, ticket, payload, capture)
        return await respond(request, worker, body.stream,
            bool(body.stream_options and body.stream_options.include_usage), app.state.ping)
    @app.post('/v1/embeddings')
    async def embeddings(body: EmbeddingRequest, request: Request):
        mid = choose(body.model, 'embeddings')
        ticket = await submit(mid)
        texts = [body.input] if isinstance(body.input, str) else body.input
        capture = request.headers.get('x-workbench-trace-content', 'true').lower() != 'false'
        def execute():
            with manager.lease(ticket, threading.Event(), lambda: tracing.child('model.load')) as handle:
                with tracing.child('embedding', {SpanAttributes.OPENINFERENCE_SPAN_KIND: 'EMBEDDING',
                                                SpanAttributes.LLM_MODEL_NAME: mid}) as span:
                    tracing.content(span, 'input.value', texts, capture)
                    vectors, count = engine.embed(handle, texts)
                    tracing.content(span, 'output.value', vectors, capture)
                    span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, count)
                    span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_TOTAL, count)
                    footprint = manager.slots[mid].footprint
                    span.set_attribute('workbench.memory.ram_bytes', footprint.ram)
                    span.set_attribute('workbench.memory.vram_bytes', footprint.vram)
                    span.set_attribute('workbench.memory.source', manager.slots[mid].source)
            if len(vectors) != len(texts) or any(not math.isfinite(x) for v in vectors for x in v):
                raise RuntimeError('Invalid embedding output')
            values = vectors if body.encoding_format == 'float' else [
                base64.b64encode(struct.pack('<'+'f'*len(v), *v)).decode() for v in vectors]
            return {'object': 'list', 'model': mid, 'data': [{'object': 'embedding', 'index': i, 'embedding': v}
                for i, v in enumerate(values)], 'usage': {'prompt_tokens': count, 'total_tokens': count}}
        return await execute_safe(execute)
    @app.post('/v1/rerank')
    async def rerank(body: RerankRequest, request: Request):
        mid = choose(body.model, 'rerank')
        ticket = await submit(mid)
        docs = [d if isinstance(d, str) else d.text for d in body.documents]
        capture = request.headers.get('x-workbench-trace-content', 'true').lower() != 'false'
        def execute():
            with manager.lease(ticket, threading.Event(), lambda: tracing.child('model.load')) as handle:
                with tracing.child('rerank', {SpanAttributes.OPENINFERENCE_SPAN_KIND: 'RERANKER',
                                             SpanAttributes.LLM_MODEL_NAME: mid}) as span:
                    tracing.content(span, 'input.value', {'query': body.query, 'documents': docs}, capture)
                    scores = engine.rank(handle, body.query, docs)
                    tracing.content(span, 'output.value', scores, capture)
                    footprint = manager.slots[mid].footprint
                    span.set_attribute('workbench.memory.ram_bytes', footprint.ram)
                    span.set_attribute('workbench.memory.vram_bytes', footprint.vram)
                    span.set_attribute('workbench.memory.source', manager.slots[mid].source)
                    count = engine.stats(handle).get('usage', {}).get('prompt_tokens')
                    if count is not None:
                        span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, count)
            if len(scores) != len(docs) or any(not math.isfinite(s) for s in scores):
                raise RuntimeError('Invalid reranker output')
            order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:body.top_n]
            return {'model': mid, 'results': [dict(index=i, relevance_score=scores[i],
                **({'document': {'text': docs[i]}} if body.return_documents else {})) for i in order]}
        return await execute_safe(execute)
    return app
