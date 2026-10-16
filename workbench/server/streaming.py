from __future__ import annotations
import asyncio, contextvars, json, queue, threading
import anyio
from sse_starlette.sse import EventSourceResponse, ServerSentEvent
from workbench.server.manager import ManagerError

class Worker:
    def __init__(self, service, ticket, body, capture):
        from workbench.server.app import Run
        self.service, self.ticket, self.body, self.capture = service, ticket, body, capture
        self.run = Run(service.engine)
        self.events = queue.Queue(maxsize=32)
        self.done = threading.Event()
        self.result, self.failure = None, None
        self.context = contextvars.copy_context()
        self.thread = threading.Thread(target=lambda: self.context.run(self.execute), daemon=False)
    def emit(self, chunk):
        while not self.run.event.is_set():
            try:
                self.events.put(chunk, timeout=.05)
                return
            except queue.Full:
                continue
        raise ManagerError(499, 'Request cancelled')
    def execute(self):
        try:
            self.result = self.service.chat(self.ticket, self.body, self.emit, self.run, self.capture)
        except Exception as exc:
            self.failure = exc
        finally:
            self.done.set()
    def start(self):
        self.thread.start()
    def cancel(self):
        self.run.cancel()
    def join(self):
        self.thread.join()

async def finish(worker):
    # AnyIO cancels the SSE generator on disconnect. Shield cleanup until the
    # native worker has stopped; otherwise a subsequent lease could race it.
    with anyio.CancelScope(shield=True):
        while not worker.done.is_set():
            if worker.run.event.is_set():
                # The native generator clears its abort flag at entry. Repeat
                # the signal until it observes cancellation or finishes.
                worker.cancel()
            await asyncio.sleep(.05)
        await asyncio.to_thread(worker.join)

def failure_payload(exc):
    if isinstance(exc, ManagerError):
        return {'error': {'message': str(exc), 'type': 'server_error', 'code': str(exc.status)}}
    if isinstance(exc, ValueError):
        return {'error': {'message': 'Input is invalid or exceeds the model context',
                          'type': 'invalid_request_error', 'code': '400'}}
    return {'error': {'message': 'Inference failed', 'type': 'server_error', 'code': '500'}}

async def respond(request, worker, stream, include_usage, ping=10):
    workers = request.app.state.workers
    workers.add(worker)
    worker.start()
    async def pieces():
        try:
            yield ServerSentEvent(comment=worker.run.stage)
            while not worker.done.is_set() or not worker.events.empty():
                try:
                    item = worker.events.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(.01)
                    continue
                yield {'data': json.dumps(item, ensure_ascii=False)}
            if worker.failure:
                yield {'data': json.dumps(failure_payload(worker.failure))}
            elif include_usage:
                result = worker.result
                yield {'data': json.dumps({'id': result['id'], 'object': 'chat.completion.chunk',
                    'created': result['created'], 'model': result['model'], 'choices': [], 'usage': result['usage']})}
            yield {'data': '[DONE]'}
        finally:
            if not worker.done.is_set():
                worker.cancel()
            await finish(worker)
            workers.discard(worker)
    if stream:
        class WorkerResponse(EventSourceResponse):
            async def __call__(self, scope, receive, send):
                try:
                    await super().__call__(scope, receive, send)
                finally:
                    # Headers or a blocked send can be cancelled before the
                    # generator enters (or resumes) its own cleanup block.
                    if not worker.done.is_set():
                        worker.cancel()
                    await finish(worker)
                    workers.discard(worker)
        return WorkerResponse(pieces(), ping=ping, send_timeout=30,
            ping_message_factory=lambda: ServerSentEvent(comment=worker.run.stage),
            headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})
    try:
        while not worker.done.is_set():
            # Drain the same bounded queue for a JSON caller; Service retains
            # the final answer, and backpressure must not deadlock the producer.
            while not worker.events.empty():
                worker.events.get_nowait()
            if await request.is_disconnected():
                worker.cancel()
                raise ManagerError(499, 'Request cancelled')
            await asyncio.sleep(.01)
        if worker.failure:
            failure = failure_payload(worker.failure)['error']
            raise ManagerError(int(failure['code']), failure['message']) from None
        return worker.result
    finally:
        if not worker.done.is_set():
            worker.cancel()
        await finish(worker)
        workers.discard(worker)

async def shutdown(workers):
    current = list(workers)
    for worker in current:
        worker.cancel()
    for worker in current:
        await finish(worker)
    workers.clear()
