from __future__ import annotations
import contextvars, json, pathlib, shutil, threading, time, uuid
from concurrent.futures import ThreadPoolExecutor
from pydantic import model_validator
from typing import Literal
from workbench import fetch
from workbench.server.manager import ManagerError
from workbench.server.schemas import Input

class ModelAction(Input):
    action: Literal['install', 'remove', 'cancel']
    model: str | None = None
    job_id: str | None = None
    @model_validator(mode='after')
    def valid(self):
        if self.action == 'cancel' and (not self.job_id or self.model):
            raise ValueError('cancel requires only job_id')
        if self.action != 'cancel' and (not self.model or self.job_id):
            raise ValueError('install/remove require only model')
        return self

class Cancelled(Exception):
    pass

class Jobs:
    def __init__(self, config, manager, tracing, download=fetch.fetch_model):
        self.config, self.manager, self.tracing, self.download = config, manager, tracing, download
        self.folder = config.root/'data/jobs'
        self.folder.mkdir(parents=True, exist_ok=True)
        self.lock, self.rows, self.events = threading.RLock(), {}, {}
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='workbench-jobs')
        self.closed = False
        for path in sorted(self.folder.glob('*.json')):
            row = json.loads(path.read_text())
            self.rows[row['id']] = row
            if row['state'] in ('queued', 'running'):
                row['state'] = 'interrupted'
                row['error'] = 'Server stopped; repeat the model action to resume.'
                self.save(row)

    def save(self, row):
        path = self.folder/(row['id']+'.json')
        temp = path.with_suffix('.json.tmp')
        temp.write_text(json.dumps(row, indent=2), encoding='utf-8')
        temp.replace(path)

    def submit(self, action, mid):
        if action not in ('install', 'remove'):
            raise ManagerError(400, 'Unsupported model action')
        with self.lock:
            if self.closed:
                raise ManagerError(503, 'Job worker is shutting down')
            self.manager.reserve_change(mid)
            try:
                model = self.manager.models[mid]
                if action == 'install':
                    self.config.models.mkdir(parents=True, exist_ok=True)
                    need = sum(fetch.remaining_bytes(f, self.config.models/model.folder) for f in model.files)
                    scratch = max((c.bytes for f in model.files for c in f.chunks), default=0)
                    if need and need*1.02+scratch > shutil.disk_usage(self.config.models).free:
                        raise ManagerError(409, 'Not enough disk for model and one scratch chunk')
                ident = uuid.uuid4().hex
                row = {'id': ident, 'action': action, 'model': mid, 'state': 'queued',
                       'created': time.time(), 'progress': {}, 'result': None, 'error': None}
                self.rows[ident], self.events[ident] = row, threading.Event()
                self.save(row)
                context = contextvars.copy_context()
                self.pool.submit(context.run, self.execute, ident)
                return dict(row)
            except BaseException:
                self.manager.release_change(mid)
                raise

    def execute(self, ident):
        with self.lock:
            row, event = self.rows[ident], self.events[ident]
        model = self.manager.models[row['model']]
        def progress(name, done, total):
            if event.is_set():
                raise Cancelled()
            with self.lock:
                row['progress'] = {'file': name, 'done': done, 'total': total}
                self.save(row)
        try:
            try:
                with self.lock:
                    row['state'] = 'running'
                    self.save(row)
                if event.is_set():
                    raise Cancelled()
                with self.tracing.child('job.'+row['action'], {'workbench.model': model.id}):
                    if row['action'] == 'install':
                        self.download(model, self.config.models, self.config.root/'.workbench/fetch', progress=progress)
                    else:
                        folder = self.config.models/model.folder
                        for file in model.files:
                            if event.is_set():
                                raise Cancelled()
                            fetch.remove_file(file, folder)
                        if folder.is_dir() and not any(folder.iterdir()):
                            folder.rmdir()
                state, failure = 'succeeded', None
            except Cancelled:
                state, failure = 'cancelled', None
            except Exception as exc:
                state, failure = 'failed', type(exc).__name__+': model action failed; retry to resume'
            with self.lock:
                row.update(state=state, error=failure, finished=time.time(),
                           result={'installed': fetch.is_installed(model, self.config.models)})
                try:
                    self.save(row)
                except OSError:
                    row.update(state='failed', error='Unable to persist the job result; check free disk and permissions')
        finally:
            self.manager.release_change(model.id)

    def cancel(self, ident):
        with self.lock:
            if ident not in self.rows:
                raise ManagerError(404, 'Unknown job')
            if self.rows[ident]['state'] in ('queued', 'running'):
                self.events[ident].set()
            return dict(self.rows[ident])

    def snapshot(self):
        with self.lock:
            return json.loads(json.dumps(list(self.rows.values())))

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            for event in self.events.values():
                event.set()
        self.pool.shutdown(wait=True)
