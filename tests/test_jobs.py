import json, threading, time, unittest
from tests.server_support import Runtime
from workbench.server.manager import Manager, ManagerError
from workbench.server.jobs import Jobs
from workbench.trace import Tracing

class JobsTest(unittest.TestCase):
    def test_reservation_held_until_terminal_result_is_saved(self):
        r = Runtime()
        self.addCleanup(r.close)
        mgr = Manager(r.config, r.profile, r.models, r.engine)
        self.addCleanup(mgr.close)
        trace = Tracing(r.config, [])
        self.addCleanup(trace.close)
        saving, release = threading.Event(), threading.Event()
        jobs = Jobs(r.config, mgr, trace, lambda *a, **kw: None)
        self.addCleanup(jobs.close)
        self.addCleanup(release.set)
        save = jobs.save
        def paused_save(row):
            if row['state'] == 'succeeded':
                saving.set()
                release.wait(2)
            save(row)
        jobs.save = paused_save
        row = jobs.submit('install', 'chat')
        self.assertTrue(saving.wait(1))
        with self.assertRaises(ManagerError) as raised:
            mgr.submit('chat')
        self.assertEqual(raised.exception.status, 409)
        release.set()
        jobs.close()
        self.assertNotIn('chat', mgr.blocked)
        saved = json.loads((r.root/'data/jobs'/f"{row['id']}.json").read_text())
        self.assertEqual(saved['state'], 'succeeded')
        self.assertTrue(saved['result']['installed'])

    def test_cancel_and_restart_preserve_results_and_partial(self):
        r = Runtime()
        self.addCleanup(r.close)
        mgr = Manager(r.config, r.profile, r.models, r.engine)
        self.addCleanup(mgr.close)
        trace = Tracing(r.config, [])
        self.addCleanup(trace.close)
        started, release = threading.Event(), threading.Event()
        def download(model, models, work, progress):
            (models/model.folder/'model.gguf.partial').write_bytes(b'partial')
            started.set()
            release.wait(2)
            progress('model.gguf', 1, 2)
            return models/model.folder
        jobs = Jobs(r.config, mgr, trace, download)
        row = jobs.submit('install', 'chat')
        self.assertTrue(started.wait(1))
        with self.assertRaises(ManagerError):
            mgr.submit('chat')
        jobs.cancel(row['id'])
        release.set(); jobs.close()
        self.assertEqual(jobs.snapshot()[0]['state'], 'cancelled')
        self.assertTrue((r.config.models/'chat/model.gguf.partial').exists())
        path = r.root/'data/jobs/crashed.json'
        path.write_text(json.dumps({'id': 'crashed', 'state': 'running', 'model': 'chat', 'action': 'install'}))
        restarted = Jobs(r.config, mgr, trace, download)
        self.addCleanup(restarted.close)
        states = {j['id']: j['state'] for j in restarted.snapshot()}
        self.assertEqual(states[row['id']], 'cancelled')
        self.assertEqual(states['crashed'], 'interrupted')

    def test_remove_and_failed_download_release_model(self):
        r = Runtime()
        self.addCleanup(r.close)
        mgr = Manager(r.config, r.profile, r.models, r.engine)
        self.addCleanup(mgr.close)
        trace = Tracing(r.config, [])
        self.addCleanup(trace.close)
        def broken(*args, **kwargs):
            raise OSError('disk full')
        jobs = Jobs(r.config, mgr, trace, broken)
        bad = jobs.submit('install', 'chat')
        deadline = time.monotonic()+2
        while jobs.snapshot()[0]['state'] in ('queued', 'running'):
            self.assertLess(time.monotonic(), deadline)
            time.sleep(.01)
        jobs.close()
        self.assertEqual(jobs.snapshot()[0]['state'], 'failed')
        again = Jobs(r.config, mgr, trace, broken)
        row = again.submit('remove', 'chat')
        deadline = time.monotonic()+2
        while next(j for j in again.snapshot() if j['id'] == row['id'])['state'] in ('queued', 'running'):
            self.assertLess(time.monotonic(), deadline)
            time.sleep(.01)
        again.close()
        self.assertFalse((r.config.models/'chat/model.gguf').exists())
        self.assertNotIn('chat', mgr.blocked)
