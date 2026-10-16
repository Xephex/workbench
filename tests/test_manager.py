import threading, unittest
from concurrent.futures import ThreadPoolExecutor
from tests.server_support import Runtime, model
from workbench.server.manager import Manager, ManagerError

class ManagerTest(unittest.TestCase):
    def test_duplicate_lease_does_not_consume_the_owners_ticket(self):
        r = Runtime([model('a')])
        self.addCleanup(r.close)
        mgr = Manager(r.config, r.profile, r.models, r.engine)
        self.addCleanup(mgr.close)
        ticket = mgr.submit('a')
        cancelled = threading.Event(); cancelled.set()
        with mgr.lease(ticket, threading.Event()):
            follower = mgr.submit('a')
            with self.assertRaises(ManagerError) as error:
                with mgr.lease(ticket, cancelled):
                    self.fail('duplicate ticket ran')
            self.assertEqual(mgr.snapshot()['queued'], 1)
            self.assertEqual(error.exception.status, 409)
        with mgr.lease(follower, threading.Event()):
            pass
        with self.assertRaisesRegex(ManagerError, 'already been consumed'):
            with mgr.lease(ticket, threading.Event()):
                self.fail('consumed ticket ran')
        self.assertEqual(len(r.engine.loads), 1)

    def test_fifo_busy_lru_and_failed_load(self):
        r = Runtime([model('a', ram=3), model('b', ram=3), model('c', ram=99)], ram=10)
        self.addCleanup(r.close)
        mgr = Manager(r.config, r.profile, r.models, r.engine)
        self.addCleanup(mgr.close)
        cancel = threading.Event()
        first = mgr.submit('a')
        second, third = mgr.submit('a'), mgr.submit('a')
        order = []
        def run(ticket):
            with mgr.lease(ticket, cancel):
                order.append(ticket.number)
        with mgr.lease(first, cancel):
            with self.assertRaises(ManagerError):
                mgr.reserve_change('a')
            pool = ThreadPoolExecutor(2)
            f3 = pool.submit(run, third)
            f2 = pool.submit(run, second)
            self.assertEqual(r.engine.unloads, [])
        f2.result(2); f3.result(2); pool.shutdown()
        self.assertEqual(order, [second.number, third.number])
        with self.assertRaisesRegex(ManagerError, 'budget'):
            with mgr.lease(mgr.submit('c'), cancel):
                self.fail('oversized model admitted')
        r.engine.fail.add('b')
        with self.assertRaisesRegex(RuntimeError, 'load failed'):
            with mgr.lease(mgr.submit('b'), cancel):
                self.fail('failed model admitted')
        r.engine.fail.clear()
        with mgr.lease(mgr.submit('b'), cancel):
            self.assertIn('a', r.engine.unloads)
        self.assertEqual(mgr.snapshot()['queued'], 0)

    def test_cancelled_ticket_does_not_block_followers(self):
        r = Runtime([model('a')])
        self.addCleanup(r.close)
        mgr = Manager(r.config, r.profile, r.models, r.engine)
        self.addCleanup(mgr.close)
        first, second, third = mgr.submit('a'), mgr.submit('a'), mgr.submit('a')
        cancelled = threading.Event(); cancelled.set()
        with mgr.lease(first, threading.Event()):
            with self.assertRaises(ManagerError):
                with mgr.lease(second, cancelled):
                    self.fail('cancelled ticket ran')
        with mgr.lease(third, threading.Event()):
            pass
        self.assertEqual(len(r.engine.loads), 1)
        self.assertEqual(mgr.snapshot()['queued'], 0)

    def test_lru_uses_last_completed_lease(self):
        r = Runtime([model('a'), model('b'), model('c')], ram=10)
        self.addCleanup(r.close)
        mgr = Manager(r.config, r.profile, r.models, r.engine)
        self.addCleanup(mgr.close)
        for mid in ('a', 'b', 'a', 'c'):
            with mgr.lease(mgr.submit(mid), threading.Event()):
                pass
        self.assertEqual(r.engine.unloads, ['b'])

    def test_status_and_admission_do_not_block_loading(self):
        r = Runtime([model('a')])
        self.addCleanup(r.close)
        mgr = Manager(r.config, r.profile, r.models, r.engine)
        self.addCleanup(mgr.close)
        started, release = threading.Event(), threading.Event()
        original = r.engine.load
        def delayed(entry, settings):
            started.set()
            release.wait(2)
            return original(entry, settings)
        r.engine.load = delayed
        def use():
            with mgr.lease(mgr.submit('a'), threading.Event()):
                pass
        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(use)
            try:
                self.assertTrue(started.wait(1))
                self.assertEqual(mgr.snapshot()['loading'], ['a'])
                ticket = mgr.submit('a')
                cancelled = threading.Event(); cancelled.set()
                with self.assertRaises(ManagerError):
                    with mgr.lease(ticket, cancelled):
                        self.fail('cancelled ticket ran')
            finally:
                release.set()
            future.result(2)

    def test_verified_backend_fallback_reloads_with_cpu_budget(self):
        r = Runtime([model('chat', ram=2, vram=6)], vram=8)
        self.addCleanup(r.close)
        original = r.engine.load
        def fallback(entry, settings):
            handle = original(entry, settings)
            handle.placement = 'cpu'
            return handle
        r.engine.load = fallback
        mgr = Manager(r.config, r.profile, r.models, r.engine)
        self.addCleanup(mgr.close)
        with mgr.lease(mgr.submit('chat'), threading.Event()) as handle:
            self.assertEqual(handle.placement, 'cpu')
            self.assertGreater(mgr.slots['chat'].footprint.ram, 8_000_000_000)
        self.assertEqual([settings['placement'] for mid, settings in r.engine.loads], ['gpu', 'cpu'])
        self.assertEqual(r.engine.unloads, ['chat'])

    def test_small_model_cpu_fallback(self):
        r = Runtime([model('chat', ram=2, vram=6), model('embed', ('embeddings',), ram=2, vram=2)], vram=8)
        self.addCleanup(r.close)
        mgr = Manager(r.config, r.profile, r.models, r.engine)
        self.addCleanup(mgr.close)
        with mgr.lease(mgr.submit('chat'), threading.Event()):
            with mgr.lease(mgr.submit('embed'), threading.Event()) as h:
                self.assertEqual(h.placement, 'cpu')
        self.assertEqual(r.engine.unloads, [])
