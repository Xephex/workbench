from __future__ import annotations
import collections, contextlib, dataclasses, threading, time
from workbench import fetch
from workbench.runtime import GB, CACHE_BYTES, Footprint, load_tuning

class ManagerError(RuntimeError):
    def __init__(self, status, message):
        self.status = status
        super().__init__(message)

@dataclasses.dataclass(frozen=True)
class Ticket:
    number: int
    mid: str

@dataclasses.dataclass
class Slot:
    handle: object
    footprint: Footprint
    lock: object = dataclasses.field(default_factory=threading.Lock)
    active: bool = False
    used: float = 0
    source: str = 'catalog estimate'

class Manager:
    def __init__(self, config, profile, models, engine):
        self.config, self.profile, self.engine = config, profile, engine
        self.models = {m.id: m for m in models}
        self.budget = Footprint(max(0, profile.ram_bytes-4*GB), max(0, profile.gpu_budget_bytes-800_000_000))
        self.cv = threading.Condition()
        self.slots = {}
        self.loading = {}
        self.loader = threading.Lock()
        self.queues = collections.defaultdict(collections.deque)
        self.claimed = set()
        self.blocked = set()
        self.cpu_only = set()
        self.serial = 0
        self.closed = False

    def submit(self, mid):
        with self.cv:
            if self.closed:
                raise ManagerError(503, 'Server is shutting down')
            if mid not in self.models:
                raise ManagerError(404, f'Unknown model {mid}')
            if mid in self.blocked:
                raise ManagerError(409, 'Model installation is changing')
            if not fetch.is_installed(self.models[mid], self.config.models):
                raise ManagerError(409, f'{mid} is not installed; run models --add {mid}')
            self.serial += 1
            ticket = Ticket(self.serial, mid)
            self.queues[mid].append(ticket)
            self.cv.notify_all()
            return ticket

    def cache_capacity(self, m, placement, row):
        if {'embeddings', 'rerank'} & set(m.kinds):
            return 0
        base = row['footprint']['ram'] if row else int((m.ram_gb + (m.vram_gb if placement == 'cpu' else 0))*GB)
        return min(CACHE_BYTES, max(0, (self.budget.ram-base)//3))

    def cost(self, m, placement):
        row = load_tuning(self.config.root, m, placement)
        # Reserve resident cache plus two transient copies made by save_state.
        cache = 3*self.cache_capacity(m, placement, row)
        if row:
            f = row['footprint']
            return Footprint(f['ram']+cache, f['vram']), row
        gpu = int((m.vram_gb*GB) if m.vram_gb > 0 else m.total_bytes*1.2) if placement == 'gpu' else 0
        ram = int((m.ram_gb + (m.vram_gb if placement == 'cpu' else 0))*GB)
        return Footprint(ram+cache, gpu), {}

    def fits(self, cost):
        return (sum(s.footprint.ram for s in self.slots.values())+sum(f.ram for f in self.loading.values())+cost.ram <= self.budget.ram and
                sum(s.footprint.vram for s in self.slots.values())+sum(f.vram for f in self.loading.values())+cost.vram <= self.budget.vram)

    def possible(self, cost):
        return cost.ram <= self.budget.ram and cost.vram <= self.budget.vram

    def evict(self, mid):
        slot = self.slots[mid]
        self.engine.unload(slot.handle)
        del self.slots[mid]

    def admit(self, m, loading):
        placement = 'gpu' if self.budget.vram and m.id not in self.cpu_only else 'cpu'
        cost, row = self.cost(m, placement)
        small = bool({'embeddings', 'rerank'} & set(m.kinds))
        if placement == 'gpu' and (not self.possible(cost) or (small and not self.fits(cost))):
            placement = 'cpu'
            cost, row = self.cost(m, placement)
        if not self.possible(cost):
            raise ManagerError(503, f'{m.id} exceeds the RAM/VRAM budget')
        while not self.fits(cost):
            candidates = [(s.used, mid) for mid, s in self.slots.items() if not s.active and not self.queues[mid]]
            if not candidates:
                return None
            self.evict(min(candidates)[1])
        settings = {**m.settings, **row.get('settings', {}), 'models_dir': self.config.models,
                    'placement': placement, 'n_threads': self.profile.cpu_cores,
                    'cache_bytes': self.cache_capacity(m, placement, row), 'think': self.profile.tier != 'cpu'}
        self.loading[m.id] = cost
        self.cv.release()
        try:
            with self.loader, loading():
                handle = self.engine.load(m, settings)
        finally:
            self.cv.acquire()
            del self.loading[m.id]
            self.cv.notify_all()
        if handle.placement != placement:
            self.engine.unload(handle)
            self.cpu_only.add(m.id)
            # Reload with CPU-specific settings, footprint and cache allowance.
            # A GPU tuning record cannot describe the silently fallen-back load.
            return self.admit(m, loading)
        slot = Slot(handle, cost, source='tuned' if row else 'catalog estimate')
        self.slots[m.id] = slot
        return slot

    @contextlib.contextmanager
    def lease(self, ticket, cancel, loading=contextlib.nullcontext):
        slot = None
        acquired = False
        owned = False
        failed = True
        try:
            with self.cv:
                queue = self.queues[ticket.mid]
                if ticket not in queue or ticket in self.claimed:
                    raise ManagerError(409, 'Ticket has already been consumed')
                # Claim before waiting or releasing the condition for native loading.
                self.claimed.add(ticket)
                owned = True
                while True:
                    if cancel.is_set() or self.closed:
                        raise ManagerError(499, 'Request cancelled')
                    queue = self.queues[ticket.mid]
                    if not queue or ticket not in queue:
                        raise ManagerError(409, 'Ticket has already been consumed')
                    current = self.slots.get(ticket.mid)
                    if queue[0] == ticket and (current is None or not current.active):
                        slot = current or self.admit(self.models[ticket.mid], loading)
                        if slot is not None:
                            if cancel.is_set() or self.closed:
                                self.evict(ticket.mid)
                                raise ManagerError(499, 'Request cancelled')
                            slot.active = True
                            slot.lock.acquire()
                            acquired = True
                            break
                    self.cv.wait(.05)
            yield slot.handle
            failed = False
        finally:
            with self.cv:
                queue = self.queues[ticket.mid]
                if owned:
                    queue.remove(ticket)
                    self.claimed.remove(ticket)
                if acquired:
                    slot.active = False
                    slot.used = time.monotonic()
                    slot.lock.release()
                    if cancel.is_set() or failed:
                        self.evict(ticket.mid)
                self.cv.notify_all()

    def reserve_change(self, mid):
        with self.cv:
            if mid not in self.models:
                raise ManagerError(404, f'Unknown model {mid}')
            if mid in self.blocked or self.queues[mid] or (mid in self.slots and self.slots[mid].active):
                raise ManagerError(409, 'Model is busy')
            if mid in self.slots:
                self.evict(mid)
            self.blocked.add(mid)

    def release_change(self, mid):
        with self.cv:
            self.blocked.discard(mid)
            self.cv.notify_all()

    def snapshot(self):
        with self.cv:
            return {'budget': dataclasses.asdict(self.budget),
                'queued': max(0, sum(map(len, self.queues.values()))-sum(s.active for s in self.slots.values())-len(self.loading)),
                'loading': list(self.loading), 'loaded': [{'id': mid, 'busy': s.active, 'footprint': dataclasses.asdict(s.footprint),
                    'footprint_source': s.source, **self.engine.stats(s.handle)} for mid, s in self.slots.items()]}

    def close(self):
        with self.cv:
            self.closed = True
            self.cv.notify_all()
            while self.loading or any(s.active for s in self.slots.values()):
                self.cv.wait(.05)
            for mid in list(self.slots):
                self.evict(mid)
