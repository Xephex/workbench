"""The model catalog: what can be downloaded, and what this machine can run."""

from __future__ import annotations

import dataclasses, pathlib, tomllib

from workbench import hw

GB = 1_000_000_000
REQUIRED = ("id", "kinds", "engine", "ram_gb", "vram_gb", "recommended_for", "license", "source", "files")


class CatalogError(ValueError):
    pass


@dataclasses.dataclass(frozen=True)
class Chunk:
    blob: str
    bytes: int


@dataclasses.dataclass(frozen=True)
class ModelFile:
    name: str
    sha256: str
    bytes: int
    repo: str
    chunks: tuple[Chunk, ...]
    role: str


@dataclasses.dataclass(frozen=True)
class Model:
    id: str
    kinds: tuple[str, ...]
    engine: str
    files: tuple[ModelFile, ...]
    ram_gb: float
    vram_gb: float
    recommended_for: tuple[str, ...]
    license: str
    source: str
    settings: dict[str, int] = dataclasses.field(default_factory=dict)

    @property
    def total_bytes(self) -> int:
        return sum(f.bytes for f in self.files)

    @property
    def folder(self) -> str:
        return self.id


@dataclasses.dataclass
class Offer:
    model: Model
    runnable: bool
    reason: str | None


def _file(mid: str, raw: dict) -> ModelFile:
    try:
        chunks = tuple(Chunk(c["blob"], int(c["bytes"])) for c in raw["chunks"])
        f = ModelFile(raw["name"], raw["sha256"], int(raw["bytes"]), raw["repo"], chunks, raw.get("role", "main"))
    except KeyError as e:
        raise CatalogError(f"model {mid}: file entry is missing {e.args[0]}") from None
    if sum(c.bytes for c in chunks) != f.bytes:
        raise CatalogError(f"model {mid}: chunks of {f.name} add up to {sum(c.bytes for c in chunks)} "
                           f"bytes, the file is {f.bytes}")
    return f


def load(path: pathlib.Path) -> list[Model]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    models = []
    for raw in data.get("model", []):
        mid = raw.get("id", "<no id>")
        for key in REQUIRED:
            if key not in raw:
                raise CatalogError(f"model {mid}: missing field {key}")
        settings = raw.get('settings', {})
        allowed = {'n_ctx', 'n_batch', 'n_ubatch', 'n_cpu_moe'}
        if not isinstance(settings, dict) or set(settings) - allowed:
            raise CatalogError(f'model {mid}: unsupported settings')
        if any(type(v) is not int or v < 0 for v in settings.values()):
            raise CatalogError(f'model {mid}: settings must be nonnegative integers')
        models.append(Model(
            id=raw["id"], kinds=tuple(raw["kinds"]), engine=raw["engine"],
            files=tuple(_file(mid, f) for f in raw["files"]),
            ram_gb=float(raw["ram_gb"]), vram_gb=float(raw["vram_gb"]),
            recommended_for=tuple(raw["recommended_for"]), license=raw["license"], source=raw["source"],
            settings=dict(settings),
        ))
    return models


def _fmt(n: float) -> str:
    return f"{n:.0f}" if abs(n - round(n)) < 0.05 else f"{n:.1f}"


def _offer(m: Model, p: hw.Profile) -> Offer:
    ram = p.ram_bytes / GB
    budget = p.gpu_budget_bytes / GB
    if m.vram_gb == 0:
        if ram >= m.ram_gb:
            return Offer(m, True, None)
        return Offer(m, False, _ram_reason(m.ram_gb, ram, p))
    if budget >= m.vram_gb and ram >= m.ram_gb:
        return Offer(m, True, None)
    if ram >= m.ram_gb + m.vram_gb:  # a GPU too small for the model must not lose the CPU option
        return Offer(m, True, "runs on CPU only; slow" if budget == 0 else
                     "too big for this GPU; runs on CPU only; slow")
    if budget and budget < m.vram_gb:
        return Offer(m, False, f"needs {_fmt(m.vram_gb)} GB of video memory; this machine has {_fmt(budget)}")
    need = m.ram_gb if budget else m.ram_gb + m.vram_gb
    return Offer(m, False, _ram_reason(need, ram, p))


def _ram_reason(need: float, have: float, p: hw.Profile) -> str:
    reason = f"needs {_fmt(need)} GB of RAM; this machine has {_fmt(have)}"
    if p.wsl and have * 2 >= need:
        reason += "; WSL can use more memory if you raise memory= in %UserProfile%\\.wslconfig"
    return reason


def offered(models: list[Model], profile: hw.Profile) -> list[Offer]:
    return [_offer(m, profile) for m in models]


def recommended(offers: list[Offer], tier: str) -> list[Model]:
    """Models that fit the memory detected on this machine.

    The static recommended_for tag is not the fit check. A runnable model
    with a reason is the slow CPU fallback: it stays visible, and it is not
    part of this default set. `tier` stays in the signature for callers.
    """
    del tier
    return [o.model for o in offers if o.runnable and o.reason is None]
