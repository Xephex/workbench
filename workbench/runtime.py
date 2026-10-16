from __future__ import annotations
import dataclasses, ipaddress, json, pathlib
from workbench.catalog import Model

GB = 1_000_000_000
CACHE_BYTES = 512 * 1024**2
LOAD_KEYS = {'n_ctx', 'n_batch', 'n_ubatch', 'n_cpu_moe'}

def is_loopback(host: str) -> bool:
    if host == 'localhost':
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False

@dataclasses.dataclass(frozen=True)
class Footprint:
    ram: int
    vram: int

@dataclasses.dataclass(frozen=True)
class Config:
    root: pathlib.Path
    models: pathlib.Path
    host: str = '127.0.0.1'
    port: int = 8080
    key: str | None = None
    capture: bool = True
    phoenix: bool = True
    default_model: str | None = None

    def __post_init__(self):
        if self.host == 'localhost':
            object.__setattr__(self, 'host', '127.0.0.1')
        if not is_loopback(self.host) and not self.key:
            raise ValueError('A non-loopback bind requires an API key; use serve --share.')
        if not 0 < self.port < 65536:
            raise ValueError('Port must be between 1 and 65535.')

def load_tuning(root: pathlib.Path, model: Model, placement: str) -> dict:
    try:
        row = json.loads((root / 'data' / 'tuning.json').read_text())[model.id][placement]
        hashes = {f.name: f.sha256 for f in model.files}
        if row['sha256'] != hashes:
            return {}
        settings = row.get('settings', {})
        footprint = row.get('footprint', {})
        if not isinstance(settings, dict) or not isinstance(footprint, dict):
            return {}
        if set(settings) - LOAD_KEYS or any(type(v) is not int or v < 0 for v in settings.values()):
            return {}
        if any(type(footprint.get(k)) is not int or footprint[k] < 0 for k in ('ram', 'vram')):
            return {}
        return row
    except (OSError, ValueError, KeyError, TypeError):
        return {}
