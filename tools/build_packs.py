"""Maintainer tool: build a platform pack (spec A8, B4). Users never run this.

    python tools/build_packs.py lock   --pack linux-x64-vulkan [--wheels DIR]
    python tools/build_packs.py fetch  --pack linux-x64-vulkan [--wheels DIR]
    python tools/build_packs.py verify --pack linux-x64-vulkan
    python tools/build_packs.py all    --pack linux-x64-vulkan [--wheels DIR]

lock:   uv resolves core + llama + tracing together for the target platform
        (environment markers evaluated for the target, not this machine), then
        each layer alone, constrained to that full resolve, with hashes:
        packs/<pack>/<layer>/requirements.lock.
fetch:  pip (from packs/pip) downloads exactly the locked, hash-checked wheels
        for the target platform. Each wheel is stored once, in the first layer
        that locks it; files over 95 MiB are split; SHA256SUMS is written.
verify: chunks join and match, every locked wheel is stored, and uv installs
        every layer offline for the target platform.

--wheels DIR adds a folder of our own llama.cpp wheels (built in CI). JamePeng's
CUDA wheels are downloaded from the pinned release and checked against the
sha256 digests GitHub lists for each asset.
"""

from __future__ import annotations

import argparse, dataclasses, hashlib, json, os, pathlib, re, shutil, subprocess, sys, tempfile
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from workbench import install  # noqa: E402

PY = "3.14"
SPLIT_BYTES = 95 * 1024 * 1024
LAYERS = install.LAYERS
JAMEPENG = "JamePeng/llama-cpp-python"


class BuildError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class Pin:
    name: str
    version: str
    hashes: list[str]


@dataclasses.dataclass(frozen=True)
class Target:
    uv_platform: str
    pip_platforms: tuple[str, ...]
    jamepeng_tag: str | None
    windows: bool


def _linux(arch: str, glibc_minor: int) -> tuple[str, ...]:
    tags = [f"manylinux_2_{m}_{arch}" for m in range(glibc_minor, 16, -1)]
    return (*tags, f"manylinux2014_{arch}", f"linux_{arch}")


PACKS = {
    "win-x64-cuda12": Target("x86_64-pc-windows-msvc", ("win_amd64",), "v0.4.2-cu128-win-20261003", True),
    "win-x64-vulkan": Target("x86_64-pc-windows-msvc", ("win_amd64",), None, True),
    "linux-x64-cuda13": Target("x86_64-manylinux_2_34", _linux("x86_64", 34), "v0.4.2-cu131-linux-20261003", False),
    "linux-x64-vulkan": Target("x86_64-manylinux_2_28", _linux("x86_64", 28), None, False),
    "linux-arm64-cpu": Target("aarch64-manylinux_2_28", _linux("aarch64", 28), None, False),
}


@dataclasses.dataclass
class Ctx:
    pack: str
    target: Target
    req_dir: pathlib.Path
    out_dir: pathlib.Path
    pip_wheel: pathlib.Path
    find_links: list[pathlib.Path]
    index: bool = True
    staging: pathlib.Path | None = None


def say(msg: str = "") -> None:
    print(msg, flush=True)


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


# ---------- pure helpers ----------

def read_lock(text: str) -> dict[str, Pin]:
    """Parse a uv/pip hashed requirements file into {normalised name: Pin}."""
    pins: dict[str, Pin] = {}
    current = None
    for raw in text.replace("\\\n", " ").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        m = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==(\S+)(.*)$", line)
        if not m:
            raise BuildError(f"cannot parse lock line {raw.strip()!r}")
        hashes = re.findall(r"--hash=sha256:([0-9a-f]{64})", m[3])
        if not hashes:
            raise BuildError(f"{m[1]}=={m[2]} has no --hash; locks must be generated with hashes")
        current = Pin(norm(m[1]), m[2], hashes)
        pins[current.name] = current
    return pins


def assign_layers(locks: dict[str, dict[str, Pin]]) -> dict[str, str]:
    """Each package is stored in the first layer (core, llama, tracing) that locks it."""
    out: dict[str, str] = {}
    for layer in LAYERS:
        for name in locks.get(layer, {}):
            out.setdefault(name, layer)
    return out


def split_big(directory: pathlib.Path) -> None:
    """Split every wheel over SPLIT_BYTES into .partNNN chunks plus a .sha256 sidecar."""
    for w in sorted(directory.glob("*.whl")):
        size = w.stat().st_size
        if size <= SPLIT_BYTES:
            continue
        digest = sha256(w)
        with w.open("rb") as f:
            n = 0
            while block := f.read(SPLIT_BYTES):
                (directory / f"{w.name}.part{n:03d}").write_bytes(block)
                n += 1
        (directory / f"{w.name}.sha256").write_text(f"{digest}  {w.name}\n")
        w.unlink()
        say(f"  split {w.name}: {n} chunks")


def write_sums(directory: pathlib.Path, digests: dict[str, str]) -> None:
    (directory / "SHA256SUMS").write_text("".join(f"{digests[n]}  {n}\n" for n in sorted(digests)))


def stored_wheels(directory: pathlib.Path) -> set[str]:
    names = {p.name for p in directory.glob("*.whl")}
    names |= {p.name[:-len(".sha256")] for p in directory.glob("*.whl.sha256")}
    return names


def wheel_project(filename: str) -> str:
    return norm(filename.split("-", 1)[0])


# ---------- tools ----------

def _uv(args: list[str]) -> subprocess.CompletedProcess:
    exe = shutil.which("uv")
    if not exe:
        raise BuildError("uv is not installed (https://docs.astral.sh/uv/)")
    env = dict(os.environ, UV_NO_CONFIG="1")
    env.pop("VIRTUAL_ENV", None)
    return subprocess.run([exe, *args], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def _run(cmd: list[str], what: str) -> None:
    if cmd[0] == "uv":
        r = _uv(cmd[1:])
    else:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.returncode != 0:
        raise BuildError(f"{what} failed:\n{r.stdout.strip()[-3000:]}")


def _links(c: Ctx) -> list[str]:
    return [a for d in c.find_links for a in ("--find-links", str(d))] + ([] if c.index else ["--no-index"])


def jamepeng_wheels(tag: str, dest: pathlib.Path) -> pathlib.Path:
    """Download the cp314 wheels of one JamePeng release into dest, checked against GitHub's digests."""
    dest.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(f"https://api.github.com/repos/{JAMEPENG}/releases/tags/{tag}",
                                 headers={"Accept": "application/json", "User-Agent": "workbench-build-packs"})
    with urllib.request.urlopen(req, timeout=60) as r:
        assets = json.load(r)["assets"]
    for a in assets:
        if "-cp314-" not in a["name"] or not a["name"].endswith(".whl"):
            continue
        want = (a.get("digest") or "").removeprefix("sha256:")
        target = dest / a["name"]
        if target.exists() and sha256(target) == want:
            continue
        say(f"  download {a['name']} ({a['size'] / 1e6:.0f} MB)")
        tmp = target.with_name(target.name + ".download")
        with urllib.request.urlopen(urllib.request.Request(a["browser_download_url"],
                                                           headers={"User-Agent": "workbench-build-packs"}),
                                    timeout=120) as r, tmp.open("wb") as out:
            shutil.copyfileobj(r, out, 1 << 22)
        if not want or sha256(tmp) != want:
            tmp.unlink()
            raise BuildError(f"{a['name']}: sha256 does not match the release digest")
        tmp.replace(target)
    return dest


# ---------- commands ----------

def lock(c: Ctx) -> None:
    say(f"lock {c.pack} for {c.target.uv_platform}, Python {PY}")
    with tempfile.TemporaryDirectory() as tmp:
        full = pathlib.Path(tmp) / "full.lock"
        common = ["--python-platform", c.target.uv_platform, "--python-version", PY, "--no-build",
                  "--generate-hashes", "--no-header", "--quiet", *_links(c)]
        ins = [c.req_dir / "core.in", c.req_dir / f"llama-{c.pack}.in", c.req_dir / "tracing.in"]
        _run(["uv", "pip", "compile", *map(str, ins), "--overrides", str(c.req_dir / "overrides.txt"),
              *common, "-o", str(full)], f"resolving {c.pack}")
        for layer, src in zip(LAYERS, ins):
            out = c.out_dir / layer / "requirements.lock"
            out.parent.mkdir(parents=True, exist_ok=True)
            _run(["uv", "pip", "compile", str(src), "-c", str(full), "--overrides",
                  str(c.req_dir / "overrides.txt"), *common, "-o", str(out)], f"resolving {c.pack}/{layer}")
            say(f"  {layer}: {len(read_lock(out.read_text()))} packages")


def _locks(c: Ctx) -> dict[str, dict[str, Pin]]:
    return {layer: read_lock((c.out_dir / layer / "requirements.lock").read_text()) for layer in LAYERS}


def fetch(c: Ctx) -> None:
    say(f"fetch {c.pack}")
    locks = _locks(c)
    home = assign_layers(locks)
    staging = c.staging or c.out_dir.parent.parent / ".workbench" / "build" / c.pack
    staging.mkdir(parents=True, exist_ok=True)
    full = staging / "all.lock"
    merged = {name: pin for layer in LAYERS for name, pin in locks[layer].items()}
    full.write_text("".join(f"{p.name}=={p.version} " + " ".join(f"--hash=sha256:{h}" for h in p.hashes) + "\n"
                            for p in merged.values()))
    plats = [a for p in c.target.pip_platforms for a in ("--platform", p)]
    _run([sys.executable, "-I", str(c.pip_wheel / "pip"), "download", "--isolated", "--no-deps",
          "--only-binary", ":all:", "--require-hashes", "--implementation", "cp", "--python-version", PY,
          "--abi", "cp314", "--abi", "abi3", "--abi", "none", *plats, *_links(c),
          "--disable-pip-version-check", "-d", str(staging), "-r", str(full)], f"downloading {c.pack}")
    by_project: dict[str, pathlib.Path] = {}
    for w in staging.glob("*.whl"):
        name = wheel_project(w.name)
        if name in merged:
            by_project[name] = w
    missing = sorted(set(merged) - set(by_project))
    if missing:
        raise BuildError(f"no wheel downloaded for: {', '.join(missing)}")
    for layer in LAYERS:
        d = c.out_dir / layer
        d.mkdir(parents=True, exist_ok=True)
        digests = {}
        for name, w in sorted(by_project.items()):
            if home[name] != layer:
                continue
            digests[w.name] = sha256(w)
            if w.name not in stored_wheels(d):
                shutil.copy2(w, d / w.name)
        split_big(d)
        write_sums(d, digests)
        stale = sorted(stored_wheels(d) - set(digests))
        if stale:
            say(f"  {layer}: no longer locked (remove by hand, with their .partNNN chunks): {', '.join(stale)}")
        say(f"  {layer}: {len(digests)} wheels")


def verify(c: Ctx) -> None:
    say(f"verify {c.pack}")
    locks = _locks(c)
    problems = []
    if c.target.windows:
        for layer, pins in locks.items():
            problems += [f"{layer}: {n} is locked for Windows (spec B6 leaves monty out)" for n in pins if "monty" in n]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        links = []
        for layer in LAYERS:
            d = c.out_dir / layer
            try:
                install.join_chunks(d, tmp / "joined" / layer)
                install.check_sums(d, tmp / "joined" / layer)
            except install.InstallError as e:
                problems.append(f"{layer}: {e}")
            stored = {wheel_project(n) for n in stored_wheels(d)}
            home = assign_layers(locks)
            for name in locks[layer]:
                if home[name] == layer and name not in stored:
                    problems.append(f"{layer}: {name} is locked but no wheel is stored")
            for name in stored - set(locks[layer]):
                problems.append(f"{layer}: a wheel for {name} is stored but not locked")
            links += [d, tmp / "joined" / layer]
        if problems:
            raise BuildError("\n".join(problems))
        cmd = ["uv", "pip", "install", "--target", str(tmp / "site"), "--python-platform", c.target.uv_platform,
               "--python-version", PY, "--no-index", "--no-deps", "--require-hashes", "--no-build", "--no-cache",
               "--python", sys.executable]
        for d in links:
            if d.is_dir():
                cmd += ["--find-links", str(d)]
        for layer in LAYERS:
            cmd += ["-r", str(c.out_dir / layer / "requirements.lock")]
        _run(cmd, f"offline install of {c.pack} for {c.target.uv_platform}")
    n = sum(len(p) for p in locks.values())
    say(f"verify OK: {c.pack} installs offline ({len(assign_layers(locks))} packages, {n} lock entries)")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("lock", "fetch", "verify", "all"))
    ap.add_argument("--pack", required=True, choices=sorted(PACKS))
    ap.add_argument("--wheels", type=pathlib.Path, help="folder with our own llama.cpp wheels (from CI)")
    a = ap.parse_args(argv)
    t = PACKS[a.pack]
    links = []
    if t.jamepeng_tag:
        links.append(jamepeng_wheels(t.jamepeng_tag, ROOT / ".workbench" / "build" / "jamepeng" / t.jamepeng_tag))
    if a.wheels:
        links.append(a.wheels.resolve())
    c = Ctx(a.pack, t, ROOT / "packs" / "requirements", ROOT / "packs" / a.pack, ROOT / install.PIP_WHEEL, links)
    try:
        if a.cmd in ("lock", "all"):
            lock(c)
        if a.cmd in ("fetch", "all"):
            fetch(c)
        if a.cmd in ("verify", "all"):
            verify(c)
    except BuildError as e:
        say(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
