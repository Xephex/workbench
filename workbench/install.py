"""Install this machine's platform pack offline (spec A5 steps 3-4, A8, B4).

The pack's wheels are in the repo itself (fetched with git sparse checkout).
Pip runs from its own wheel into a fresh folder, and that folder replaces
.workbench/site only after the install succeeded, so an interrupted or failed
install never breaks a working one. Pip's .exe launchers are deleted (R6).
"""

from __future__ import annotations

import hashlib, json, os, pathlib, re, shutil, site, subprocess, sys, time, zipfile

LAYERS = ("core", "llama", "tracing")
OPTIONAL = frozenset({"tracing"})
PIP_WHEEL = pathlib.Path("packs/pip/pip-26.2.1-py3-none-any.whl")
DEEPEST = 120  # longest path pip writes below the site folder (Phoenix: 113) plus margin
MAX_PATH = 260
PART = re.compile(r"^(?P<name>.+)\.part(?P<n>\d{3})$")


class InstallError(RuntimeError):
    pass


def site_dir(root: pathlib.Path) -> pathlib.Path:
    return root / ".workbench" / "site"


def _marker(root: pathlib.Path) -> pathlib.Path:
    return root / ".workbench" / "site.json"


def _load_dll(path: pathlib.Path) -> None:
    import ctypes
    ctypes.WinDLL(str(path))


def activate(root: pathlib.Path) -> bool:
    """Make the installed packages importable in this process, ahead of anything else on sys.path,
    so the pinned, checked pack wins over packages the user installed elsewhere."""
    s = site_dir(root)
    if not s.is_dir():
        return False
    before = list(sys.path)
    site.addsitedir(str(s))  # also runs the site's .pth files
    added = [p for p in sys.path if p not in before]
    sys.path[:] = added + before
    if sys.platform == "win32":
        # llama.cpp's DLLs import msvcp140.dll, which python.org's Python does not ship; the pack's
        # msvc-runtime wheel puts it in the site folder, which Windows does not search (spec B4).
        dll = s / "msvcp140.dll"
        if dll.is_file():
            _load_dll(dll)
    return True


def installed_pack(root: pathlib.Path) -> str | None:
    try:
        return json.loads(_marker(root).read_text())["pack"]
    except (OSError, ValueError, KeyError, TypeError):
        return None


def fetch_pack(root: pathlib.Path, pack: str, layers: list[str]) -> None:
    """Have git check out this pack's layer folders (a sparse clone has only top-level files)."""
    r = subprocess.run(["git", "-C", str(root), "config", "--get", "core.sparseCheckout"],
                       stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    if r.stdout.strip() != "true":
        return  # a full checkout (or a copy without git) already has every folder
    paths = ["packs/pip"] + [f"packs/{pack}/{layer}" for layer in layers]
    r = subprocess.run(["git", "-C", str(root), "sparse-checkout", "add", *paths],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        lines = r.stderr.decode(errors="replace").strip().splitlines()
        raise InstallError("git could not fetch the packages: " + (lines[-1] if lines else f"exit {r.returncode}")
                           + ". Check the network and run setup again.")


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def join_chunks(layer_dir: pathlib.Path, out_dir: pathlib.Path) -> None:
    """Rebuild every <name>.partNNN group in layer_dir as out_dir/<name>, checked against <name>.sha256."""
    groups: dict[str, dict[int, pathlib.Path]] = {}
    for p in layer_dir.iterdir():
        m = PART.match(p.name)
        if m:
            groups.setdefault(m["name"], {})[int(m["n"])] = p
    for name, parts in sorted(groups.items()):
        if sorted(parts) != list(range(len(parts))):
            raise InstallError(f"{name}: some of its pieces are missing ({len(parts)} found). "
                               "Run git pull, then setup again.")
        side = layer_dir / f"{name}.sha256"
        try:
            want = side.read_text().split()[0]
        except (OSError, IndexError):
            raise InstallError(f"{name}: its checksum file is missing. Run git pull, then setup again.") from None
        target = out_dir / name
        if target.exists() and _sha256(target) == want:
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as out:
            for i in range(len(parts)):
                with parts[i].open("rb") as f:
                    shutil.copyfileobj(f, out, 1 << 22)
        if _sha256(target) != want:
            target.unlink()
            raise InstallError(f"{name}: the joined pieces do not match its checksum. "
                               "Run git pull, then setup again.")


def check_sums(layer_dir: pathlib.Path, joined_dir: pathlib.Path) -> None:
    """Every wheel in SHA256SUMS exists (whole, or joined from pieces) and matches."""
    try:
        lines = (layer_dir / "SHA256SUMS").read_text().splitlines()
    except OSError:
        raise InstallError(f"{layer_dir.name}: SHA256SUMS is missing. Run git pull, then setup again.") from None
    for line in lines:
        if not line.strip():
            continue
        want, name = line.split(None, 1)
        name = name.strip().lstrip("*")
        path = layer_dir / name if (layer_dir / name).exists() else joined_dir / name
        if not path.exists():
            raise InstallError(f"{name} is missing from the {layer_dir.name} packages. Run git pull, then setup again.")
        if _sha256(path) != want.lower():
            raise InstallError(f"{name} does not match its checksum. Run git pull, then setup again.")


def remove_exe(site_path: pathlib.Path) -> list[pathlib.Path]:
    """Delete every .exe pip put in the site folder (launchers and any shipped tools; spec R6)."""
    exes = [p for p in site_path.rglob("*") if p.is_file() and p.suffix.lower() == ".exe"]
    for p in exes:
        p.unlink()
    return exes


def _long_paths_enabled() -> bool | None:
    if sys.platform != "win32":
        return None
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem") as k:
            return winreg.QueryValueEx(k, "LongPathsEnabled")[0] == 1
    except OSError:
        return False


def long_path_problem(site_path, enabled: bool | None) -> str | None:
    """On Windows without long paths, a deep install folder makes some package files unreachable."""
    if enabled is None or enabled:
        return None
    if len(str(site_path)) + DEEPEST < MAX_PATH:
        return None
    return (f"This folder is too deep for Windows: {site_path} is {len(str(site_path))} characters and some "
            f"package files add {DEEPEST} more, past Windows' {MAX_PATH}-character limit. Move the workbench "
            "to a shorter folder (for example C:\\workbench) and run setup again.")


def _digest(pack: str, layers: list[str], dirs: list[pathlib.Path]) -> str:
    h = hashlib.sha256(json.dumps([pack, layers]).encode())
    for d in dirs:
        for name in ("requirements.lock", "SHA256SUMS"):
            h.update((d / name).read_bytes())
    return h.hexdigest()


def _remove_tree(path: pathlib.Path) -> None:
    """Remove one of the fixed .workbench folders (site.new, site.old, wheels, tmp)."""
    if path.parent.name == ".workbench" and path.name in ("site.new", "site.old", "wheels", "tmp") and path.exists():
        shutil.rmtree(path)


def _rename(src: pathlib.Path, dst: pathlib.Path) -> None:
    """Rename, retrying briefly: on Windows antivirus or an indexer can hold new files open for a moment."""
    for attempt in range(5):
        try:
            src.replace(dst)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.5 * (attempt + 1))


def _unpacked_bytes(paths: list[pathlib.Path]) -> int:
    total = 0
    for p in paths:
        try:
            with zipfile.ZipFile(p) as z:
                total += sum(i.file_size for i in z.infolist())
        except (OSError, zipfile.BadZipFile):
            total += p.stat().st_size
    return total


def _pip_env(root: pathlib.Path) -> dict[str, str]:
    """pip's environment: no PIP_* settings and no config files at all (--isolated still reads the machine-wide
    and site config), and its temp folder on the same drive as the install, where the disk check looked."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("PIP_")}
    env["PIP_CONFIG_FILE"] = os.devnull
    tmp = str(root / ".workbench" / "tmp")
    env.update(TMPDIR=tmp, TEMP=tmp, TMP=tmp)
    return env


def _pip(root: pathlib.Path, new_site: pathlib.Path, find_links: list[pathlib.Path],
         locks: list[pathlib.Path]) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-I", str(root / PIP_WHEEL / "pip"), "install", "--isolated", "--no-index",
           "--require-hashes", "--only-binary", ":all:", "--no-deps", "--disable-pip-version-check",
           "--no-warn-script-location", "--no-compile", "--target", str(new_site)]
    for d in find_links:
        cmd += ["--find-links", str(d)]
    for lock in locks:
        cmd += ["-r", str(lock)]
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=_pip_env(root))


def install_pack(root: pathlib.Path, pack: str, with_tracing: bool = True) -> str:
    """Install `pack` into .workbench/site. Returns "installed" or "up to date"."""
    layers = [x for x in LAYERS if with_tracing or x not in OPTIONAL]
    fetch_pack(root, pack, layers)
    dirs = [root / "packs" / pack / layer for layer in layers]
    if not (root / PIP_WHEEL).is_file() or not all((d / "requirements.lock").is_file() for d in dirs):
        raise InstallError(f"Packages for {pack} are not published yet.")
    work = root / ".workbench"
    problem = long_path_problem(work / "site.new", _long_paths_enabled())
    if problem:
        raise InstallError(problem)
    digest = _digest(pack, layers, dirs)
    try:
        current = json.loads(_marker(root).read_text()).get("digest")
    except (OSError, ValueError, AttributeError):
        current = None
    if current == digest and site_dir(root).is_dir():
        return "up to date"

    try:
        _install(root, pack, layers, dirs, digest)
    except OSError as e:
        raise InstallError(f"Installing the packages failed: {e}. Close any running workbench (server, app or "
                           "command), make sure the disk has free space, then run setup again.") from None
    return "installed"


def _install(root, pack, layers, dirs, digest) -> None:
    work = root / ".workbench"
    work.mkdir(parents=True, exist_ok=True)
    joined_root, new_site, old_site, tmp = work / "wheels", work / "site.new", work / "site.old", work / "tmp"
    stored = [p for d in dirs for p in d.iterdir() if p.is_file()]
    chunks = sum(p.stat().st_size for p in stored if PART.match(p.name))
    unpacked = _unpacked_bytes([p for p in stored if p.suffix == ".whl"])
    # joined chunks, then pip's staging copy and the new site, while the old site still exists
    need = chunks + unpacked * 2 + chunks * 3
    free = shutil.disk_usage(work).free
    if free < need:
        raise InstallError(f"Not enough disk to install the packages: need about {need / 1e9:.1f} GB free, "
                           f"have {free / 1e9:.1f} GB.")
    for d in dirs:
        join_chunks(d, joined_root / d.name)
        check_sums(d, joined_root / d.name)

    _remove_tree(new_site)
    _remove_tree(tmp)
    tmp.mkdir()
    find_links = [p for d in dirs for p in (d, joined_root / d.name) if p.is_dir()]
    r = _pip(root, new_site, find_links, [d / "requirements.lock" for d in dirs])
    _remove_tree(tmp)
    if r.returncode != 0:
        _remove_tree(new_site)
        tail = "\n".join("    " + line for line in r.stdout.strip().splitlines()[-15:])
        raise InstallError(f"pip could not install the packages:\n{tail}")
    remove_exe(new_site)
    _remove_tree(old_site)
    had_site = site_dir(root).exists()
    if had_site:
        _rename(site_dir(root), old_site)
    try:
        _rename(new_site, site_dir(root))
    except OSError:
        if had_site:
            _rename(old_site, site_dir(root))
        raise
    _remove_tree(old_site)
    _remove_tree(joined_root)
    marker = _marker(root)
    tmp_marker = marker.with_name("site.json.tmp")
    tmp_marker.write_text(json.dumps({"pack": pack, "layers": layers, "digest": digest}))
    tmp_marker.replace(marker)
