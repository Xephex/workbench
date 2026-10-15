"""Fetch model files from GitHub one chunk at a time, by git blob ID.

Git is the only thing that touches the network. Each chunk is fetched into a
throwaway bare repo (partial clone, so only that object downloads), streamed
onto the end of <name>.partial, and the repo is deleted after each chunk so
extra disk use stays at about one chunk. Re-running resumes from the last whole chunk.
"""

from __future__ import annotations

import hashlib, os, pathlib, re, shutil, stat, subprocess, time
from typing import BinaryIO, Callable

from workbench import catalog

STATE_SUFFIXES = ("", ".ok", ".partial", ".fetching")


class FetchError(RuntimeError):
    pass


def repo_url(repo: str) -> str:
    base = os.environ.get("WORKBENCH_REPO_BASE")
    if base:
        return f"{base.rstrip('/')}/{repo}.git"
    return f"https://github.com/{repo}.git"


def _git(*args: str, cwd: pathlib.Path) -> subprocess.CompletedProcess:
    r = subprocess.run(["git", *args], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        lines = r.stderr.decode(errors="replace").strip().splitlines()
        raise FetchError(f"git {args[0]} failed: {lines[-1] if lines else 'exit ' + str(r.returncode)}")
    return r


def _ready(bare: pathlib.Path) -> bool:
    """A usable throwaway repo has HEAD and a remote (Ctrl+C can stop init half-way)."""
    try:
        return (bare / "HEAD").exists() and "[remote \"origin\"]" in (bare / "config").read_text()
    except OSError:
        return False


def _bare(work_dir: pathlib.Path, repo: str) -> pathlib.Path:
    bare = work_dir / (re.sub(r"[^A-Za-z0-9._-]", "_", repo) + ".git")
    if not _ready(bare):
        _reset_bare(bare)
        bare.mkdir(parents=True, exist_ok=True)
        _git("init", "-q", "--bare", cwd=bare)
        _git("config", "remote.origin.promisor", "true", cwd=bare)
        _git("config", "remote.origin.partialclonefilter", "blob:none", cwd=bare)
        _git("config", "remote.origin.url", repo_url(repo), cwd=bare)  # last: marks the repo ready
    return bare


def _clear_readonly(func, path, exc) -> None:
    """Windows refuses to delete read-only files, and git marks its objects read-only."""
    os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
    func(path)


def _reset_bare(bare: pathlib.Path) -> None:
    if not (bare.name.endswith(".git") and bare.is_dir()):
        return
    for attempt in range(5):  # antivirus can hold a new pack file open for a moment
        try:
            shutil.rmtree(bare, onexc=_clear_readonly)
            return
        except OSError:
            if attempt == 4:
                raise
            time.sleep(0.5 * (attempt + 1))


def _append_chunk(bare: pathlib.Path, chunk: catalog.Chunk, out: BinaryIO) -> None:
    _git("fetch", "-q", "--no-tags", "--filter=blob:none", "origin", chunk.blob, cwd=bare)
    start = out.tell()
    with subprocess.Popen(["git", "cat-file", "blob", chunk.blob], cwd=bare,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE) as p:
        shutil.copyfileobj(p.stdout, out, 1 << 22)
        err = p.stderr.read()
    if p.returncode != 0:
        raise FetchError(f"git cat-file failed for {chunk.blob}: {err.decode(errors='replace').strip()}")
    if out.tell() - start != chunk.bytes:
        raise FetchError(f"chunk {chunk.blob} has size {out.tell() - start}, catalog says {chunk.bytes}")


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def _done_ok(f: catalog.ModelFile, dest_dir: pathlib.Path) -> bool:
    target = dest_dir / f.name
    return (dest_dir / f"{f.name}.ok").exists() and target.exists() and target.stat().st_size == f.bytes


def _write_atomic(path: pathlib.Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _chunks_done(f: catalog.ModelFile, dest_dir: pathlib.Path) -> int:
    """Whole chunks safely on disk: what .fetching says, capped by the bytes in .partial."""
    try:
        claimed = int((dest_dir / f"{f.name}.fetching").read_text().strip() or 0)
        have = (dest_dir / f"{f.name}.partial").stat().st_size
    except (OSError, ValueError):
        return 0
    done, total = 0, 0
    for c in f.chunks[:max(claimed, 0)]:
        if total + c.bytes > have:
            break
        total += c.bytes
        done += 1
    return done


def remaining_bytes(f: catalog.ModelFile, dest_dir: pathlib.Path) -> int:
    if _done_ok(f, dest_dir):
        return 0
    return sum(c.bytes for c in f.chunks[_chunks_done(f, dest_dir):])


def remove_file(f: catalog.ModelFile, dest_dir: pathlib.Path) -> None:
    for suffix in STATE_SUFFIXES:
        (dest_dir / f"{f.name}{suffix}").unlink(missing_ok=True)
    (dest_dir / f"{f.name}.fetching.tmp").unlink(missing_ok=True)


def fetch_file(f: catalog.ModelFile, dest_dir: pathlib.Path, work_dir: pathlib.Path,
               progress: Callable[[int, int], None] | None = None) -> pathlib.Path:
    target = dest_dir / f.name
    ok = dest_dir / f"{f.name}.ok"
    if _done_ok(f, dest_dir):
        return target
    if target.exists() and target.stat().st_size == f.bytes and _sha256(target) == f.sha256:
        ok.write_text(f.sha256 + "\n")  # crashed between the rename and the marker
        return target
    dest_dir.mkdir(parents=True, exist_ok=True)
    partial = dest_dir / f"{f.name}.partial"
    state = dest_dir / f"{f.name}.fetching"
    done = _chunks_done(f, dest_dir)
    done_bytes = sum(c.bytes for c in f.chunks[:done])
    with open(partial, "r+b" if partial.exists() else "w+b") as out:
        out.truncate(done_bytes)  # drop any half-written chunk
        out.seek(done_bytes)
        for i in range(done, len(f.chunks)):
            bare = _bare(work_dir, f.repo)
            _append_chunk(bare, f.chunks[i], out)
            out.flush()
            os.fsync(out.fileno())
            _write_atomic(state, f"{i + 1}\n")
            _reset_bare(bare)  # keep extra disk use at about one chunk (spec A5)
            done_bytes += f.chunks[i].bytes
            if progress:
                progress(done_bytes, f.bytes)
    if _sha256(partial) != f.sha256:
        partial.unlink()
        state.unlink(missing_ok=True)
        raise FetchError(f"{f.name}: sha256 does not match the catalog; the download was discarded. "
                         "Run the same command again to retry.")
    partial.replace(target)
    ok.write_text(f.sha256 + "\n")
    state.unlink(missing_ok=True)
    return target


def fetch_model(m: catalog.Model, models_dir: pathlib.Path, work_dir: pathlib.Path,
                progress: Callable[[str, int, int], None] | None = None) -> pathlib.Path:
    folder = models_dir / m.folder
    for f in m.files:
        cb = (lambda done, total, name=f.name: progress(name, done, total)) if progress else None
        fetch_file(f, folder, work_dir, cb)
    return folder


def is_installed(m: catalog.Model, models_dir: pathlib.Path) -> bool:
    return all(_done_ok(f, models_dir / m.folder) for f in m.files)
