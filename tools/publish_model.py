"""Maintainer tool: publish model files to Xephex/workbench-models (spec B5). Users never run this.

For each model in tools/publish.toml:
  1. download every file from Hugging Face at the pinned revision (resumable),
     check its size and sha256;
  2. rerankers: read the GGUF header and refuse files without the rank head;
  3. split into 95 MiB parts, add <file>.sha256, LICENSE and SOURCE;
  4. commit into a blobless bare clone with git plumbing and push in slices
     (at most 15 files, so under GitHub's 2 GB per push); never force;
  5. check the remote tree, fetch every file back with the user-side fetcher,
     and write the catalog entry.

A re-run continues from what the remote already has: files whose path and blob
id are already on main are neither committed nor pushed again.

Runs on miniRUD (Python 3.13+, stdlib only):
    printf 'protocol=https\\nhost=github.com\\n\\n' | git credential fill | sed -n 's/^password=//p' |
      ssh minirud 'cd ~/Work/scratch/workbench-publish/wb && python3 tools/publish_model.py \\
        --spec tools/publish.toml --work ../work --repo Xephex/workbench-models --catalog catalog.toml --token-stdin'

The token is read from one line of standard input and only ever handed to the
`git push` child through its environment (an inline credential helper reads it).
"""

from __future__ import annotations

import argparse, dataclasses, hashlib, http.client, math, os, pathlib, re, subprocess, sys, time, tomllib
import urllib.error, urllib.request

TOOLS = pathlib.Path(__file__).resolve().parent
ROOT = TOOLS.parent
for p in (str(TOOLS), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import catalog_write, gguf  # noqa: E402
from workbench import catalog, fetch  # noqa: E402

HF_BASE = "https://huggingface.co"
CHUNK = 99_614_720  # 95 MiB, llm-kit split_tool.py format
MAX_FILES = 15      # per commit and push: 15 x 95 MiB is about 1.5 GB, under GitHub's 2 GB push limit
TRIES = 3           # per push
DOWNLOAD_TRIES = 8
RETRY_WAIT = 10     # seconds, multiplied by the attempt number
LICENSE_FILE = ROOT / "LICENSE"
AUTHOR = ("Xephex", "Xephex@users.noreply.github.com")
TRAILER = "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
HELPER = '!f() { echo username=Xephex; echo "password=$WB_PUSH_TOKEN"; }; f'
USER_AGENT = "workbench-publish/0.1"

ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]*\Z")
NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
SOURCE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*@[0-9a-f]{40}\Z")
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")


class PublishError(RuntimeError):
    pass


class PushError(PublishError):
    pass


@dataclasses.dataclass(frozen=True)
class SpecFile:
    name: str
    role: str
    sha256: str
    bytes: int


@dataclasses.dataclass(frozen=True)
class Spec:
    id: str
    kinds: tuple[str, ...]
    ram_gb: float
    vram_gb: float
    recommended_for: tuple[str, ...]
    license: str
    source: str
    files: tuple[SpecFile, ...]
    engine: str = "llama.cpp"

    @property
    def repo(self) -> str:
        return self.source.split("@")[0]

    @property
    def rev(self) -> str:
        return self.source.split("@")[1]


@dataclasses.dataclass(frozen=True)
class Part:
    name: str
    path: pathlib.Path
    bytes: int
    blob: str


@dataclasses.dataclass
class Context:
    repo: str
    url: str
    work: pathlib.Path
    catalog: pathlib.Path
    token: str | None
    fetch_back: bool


def say(*args) -> None:
    print(time.strftime("%H:%M:%S"), *args, flush=True)


# spec ----------------------------------------------------------------------------------------------

def _safe_name(name, what: str) -> str:
    if not isinstance(name, str) or not NAME_RE.match(name) or ".." in name or name in ("LICENSE", "SOURCE"):
        raise PublishError(f"{what} {name!r} is not a plain file name")
    return name


def load_spec(path: pathlib.Path) -> list[Spec]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    out, seen = [], set()
    for raw in data.get("model", []):
        mid = raw.get("id")
        if not isinstance(mid, str) or not ID_RE.match(mid) or ".." in mid or mid in seen:
            raise PublishError(f"model id {mid!r} is not a unique plain folder name")
        seen.add(mid)
        if not SOURCE_RE.match(str(raw.get("source", ""))):
            raise PublishError(f"{mid}: source must be owner/repo@<40-hex revision>, got {raw.get('source')!r}")
        if raw.get("license") != "Apache-2.0":
            raise PublishError(f"{mid}: only Apache-2.0 models can be published (the LICENSE text shipped is Apache-2.0)")
        files, names = [], set()
        for f in raw.get("files", []):
            name = _safe_name(f.get("name"), f"{mid}: file")
            if name in names or not SHA_RE.match(str(f.get("sha256", ""))) \
                    or not isinstance(f.get("bytes"), int) or f["bytes"] <= 0:
                raise PublishError(f"{mid}: file {name} needs a unique name, a sha256 and a byte count")
            names.add(name)
            files.append(SpecFile(name, str(f.get("role", "main")), f["sha256"], f["bytes"]))
        if not files:
            raise PublishError(f"{mid}: no files")
        out.append(Spec(mid, tuple(raw["kinds"]), raw["ram_gb"], raw["vram_gb"], tuple(raw["recommended_for"]),
                        raw["license"], raw["source"], tuple(files), raw.get("engine", "llama.cpp")))
    return out


# hashing and splitting -----------------------------------------------------------------------------

def blob_of_bytes(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def blob_id(path: pathlib.Path) -> str:
    """git's blob id: sha1("blob <size>\\0" + content), the same as `git hash-object`."""
    h = hashlib.sha1(b"blob %d\0" % path.stat().st_size)
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def split(path: pathlib.Path, out_dir: pathlib.Path, have: dict[str, str] | None = None) -> list[Part]:
    """Cut `path` into CHUNK-byte parts <name>.partNNN in out_dir.

    A part whose name and blob id are in `have` (already on the remote) is not written.
    """
    if path.stat().st_size == 0:
        raise PublishError(f"{path.name} is empty")
    out_dir.mkdir(parents=True, exist_ok=True)
    have = have or {}
    parts = []
    with path.open("rb") as f:
        for i in range(math.ceil(path.stat().st_size / CHUNK)):
            data = f.read(CHUNK)
            name = f"{path.name}.part{i:03d}"
            blob = blob_of_bytes(data)
            target = out_dir / name
            if have.get(name) != blob:
                tmp = out_dir / f"{name}.tmp"
                tmp.write_bytes(data)
                tmp.replace(target)
            parts.append(Part(name, target, len(data), blob))
    return parts


def slices(paths: list, max_files: int = MAX_FILES) -> list[list]:
    return [paths[i:i + max_files] for i in range(0, len(paths), max_files)]


# download --------------------------------------------------------------------------------------------

def _get(url: str, part: pathlib.Path, have: int, total: int) -> None:
    headers = {"User-Agent": USER_AGENT}
    if have:
        headers["Range"] = f"bytes={have}-"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
        if have and r.status != 206:
            have = 0  # the server ignored the Range header: start again
        elif have and not r.headers.get("Content-Range", "").startswith(f"bytes {have}-"):
            raise PublishError(f"{url}: unexpected Content-Range {r.headers.get('Content-Range')!r}")
        step = max(total // 10, 1)
        mark = have // step
        with part.open("ab" if have else "wb") as out:
            done = have
            while block := r.read(1 << 20):
                out.write(block)
                done += len(block)
                if done // step > mark:
                    mark = done // step
                    say(f"  download {part.name.removesuffix('.download')}: {done * 100 // total}%")


def download(m: Spec, f: SpecFile, dest: pathlib.Path) -> pathlib.Path:
    """Download one file at the pinned revision into dest, resuming a .download file; check bytes and sha256."""
    dest.mkdir(parents=True, exist_ok=True)
    target, ok, part = dest / f.name, dest / f"{f.name}.ok", dest / f"{f.name}.download"
    if target.exists() and ok.exists() and target.stat().st_size == f.bytes and ok.read_text().strip() == f.sha256:
        return target
    url = f"{HF_BASE}/{m.repo}/resolve/{m.rev}/{f.name}"
    say(f"  downloading {m.repo}@{m.rev[:12]} {f.name} ({f.bytes / 1e9:.2f} GB)")
    for attempt in range(DOWNLOAD_TRIES):
        have = part.stat().st_size if part.exists() else 0
        if have > f.bytes:
            part.unlink()
            have = 0
        if have == f.bytes:
            break
        try:
            _get(url, part, have, f.bytes)
        except urllib.error.HTTPError as e:
            if 400 <= e.code < 500 and e.code not in (408, 429):
                raise PublishError(f"{f.name}: {url} answered {e.code} {e.reason}") from None
            say(f"  download of {f.name} stopped ({e}); resuming")
            time.sleep(RETRY_WAIT * (attempt + 1))
        except (urllib.error.URLError, http.client.HTTPException, OSError) as e:
            say(f"  download of {f.name} stopped ({e}); resuming")
            time.sleep(RETRY_WAIT * (attempt + 1))
    size = part.stat().st_size if part.exists() else 0
    if size != f.bytes:
        raise PublishError(f"{f.name}: got {size} bytes after {DOWNLOAD_TRIES} tries, expected {f.bytes}")
    digest = _sha256(part)
    if digest != f.sha256:
        part.unlink()
        raise PublishError(f"{f.name}: sha256 {digest} does not match the pinned {f.sha256}; the download was discarded")
    part.replace(target)
    ok.write_text(f.sha256 + "\n")
    return target


def check_reranker(path: pathlib.Path) -> None:
    try:
        with path.open("rb") as fh:
            problem = gguf.rerank_problem(*gguf.read_header(fh))
    except ValueError as e:
        problem = str(e)
    if problem:
        raise PublishError(f"{path.name} cannot be published as a reranker: {problem}")


# git ---------------------------------------------------------------------------------------------------

def _env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "WB_PUSH_TOKEN"}
    env.update(GIT_NO_LAZY_FETCH="1", GIT_TERMINAL_PROMPT="0",
               GIT_AUTHOR_NAME=AUTHOR[0], GIT_AUTHOR_EMAIL=AUTHOR[1],
               GIT_COMMITTER_NAME=AUTHOR[0], GIT_COMMITTER_EMAIL=AUTHOR[1], **extra)
    return env


def _git(cwd: pathlib.Path, *args: str, env: dict | None = None, stdin=None, input: bytes | None = None) -> str:
    kw = {"input": input} if input is not None else {"stdin": stdin}
    r = subprocess.run(["git", *args], cwd=cwd, env=env or _env(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kw)
    if r.returncode != 0:
        lines = r.stderr.decode(errors="replace").strip().splitlines()
        raise PublishError(f"git {args[0]} failed: {lines[-1] if lines else 'exit ' + str(r.returncode)}")
    return r.stdout.decode().strip()


def _ready(bare: pathlib.Path) -> bool:
    try:
        return (bare / "HEAD").exists() and '[remote "origin"]' in (bare / "config").read_text()
    except OSError:
        return False


def ensure_clone(ctx: Context) -> pathlib.Path:
    """A blobless bare clone: commits and trees only, so existing chunks are never downloaded."""
    bare = ctx.work / "git" / (re.sub(r"[^A-Za-z0-9._-]", "_", ctx.repo) + ".git")
    if not _ready(bare):
        fetch._reset_bare(bare)
        bare.parent.mkdir(parents=True, exist_ok=True)
        _git(bare.parent, "clone", "-q", "--bare", "--filter=blob:none", "--no-tags", ctx.url, bare.name)
    return bare


def remote_main(bare: pathlib.Path) -> str | None:
    out = _git(bare, "ls-remote", "origin", "refs/heads/main")
    return out.split()[0] if out else None


def refresh(bare: pathlib.Path) -> str | None:
    """Fetch the remote main (commits and trees) and return its commit, or None for an empty repo."""
    if remote_main(bare) is None:
        return None
    _git(bare, "fetch", "-q", "--no-tags", "origin", "+refs/heads/main:refs/remotes/origin/main")
    return _git(bare, "rev-parse", "refs/remotes/origin/main")


def ls_tree(bare: pathlib.Path, commit: str | None, folder: str) -> dict[str, str]:
    """path -> blob id under folder/ (names and ids only; no blob is downloaded)."""
    if commit is None:
        return {}
    out = {}
    for line in _git(bare, "ls-tree", "-r", commit, "--", f"{folder}/").splitlines():
        meta, path = line.split("\t", 1)
        out[path] = meta.split()[2]
    return out


def commit_files(bare: pathlib.Path, parent: str | None, files: dict[str, pathlib.Path], message: str) -> str:
    """Commit `files` (repo path -> local file) on top of parent without needing the parent's blobs."""
    index = bare / "wb-publish.index"
    index.unlink(missing_ok=True)
    env = _env(GIT_INDEX_FILE=str(index))
    if parent:
        _git(bare, "read-tree", parent, env=env)
    for path, src in files.items():
        with src.open("rb") as fh:
            blob = _git(bare, "hash-object", "-w", "--stdin", env=env, stdin=fh)
        _git(bare, "update-index", "--add", "--cacheinfo", f"100644,{blob},{path}", env=env)
    tree = _git(bare, "write-tree", "--missing-ok", env=env)  # plain write-tree would try to fetch the parent's blobs
    commit = _git(bare, "commit-tree", tree, *(["-p", parent] if parent else []), "-F", "-", env=env,
                  input=f"{message}\n\n{TRAILER}\n".encode())
    index.unlink(missing_ok=True)
    return commit


def push(bare: pathlib.Path, commit: str, url: str, token: str | None) -> None:
    """Fast-forward main to commit. Never forces. The token, if any, reaches git only through its environment."""
    args, env = ["git"], _env()
    if token:
        args += ["-c", "credential.helper=", "-c", f"credential.helper={HELPER}"]
        env["WB_PUSH_TOKEN"] = token
    # chunks are incompressible: storing them without a delta search saves minutes of CPU per push
    args += ["-c", "core.bigFileThreshold=16m", "-c", "pack.compression=1", "push", "-q", url, f"{commit}:refs/heads/main"]
    r = subprocess.run(args, cwd=bare, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        lines = [x for x in r.stderr.decode(errors="replace").strip().splitlines() if not x.startswith("hint:")]
        msg = " | ".join(lines[-3:]) or f"exit {r.returncode}"
        raise PushError(msg.replace(token, "***") if token else msg)


def publish_slice(bare: pathlib.Path, ctx: Context, folder: str, group: list[tuple[str, pathlib.Path, str]],
                  message: str) -> str:
    """Commit and push one slice on top of the current remote main, retrying; returns the remote commit."""
    last: Exception | None = None
    for attempt in range(TRIES):
        head = refresh(bare)
        remote = ls_tree(bare, head, folder)
        todo = {path: local for path, local, blob in group if remote.get(path) != blob}
        if not todo:
            return head  # already there (for example the push went through but the reply was lost)
        commit = commit_files(bare, head, todo, message)
        try:
            push(bare, commit, ctx.url, ctx.token)
        except PushError as e:
            last = e
            say(f"  push rejected or failed ({e}); fetching main and trying again")
            time.sleep(RETRY_WAIT * (attempt + 1))
            continue
        now = remote_main(bare)
        if now == commit:
            say(f"  pushed {commit[:12]} {message}")
            return commit
        last = PushError(f"after the push main is at {now}, expected {commit}")
    raise PushError(f"{message}: push failed {TRIES} times: {last}")


# catalog ----------------------------------------------------------------------------------------------

def source_text(m: Spec) -> str:
    lines = [f"model: {m.id}",
             f"source: https://huggingface.co/{m.repo} at revision {m.rev}",
             f"license: {m.license} (text in LICENSE; the source repository ships no license file)",
             f"changes: workbench (tools/publish_model.py) split each file into {CHUNK}-byte parts named "
             "<file>.partNNN; joining the parts in order gives back the original file.",
             "files:"]
    lines += [f"  {f.name}  {f.bytes} bytes  sha256 {f.sha256}" for f in m.files]
    return "\n".join(lines) + "\n"


def small_files(m: Spec) -> dict[str, bytes]:
    """name (inside the model folder) -> content, for the files besides the parts."""
    out = {f"{f.name}.sha256": f"{f.sha256}  {f.name}\n".encode() for f in m.files}
    out["LICENSE"] = LICENSE_FILE.read_bytes()
    out["SOURCE"] = source_text(m).encode()
    return out


def entry(m: Spec, repo: str, chunks: dict[str, list[dict]]) -> dict:
    return {"id": m.id, "kinds": list(m.kinds), "engine": m.engine, "ram_gb": m.ram_gb, "vram_gb": m.vram_gb,
            "recommended_for": list(m.recommended_for), "license": m.license, "source": m.source,
            "files": [{"name": f.name, "role": f.role, "sha256": f.sha256, "bytes": f.bytes, "repo": repo,
                       "chunks": chunks[f.name]} for f in m.files]}


def published_entry(m: Spec, ctx: Context, remote: dict[str, str]) -> dict | None:
    """The catalog entry, when the catalog already lists this model and the remote has every file of it."""
    data = tomllib.loads(ctx.catalog.read_text(encoding="utf-8"))
    old = next((x for x in data.get("model", []) if x.get("id") == m.id), None)
    if not old:
        return None
    by_name = {f.get("name"): f for f in old.get("files", [])}
    chunks = {}
    for f in m.files:
        o = by_name.get(f.name)
        if not o or (o.get("repo"), o.get("sha256"), o.get("bytes")) != (ctx.repo, f.sha256, f.bytes):
            return None
        cs = [{"blob": c["blob"], "bytes": c["bytes"]} for c in o.get("chunks", [])]
        if len(cs) != math.ceil(f.bytes / CHUNK) or any(
                remote.get(f"{m.id}/{f.name}.part{i:03d}") != c["blob"] for i, c in enumerate(cs)):
            return None
        chunks[f.name] = cs
    if any(remote.get(f"{m.id}/{name}") != blob_of_bytes(data) for name, data in small_files(m).items()):
        return None
    return entry(m, ctx.repo, chunks)


def write_catalog(ctx: Context, e: dict) -> None:
    catalog_write.upsert(ctx.catalog, e)
    catalog.load(ctx.catalog)  # the written catalog must still load


# one model --------------------------------------------------------------------------------------------

def _rmdir_if_empty(path: pathlib.Path) -> None:
    try:
        path.rmdir()
    except OSError:
        pass


def fetch_back(m: Spec, ctx: Context, e: dict) -> None:
    folder = ctx.work / "fetchback" / m.id
    for f in e["files"]:
        mf = catalog.ModelFile(f["name"], f["sha256"], f["bytes"], f["repo"],
                               tuple(catalog.Chunk(c["blob"], c["bytes"]) for c in f["chunks"]), f["role"])
        fetch.remove_file(mf, folder)  # never trust an earlier fetch-back
        fetch.fetch_file(mf, folder, ctx.work / "fetchback-git")
        say(f"fetch-back ok {mf.name}")
        fetch.remove_file(mf, folder)
    _rmdir_if_empty(folder)


def publish_one(m: Spec, ctx: Context) -> None:
    say(f"model {m.id}")
    bare = ensure_clone(ctx)
    remote = ls_tree(bare, refresh(bare), m.id)
    done = published_entry(m, ctx, remote)
    if done:
        write_catalog(ctx, done)
        say(f"already published {m.id}")
        return

    downloads = ctx.work / "downloads" / m.id
    paths = {f.name: download(m, f, downloads) for f in m.files}  # all checked before anything is committed
    if "rerank" in m.kinds:
        for f in m.files:
            if f.role == "main":
                check_reranker(paths[f.name])

    parts_dir = ctx.work / "parts" / m.id
    have = {path.removeprefix(f"{m.id}/"): blob for path, blob in remote.items()}
    parts = {f.name: split(paths[f.name], parts_dir, have) for f in m.files}
    wanted: list[tuple[str, pathlib.Path, str]] = []  # (path in the repo, local file, blob id), in commit order
    for f in m.files:
        wanted += [(f"{m.id}/{p.name}", p.path, p.blob) for p in parts[f.name]]
    for name, data in small_files(m).items():
        local = parts_dir / name
        blob = blob_of_bytes(data)
        if have.get(name) != blob:
            local.write_bytes(data)
        wanted.append((f"{m.id}/{name}", local, blob))

    todo = [w for w in wanted if remote.get(w[0]) != w[2]]
    groups = slices(todo, MAX_FILES)
    for k, group in enumerate(groups, 1):
        names = [path.rsplit("/", 1)[1] for path, _, _ in group]
        publish_slice(bare, ctx, m.id, group,
                      f"{m.id}: add {len(group)} files, {names[0]} .. {names[-1]} (slice {k} of {len(groups)})")
        for _, local, _ in group:
            local.unlink(missing_ok=True)  # on GitHub now; parts_dir plus a validated name

    head = refresh(bare)
    remote = ls_tree(bare, head, m.id)
    missing = [path for path, _, blob in wanted if remote.get(path) != blob]
    if missing:
        raise PublishError(f"{m.id}: main {head} is missing {', '.join(missing[:5])}")
    say(f"pushed {m.id}: {len(todo)} new files; main at {head}")
    for _, local, _ in wanted:
        local.unlink(missing_ok=True)
    _rmdir_if_empty(parts_dir)
    fetch._reset_bare(bare)  # drop the pushed loose objects; the next model clones afresh (commits and trees only)

    e = entry(m, ctx.repo, {n: [{"blob": p.blob, "bytes": p.bytes} for p in ps] for n, ps in parts.items()})
    if ctx.fetch_back:
        fetch_back(m, ctx, e)
    write_catalog(ctx, e)
    for f in m.files:
        (downloads / f.name).unlink(missing_ok=True)
        (downloads / f"{f.name}.ok").unlink(missing_ok=True)
    _rmdir_if_empty(downloads)
    say(f"published {m.id}: " + ", ".join(f"{n} {len(ps)} chunks" for n, ps in parts.items()))


# main -------------------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Publish model files to the workbench models repo (maintainer only).")
    ap.add_argument("--spec", type=pathlib.Path, default=TOOLS / "publish.toml")
    ap.add_argument("--work", type=pathlib.Path, required=True, help="downloads, parts and the blobless clone")
    ap.add_argument("--repo", default="Xephex/workbench-models")
    ap.add_argument("--catalog", type=pathlib.Path, default=ROOT / "catalog.toml")
    ap.add_argument("--only", nargs="+", metavar="ID")
    ap.add_argument("--no-fetch-back", action="store_true")
    ap.add_argument("--token-stdin", action="store_true", help="read the GitHub token from one line of standard input")
    a = ap.parse_args(argv)

    token = None
    if a.token_stdin:
        token = sys.stdin.readline().strip()
        if not token:
            say("error: --token-stdin given but standard input held no token")
            return 2
    try:
        models = load_spec(a.spec)
        if a.only:
            unknown = sorted(set(a.only) - {m.id for m in models})
            if unknown:
                say(f"error: not in {a.spec}: {', '.join(unknown)}")
                return 2
            models = [m for m in models if m.id in a.only]
        ctx = Context(a.repo, fetch.repo_url(a.repo), a.work.resolve(), a.catalog.resolve(), token, not a.no_fetch_back)
        for m in models:
            publish_one(m, ctx)
    except KeyboardInterrupt:
        say("interrupted; run the same command again to continue where it stopped")
        return 130
    except (PublishError, fetch.FetchError, catalog.CatalogError, OSError, ValueError) as e:
        msg = str(e)
        say(f"error: {msg.replace(token, '***') if token else msg}")
        return 1
    say(f"done: {len(models)} models")
    return 0


if __name__ == "__main__":
    sys.exit(main())
