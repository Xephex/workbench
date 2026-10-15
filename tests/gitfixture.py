"""Build a local git repo with chunked files, served over file:// like GitHub."""

import hashlib, os, pathlib, subprocess

from workbench import catalog


def git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def make_repo(base: pathlib.Path, repo: str, files: dict[str, bytes], chunk: int) -> list[catalog.ModelFile]:
    """Create base/<repo>.git with each file split into `chunk`-byte .partNNN pieces.

    Returns catalog.ModelFile entries whose blob IDs point at the committed chunks.
    """
    work = base / "work" / repo.replace("/", "_")
    work.mkdir(parents=True)
    git("init", "-q", "-b", "main", cwd=work)
    for name, data in files.items():
        for i in range(0, max(len(data), 1), chunk):
            (work / f"{name}.part{i // chunk:03d}").write_bytes(data[i:i + chunk])
    git("add", ".", cwd=work)
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "chunks", cwd=work)
    bare = base / f"{repo}.git"
    bare.parent.mkdir(parents=True, exist_ok=True)
    git("clone", "-q", "--bare", str(work), str(bare), cwd=base)
    git("config", "uploadpack.allowFilter", "true", cwd=bare)
    git("config", "uploadpack.allowAnySHA1InWant", "true", cwd=bare)
    out = []
    for name, data in files.items():
        chunks = []
        for i in range(0, max(len(data), 1), chunk):
            part = f"{name}.part{i // chunk:03d}"
            blob = subprocess.run(["git", "rev-parse", f"HEAD:{part}"], cwd=work, check=True,
                                  capture_output=True, text=True).stdout.strip()
            chunks.append(catalog.Chunk(blob, len(data[i:i + chunk])))
        out.append(catalog.ModelFile(name, hashlib.sha256(data).hexdigest(), len(data), repo,
                                     tuple(chunks), "main"))
    return out
