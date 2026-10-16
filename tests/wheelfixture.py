"""Build tiny pure-Python wheels, hashed lock lines and chunked copies for install tests."""

import base64, hashlib, pathlib, zipfile


def make_wheel(out_dir: pathlib.Path, name: str, version: str, script: bool = False,
               extra_files: dict[str, bytes] | None = None, requires: tuple[str, ...] = ()) -> pathlib.Path:
    mod = name.replace("-", "_")
    files = {f"{mod}/__init__.py": f"VERSION = {version!r}\n\n\ndef main():\n    print({name!r})\n".encode()}
    files.update(extra_files or {})
    info = f"{mod}-{version}.dist-info"
    meta = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n" + "".join(f"Requires-Dist: {r}\n" for r in requires)
    files[f"{info}/METADATA"] = meta.encode()
    files[f"{info}/WHEEL"] = b"Wheel-Version: 1.0\nGenerator: wheelfixture\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    if script:
        files[f"{info}/entry_points.txt"] = f"[console_scripts]\n{name} = {mod}:main\n".encode()
    record = []
    path = out_dir / f"{mod}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for arc, data in files.items():
            z.writestr(arc, data)
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            record.append(f"{arc},sha256={digest},{len(data)}")
        record.append(f"{info}/RECORD,,")
        z.writestr(f"{info}/RECORD", "\n".join(record) + "\n")
    return path


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lock_line(wheel: pathlib.Path) -> str:
    name, version = wheel.name.split("-")[:2]
    return f"{name}=={version} \\\n    --hash=sha256:{sha256(wheel)}\n"


def split(path: pathlib.Path, out_dir: pathlib.Path, size: int) -> list[pathlib.Path]:
    """Store `path` as <name>.partNNN chunks plus a <name>.sha256 sidecar (split_tool format)."""
    data = path.read_bytes()
    parts = []
    for i in range(0, len(data), size):
        part = out_dir / f"{path.name}.part{i // size:03d}"
        part.write_bytes(data[i:i + size])
        parts.append(part)
    (out_dir / f"{path.name}.sha256").write_text(f"{sha256(path)}  {path.name}\n")
    return parts
